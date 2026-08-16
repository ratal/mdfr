//! Exporting mdf to hdf5 files.
use anyhow::{Context, Error, Result};
use arrow::{
    array::{Array, ArrowPrimitiveType},
    datatypes::{
        Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
        UInt32Type, UInt64Type,
    },
};
use log::info;
use rust_hdf5::{
    FilterPipeline, H5File, H5Group, H5Type,
    dataset::H5Dataset,
    types::{Complex32, Complex64, VarLenUnicode},
};

use crate::mdfreader::Mdf;
use crate::{
    data_holder::channel_data::ChannelData,
    mdfinfo::{
        MdfInfo,
        mdfinfo3::{Cg3, Cn3, MdfInfo3},
        mdfinfo4::{Cg4, Cn4, Dg4, MdfInfo4},
    },
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// writes mdf into hdf5 file
pub fn export_to_hdf5(mdf: &Mdf, file_name: &str, compression: Option<&str>) -> Result<(), Error> {
    let mut file = H5File::create(file_name).context("failed creating hdf5 file")?;
    let hdf5_compression = hdf5_compression_from_string(compression);
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdf4_metadata(&mut file, mdfinfo4).context("failed creating metadata for mdf4")?;
            mdfinfo4.dg.iter().try_for_each(
                |(_dg_block_position, dg): (&i64, &Dg4)| -> Result<(), Error> {
                    dg.cg.iter().try_for_each(
                        |(_rec_id, cg): (&u64, &Cg4)| -> Result<(), Error> {
                            mdf4_cg_to_hdf5(&mut file, mdfinfo4, cg, &hdf5_compression)
                                .context("failed converting Channel Group 4 to hdf5")?;
                            Ok(())
                        },
                    )?;
                    Ok(())
                },
            )?;
        }
        MdfInfo::V3(mdfinfo3) => {
            mdf3_metadata(&mut file, mdfinfo3).context("failed creating metadata for mdf3")?;
            for (_dg_block_position, dg) in mdfinfo3.dg.iter() {
                for (_rec_id, cg) in dg.cg.iter() {
                    mdf3_cg_to_hdf5(&mut file, mdfinfo3, cg, &hdf5_compression)
                        .context("failed converting Channel Group 3 to hdf5")?;
                }
            }
        }
    }
    Ok(())
}

/// writes a dataframe or channel group defined by a given channel into a hdf5 file
pub fn export_dataframe_to_hdf5(
    mdf: &Mdf,
    channel_name: &str,
    file_name: &str,
    compression: Option<&str>,
) -> Result<(), Error> {
    let mut file = H5File::create(file_name).context("failed creating hdf5 file")?;
    let hdf5_compression = hdf5_compression_from_string(compression);
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdf4_metadata(&mut file, mdfinfo4).context("failed creating metadata for mdf4")?;
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, _rec_pos))) =
                mdfinfo4.get_channel_id(channel_name)
            {
                if let Some(dg) = mdfinfo4.dg.get(dg_pos) {
                    if let Some(cg) = dg.cg.get(rec_id) {
                        mdf4_cg_to_hdf5(&mut file, mdfinfo4, cg, &hdf5_compression).context(
                            "failed converting Channel Group 4 to hdf5 containing channel",
                        )?;
                    }
                }
            }
        }
        MdfInfo::V3(mdfinfo3) => {
            mdf3_metadata(&mut file, mdfinfo3).context("failed creating metadata for mdf3")?;
            if let Some((_master, dg_pos, (_cg_pos, rec_id), _cn_pos)) =
                mdfinfo3.get_channel_id(channel_name)
            {
                if let Some(dg) = mdfinfo3.dg.get(dg_pos) {
                    if let Some(cg) = dg.cg.get(rec_id) {
                        mdf3_cg_to_hdf5(&mut file, mdfinfo3, cg, &hdf5_compression).context(
                            "failed converting Channel Group 3 to hdf5 containing channel",
                        )?;
                    }
                }
            }
        }
    }
    file.close().context("failed closing hdf5 file")
}

/// create a hdf5 file for the given CG4 block
#[inline]
pub fn mdf4_cg_to_hdf5(
    file: &mut H5File,
    mdfinfo4: &MdfInfo4,
    cg: &Cg4,
    compression: &Hdf5Compression,
) -> Result<()> {
    let master_channel = cg
        .master_channel_name
        .clone()
        .unwrap_or(format!("no_master_channel_{}", cg.block_position));
    let group = file
        .create_group(&master_channel)
        .with_context(|| format!("failed creating group {:?}", master_channel))?;
    cg.cn
        .par_iter()
        .try_for_each(|(_rec_pos, cn): (&i32, &Cn4)| -> Result<(), Error> {
            if !cn.data.is_empty() {
                mdf4_cn_to_hdf5(mdfinfo4, cn, compression, &group)
                    .context("failed writing dataset")?;
            }
            Ok(())
        })
        .context("failed extracting data")?;
    Ok(())
}

#[inline]
pub fn mdf4_cn_to_hdf5(
    mdfinfo4: &MdfInfo4,
    cn: &Cn4,
    compression: &Hdf5Compression,
    group: &H5Group,
) -> Result<(), Error> {
    let dataset =
        convert_channel_data_into_h5dataset(compression, group, &cn.data, &cn.unique_name)
            .with_context(|| format!("failed writing channel {} dataset", cn.unique_name))?;
    // writing channel unit if existing
    if let Ok(Some(unit)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_unit) {
        if !unit.is_empty() {
            create_str_attr(&dataset, "unit", &unit).with_context(|| {
                format!(
                    "failed writing unit attribute for channel {}",
                    cn.unique_name
                )
            })?;
        }
    }
    // writing channel description if existing
    if let Ok(Some(desc)) = mdfinfo4.sharable.get_tx(cn.block.cn_md_comment) {
        if !desc.is_empty() {
            create_str_attr(&dataset, "description", &desc).with_context(|| {
                format!(
                    "failed writing description attribute for channel {}",
                    cn.unique_name
                )
            })?;
        }
    };
    // sync type
    create_scalar_attr(&dataset, "sync_type", &cn.block.cn_sync_type).with_context(|| {
        format!(
            "failed writing sync type attribute for channel {}",
            cn.unique_name
        )
    })?;
    Ok(())
}

/// create a hdf5 file for the given CG3 block
#[inline]
pub fn mdf3_cg_to_hdf5(
    file: &mut H5File,
    mdfinfo3: &MdfInfo3,
    cg: &Cg3,
    compression: &Hdf5Compression,
) -> Result<()> {
    let master_channel = cg
        .master_channel_name
        .clone()
        .unwrap_or(format!("no_master_channel_{}", cg.block_position));
    let group = file
        .create_group(&master_channel)
        .with_context(|| format!("failed creating group {:?}", cg.master_channel_name))?;
    cg.cn
        .par_iter()
        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
            if !cn.data.is_empty() {
                mdf3_cn_to_hdf5(mdfinfo3, cn, compression, &group)
                    .context("failed writing dataset")?;
            }
            Ok(())
        })
        .context("failed extracting data")?;
    Ok(())
}

#[inline]
fn mdf3_cn_to_hdf5(
    mdfinfo3: &MdfInfo3,
    cn: &Cn3,
    compression: &Hdf5Compression,
    group: &H5Group,
) -> Result<(), Error> {
    let dataset =
        convert_channel_data_into_h5dataset(compression, group, &cn.data, &cn.unique_name)
            .with_context(|| format!("failed writing channel {} dataset", cn.unique_name))?;
    // writing channel unit if existing
    if let Some(unit) = mdfinfo3._get_unit(&cn.block1.cn_cc_conversion) {
        if !unit.is_empty() {
            create_str_attr(&dataset, "unit", &unit).with_context(|| {
                format!(
                    "failed writing unit attribute for channel {}",
                    cn.unique_name
                )
            })?;
        }
    }
    // writing channel description if existing
    create_str_attr(&dataset, "description", &cn.description).with_context(|| {
        format!(
            "failed writing description attribute for channel {}",
            cn.unique_name
        )
    })?;
    // sync type
    create_scalar_attr(&dataset, "sync_type", &cn.block1.cn_type).with_context(|| {
        format!(
            "failed writing sync type attribute for channel {}",
            cn.unique_name
        )
    })?;
    Ok(())
}

fn mdf4_metadata(file: &mut H5File, mdfinfo4: &MdfInfo4) -> Result<()> {
    file.set_attr_numeric::<u64>("start_time_ns", &mdfinfo4.hd_block.hd_start_time_ns)
        .with_context(|| {
            format!(
                "failed writing attribute start_time_ns with value {}",
                mdfinfo4.hd_block.hd_start_time_ns
            )
        })?;
    if let Some(hd) = mdfinfo4
        .sharable
        .get_hd_comments(mdfinfo4.hd_block.hd_md_comment)
    {
        if let Some(tx) = &hd.tx {
            file.set_attr_string("TX", tx)
                .context("failed writing HD TX attribute")?;
        }
        if let Some(ts) = &hd.time_source {
            file.set_attr_string("time_source", ts)
                .context("failed writing HD time_source attribute")?;
        }
        for (name, value) in &hd.constants {
            file.set_attr_string(name, value)
                .with_context(|| format!("failed writing HD constant {name} with value {value}"))?;
        }
        for (name, value) in &hd.common_properties {
            file.set_attr_string(name, &format!("{value}"))
                .with_context(|| format!("failed writing HD property {name}"))?;
        }
    }
    Ok(())
}

fn mdf3_metadata(file: &mut H5File, mdfinfo3: &MdfInfo3) -> Result<()> {
    let time = mdfinfo3.hd_block.hd_start_time_ns.unwrap_or(0);
    file.set_attr_numeric::<u64>("start_time_ns", &time)
        .with_context(|| format!("failed writing attribute start_time_ns with value {}", time))?;
    file.set_attr_string("Author", &mdfinfo3.hd_block.hd_author)
        .with_context(|| {
            format!(
                "failed writing attribute author {}",
                mdfinfo3.hd_block.hd_author
            )
        })?;
    file.set_attr_string("Project", &mdfinfo3.hd_block.hd_project)
        .with_context(|| {
            format!(
                "failed writing attribute project {}",
                mdfinfo3.hd_block.hd_project
            )
        })?;
    file.set_attr_string("Subject", &mdfinfo3.hd_block.hd_subject)
        .with_context(|| {
            format!(
                "failed writing attribute subject {}",
                mdfinfo3.hd_block.hd_subject
            )
        })?;
    file.set_attr_string("Organization", &mdfinfo3.hd_block.hd_organization)
        .with_context(|| {
            format!(
                "failed writing attribute organization {}",
                mdfinfo3.hd_block.hd_organization
            )
        })?;
    Ok(())
}

#[inline]
fn create_str_attr(location: &H5Dataset, name: &str, value: &str) -> Result<()> {
    let attr = location
        .new_attr::<VarLenUnicode>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    let value: VarLenUnicode = value.parse().unwrap_or("None".parse().unwrap());
    attr.write_scalar(&value)
        .with_context(|| format!("failed writing attribute {} with value {}", name, value))
}

#[inline]
fn create_scalar_attr<N>(location: &H5Dataset, name: &str, value: &N) -> Result<()>
where
    N: H5Type + std::fmt::Debug,
{
    let attr = location
        .new_attr::<N>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    attr.write_numeric(value)
        .with_context(|| format!("failed writing attribute {} with value {:?}", name, value))
}

#[inline]
fn convert_channel_data_into_h5dataset(
    compression: &Hdf5Compression,
    group: &H5Group,
    cdata: &ChannelData,
    name: &str,
) -> Result<H5Dataset, Error> {
    let ndim = &cdata.shape().0;
    match cdata {
        ChannelData::Int8(data) => Ok(create_dataset::<i8, Int8Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::UInt8(data) => Ok(create_dataset::<u8, UInt8Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Int16(data) => Ok(create_dataset::<i16, Int16Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::UInt16(data) => Ok(create_dataset::<u16, UInt16Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Int32(data) => Ok(create_dataset::<i32, Int32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::UInt32(data) => Ok(create_dataset::<u32, UInt32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Float32(data) => Ok(create_dataset::<f32, Float32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Int64(data) => Ok(create_dataset::<i64, Int64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::UInt64(data) => Ok(create_dataset::<u64, UInt64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Float64(data) => Ok(create_dataset::<f64, Float64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Complex32(data) => {
            let filter = get_filter(compression);
            let dataset = group
                .new_dataset::<Complex32>()
                .filter_pipeline(filter)
                .shape(ndim)
                .create(name)?;
            dataset.write_raw(data.values_slice())?;
            Ok(dataset)
        }
        ChannelData::Complex64(data) => {
            let filter = get_filter(compression);
            let dataset = group
                .new_dataset::<Complex64>()
                .filter_pipeline(filter)
                .shape(ndim)
                .create(name)?;
            dataset.write_raw(data.values_slice())?;
            Ok(dataset)
        }
        ChannelData::Utf8(data) => {
            let filter = get_filter(compression);
            let array = data.finish_cloned();
            let strings: Vec<&str> = array.iter().map(|opt| opt.unwrap_or("")).collect();
            let chunk_size = strings.len().max(1);
            Ok(group.write_vlen_strings_compressed(name, &strings, chunk_size, filter)?)
        }
        ChannelData::VariableSizeByteArray(data) => {
            // Convert LargeBinaryBuilder -> LargeBinaryArray -> Vec<&[u8]>
            let array = data.finish_cloned();
            let bytes: Vec<&[u8]> = array.iter().map(|opt| opt.unwrap_or(&[])).collect();
            Ok(group.write_vlen_bytes(name, &bytes)?)
        }
        ChannelData::FixedSizeByteArray(data) => {
            let fixed_binary = data.finish_cloned();
            let value_length = fixed_binary.value_length();
            let vector = fixed_binary.value_data().to_vec();
            let shape = vec![fixed_binary.len(), value_length as usize];
            let filter = get_filter(compression);
            let dataset = group
                .new_dataset::<u8>()
                .filter_pipeline(filter)
                .shape(shape)
                .create(name)?;
            dataset.write_raw(&vector)?;
            Ok(dataset)
        }
        ChannelData::ArrayDInt8(data) => Ok(create_dataset::<i8, Int8Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDUInt8(data) => Ok(create_dataset::<u8, UInt8Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDInt16(data) => Ok(create_dataset::<i16, Int16Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDUInt16(data) => Ok(create_dataset::<u16, UInt16Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDInt32(data) => Ok(create_dataset::<i32, Int32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDUInt32(data) => Ok(create_dataset::<u32, UInt32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDFloat32(data) => Ok(create_dataset::<f32, Float32Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDInt64(data) => Ok(create_dataset::<i64, Int64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDUInt64(data) => Ok(create_dataset::<u64, UInt64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::ArrayDFloat64(data) => Ok(create_dataset::<f64, Float64Type>(
            group,
            name,
            ndim,
            compression,
            data.values_slice(),
        )?),
        ChannelData::Union(data) => {
            info!("Union channel {} data exported as raw bytes", name);
            let filter = get_filter(compression);
            let union_byte_count = cdata.byte_count() as usize;
            let shape = vec![data.len(), union_byte_count];
            let dataset = group
                .new_dataset::<u8>()
                .filter_pipeline(filter)
                .shape(shape)
                .create(name)?;
            let bytes = cdata.to_bytes()?;
            dataset.write_raw(&bytes)?;
            Ok(dataset)
        }
    }
}

/// create HDF5 dataset from group, data and compression
#[inline]
fn create_dataset<T: H5Type, A>(
    group: &H5Group,
    name: &str,
    ndim: &[usize],
    compression: &Hdf5Compression,
    data: &[A::Native],
) -> Result<H5Dataset, Error>
where
    A: ArrowPrimitiveType,
    A: ArrowPrimitiveType<Native = T>,
{
    let filter = get_filter(compression);
    let dataset = group
        .new_dataset::<T>()
        .filter_pipeline(filter)
        .shape(ndim)
        .create(name)?;
    dataset.write_raw(data)?;
    Ok(dataset)
}

/// create compression filter for HDF5 dataset
#[inline]
fn get_filter(compression: &Hdf5Compression) -> FilterPipeline {
    match compression {
        Hdf5Compression::Deflate(level) => FilterPipeline::deflate(*level),
        Hdf5Compression::Zstd(level) => FilterPipeline::zstd(*level),
        Hdf5Compression::Lz4 => FilterPipeline::lz4(),
        Hdf5Compression::Uncompressed => FilterPipeline::none(),
    }
}

/// converts a clap compression string into a Hdf5Compression enum
#[inline]
pub fn hdf5_compression_from_string(compression_option: Option<&str>) -> Hdf5Compression {
    match compression_option {
        Some(option) => match option {
            "deflate" => Hdf5Compression::Deflate(8),
            "zstd" => Hdf5Compression::Zstd(22),
            "lz4" => Hdf5Compression::Lz4,
            _ => Hdf5Compression::Uncompressed,
        },
        None => Hdf5Compression::Uncompressed,
    }
}

pub enum Hdf5Compression {
    Deflate(u32),
    Zstd(u32),
    Lz4,
    Uncompressed,
}
