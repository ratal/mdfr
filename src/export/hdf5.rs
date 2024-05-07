//! Exporting mdf to hdf5 files.
use anyhow::{Context, Error, Result};
use arrow::array::Array;
use hdf5::{
    file::File,
    types::{VarLenArray, VarLenUnicode},
    Dataset, DatasetBuilder, H5Type,
};
use log::info;
use ndarray::{Array as NdArray, IxDyn};

use crate::mdfreader::Mdf;
use crate::{
    data_holder::channel_data::ChannelData,
    mdfinfo::{
        mdfinfo3::{Cg3, Cn3, MdfInfo3},
        mdfinfo4::{Cg4, Cn4, Dg4, MdfInfo4},
        MdfInfo,
    },
};

/// writes mdf into hdf5 file
pub fn export_to_hdf5(mdf: &Mdf, file_name: &str) -> Result<(), Error> {
    let mut file = File::create(file_name).context("failed creating hdf5 file")?;
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdf4_metadata(&mut file, mdfinfo4).context("failed creating metadata for mdf4")?;
            mdfinfo4.dg.iter().try_for_each(
                |(_dg_block_position, dg): (&i64, &Dg4)| -> Result<(), Error> {
                    dg.cg.iter().try_for_each(
                        |(_rec_id, cg): (&u64, &Cg4)| -> Result<(), Error> {
                            mdf4_cg_to_hdf5(&mut file, mdfinfo4, cg)
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
                    mdf3_cg_to_hdf5(&mut file, mdfinfo3, cg)
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
) -> Result<(), Error> {
    let mut file = File::create(file_name).context("failed creating hdf5 file")?;
    match &mdf.mdf_info {
        MdfInfo::V4(mdfinfo4) => {
            mdf4_metadata(&mut file, mdfinfo4).context("failed creating metadata for mdf4")?;
            if let Some((_master, dg_pos, (_cg_pos, rec_id), (_cn_pos, _rec_pos))) =
                mdfinfo4.get_channel_id(channel_name)
            {
                if let Some(dg) = mdfinfo4.dg.get(dg_pos) {
                    if let Some(cg) = dg.cg.get(rec_id) {
                        mdf4_cg_to_hdf5(&mut file, mdfinfo4, cg).context(
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
                        mdf3_cg_to_hdf5(&mut file, mdfinfo3, cg).context(
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
pub fn mdf4_cg_to_hdf5(file: &mut File, mdfinfo4: &MdfInfo4, cg: &Cg4) -> Result<()> {
    let master_channel = cg
        .master_channel_name
        .clone()
        .unwrap_or(format!("no_master_channel_{}", cg.block_position));
    let group = file
        .create_group(&master_channel)
        .with_context(|| format!("failed creating group {:?}", master_channel))?;
    cg.cn
        .iter()
        .try_for_each(|(_rec_pos, cn): (&i32, &Cn4)| -> Result<(), Error> {
            if !cn.data.is_empty() {
                let builder = group.new_dataset_builder();
                let dataset = convert_channel_data_into_ndarray(builder, &cn.data, &cn.unique_name)
                    .with_context(|| {
                        format!("failed writing channel {} dataset", cn.unique_name)
                    })?;
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
                create_scalar_attr(&dataset, "sync_type", &cn.block.cn_sync_type).with_context(
                    || {
                        format!(
                            "failed writing sync type attribute for channel {}",
                            cn.unique_name
                        )
                    },
                )?;
            }
            Ok(())
        })
        .context("failed extracting data")?;
    Ok(())
}

/// create a hdf5 file for the given CG3 block
pub fn mdf3_cg_to_hdf5(file: &mut File, mdfinfo3: &MdfInfo3, cg: &Cg3) -> Result<()> {
    let master_channel = cg
        .master_channel_name
        .clone()
        .unwrap_or(format!("no_master_channel_{}", cg.block_position));
    let group = file
        .create_group(&master_channel)
        .with_context(|| format!("failed creating group {:?}", cg.master_channel_name))?;
    cg.cn
        .iter()
        .try_for_each(|(_rec_pos, cn): (&u32, &Cn3)| -> Result<(), Error> {
            if !cn.data.is_empty() {
                let builder = group.new_dataset_builder();
                let dataset = convert_channel_data_into_ndarray(builder, &cn.data, &cn.unique_name)
                    .with_context(|| {
                        format!("failed writing channel {} dataset", cn.unique_name)
                    })?;
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
                create_scalar_attr(&dataset, "sync_type", &cn.block1.cn_type).with_context(
                    || {
                        format!(
                            "failed writing sync type attribute for channel {}",
                            cn.unique_name
                        )
                    },
                )?;
            }
            Ok(())
        })
        .context("failed extracting data")?;
    Ok(())
}

fn mdf4_metadata(file: &mut File, mdfinfo4: &MdfInfo4) -> Result<()> {
    create_scalar_group_attr::<File, u64>(
        &file,
        "start_time_ns",
        &mdfinfo4.hd_block.hd_start_time_ns,
    )
    .with_context(|| {
        format!(
            "failed writing attribute start_time_ns with value {}",
            mdfinfo4.hd_block.hd_start_time_ns
        )
    })?;
    let comments = mdfinfo4
        .sharable
        .get_hd_comments(mdfinfo4.hd_block.hd_md_comment);
    comments
        .iter()
        .try_for_each(|(name, comment)| -> Result<(), Error> {
            create_str_group_attr::<File>(&file, name, comment).with_context(|| {
                format!("failed writing attribute {} with value {}", name, comment,)
            })?;
            Ok(())
        })
        .context("failed writing hd comments")?;
    Ok(())
}

fn mdf3_metadata(file: &mut File, mdfinfo3: &MdfInfo3) -> Result<()> {
    let time = mdfinfo3.hd_block.hd_start_time_ns.unwrap_or(0);
    create_scalar_group_attr::<File, u64>(&file, "start_time_ns", &time)
        .with_context(|| format!("failed writing attribute start_time_ns with value {}", time))?;
    create_str_group_attr::<File>(&file, "Author", &mdfinfo3.hd_block.hd_author).with_context(
        || {
            format!(
                "failed writing attribute author {}",
                mdfinfo3.hd_block.hd_author
            )
        },
    )?;
    create_str_group_attr::<File>(&file, "Project", &mdfinfo3.hd_block.hd_project).with_context(
        || {
            format!(
                "failed writing attribute project {}",
                mdfinfo3.hd_block.hd_project
            )
        },
    )?;
    create_str_group_attr::<File>(&file, "Subject", &mdfinfo3.hd_block.hd_subject).with_context(
        || {
            format!(
                "failed writing attribute subject {}",
                mdfinfo3.hd_block.hd_subject
            )
        },
    )?;
    create_str_group_attr::<File>(&file, "Organization", &mdfinfo3.hd_block.hd_organization)
        .with_context(|| {
            format!(
                "failed writing attribute organization {}",
                mdfinfo3.hd_block.hd_organization
            )
        })?;
    Ok(())
}

fn create_str_attr<T>(location: &T, name: &str, value: &str) -> Result<()>
where
    T: std::ops::Deref<Target = hdf5::Container>,
{
    let attr = location
        .new_attr::<VarLenUnicode>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    let value: VarLenUnicode = value.parse().unwrap_or("None".parse().unwrap());
    attr.write_scalar(&value)
        .with_context(|| format!("failed writing attribute {} with value {}", name, value))
}

fn create_str_group_attr<T>(location: &T, name: &str, value: &str) -> Result<()>
where
    T: std::ops::Deref<Target = hdf5::Group>,
{
    let attr = location
        .new_attr::<VarLenUnicode>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    let value: VarLenUnicode = value.parse().unwrap_or("None".parse().unwrap());
    attr.write_scalar(&value)
        .with_context(|| format!("failed writing attribute {} with value {}", name, value))
}

fn create_scalar_attr<T, N>(location: &T, name: &str, value: &N) -> Result<()>
where
    T: std::ops::Deref<Target = hdf5::Container>,
    N: H5Type + std::fmt::Debug,
{
    let attr = location
        .new_attr::<N>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    attr.write_scalar(value)
        .with_context(|| format!("failed writing attribute {} with value {:?}", name, value))
}

fn create_scalar_group_attr<T, N>(location: &T, name: &str, value: &N) -> Result<()>
where
    T: std::ops::Deref<Target = hdf5::Group>,
    N: H5Type + std::fmt::Debug,
{
    let attr = location
        .new_attr::<N>()
        .create(name)
        .with_context(|| format!("failed creating attribute {}", name))?;
    attr.write_scalar(value)
        .with_context(|| format!("failed writing attribute {} with value {:?}", name, value))
}

fn convert_channel_data_into_ndarray(
    builder: DatasetBuilder,
    data: &ChannelData,
    name: &str,
) -> Result<Dataset, Error> {
    match data {
        ChannelData::Int8(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::UInt8(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Int16(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::UInt16(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Int32(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::UInt32(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Float32(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Int64(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::UInt64(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Float64(data) => Ok(builder.with_data(data.values_slice()).create(name)?),
        ChannelData::Complex32(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData f32 complex into ndarray")?,
            )
            .create(name)?),
        ChannelData::Complex64(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData f64 complex into ndarray")?,
            )
            .create(name)?),
        ChannelData::Utf8(data) => {
            let string_vect: Vec<VarLenUnicode> = data
                .finish_cloned()
                .iter()
                .map(|x| match x {
                    Some(x) => match x.parse() {
                        Ok(s) => s,
                        Err(e) => {
                            info!("failed parsing value {:?}, error {}", x, e);
                            "null".parse().unwrap()
                        }
                    },
                    None => "null".parse().unwrap(),
                })
                .collect();
            Ok(builder.with_data(&string_vect).create(name)?)
        }
        ChannelData::VariableSizeByteArray(data) => {
            let bytes_vect: Vec<VarLenArray<u8>> = data
                .finish_cloned()
                .iter()
                .map(|x| match x {
                    Some(x) => VarLenArray::from_slice(x),
                    None => VarLenArray::from_slice(&[0]),
                })
                .collect();
            Ok(builder.with_data(&bytes_vect).create(name)?)
        }
        ChannelData::FixedSizeByteArray(data) => {
            let fixed_binary = data.finish_cloned();
            let value_length = fixed_binary.value_length();
            let vector = fixed_binary.value_data().to_vec();
            let shape = vec![fixed_binary.len(), value_length as usize];
            Ok(builder
                .with_data(
                    &NdArray::from_shape_vec(IxDyn(&shape), vector)
                        .context("Failed reshaping byteArray arrow into ndarray")?,
                )
                .create(name)?)
        }
        ChannelData::ArrayDInt8(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd i8 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDUInt8(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd u8 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDInt16(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd i16 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDUInt16(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd u16 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDInt32(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd i32 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDUInt32(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd u32 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDFloat32(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd f32 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDInt64(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd i64 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDUInt64(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd u64 into ndarray")?,
            )
            .create(name)?),
        ChannelData::ArrayDFloat64(data) => Ok(builder
            .with_data(
                &data
                    .to_ndarray()
                    .context("Failed converting channelData nd f64 into ndarray")?,
            )
            .create(name)?),
    }
}
