//! Exporting mdf to Parquet files.
use crate::{mdfinfo::mdfinfo4::MdfInfo4, mdfreader::channel_data::ChannelData};
use parquet2::{
    compression::CompressionOptions,
    encoding::Encoding,
    error::Error,
    metadata::{Descriptor, SchemaDescriptor},
    page::CompressedPage,
    page::{DataPage, DataPageHeader, DataPageHeaderV1, EncodedPage},
    schema::{
        types::{
            FieldInfo, IntegerType, ParquetType, PhysicalType, PrimitiveConvertedType,
            PrimitiveLogicalType, PrimitiveType,
        },
        Repetition,
    },
    write::{Compressor, DynIter, DynStreamingIterator, FileWriter, Version, WriteOptions},
};
use std::{fs, path::Path};

pub fn export_to_parquet(info: &MdfInfo4, file_name: &str, compression: CompressionOptions) {
    // create schema
    let mut fields: Vec<ParquetType> = Vec::with_capacity(info.channel_names_set.len());
    info.dg.iter().for_each(|(_dg_block_position, dg)| {
        for (_rec_id, cg) in dg.cg.iter() {
            for (_rec_pos, cn) in cg.cn.iter() {
                fields.push(ParquetType::PrimitiveType(cn.data.parquet_primitive_type(
                    cn.unique_name.clone(),
                    Repetition::Optional,
                    None,
                )));
            }
        }
    });
    // Create file
    let path = Path::new(file_name);
    let schema = SchemaDescriptor::new(file_name.to_string(), fields);
    let options = WriteOptions {
        write_statistics: false,
        version: Version::V1,
    };
    let file = fs::File::create(&path).unwrap();
    let mut writer = FileWriter::new(file, schema, options, Some(r"mdfr".to_string()));
    writer
        .start()
        .expect("Error while starting the parquet writer");
    // write data in file
    info.dg.iter().for_each(|(_dg_block_position, dg)| {
        for (_rec_id, cg) in dg.cg.iter() {
            let mut columns: Vec<Result<DynStreamingIterator<CompressedPage, Error>, Error>> =
                Vec::new();
            for (_rec_pos, cn) in cg.cn.iter() {
                let field = cn.data.parquet_primitive_type(
                    cn.unique_name.clone(),
                    Repetition::Optional,
                    None,
                );
                let pages = Ok(DynStreamingIterator::new(Compressor::new_from_vec(
                    DynIter::new(std::iter::once(
                        cn.data.parquet_encoded_page(field),
                    )),
                    compression,
                    vec![],
                )));
                columns.push(pages);
            }
            writer
                .write(DynIter::new(columns.into_iter()))
                .expect("could not write rowgroup");
        }
    });

    writer
        .end(None)
        .expect("Error while ending the parquet writer");
}

impl ChannelData {
    /// returns parquet type
    fn parquet_primitive_type(
        &self,
        name: String,
        repetition: Repetition,
        id: Option<i32>,
    ) -> PrimitiveType {
        let field_info = FieldInfo {
            name,
            repetition,
            id,
        };
        match self {
            ChannelData::Int8(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int8),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int8)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::UInt8(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint8),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt8)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::Int16(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int16),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int16)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::UInt16(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint16),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt16)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::Float16(_) => PrimitiveType {
                field_info,
                converted_type: None,
                logical_type: None,
                physical_type: PhysicalType::Float,
            },
            ChannelData::Int24(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int32),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int32)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::UInt24(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint32),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt32)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::Int32(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int32),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int32)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::UInt32(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint32),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt32)),
                physical_type: PhysicalType::Int32,
            },
            ChannelData::Float32(_) => PrimitiveType {
                field_info,
                converted_type: None,
                logical_type: None,
                physical_type: PhysicalType::Float,
            },
            ChannelData::Int48(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int64),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int64)),
                physical_type: PhysicalType::Int64,
            },
            ChannelData::UInt48(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint64),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt64)),
                physical_type: PhysicalType::Int64,
            },
            ChannelData::Int64(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Int64),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::Int64)),
                physical_type: PhysicalType::Int64,
            },
            ChannelData::UInt64(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Uint64),
                logical_type: Some(PrimitiveLogicalType::Integer(IntegerType::UInt64)),
                physical_type: PhysicalType::Int64,
            },
            ChannelData::Float64(_) => PrimitiveType {
                field_info,
                converted_type: None,
                logical_type: None,
                physical_type: PhysicalType::Double,
            },
            ChannelData::Complex16(_) => todo!(),
            ChannelData::Complex32(_) => todo!(),
            ChannelData::Complex64(_) => todo!(),
            ChannelData::StringSBC(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Utf8),
                logical_type: Some(PrimitiveLogicalType::String),
                physical_type: PhysicalType::ByteArray,
            },
            ChannelData::StringUTF8(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Utf8),
                logical_type: Some(PrimitiveLogicalType::String),
                physical_type: PhysicalType::ByteArray,
            },
            ChannelData::StringUTF16(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Utf8),
                logical_type: Some(PrimitiveLogicalType::String),
                physical_type: PhysicalType::ByteArray,
            },
            ChannelData::ByteArray(_) => PrimitiveType {
                field_info,
                converted_type: Some(PrimitiveConvertedType::Enum),
                logical_type: Some(PrimitiveLogicalType::Enum),
                physical_type: PhysicalType::ByteArray,
            },
            ChannelData::ArrayDInt8(_) => todo!(),
            ChannelData::ArrayDUInt8(_) => todo!(),
            ChannelData::ArrayDInt16(_) => todo!(),
            ChannelData::ArrayDUInt16(_) => todo!(),
            ChannelData::ArrayDFloat16(_) => todo!(),
            ChannelData::ArrayDInt24(_) => todo!(),
            ChannelData::ArrayDUInt24(_) => todo!(),
            ChannelData::ArrayDInt32(_) => todo!(),
            ChannelData::ArrayDUInt32(_) => todo!(),
            ChannelData::ArrayDFloat32(_) => todo!(),
            ChannelData::ArrayDInt48(_) => todo!(),
            ChannelData::ArrayDUInt48(_) => todo!(),
            ChannelData::ArrayDInt64(_) => todo!(),
            ChannelData::ArrayDUInt64(_) => todo!(),
            ChannelData::ArrayDFloat64(_) => todo!(),
            ChannelData::ArrayDComplex16(_) => todo!(),
            ChannelData::ArrayDComplex32(_) => todo!(),
            ChannelData::ArrayDComplex64(_) => todo!(),
        }
    }
    fn parquet_encoded_page(
        &self,
        primitive_type: PrimitiveType,
    ) -> Result<EncodedPage, Error> {
        let data_length = self.len();
        let header = DataPageHeaderV1 {
            num_values: data_length as i32,
            encoding: Encoding::Plain.into(),
            definition_level_encoding: Encoding::Rle.into(),
            repetition_level_encoding: Encoding::Rle.into(),
            statistics: None,
        };
        Ok(EncodedPage::Data(DataPage::new(
            DataPageHeader::V1(header),
            self.to_bytes(),
            None,
            Descriptor {
                primitive_type,
                max_def_level: 0,
                max_rep_level: 0,
            },
            Some(data_length),
        )))
    }
}
