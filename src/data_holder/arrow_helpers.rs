//! helpers for arrow
use arrow::array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
use arrow::datatypes::DataType;

/// returns the number of bits corresponding to the array's datatype
pub fn arrow_bit_count(array: &dyn Array) -> u32 {
    let data_type = array.data_type();
    bit_count(array, data_type)
}

fn bit_count(array: &dyn Array, data_type: &DataType) -> u32 {
    match data_type {
        DataType::Null => 0,
        DataType::Boolean => 8,
        DataType::Int8 => 8,
        DataType::Int16 => 16,
        DataType::Int32 => 32,
        DataType::Int64 => 64,
        DataType::UInt8 => 8,
        DataType::UInt16 => 16,
        DataType::UInt32 => 32,
        DataType::UInt64 => 64,
        DataType::Float16 => 16,
        DataType::Float32 => 32,
        DataType::Float64 => 64,
        DataType::Timestamp(_, _) => 64,
        DataType::Date32 => 32,
        DataType::Date64 => 64,
        DataType::Time32(_) => 32,
        DataType::Time64(_) => 64,
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeBinary(size) => 8 * *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("could not downcast to long utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
                * 8
        }
        DataType::FixedSizeList(field, size) => match field.data_type() {
            DataType::Float32 => 32 * *size as u32,
            DataType::Float64 => 64 * *size as u32,
            _ => panic!("unsupported type"),
        },
        _ => panic!("unsupported type"),
    }
}

/// returns the number of bytes corresponding to the array's datatype
pub fn arrow_byte_count(array: &dyn Array) -> u32 {
    let data_type = array.data_type();
    byte_count(array, data_type)
}
fn byte_count(array: &dyn Array, data_type: &DataType) -> u32 {
    match data_type {
        DataType::Null => 0,
        DataType::Boolean => 1,
        DataType::Int8 => 1,
        DataType::Int16 => 2,
        DataType::Int32 => 4,
        DataType::Int64 => 8,
        DataType::UInt8 => 1,
        DataType::UInt16 => 2,
        DataType::UInt32 => 4,
        DataType::UInt64 => 8,
        DataType::Float16 => 2,
        DataType::Float32 => 4,
        DataType::Float64 => 8,
        DataType::Timestamp(_, _) => 8,
        DataType::Date32 => 4,
        DataType::Date64 => 8,
        DataType::Time32(_) => 4,
        DataType::Time64(_) => 8,
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeBinary(size) => *size as u32,
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("could not downcast to utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("could not downcast to long utf8 array");
            array
                .iter()
                .map(|s| s.unwrap_or_default().len() as u32)
                .max()
                .unwrap_or(0)
        }
        DataType::FixedSizeList(field, size) => match field.data_type() {
            DataType::Float32 => 4 * *size as u32,
            DataType::Float64 => 8 * *size as u32,
            _ => panic!("unsupported type"),
        },
        _ => panic!("unsupported type"),
    }
}

/// returns mdf4 data type from arrow array
pub fn arrow_to_mdf_data_type(array: &dyn Array, endian: bool) -> u8 {
    mdf_data_type(array.data_type(), endian)
}

fn mdf_data_type(data_type: &DataType, endian: bool) -> u8 {
    if endian {
        // BE
        match data_type {
            DataType::Null => 1,
            DataType::Boolean => 1,
            DataType::Int8 => 3,
            DataType::Int16 => 3,
            DataType::Int32 => 3,
            DataType::Int64 => 3,
            DataType::UInt8 => 1,
            DataType::UInt16 => 1,
            DataType::UInt32 => 1,
            DataType::UInt64 => 1,
            DataType::Float16 => 5,
            DataType::Float32 => 5,
            DataType::Float64 => 5,
            DataType::Timestamp(_, _) => 3,
            DataType::Date32 => 3,
            DataType::Date64 => 3,
            DataType::Time32(_) => 3,
            DataType::Time64(_) => 3,
            DataType::Duration(_) => 3,
            DataType::Interval(_) => 3,
            DataType::Binary => 10,
            DataType::FixedSizeBinary(_) => 10,
            DataType::LargeBinary => 10,
            DataType::Utf8 => 7,
            DataType::LargeUtf8 => 7,
            DataType::List(_) => 16,
            DataType::FixedSizeList(_, _) => 16,
            DataType::LargeList(_) => 16,
            _ => panic!("unsupported type"),
        }
    } else {
        // LE
        match data_type {
            DataType::Null => 0,
            DataType::Boolean => 0,
            DataType::Int8 => 2,
            DataType::Int16 => 2,
            DataType::Int32 => 2,
            DataType::Int64 => 2,
            DataType::UInt8 => 0,
            DataType::UInt16 => 0,
            DataType::UInt32 => 0,
            DataType::UInt64 => 0,
            DataType::Float16 => 4,
            DataType::Float32 => 4,
            DataType::Float64 => 4,
            DataType::Timestamp(_, _) => 2,
            DataType::Date32 => 2,
            DataType::Date64 => 2,
            DataType::Time32(_) => 2,
            DataType::Time64(_) => 2,
            DataType::Duration(_) => 2,
            DataType::Interval(_) => 2,
            DataType::Binary => 10,
            DataType::FixedSizeBinary(_) => 10,
            DataType::LargeBinary => 10,
            DataType::Utf8 => 7,
            DataType::LargeUtf8 => 7,
            DataType::List(_) => 15,
            DataType::FixedSizeList(_, _) => 15,
            DataType::LargeList(_) => 15,
            _ => panic!("unsupported type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryBuilder, BooleanArray, Date32Array, Date64Array, FixedSizeBinaryBuilder,
        Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        LargeBinaryBuilder, LargeStringBuilder, NullArray, StringBuilder,
        TimestampNanosecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };

    // ── bit_count tests ──

    #[test]
    fn test_bit_count_primitives() {
        assert_eq!(arrow_bit_count(&NullArray::new(1)), 0);
        assert_eq!(arrow_bit_count(&BooleanArray::from(vec![true])), 8);
        assert_eq!(arrow_bit_count(&Int8Array::from(vec![1])), 8);
        assert_eq!(arrow_bit_count(&Int16Array::from(vec![1])), 16);
        assert_eq!(arrow_bit_count(&Int32Array::from(vec![1])), 32);
        assert_eq!(arrow_bit_count(&Int64Array::from(vec![1])), 64);
        assert_eq!(arrow_bit_count(&UInt8Array::from(vec![1])), 8);
        assert_eq!(arrow_bit_count(&UInt16Array::from(vec![1])), 16);
        assert_eq!(arrow_bit_count(&UInt32Array::from(vec![1])), 32);
        assert_eq!(arrow_bit_count(&UInt64Array::from(vec![1])), 64);
        assert_eq!(arrow_bit_count(&Float32Array::from(vec![1.0])), 32);
        assert_eq!(arrow_bit_count(&Float64Array::from(vec![1.0])), 64);
    }

    #[test]
    fn test_bit_count_temporal() {
        assert_eq!(
            arrow_bit_count(&TimestampNanosecondArray::from(vec![1])),
            64
        );
        assert_eq!(arrow_bit_count(&Date32Array::from(vec![1])), 32);
        assert_eq!(arrow_bit_count(&Date64Array::from(vec![1])), 64);
    }

    #[test]
    fn test_bit_count_strings() {
        let mut builder = StringBuilder::new();
        builder.append_value("hello"); // 5 bytes
        builder.append_value("hi"); // 2 bytes
        let array = builder.finish();
        assert_eq!(arrow_bit_count(&array), 5 * 8); // max is 5 bytes

        let mut builder = LargeStringBuilder::new();
        builder.append_value("abc");
        let array = builder.finish();
        assert_eq!(arrow_bit_count(&array), 3 * 8);
    }

    #[test]
    fn test_bit_count_binary() {
        let mut builder = BinaryBuilder::new();
        builder.append_value(b"abcd");
        let array = builder.finish();
        assert_eq!(arrow_bit_count(&array), 4 * 8);

        let mut builder = LargeBinaryBuilder::new();
        builder.append_value(b"ab");
        let array = builder.finish();
        assert_eq!(arrow_bit_count(&array), 2 * 8);

        let mut builder = FixedSizeBinaryBuilder::new(3);
        builder.append_value(b"abc").unwrap();
        let array = builder.finish();
        assert_eq!(arrow_bit_count(&array), 3 * 8);
    }

    // ── byte_count tests ──

    #[test]
    fn test_byte_count_primitives() {
        assert_eq!(arrow_byte_count(&NullArray::new(1)), 0);
        assert_eq!(arrow_byte_count(&BooleanArray::from(vec![true])), 1);
        assert_eq!(arrow_byte_count(&Int8Array::from(vec![1])), 1);
        assert_eq!(arrow_byte_count(&Int16Array::from(vec![1])), 2);
        assert_eq!(arrow_byte_count(&Int32Array::from(vec![1])), 4);
        assert_eq!(arrow_byte_count(&Int64Array::from(vec![1])), 8);
        assert_eq!(arrow_byte_count(&UInt8Array::from(vec![1])), 1);
        assert_eq!(arrow_byte_count(&UInt16Array::from(vec![1])), 2);
        assert_eq!(arrow_byte_count(&UInt32Array::from(vec![1])), 4);
        assert_eq!(arrow_byte_count(&UInt64Array::from(vec![1])), 8);
        assert_eq!(arrow_byte_count(&Float32Array::from(vec![1.0])), 4);
        assert_eq!(arrow_byte_count(&Float64Array::from(vec![1.0])), 8);
    }

    #[test]
    fn test_byte_count_strings() {
        let mut builder = StringBuilder::new();
        builder.append_value("hello"); // 5 bytes
        let array = builder.finish();
        assert_eq!(arrow_byte_count(&array), 5);

        let mut builder = LargeStringBuilder::new();
        builder.append_value("abc");
        let array = builder.finish();
        assert_eq!(arrow_byte_count(&array), 3);
    }

    // ── mdf_data_type tests ──

    #[test]
    fn test_mdf_data_type_le() {
        // LE (endian=false)
        assert_eq!(arrow_to_mdf_data_type(&NullArray::new(1), false), 0);
        assert_eq!(
            arrow_to_mdf_data_type(&BooleanArray::from(vec![true]), false),
            0
        );
        assert_eq!(arrow_to_mdf_data_type(&Int32Array::from(vec![1]), false), 2);
        assert_eq!(
            arrow_to_mdf_data_type(&UInt32Array::from(vec![1]), false),
            0
        );
        assert_eq!(
            arrow_to_mdf_data_type(&Float64Array::from(vec![1.0]), false),
            4
        );

        let mut builder = StringBuilder::new();
        builder.append_value("x");
        let array = builder.finish();
        assert_eq!(arrow_to_mdf_data_type(&array, false), 7); // UTF-8

        let mut builder = BinaryBuilder::new();
        builder.append_value(b"x");
        let array = builder.finish();
        assert_eq!(arrow_to_mdf_data_type(&array, false), 10); // Byte Array
    }

    #[test]
    fn test_mdf_data_type_be() {
        // BE (endian=true)
        assert_eq!(arrow_to_mdf_data_type(&NullArray::new(1), true), 1);
        assert_eq!(arrow_to_mdf_data_type(&Int32Array::from(vec![1]), true), 3);
        assert_eq!(arrow_to_mdf_data_type(&UInt32Array::from(vec![1]), true), 1);
        assert_eq!(
            arrow_to_mdf_data_type(&Float64Array::from(vec![1.0]), true),
            5
        );

        let mut builder = StringBuilder::new();
        builder.append_value("x");
        let array = builder.finish();
        assert_eq!(arrow_to_mdf_data_type(&array, true), 7); // UTF-8 same for both

        let mut builder = BinaryBuilder::new();
        builder.append_value(b"x");
        let array = builder.finish();
        assert_eq!(arrow_to_mdf_data_type(&array, true), 10); // Byte Array same
    }
}
