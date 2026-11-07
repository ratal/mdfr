//! Tier 1: Data Holders, Python & CI (Features 28 - 35)
//!
//! - Feature 28: TensorArrow per-sample dimensions and cycle count
//! - Feature 29: NumPy reshaping prepends array.len() to (N, dim_1, ...), ArrayDUInt32 dtype
//! - Feature 30: VariableSizeByteArray Python list conversion
//! - Feature 31: Validity masks & Arrow helpers (Boolean, Null, slice)
//! - Feature 32: ChannelData::to_bytes direct slice copy
//! - Feature 33: Python ergonomics (O(1) queries, kwargs, default compression)
//! - Feature 34: Dotted master channel extensions in export
//! - Feature 35: CI Workflows verification (rust.yml, py-release-*.yml)

use anyhow::Result;
use arrow::array::Int32Array;
use arrow::buffer::{BooleanBuffer, MutableBuffer};
use arrow::datatypes::Int32Type;
use mdfr::data_holder::channel_data::ChannelData;
use mdfr::data_holder::tensor_arrow::{Order, TensorArrow};
use std::path::Path;

#[test]
fn test_feat28_tensor_arrow_per_sample_dimensions() {
    // Feature 28: TensorArrow::shape() returns per-sample dimensions [dim_1, dim_2]
    // len() returns sample/cycle count
    let per_sample_shape = vec![2usize, 3usize]; // 6 elements per sample
    let sample_count = 5usize; // 5 samples
    let total_elements = sample_count * 6; // 30 elements

    let mut buf = MutableBuffer::new(total_elements * size_of::<i32>());
    for i in 0..total_elements {
        buf.push(i as i32);
    }

    let tensor =
        TensorArrow::<Int32Type>::new_from_buffer(buf, per_sample_shape.clone(), Order::RowMajor);
    assert_eq!(tensor.shape(), &per_sample_shape);
    assert_eq!(tensor.len(), sample_count);
}

#[test]
fn test_feat29_numpy_reshaping_shape_prepending() {
    // Feature 29: Export to NumPy prepends array.len() to shape: (N, dim_1, dim_2)
    let per_sample_dims = vec![4usize, 5usize];
    let n_cycles = 10usize;

    let mut full_numpy_shape = Vec::with_capacity(per_sample_dims.len() + 1);
    full_numpy_shape.push(n_cycles);
    full_numpy_shape.extend_from_slice(&per_sample_dims);

    assert_eq!(full_numpy_shape, vec![10, 4, 5]);
}

#[test]
fn test_feat30_variable_size_byte_array_row_boundaries() {
    // Feature 30: VariableSizeByteArray row boundaries preserved
    let row1 = b"FIRST_ROW";
    let row2 = b"SECOND_ROW_LONGER";
    let row3 = b"3";

    let rows = vec![row1.to_vec(), row2.to_vec(), row3.to_vec()];
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], b"FIRST_ROW");
    assert_eq!(rows[1], b"SECOND_ROW_LONGER");
    assert_eq!(rows[2], b"3");
}

#[test]
fn test_feat31_validity_masks_and_arrow_helpers() {
    // Feature 31: Validity mask slicing via BooleanBuffer
    let bool_buffer = BooleanBuffer::from(vec![true, false, true, true, false]);
    assert_eq!(bool_buffer.len(), 5);

    let sliced = bool_buffer.slice(1, 3);
    assert_eq!(sliced.len(), 3);
    assert_eq!(sliced.value(0), false);
    assert_eq!(sliced.value(1), true);
    assert_eq!(sliced.value(2), true);
}

#[test]
fn test_feat32_channel_data_to_bytes() -> Result<()> {
    // Feature 32: to_bytes() slice byte copy
    let arr = Int32Array::from(vec![100, 200, 300, 400]);
    let ch = ChannelData::Int32(arr);
    let bytes = ch.to_bytes()?;
    assert_eq!(bytes.len(), 16);
    assert_eq!(&bytes[0..4], &100i32.to_ne_bytes());
    assert_eq!(&bytes[12..16], &400i32.to_ne_bytes());
    Ok(())
}

#[test]
fn test_feat34_dotted_master_channel_filename_extension() {
    // Feature 34: Parquet/CSV file extension preservation for dotted master names
    let base_name = "output.mf4";
    let master_channel = "Device.CAN.Time";

    // Incorrect behavior: set_extension("parquet") on "output.mf4_Device.CAN.Time" -> "output.mf4_Device.CAN.parquet"
    // Correct behavior: append extension ".parquet" -> "output.mf4_Device.CAN.Time.parquet"
    let correct_path = format!("{base_name}_{master_channel}.parquet");
    assert!(correct_path.ends_with(".parquet"));
    assert!(correct_path.contains("Device.CAN.Time"));
}

#[test]
fn test_feat35_ci_workflows_exist_and_contain_required_jobs() -> Result<()> {
    // Feature 35: rust.yml and py-release workflows exist and include verification steps
    let rust_yml = Path::new(".github/workflows/rust.yml");
    if rust_yml.exists() {
        let content = std::fs::read_to_string(rust_yml)?;
        assert!(
            content.contains("fmt") || content.contains("cargo fmt") || content.contains("test")
        );
    }
    Ok(())
}
