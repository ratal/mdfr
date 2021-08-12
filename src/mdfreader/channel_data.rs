use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use num::Complex;

/// channel data type enum
#[derive(Debug, Clone)]
pub enum ChannelData {
    Int8(Array1<i8>),
    UInt8(Array1<u8>),
    Int16(Array1<i16>),
    UInt16(Array1<u16>),
    Float16(Array1<f32>),
    Int24(Array1<i32>),
    UInt24(Array1<u32>),
    Int32(Array1<i32>),
    UInt32(Array1<u32>),
    Float32(Array1<f32>),
    Int48(Array1<i64>),
    UInt48(Array1<u64>),
    Int64(Array1<i64>),
    UInt64(Array1<u64>),
    Float64(Array1<f64>),
    Complex16(Array1<Complex<f32>>),
    Complex32(Array1<Complex<f32>>),
    Complex64(Array1<Complex<f64>>),
    StringSBC(Vec<String>),
    StringUTF8(Vec<String>),
    StringUTF16(Vec<String>),
    ByteArray(Vec<u8>),
}

impl ChannelData {
    pub fn zeros(&self, cycle_count: u64, n_bytes: u32) -> ChannelData {
        match self {
            ChannelData::Int8(_) => ChannelData::Int8(
                ArrayBase::<OwnedRepr<i8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt8(_) => ChannelData::UInt8(
                ArrayBase::<OwnedRepr<u8>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::Int16(_) => ChannelData::Int16(
                ArrayBase::<OwnedRepr<i16>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt16(_) => {
                ChannelData::UInt16(ArrayBase::<OwnedRepr<u16>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float16(_) => {
                ChannelData::Float16(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int24(_) => ChannelData::Int24(
                ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt24(_) => {
                ChannelData::UInt24(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int32(_) => ChannelData::Int32(
                ArrayBase::<OwnedRepr<i32>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt32(_) => {
                ChannelData::UInt32(ArrayBase::<OwnedRepr<u32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float32(_) => {
                ChannelData::Float32(ArrayBase::<OwnedRepr<f32>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int48(_) => ChannelData::Int48(
                ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt48(_) => {
                ChannelData::UInt48(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Int64(_) => ChannelData::Int64(
                ArrayBase::<OwnedRepr<i64>, Dim<[usize; 1]>>::zeros((cycle_count as usize,)),
            ),
            ChannelData::UInt64(_) => {
                ChannelData::UInt64(ArrayBase::<OwnedRepr<u64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Float64(_) => {
                ChannelData::Float64(ArrayBase::<OwnedRepr<f64>, Dim<[usize; 1]>>::zeros((
                    cycle_count as usize,
                )))
            }
            ChannelData::Complex16(_) => ChannelData::Complex16(ArrayBase::<
                OwnedRepr<Complex<f32>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::Complex32(_) => ChannelData::Complex32(ArrayBase::<
                OwnedRepr<Complex<f32>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::Complex64(_) => ChannelData::Complex64(ArrayBase::<
                OwnedRepr<Complex<f64>>,
                Dim<[usize; 1]>,
            >::zeros((
                cycle_count as usize,
            ))),
            ChannelData::StringSBC(_) => {
                ChannelData::StringSBC(vec![String::new(); cycle_count as usize])
            }
            ChannelData::StringUTF8(_) => {
                ChannelData::StringUTF8(vec![String::new(); cycle_count as usize])
            }
            ChannelData::StringUTF16(_) => {
                ChannelData::StringUTF16(vec![String::new(); cycle_count as usize])
            }
            ChannelData::ByteArray(_) => {
                ChannelData::ByteArray(vec![0u8; (n_bytes as u64 * cycle_count) as usize])
            }
        }
    }
}

impl Default for ChannelData {
    fn default() -> Self {
        ChannelData::UInt8(Array1::<u8>::zeros((0,)))
    }
}

/// Initialises a channel array with cycle_count zeroes and correct depending of cn_type, cn_data_type and number of bytes
pub fn data_init(cn_type: u8, cn_data_type: u8, n_bytes: u32, cycle_count: u64) -> ChannelData {
    let data_type: ChannelData;
    if cn_type != 3 || cn_type != 6 {
        if cn_data_type == 0 || cn_data_type == 1 {
            // unsigned int
            if n_bytes <= 1 {
                data_type = ChannelData::UInt8(Array1::<u8>::zeros((cycle_count as usize,)));
            } else if n_bytes == 2 {
                data_type = ChannelData::UInt16(Array1::<u16>::zeros((cycle_count as usize,)));
            } else if n_bytes == 3 {
                data_type = ChannelData::UInt24(Array1::<u32>::zeros((cycle_count as usize,)));
            } else if n_bytes == 4 {
                data_type = ChannelData::UInt32(Array1::<u32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 6 {
                data_type = ChannelData::UInt48(Array1::<u64>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::UInt64(Array1::<u64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 2 || cn_data_type == 3 {
            // signed int
            if n_bytes <= 1 {
                data_type = ChannelData::Int8(Array1::<i8>::zeros((cycle_count as usize,)));
            } else if n_bytes == 2 {
                data_type = ChannelData::Int16(Array1::<i16>::zeros((cycle_count as usize,)));
            } else if n_bytes == 3 {
                data_type = ChannelData::Int24(Array1::<i32>::zeros((cycle_count as usize,)));
            } else if n_bytes == 4 {
                data_type = ChannelData::Int32(Array1::<i32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 6 {
                data_type = ChannelData::Int48(Array1::<i64>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::Int64(Array1::<i64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 4 || cn_data_type == 5 {
            // float
            if n_bytes <= 2 {
                data_type = ChannelData::Float16(Array1::<f32>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 4 {
                data_type = ChannelData::Float32(Array1::<f32>::zeros((cycle_count as usize,)));
            } else {
                data_type = ChannelData::Float64(Array1::<f64>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 15 || cn_data_type == 16 {
            // complex
            if n_bytes <= 2 {
                data_type =
                    ChannelData::Complex16(Array1::<Complex<f32>>::zeros((cycle_count as usize,)));
            } else if n_bytes <= 4 {
                data_type =
                    ChannelData::Complex32(Array1::<Complex<f32>>::zeros((cycle_count as usize,)));
            } else {
                data_type =
                    ChannelData::Complex64(Array1::<Complex<f64>>::zeros((cycle_count as usize,)));
            }
        } else if cn_data_type == 6 {
            // SBC ISO-8859-1 to be converted into UTF8
            data_type = ChannelData::StringSBC(vec![String::new(); cycle_count as usize]);
        } else if cn_data_type == 7 {
            // String UTF8
            data_type = ChannelData::StringUTF8(vec![String::new(); cycle_count as usize]);
        } else if cn_data_type == 8 || cn_data_type == 9 {
            // String UTF16 to be converted into UTF8
            data_type = ChannelData::StringUTF16(vec![String::new(); cycle_count as usize]);
        } else {
            // bytearray
            data_type = ChannelData::ByteArray(vec![0u8; (n_bytes as u64 * cycle_count) as usize]);
        }
    } else {
        // virtual channels, cn_bit_count = 0 -> n_bytes = 0, must be LE unsigned int
        data_type = ChannelData::UInt64(Array1::<u64>::from_iter(0..cycle_count));
    }
    data_type
}
