use crate::mdfinfo::mdfinfo4::{Cc4Block, Cn4, Dg4};
use crate::mdfreader::channel_data::ChannelData;
use ndarray::{Array1, Zip};
use num::Complex;
use std::collections::HashMap;

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg4, cc: &HashMap<i64, Cc4Block>) {
    for channel_group in dg.cg.values_mut() {
        for (_cn_record_position, cn) in channel_group.cn.iter_mut() {
            if !cn.data.is_empty() {  // Coudl be empty if only initialised
                if let Some(conv) = cc.get(&cn.block.cn_cc_conversion) {
                    match conv.cc_type {
                        1 => linear_conversion(cn, &conv.cc_val, &channel_group.block.cg_cycle_count),
                        2 => rational_conversion(cn, &conv.cc_val, &channel_group.block.cg_cycle_count),
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Apply linear conversion to get physical data
fn linear_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && (p2 - 1.0) < 1e-12) {
        let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int8(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int16(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt16(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float16(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int24(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt24(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int32(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt32(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float32(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int48(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt48(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int64(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt64(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float64(a) => {
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| *new_array = *a * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Complex16(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| {
                        *new_array =
                            Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                    });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::Complex32(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| {
                        *new_array =
                            Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                    });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::Complex64(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .par_for_each(|new_array, a| {
                        *new_array = Complex::<f64>::new(a.re * p2 + p1, a.im * p2 + p1)
                    });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::StringSBC(_) => {}
            ChannelData::StringUTF8(_) => {}
            ChannelData::StringUTF16(_) => {}
            ChannelData::ByteArray(_) => {}
        }
    }
}

// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int8(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m = *a as f64;
                    let m_2 = f64::powi(m, 2);
                    *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            Zip::from(&mut new_array)
                .and(a)
                .par_for_each(|new_array, a| {
                    let m_2 = f64::powi(*a, 2);
                    *new_array = (m_2 * p1 + *a * p2 + p1) / (m_2 * p4 + *a * p5 + p6)
                });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Complex16(_) => {}
        ChannelData::Complex32(_) => {}
        ChannelData::Complex64(_) => {}
        ChannelData::StringSBC(_) => {}
        ChannelData::StringUTF8(_) => {}
        ChannelData::StringUTF16(_) => {}
        ChannelData::ByteArray(_) => {}
    }
}
