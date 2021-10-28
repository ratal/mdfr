//! this modules implements functions to convert arrays into physical arrays using CCBlock
use std::collections::BTreeMap;

use crate::mdfinfo::mdfinfo4::{Cn4, Dg4, SharableBlocks};
use crate::mdfreader::channel_data::ChannelData;
use ndarray::{Array1, ArrayD, Zip};
use num::Complex;
use rayon::prelude::*;
use fasteval::Evaler;
use fasteval::Compiler; 

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg4, sharable: &SharableBlocks) {
    for channel_group in dg.cg.values_mut() {
        let cycle_count = channel_group.block.cg_cycle_count;
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| !cn.data.is_empty())
            .for_each(|(_rec_pos, cn)| {
                // Could be empty if only initialised
                if let Some(conv) = sharable.cc.get(&cn.block.cn_cc_conversion) {
                    match conv.cc_type {
                        1 => {
                            linear_conversion(cn, &conv.cc_val_real, &cycle_count);
                        }
                        2 => {
                            rational_conversion(cn, &conv.cc_val_real, &cycle_count);
                        }
                        3 => {
                            if !&conv.cc_ref.is_empty() {
                                if let Some(conv) = sharable.tx.get(&conv.cc_ref[0]) {
                                    algebraic_conversion(cn, &conv.0, &cycle_count);
                                }
                            }
                        }
                        _ => {} //TODO further implement conversions
                    }
                }
            })
    }
}

/// Apply linear conversion to get physical data
#[inline]
fn linear_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && num::abs(p2 - 1.0) < 1e-12) {
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int8(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int16(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt16(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float16(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int24(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt24(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int32(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt32(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float32(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int48(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt48(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int64(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt64(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float64(a) => {
                let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = *a * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Complex16(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::Complex32(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::Complex64(a) => {
                let mut new_array = Array1::<Complex<f64>>::zeros((*cycle_count as usize,));
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re * p2 + p1, a.im * p2 + p1)
                });
                cn.data = ChannelData::Complex64(new_array);
            }
            ChannelData::StringSBC(_) => {}
            ChannelData::StringUTF8(_) => {}
            ChannelData::StringUTF16(_) => {}
            ChannelData::ByteArray(_) => {}
            ChannelData::ArrayDUInt8(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt8(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt16(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDUInt16(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDFloat16(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt24(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDUInt24(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt32(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDUInt32(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDFloat32(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt48(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDUInt48(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDInt64(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDUInt64(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDFloat64(a) => {
                let mut new_array = ArrayD::<f64>::zeros(a.shape());
                Zip::from(&mut new_array)
                    .and(a)
                    .for_each(|new_array, a| *new_array = *a * p2 + p1);
                cn.data = ChannelData::ArrayDFloat64(new_array);
            }
            ChannelData::ArrayDComplex16(a) => {
                let mut new_array = ArrayD::<Complex<f64>>::zeros(a.shape());
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                });
                cn.data = ChannelData::ArrayDComplex64(new_array);
            }
            ChannelData::ArrayDComplex32(a) => {
                let mut new_array = ArrayD::<Complex<f64>>::zeros(a.shape());
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re as f64 * p2 + p1, a.im as f64 * p2 + p1)
                });
                cn.data = ChannelData::ArrayDComplex64(new_array);
            }
            ChannelData::ArrayDComplex64(a) => {
                let mut new_array = ArrayD::<Complex<f64>>::zeros(a.shape());
                Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                    *new_array = Complex::<f64>::new(a.re * p2 + p1, a.im * p2 + p1)
                });
                cn.data = ChannelData::ArrayDComplex64(new_array);
            }
        }
    }
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];

    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
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
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let m_2 = f64::powi(*a, 2);
                *new_array = (m_2 * p1 + *a * p2 + p1) / (m_2 * p4 + *a * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        }
        ChannelData::ArrayDComplex16(_) => todo!(),
        ChannelData::ArrayDComplex32(_) => todo!(),
        ChannelData::ArrayDComplex64(_) => todo!(),
    }
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn4, formulae: &String, cycle_count: &u64) {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let mut map = BTreeMap::new();
    let compiled = parser.parse(formulae, &mut slab.ps)
        .expect("error parsing formulae for conversion")
        .from(&slab.ps).compile(&slab.ps, &mut slab.cs);
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => todo!(),
        ChannelData::Complex32(_) => todo!(),
        ChannelData::Complex64(_) => todo!(),
        ChannelData::StringSBC(_) => todo!(),
        ChannelData::StringUTF8(_) => todo!(),
        ChannelData::StringUTF16(_) => todo!(),
        ChannelData::ByteArray(_) => todo!(),
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