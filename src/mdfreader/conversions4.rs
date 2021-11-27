//! this modules implements functions to convert arrays into physical arrays using CCBlock
use std::collections::{BTreeMap, HashMap};
use itertools::Itertools;

use crate::mdfinfo::mdfinfo4::{Cc4Block, CcVal, Cn4, Dg4, SharableBlocks};
use crate::mdfreader::channel_data::ChannelData;
use ndarray::{Array1, ArrayD, Zip};
use num::Complex;
use rayon::prelude::*;
use fasteval::{Evaler, Instruction, Slab};
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
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => linear_conversion(cn, cc_val, &cycle_count),
                                CcVal::Uint(_) => (),
                            }
                        }
                        2 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => rational_conversion(cn, cc_val, &cycle_count),
                                CcVal::Uint(_) => (),
                            }
                        }
                        3 => {
                            if !&conv.cc_ref.is_empty() {
                                if let Some(conv) = sharable.tx.get(&conv.cc_ref[0]) {
                                    algebraic_conversion(cn, &conv.0, &cycle_count);
                                }
                            }
                        }
                        4 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => value_to_value_with_interpolation(cn, cc_val.clone(), &cycle_count),
                                CcVal::Uint(_) => (),
                            }
                        }
                        5 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => value_to_value_without_interpolation(cn, cc_val.clone(), &cycle_count),
                                CcVal::Uint(_) => (),
                            }
                        }
                        6 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => value_range_to_value_table(cn, cc_val.clone(), &cycle_count),
                                CcVal::Uint(_) => (),
                            }
                        }
                        7 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => {
                                    value_to_text(cn, &cc_val, &conv.cc_ref, &cycle_count, sharable)
                                },
                                CcVal::Uint(_) => (),
                            }
                        }
                        8 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => {
                                    value_range_to_text(cn, &cc_val, &conv.cc_ref, &cycle_count, sharable)
                                },
                                CcVal::Uint(_) => (),
                            }
                        }
                        9 => {
                            match &conv.cc_val {
                                CcVal::Real(cc_val) => {
                                    text_to_value(cn, &cc_val, &conv.cc_ref, &cycle_count, sharable)
                                },
                                CcVal::Uint(_) => (),
                            }
                        }
                        10 => {
                            text_to_text(cn, &conv.cc_ref, &cycle_count, sharable)
                        }
                        11 => {
                            match &conv.cc_val {
                                CcVal::Real(_) => (),
                                CcVal::Uint(cc_val) => {
                                    bitfield_text_table(cn, &cc_val, &conv.cc_ref, &cycle_count, sharable)
                                },
                            }
                        }
                        _ => {} //TODO further implement conversions
                    }
                }
            })
    }
}

/// Apply linear conversion to get physical data
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
        ChannelData::Complex16(_) => todo!(),
        ChannelData::Complex32(_) => todo!(),
        ChannelData::Complex64(_) => todo!(),
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
                map.insert("X".to_string(), *a);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => todo!(),
        ChannelData::Complex32(_) => todo!(),
        ChannelData::Complex64(_) => todo!(),
        ChannelData::StringSBC(_) => (),
        ChannelData::StringUTF8(_) => (),
        ChannelData::StringUTF16(_) => (),
        ChannelData::ByteArray(_) => (),
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a as f64);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                map.insert("X".to_string(), *a);
                *new_array = compiled.eval(&slab, &mut map).expect("could not evaluate algebraic expression");
            });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDComplex16(_) => todo!(),
        ChannelData::ArrayDComplex32(_) => todo!(),
        ChannelData::ArrayDComplex64(_) => todo!(),
    }
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - *a) + y1 * (*a - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - *a) + y1 * (*a - x0)) / (x1 - x0)
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                                *y1
                            } else {
                                *y0
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                                *y1
                            } else {
                                *y0
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                                *y1
                            } else {
                                *y0
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (*a - x0) > (x1 - *a) {
                                *y1
                            } else {
                                *y0
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (a64 - x0) > (x1 - a64) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        if (*a - x0) > (x1 - *a) {
                            *y1
                        } else {
                            *y0
                        }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

/// Apply value to value without interpolation conversion to get physical data
fn value_range_to_value_table(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let mut val: Vec<(f64, f64, f64)> = Vec::new();
    for (a, b, c) in cc_val.iter().tuples::<(_, _, _)>() {
        val.push((*a, *b, *c));
    }
    let default_value = cc_val[cc_val.len() - 1];
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float32(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Float64(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && *a <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if *a <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                let a64 = *a as f64;
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if a64 <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = ArrayD::<f64>::zeros(a.shape());
            Zip::from(&mut new_array).and(a).for_each(|new_array, a| {
                *new_array =  match val.binary_search_by(|&(xi, _, _)| xi.partial_cmp(a).unwrap()) {
                    Ok(idx) => val[idx].2,
                    Err(0) => default_value,
                    Err(idx) if (idx >= val.len() && *a <= val[idx - 1].1) => val[idx - 1].2,
                    Err(idx) => {
                        if *a <= val[idx].1 {
                                val[idx].2
                            } else {
                                default_value
                            }
                        },
                    };
                });
            cn.data = ChannelData::ArrayDFloat64(new_array);
        },
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

/// value can be txt, scaling text or null
#[derive(Debug)]
enum TextOrScaleConversion {
    Txt(String),
    Scale(ConversionFunction),
    Nil,
}

/// Default value can be txt, scaling text or null
#[derive(Debug)]
enum DefaultTextOrScaleConversion {
    DefaultTxt(String),
    DefaultScale(ConversionFunction),
    Nil,
}

/// Apply value to text or scale conversion to get physical data
fn value_to_text(cn: &mut Cn4, cc_val: &Vec<f64>, cc_ref: &Vec<i64>, cycle_count: &u64, sharable: &SharableBlocks) {
    // table applicable only to integers, no canonization
    let mut table_int: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
    for (ind, val) in cc_val.iter().enumerate() {
        let val_i64 = (*val).round() as i64;
        if let Some(txt) = sharable.tx.get(&cc_ref[ind]) {
            table_int.insert(val_i64,TextOrScaleConversion::Txt(txt.0.clone()));
        } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]){
            let conv = conversion_function(cc, sharable);
            table_int.insert(val_i64,TextOrScaleConversion::Scale(conv));
        } else {
            table_int.insert(val_i64,TextOrScaleConversion::Nil);
        }
    }
    let def: DefaultTextOrScaleConversion;
    if let Some(txt) = sharable.tx.get(&cc_ref[cc_val.len()]) {
        def = DefaultTextOrScaleConversion::DefaultTxt(txt.0.clone());
    } else if let Some(cc) = sharable.cc.get(&cc_ref[cc_val.len()]){
        let conv = conversion_function(cc, sharable);
        def = DefaultTextOrScaleConversion::DefaultScale(conv);
    } else {
        def = DefaultTextOrScaleConversion::Nil;
    }

    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float16(a) => {
            // table for floating point comparison
            let mut table_float: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = (*val* 128.0).round() as i64; // Canonization
                if let Some(txt) = sharable.tx.get(&cc_ref[ind]) {
                    table_float.insert(ref_val,TextOrScaleConversion::Txt(txt.0.clone()));
                } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]){
                    let conv = conversion_function(cc, sharable);
                    table_float.insert(ref_val,TextOrScaleConversion::Scale(conv));
                } else {
                    table_float.insert(ref_val,TextOrScaleConversion::Nil);
                }
            }
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = (*a* 128.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float32(a) => {
            // table for floating point comparison
            let mut table_float: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = (*val* 1024.0 * 1024.0).round() as i64; // Canonization
                if let Some(txt) = sharable.tx.get(&cc_ref[ind]) {
                    table_float.insert(ref_val,TextOrScaleConversion::Txt(txt.0.clone()));
                } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]){
                    let conv = conversion_function(cc, sharable);
                    table_float.insert(ref_val,TextOrScaleConversion::Scale(conv));
                } else {
                    table_float.insert(ref_val,TextOrScaleConversion::Nil);
                }
            }
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = (*a* 1024.0 * 1024.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                if let Some(tosc) = table_int.get(&a) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = *a as i64;
                if let Some(tosc) = table_int.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float64(a) => {
            // table for floating point comparison
            let mut table_float: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = (*val* 1024.0 * 1024.0).round() as i64; // Canonization
                if let Some(txt) = sharable.tx.get(&cc_ref[ind]) {
                    table_float.insert(ref_val,TextOrScaleConversion::Txt(txt.0.clone()));
                } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]){
                    let conv = conversion_function(cc, sharable);
                    table_float.insert(ref_val,TextOrScaleConversion::Scale(conv));
                } else {
                    table_float.insert(ref_val,TextOrScaleConversion::Nil);
                }
            }
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let ref_val = (*a* 1024.0 * 1024.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Complex16(_) => (),
        ChannelData::Complex32(_) => (),
        ChannelData::Complex64(_) => (),
        ChannelData::StringSBC(_) => (),
        ChannelData::StringUTF8(_) => (),
        ChannelData::StringUTF16(_) => (),
        ChannelData::ByteArray(_) => (),
        ChannelData::ArrayDInt8(_) => (),
        ChannelData::ArrayDUInt8(_) => (),
        ChannelData::ArrayDInt16(_) => (),
        ChannelData::ArrayDUInt16(_) => (),
        ChannelData::ArrayDFloat16(_) => (),
        ChannelData::ArrayDInt24(_) => (),
        ChannelData::ArrayDUInt24(_) => (),
        ChannelData::ArrayDInt32(_) => (),
        ChannelData::ArrayDUInt32(_) => (),
        ChannelData::ArrayDFloat32(_) => (),
        ChannelData::ArrayDInt48(_) => (),
        ChannelData::ArrayDUInt48(_) => (),
        ChannelData::ArrayDInt64(_) => (),
        ChannelData::ArrayDUInt64(_) => (),
        ChannelData::ArrayDFloat64(_) => (),
        ChannelData::ArrayDComplex16(_) => (),
        ChannelData::ArrayDComplex32(_) => (),
        ChannelData::ArrayDComplex64(_) => (),
    }
}

/// enum of conversion functions for unique value
#[derive(Debug)]
enum ConversionFunction {
    Identity,
    Linear(f64, f64),
    Rational(f64, f64, f64, f64, f64, f64),
    Algebraic(Instruction, Slab),
}

/// conversion function of single value (not arrays)
fn conversion_function(cc: &Cc4Block, sharable: &SharableBlocks) -> ConversionFunction {
    match &cc.cc_val {
        CcVal::Real(cc_val) => {
            match cc.cc_type {
                0 => ConversionFunction::Identity,
                1 => ConversionFunction::Linear(cc_val[0], cc_val[1]),
                2 => ConversionFunction::Rational(cc_val[0], cc_val[1], cc_val[2], cc_val[3], cc_val[4], cc_val[5]),
                3 => {
                    if !&cc.cc_ref.is_empty() {
                        if let Some(formulae) = sharable.tx.get(&cc.cc_ref[0]) {
                            let parser = fasteval::Parser::new();
                            let mut slab = fasteval::Slab::new();
                            let compiled = parser.parse(&formulae.0, &mut slab.ps)
                                .expect("error parsing formulae for conversion")
                                .from(&slab.ps).compile(&slab.ps, &mut slab.cs);
                            ConversionFunction::Algebraic(compiled, slab)} else {ConversionFunction::Identity}
                        } else {ConversionFunction::Identity}
                    },
                _ => ConversionFunction::Identity,
            }
        }
        CcVal::Uint(_) => ConversionFunction::Identity,
    }
}

/// conversion function implmentation for single value (not arrays)
impl ConversionFunction {
    fn eval_to_txt(&self, a: f64) -> String {
        match self {
            ConversionFunction::Identity => a.to_string(),
            ConversionFunction::Linear(p1, p2) => (a * p2 + p1).to_string(),
            ConversionFunction::Rational(p1, p2, p3, p4, p5, p6) => {
                let a_2 = f64::powi(a, 2);
                ((a_2 * p1 + a * p2 + p3) / (a_2 * p4 + a * p5 + p6)).to_string()},
            ConversionFunction::Algebraic(compiled, slab) => {
                let mut map: BTreeMap<String, f64> = BTreeMap::new();
                map.insert("X".to_string(), a);
                let result: f64 = compiled.eval(slab, &mut map).expect("could not evaluate algebraic expression");
                result.to_string()
            },
        }
    }
}

/// keys range struct
#[derive(Debug)]
struct KeyRange {
    min: f64,
    max: f64,
}

/// Apply value range to text or scale conversion to get physical data
fn value_range_to_text(cn: &mut Cn4, cc_val: &Vec<f64>, cc_ref: &Vec<i64>, cycle_count: &u64, sharable: &SharableBlocks) {
    let n_keys = cc_val.len() / 2;
    let mut keys: Vec<KeyRange> = Vec::with_capacity(n_keys);
    for (key_min, key_max) in cc_val.into_iter().tuples() {
        let key: KeyRange = KeyRange {
            min: *key_min,
            max: *key_max,
        };
        keys.push(key);
    }
    let mut txt: Vec<TextOrScaleConversion> = Vec::with_capacity(n_keys);
    for pointer in cc_ref.iter() {
        if let Some(t) = sharable.tx.get(pointer) {
            txt.push(TextOrScaleConversion::Txt(t.0.clone()));
        } else if let Some(cc) = sharable.cc.get(pointer){
            let conv = conversion_function(cc, sharable);
            txt.push(TextOrScaleConversion::Scale(conv));
        } else {
            txt.push(TextOrScaleConversion::Nil);
        }
    }
    let def: DefaultTextOrScaleConversion;
    if let Some(t) = sharable.tx.get(&cc_ref[n_keys]) {
        def=DefaultTextOrScaleConversion::DefaultTxt(t.0.clone());
    } else if let Some(cc) = sharable.cc.get(&cc_ref[n_keys]){
        let conv = conversion_function(cc, sharable);
        def=DefaultTextOrScaleConversion::DefaultScale(conv);
    } else {
        def=DefaultTextOrScaleConversion::Nil;
    }

    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::UInt64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_array,a| {
                let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a && *a<= x.1.max });
                if let Some(key) = matched_key {
                    match &txt[key.0] {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        },
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        },
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a as f64);
                        },
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(_) => {},
        ChannelData::ArrayDUInt8(_) => {},
        ChannelData::ArrayDInt16(_) => {},
        ChannelData::ArrayDUInt16(_) => {},
        ChannelData::ArrayDFloat16(_) => {},
        ChannelData::ArrayDInt24(_) => {},
        ChannelData::ArrayDUInt24(_) => {},
        ChannelData::ArrayDInt32(_) => {},
        ChannelData::ArrayDUInt32(_) => {},
        ChannelData::ArrayDFloat32(_) => {},
        ChannelData::ArrayDInt48(_) => {},
        ChannelData::ArrayDUInt48(_) => {},
        ChannelData::ArrayDInt64(_) => {},
        ChannelData::ArrayDUInt64(_) => {},
        ChannelData::ArrayDFloat64(_) => {},
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

/// Apply text to value conversion to get physical data
fn text_to_value(cn: &mut Cn4, cc_val: &Vec<f64>, cc_ref: &Vec<i64>, cycle_count: &u64, sharable: &SharableBlocks) {
    let mut table: HashMap<String, f64> = HashMap::with_capacity(cc_ref.len());
    for (ind, ccref) in cc_ref.iter().enumerate() {
        if let Some(txt) = sharable.tx.get(ccref) {
            table.insert(txt.0.clone(), cc_val[ind]);
        }
    }
    let default = cc_val[cc_val.len() - 1];
    match &mut cn.data {
        ChannelData::Int8(_) => {},
        ChannelData::UInt8(_) => {},
        ChannelData::Int16(_) => {},
        ChannelData::UInt16(_) => {},
        ChannelData::Float16(_) => {},
        ChannelData::Int24(_) => {},
        ChannelData::UInt24(_) => {},
        ChannelData::Int32(_) => {},
        ChannelData::UInt32(_) => {},
        ChannelData::Float32(_) => {},
        ChannelData::Int48(_) => {},
        ChannelData::UInt48(_) => {},
        ChannelData::Int64(_) => {},
        ChannelData::UInt64(_) => {},
        ChannelData::Float64(_) => {},
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    *new_a = *val;
                } else {
                    *new_a = default;
                }
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::StringUTF8(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    *new_a = *val;
                } else {
                    *new_a = default;
                }
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::StringUTF16(a) => {
            let mut new_array = Array1::<f64>::zeros((*cycle_count as usize,));
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    *new_a = *val;
                } else {
                    *new_a = default;
                }
            });
            cn.data = ChannelData::Float64(new_array);
        },
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(_) => {},
        ChannelData::ArrayDUInt8(_) => {},
        ChannelData::ArrayDInt16(_) => {},
        ChannelData::ArrayDUInt16(_) => {},
        ChannelData::ArrayDFloat16(_) => {},
        ChannelData::ArrayDInt24(_) => {},
        ChannelData::ArrayDUInt24(_) => {},
        ChannelData::ArrayDInt32(_) => {},
        ChannelData::ArrayDUInt32(_) => {},
        ChannelData::ArrayDFloat32(_) => {},
        ChannelData::ArrayDInt48(_) => {},
        ChannelData::ArrayDUInt48(_) => {},
        ChannelData::ArrayDInt64(_) => {},
        ChannelData::ArrayDUInt64(_) => {},
        ChannelData::ArrayDFloat64(_) => {},
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

/// Apply text to text conversion to get physical data
fn text_to_text(cn: &mut Cn4, cc_ref: &Vec<i64>, cycle_count: &u64, sharable: &SharableBlocks) {
    let pairs: Vec<(&i64, &i64)> = cc_ref.iter().tuples().collect();
    let mut table: HashMap<String, Option<String>> = HashMap::with_capacity(cc_ref.len());
    for ccref in pairs.iter() {
        if let Some(key) = sharable.tx.get(ccref.0) {
            if let Some(txt) = sharable.tx.get(ccref.1) {
                table.insert(key.0.clone(), Some(txt.0.clone()));
            } else {
                table.insert(key.0.clone(), None);
            }
        }
    }
    let mut default: Option<String> = None;
    if let Some(txt) = sharable.tx.get(&cc_ref[(cc_ref.len() - 1) as usize]) {
        default = Some(txt.0.clone());
    }
    match &mut cn.data {
        ChannelData::Int8(_) => {},
        ChannelData::UInt8(_) => {},
        ChannelData::Int16(_) => {},
        ChannelData::UInt16(_) => {},
        ChannelData::Float16(_) => {},
        ChannelData::Int24(_) => {},
        ChannelData::UInt24(_) => {},
        ChannelData::Int32(_) => {},
        ChannelData::UInt32(_) => {},
        ChannelData::Float32(_) => {},
        ChannelData::Int48(_) => {},
        ChannelData::UInt48(_) => {},
        ChannelData::Int64(_) => {},
        ChannelData::UInt64(_) => {},
        ChannelData::Float64(_) => {},
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    if let Some(txt) = val.clone() {
                        *new_a = txt;
                    } else {
                        *new_a = a.clone();
                    }
                } else {
                    if let Some(tx) = default.clone() {
                        *new_a = tx.clone();
                    } else {
                        *new_a = a.clone();
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::StringUTF8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    if let Some(txt) = val.clone() {
                        *new_a = txt;
                    } else {
                        *new_a = a.clone();
                    }
                } else {
                    if let Some(tx) = default.clone() {
                        *new_a = tx.clone();
                    } else {
                        *new_a = a.clone();
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::StringUTF16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                if let Some(val) = table.get(a) {
                    if let Some(txt) = val.clone() {
                        *new_a = txt;
                    } else {
                        *new_a = a.clone();
                    }
                } else {
                    if let Some(tx) = default.clone() {
                        *new_a = tx.clone();
                    } else {
                        *new_a = a.clone();
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(_) => {},
        ChannelData::ArrayDUInt8(_) => {},
        ChannelData::ArrayDInt16(_) => {},
        ChannelData::ArrayDUInt16(_) => {},
        ChannelData::ArrayDFloat16(_) => {},
        ChannelData::ArrayDInt24(_) => {},
        ChannelData::ArrayDUInt24(_) => {},
        ChannelData::ArrayDInt32(_) => {},
        ChannelData::ArrayDUInt32(_) => {},
        ChannelData::ArrayDFloat32(_) => {},
        ChannelData::ArrayDInt48(_) => {},
        ChannelData::ArrayDUInt48(_) => {},
        ChannelData::ArrayDInt64(_) => {},
        ChannelData::ArrayDUInt64(_) => {},
        ChannelData::ArrayDFloat64(_) => {},
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}

enum ValueOrValueRangeToText {
    ValueToText(HashMap<i64, TextOrScaleConversion>, DefaultTextOrScaleConversion),
    ValueRangeToText(Vec<TextOrScaleConversion>, DefaultTextOrScaleConversion, Vec<KeyRange>),
}

fn bitfield_text_table(cn: &mut Cn4, cc_val: &Vec<u64>, cc_ref: &Vec<i64>, cycle_count: &u64, sharable: &SharableBlocks) {
    let mut table: Vec<(ValueOrValueRangeToText, Option<String>)> = Vec::with_capacity(cc_ref.len());
    for pointer in cc_ref.iter() {
        if let Some(cc) = sharable.cc.get(pointer){
            let name: Option<String>;
            if cc.cc_tx_name != 0 {
                if let Some(n) = sharable.tx.get(&cc.cc_tx_name) {
                    name = Some(n.0.clone());
                } else {name = None}
            } else {name = None}
            if cc.cc_type == 7 {
                match &cc.cc_val {
                    CcVal::Real(cc_val) => {
                        let mut table_int: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
                        for (ind, val) in cc_val.iter().enumerate() {
                            let val_i64 = (*val).round() as i64;
                            if let Some(txt) = sharable.tx.get(&cc.cc_ref[ind]) {
                                table_int.insert(val_i64,TextOrScaleConversion::Txt(txt.0.clone()));
                            } else if let Some(cc) = sharable.cc.get(&cc.cc_ref[ind]){
                                let conv = conversion_function(cc, sharable);
                                table_int.insert(val_i64,TextOrScaleConversion::Scale(conv));
                            } else {
                                table_int.insert(val_i64,TextOrScaleConversion::Nil);
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        if let Some(txt) = sharable.tx.get(&cc.cc_ref[cc_val.len()]) {
                            def = DefaultTextOrScaleConversion::DefaultTxt(txt.0.clone());
                        } else if let Some(cc) = sharable.cc.get(&cc.cc_ref[cc_val.len()]){
                            let conv = conversion_function(cc, sharable);
                            def = DefaultTextOrScaleConversion::DefaultScale(conv);
                        } else {
                            def = DefaultTextOrScaleConversion::Nil;
                        }
                        table.push((ValueOrValueRangeToText::ValueToText(table_int, def), name));
                    },
                    CcVal::Uint(_) => (),
                }
            } else if cc.cc_type == 8 {
                match &cc.cc_val {
                    CcVal::Real(cc_val) => {
                        let n_keys = cc_val.len() / 2;
                        let mut keys: Vec<KeyRange> = Vec::with_capacity(n_keys);
                        for (key_min, key_max) in cc_val.into_iter().tuples() {
                            let key: KeyRange = KeyRange {
                                min: *key_min,
                                max: *key_max,
                            };
                            keys.push(key);
                        }
                        let mut txt: Vec<TextOrScaleConversion> = Vec::with_capacity(n_keys);
                        for pointer in cc.cc_ref.iter() {
                            if let Some(t) = sharable.tx.get(pointer) {
                                txt.push(TextOrScaleConversion::Txt(t.0.clone()));
                            } else if let Some(ccc) = sharable.cc.get(pointer){
                                let conv = conversion_function(ccc, sharable);
                                txt.push(TextOrScaleConversion::Scale(conv));
                            } else {
                                txt.push(TextOrScaleConversion::Nil);
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        if let Some(t) = sharable.tx.get(&cc.cc_ref[n_keys]) {
                            def=DefaultTextOrScaleConversion::DefaultTxt(t.0.clone());
                        } else if let Some(ccc) = sharable.cc.get(&cc.cc_ref[n_keys]){
                            let conv = conversion_function(ccc, sharable);
                            def=DefaultTextOrScaleConversion::DefaultScale(conv);
                        } else {
                            def=DefaultTextOrScaleConversion::Nil;
                        }
                        table.push((ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name));
                    }
                    CcVal::Uint(_) => (),
                }
            }

        }
    }
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                for (ind, val) in cc_val.iter().enumerate() {
                    let masked_val = *a & (*val as u8);
                    match &table[ind] {
                        (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                            let ref_val = masked_val as i64;
                            if let Some(tosc) = table_int.get(&ref_val) {
                                match tosc {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        } else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        *new_a = txt.clone();
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        *new_a = conv.eval_to_txt(*a as f64);
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                        (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                            let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                            if let Some(key) = matched_key {
                                match &txt[key.0] {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Int8(_) => (),
        ChannelData::Int16(_) => (),
        ChannelData::UInt16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                for (ind, val) in cc_val.iter().enumerate() {
                    let masked_val = *a & (*val as u16);
                    match &table[ind] {
                        (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                            let ref_val = masked_val as i64;
                            if let Some(tosc) = table_int.get(&ref_val) {
                                match tosc {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        } else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        *new_a = txt.clone();
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        *new_a = conv.eval_to_txt(*a as f64);
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                        (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                            let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                            if let Some(key) = matched_key {
                                match &txt[key.0] {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float16(_) => {},
        ChannelData::Int24(_) => (),
        ChannelData::UInt24(_) => (),
        ChannelData::Int32(_) => (),
        ChannelData::UInt32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                for (ind, val) in cc_val.iter().enumerate() {
                    let masked_val = *a & (*val as u32);
                    match &table[ind] {
                        (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                            let ref_val = masked_val as i64;
                            if let Some(tosc) = table_int.get(&ref_val) {
                                match tosc {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        } else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        *new_a = txt.clone();
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        *new_a = conv.eval_to_txt(*a as f64);
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                        (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                            let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                            if let Some(key) = matched_key {
                                match &txt[key.0] {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float32(_) => {},
        ChannelData::Int48(_) => (),
        ChannelData::UInt48(_) => (),
        ChannelData::Int64(_) => (),
        ChannelData::UInt64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            Zip::from(&mut new_array).and(a).for_each(|new_a, a| {
                for (ind, val) in cc_val.iter().enumerate() {
                    let masked_val = *a & val;
                    match &table[ind] {
                        (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                            let ref_val = masked_val as i64;
                            if let Some(tosc) = table_int.get(&ref_val) {
                                match tosc {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        } else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        *new_a = txt.clone();
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        *new_a = conv.eval_to_txt(*a as f64);
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                        (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                            let matched_key = keys.iter().enumerate().find(|&x| { x.1.min <= *a as f64  && *a as f64 <= x.1.max });
                            if let Some(key) = matched_key {
                                match &txt[key.0] {
                                    TextOrScaleConversion::Txt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    TextOrScaleConversion::Scale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            } else {
                                match &def {
                                    DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                        } else {
                                            *new_a = format!("{} | {}", new_a, txt.clone());
                                        }
                                    },
                                    DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                        if let Some(n) = name {
                                            *new_a = format!("{} | {} = {}", new_a, n, conv.eval_to_txt(*a as f64));
                                        }  else {
                                            *new_a = format!("{} | {}", new_a, conv.eval_to_txt(*a as f64));
                                        }
                                    },
                                    _ => {
                                        *new_a = format!("{} | {}", new_a, "nothing");
                                    }
                                }
                            }
                        },
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        },
        ChannelData::Float64(_) => {},
        ChannelData::Complex16(_) => {},
        ChannelData::Complex32(_) => {},
        ChannelData::Complex64(_) => {},
        ChannelData::StringSBC(_) => {},
        ChannelData::StringUTF8(_) => {},
        ChannelData::StringUTF16(_) => {},
        ChannelData::ByteArray(_) => {},
        ChannelData::ArrayDInt8(_) => {},
        ChannelData::ArrayDUInt8(_) => {},
        ChannelData::ArrayDInt16(_) => {},
        ChannelData::ArrayDUInt16(_) => {},
        ChannelData::ArrayDFloat16(_) => {},
        ChannelData::ArrayDInt24(_) => {},
        ChannelData::ArrayDUInt24(_) => {},
        ChannelData::ArrayDInt32(_) => {},
        ChannelData::ArrayDUInt32(_) => {},
        ChannelData::ArrayDFloat32(_) => {},
        ChannelData::ArrayDInt48(_) => {},
        ChannelData::ArrayDUInt48(_) => {},
        ChannelData::ArrayDInt64(_) => {},
        ChannelData::ArrayDUInt64(_) => {},
        ChannelData::ArrayDFloat64(_) => {},
        ChannelData::ArrayDComplex16(_) => {},
        ChannelData::ArrayDComplex32(_) => {},
        ChannelData::ArrayDComplex64(_) => {},
    }
}