//! this modules implements functions to convert arrays into physical arrays using CCBlock
use itertools::Itertools;
use num::ToPrimitive;
use std::collections::BTreeMap;
use std::fmt::Display;

use crate::mdfinfo::mdfinfo3::{Cn3, Conversion, Dg3, SharableBlocks3};
use crate::mdfreader::channel_data::ChannelData;
use fasteval::Evaler;
use fasteval::{Compiler, Instruction, Slab};
use log::warn;
use rayon::prelude::*;

use crate::mdfreader::channel_data::ArrowComplex;

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg3, sharable: &SharableBlocks3) {
    for channel_group in dg.cg.values_mut() {
        let cycle_count = channel_group.block.cg_cycle_count;
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| !cn.data.is_empty())
            .for_each(|(_rec_pos, cn)| {
                // Could be empty if only initialised
                if let Some((_block, conv)) = sharable.cc.get(&cn.block1.cn_cc_conversion) {
                    match conv {
                        Conversion::Linear(cc_val) => linear_conversion(cn, cc_val, &cycle_count),
                        Conversion::TabularInterpolation(cc_val) => {
                            value_to_value_with_interpolation(cn, cc_val.clone(), &cycle_count)
                        }
                        Conversion::Tabular(cc_val) => {
                            value_to_value_without_interpolation(cn, cc_val.clone(), &cycle_count)
                        }
                        Conversion::Rational(cc_val) => {
                            rational_conversion(cn, cc_val, &cycle_count)
                        }
                        Conversion::Formula(formula) => {
                            algebraic_conversion(cn, formula, &cycle_count)
                        }
                        Conversion::Identity => {}
                        Conversion::Polynomial(cc_val) => {
                            polynomial_conversion(cn, cc_val, &cycle_count)
                        }
                        Conversion::Exponential(cc_val) => {
                            exponential_conversion(cn, cc_val, &cycle_count)
                        }
                        Conversion::Logarithmic(cc_val) => {
                            logarithmic_conversion(cn, cc_val, &cycle_count)
                        }
                        Conversion::TextTable(cc_val_ref) => {
                            value_to_text(cn, cc_val_ref, &cycle_count)
                        }
                        Conversion::TextRangeTable(cc_val_ref) => {
                            value_range_to_text(cn, cc_val_ref, &cycle_count)
                        }
                    }
                }
            })
    }
}

/// Apply linear conversion to get physical data
fn linear_conversion(cn: &mut Cn3, cc_val: &[f64], cycle_count: &u32) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && num::abs(p2 - 1.0) < 1e-12) {
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int8(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int16(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt16(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float16(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int24(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt24(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int32(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt32(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float32(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int48(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt48(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Int64(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::UInt64(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = (*a as f64) * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::Float64(a) => {
                let mut new_array = vec![0f64; *cycle_count as usize];
                new_array
                    .iter_mut()
                    .zip(a)
                    .for_each(|(new_array, a)| *new_array = *a * p2 + p1);
                cn.data = ChannelData::Float64(new_array);
            }
            ChannelData::ArrayDUInt8(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt8(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt16(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDUInt16(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDFloat16(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt24(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDUInt24(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt32(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDUInt32(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDFloat32(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt48(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDUInt48(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDInt64(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDUInt64(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a as f64 * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDFloat64(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = *a * p2 + p1;
                });
                cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()))
            }
            ChannelData::ArrayDComplex16(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array
                    .iter_mut()
                    .zip(&a.0 .0)
                    .for_each(|(new_array, a)| *new_array = *a as f64 * p2 + p1);
                cn.data =
                    ChannelData::ArrayDComplex64((ArrowComplex::<f64>(new_array), a.1.clone()));
            }
            ChannelData::ArrayDComplex32(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array
                    .iter_mut()
                    .zip(&a.0 .0)
                    .for_each(|(new_array, a)| *new_array = *a as f64 * p2 + p1);
                cn.data =
                    ChannelData::ArrayDComplex64((ArrowComplex::<f64>(new_array), a.1.clone()));
            }
            ChannelData::ArrayDComplex64(a) => {
                let mut new_array = vec![0f64; a.0.len()];
                new_array
                    .iter_mut()
                    .zip(&a.0 .0)
                    .for_each(|(new_array, a)| *new_array = *a * p2 + p1);
                cn.data =
                    ChannelData::ArrayDComplex64((ArrowComplex::<f64>(new_array), a.1.clone()));
            }
            _ => warn!(
                "not possible to apply linear conversion to the data type of channel {}",
                cn.unique_name,
            ),
        }
    }
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn3, cc_val: &[f64], cycle_count: &u32) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];

    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m_2 = f64::powi(*a, 2);
                *new_array = (m_2 * p1 + *a * p2 + p1) / (m_2 * p4 + *a * p5 + p6)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                let m_2 = f64::powi(m, 2);
                *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m_2 = f64::powi(*a, 2);
                *new_array = (m_2 * p1 + *a * p2 + p1) / (m_2 * p4 + *a * p5 + p6)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _ => warn!(
            "not possible to apply ratioanl conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply polynomial conversion to get physical data
fn polynomial_conversion(cn: &mut Cn3, cc_val: &[f64], cycle_count: &u32) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];

    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                *new_array = (p2 - (p4 * (*a - p5 - p6))) / (p3 * (*a - p5 - p6) - p1)
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let m = *a as f64;
                *new_array = (p2 - (p4 * (m - p5 - p6))) / (p3 * (m - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                *new_array = (p2 - (p4 * (*a - p5 - p6))) / (p3 * (*a - p5 - p6) - p1)
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _ => warn!(
            "not possible to apply polynomial conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply exponential conversion to get physical data
fn exponential_conversion(cn: &mut Cn3, cc_val: &[f64], cycle_count: &u32) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];

    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    *new_array = (((*a - p7) * p6 - p3) / p1).ln() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a - p7) - p6) / p4).ln() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a - p7) * p6 - p3) / p1).ln() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a - p7) - p6) / p4).ln() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _ => warn!(
            "not possible to apply exponential conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply exponential conversion to get physical data
fn logarithmic_conversion(cn: &mut Cn3, cc_val: &[f64], cycle_count: &u32) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];

    match &mut cn.data {
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = (((m - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    let m = *a as f64;
                    *new_array = ((p3 / (m - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            if p4 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    *new_array = (((*a - p7) * p6 - p3) / p1).exp() / p2;
                });
                cn.data = ChannelData::Float64(new_array);
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a - p7) - p6) / p4).exp() / p5;
                });
                cn.data = ChannelData::Float64(new_array);
            }
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a as f64 - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a as f64 - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            if p4 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = (((*a - p7) * p6 - p3) / p1).exp() / p2;
                });
            } else if p1 == 0.0 {
                new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                    *new_array = ((p3 / (*a - p7) - p6) / p4).exp() / p5;
                });
            }
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _ => warn!(
            "not possible to apply logarithmic conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Generic function calculating algebraic expression
#[inline]
fn alegbraic_conversion_calculation<T: ToPrimitive + Display>(
    array: &Vec<T>,
    compiled: &Instruction,
    slab: &Slab,
    cycle_count: &usize,
    formulae: &str,
    name: &str,
) -> Vec<f64> {
    let mut map = BTreeMap::new();
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_array, a)| {
        map.insert("X".to_string(), a.to_f64().unwrap_or_default());
        match compiled.eval(slab, &mut map) {
            Ok(res) => *new_array = res,
            Err(error_message) => {
                *new_array = a.to_f64().unwrap_or_default();
                warn!(
                    "{}\n Could not compute formulae {} for channel {} and value {}",
                    error_message, formulae, name, a
                );
            }
        }
    });
    new_array
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn3, formulae: &str, cycle_count: &u32) {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let compiled_instruction = parser.parse(formulae, &mut slab.ps);
    if let Ok(compiled_instruct) = compiled_instruction {
        let compiled = compiled_instruct
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs);
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int8(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int16(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::UInt16(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Float16(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int24(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::UInt24(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int32(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::UInt32(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Float32(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int48(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::UInt48(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Int64(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::UInt64(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::Float64(a) => {
                cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                    a,
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                ));
            }
            ChannelData::ArrayDInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDFloat16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDInt24(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt24(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDFloat32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDInt48(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt48(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDUInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            ChannelData::ArrayDFloat64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    alegbraic_conversion_calculation(
                        &a.0,
                        &compiled,
                        &slab,
                        &(*cycle_count as usize),
                        formulae,
                        &cn.unique_name,
                    ),
                    a.1.clone(),
                ));
            }
            _ => warn!(
                "not possible to apply algebraic conversion to the data type of channel {}",
                cn.unique_name,
            ),
        }
    } else if let Err(error_message) = compiled_instruction {
        // could not parse the formulae, probably some function or syntax not yet implementated by fasteval
        warn!(
            "{}\n Could not parse formulae {} for channel {}",
            error_message, formulae, cn.unique_name
        );
    }
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(cn: &mut Cn3, cc_val: Vec<f64>, cycle_count: &u32) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(a).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - *a) + y1 * (*a - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(a).expect("Could not compare values")
                }) {
                    Ok(idx) => *val[idx].1,
                    Err(0) => *val[0].1,
                    Err(idx) if idx >= val.len() => *val[idx - 1].1,
                    Err(idx) => {
                        let (x0, y0) = val[idx - 1];
                        let (x1, y1) = val[idx];
                        (y0 * (x1 - *a) + y1 * (*a - x0)) / (x1 - x0)
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _=> warn!(
            "not possible to apply value to value with interpolation conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(cn: &mut Cn3, cc_val: Vec<f64>, cycle_count: &u32) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt8(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![0f64; *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(a).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::Float64(new_array);
        }
        ChannelData::ArrayDInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt8(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat16(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt24(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat32(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt48(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDUInt64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                let a64 = *a as f64;
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(&a64).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        ChannelData::ArrayDFloat64(a) => {
            let mut new_array = vec![0f64; a.0.len()];
            new_array.iter_mut().zip(&a.0).for_each(|(new_array, a)| {
                *new_array = match val.binary_search_by(|&(xi, _)| {
                    xi.partial_cmp(a).expect("Could not compare values")
                }) {
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
                    }
                };
            });
            cn.data = ChannelData::ArrayDFloat64((new_array, a.1.clone()));
        }
        _ => warn!(
            "not possible to apply value to value without interpolation conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply value to text or scale conversion to get physical data
fn value_to_text(cn: &mut Cn3, cc_val_ref: &[(f64, String)], cycle_count: &u32) {
    // identify max string length in cc_val_ref
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_str, val)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *val as f64);
                if let Some(key) = matched_key {
                    *new_str = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt8(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_str, val)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *val as f64);
                if let Some(key) = matched_key {
                    *new_str = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a as f64);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![String::with_capacity(32); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref.iter().find(|&x| x.0 == *a);
                if let Some(key) = matched_key {
                    *new_array = key.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        _ => warn!(
            "not possible to apply value to text conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}

/// Apply value range to text or scale conversion to get physical data
fn value_range_to_text(
    cn: &mut Cn3,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
    cycle_count: &u32,
) {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt8(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float16(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt24(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float32(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt48(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Int64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::UInt64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= (*a as f64)) && ((*a as f64) < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        ChannelData::Float64(a) => {
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let matched_key = cc_val_ref
                    .0
                    .iter()
                    .enumerate()
                    .find(|&x| (x.1 .0 <= *a) && (*a < x.1 .1));
                if let Some(key) = matched_key {
                    *new_array = key.1 .2.clone();
                } else {
                    *new_array = cc_val_ref.1.clone();
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        _ => warn!(
            "not possible to apply value to text conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
}
