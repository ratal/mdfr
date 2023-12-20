//! this modules implements functions to convert arrays into physical arrays using CCBlock
use itertools::Itertools;
use log::warn;
use num::{Integer, ToPrimitive};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Display;

use crate::mdfinfo::mdfinfo4::{Cc4Block, CcVal, Cn4, Dg4, SharableBlocks};
use crate::mdfreader::channel_data::ChannelData;
use fasteval::{Compiler, Evaler, Instruction, Slab};
use rayon::prelude::*;

use crate::mdfreader::channel_data::ArrowComplex;

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
                        1 => match &conv.cc_val {
                            CcVal::Real(cc_val) => linear_conversion(cn, cc_val, &cycle_count),
                            CcVal::Uint(_) => (),
                        },
                        2 => match &conv.cc_val {
                            CcVal::Real(cc_val) => rational_conversion(cn, cc_val, &cycle_count),
                            CcVal::Uint(_) => (),
                        },
                        3 => {
                            if !&conv.cc_ref.is_empty() {
                                if let Ok(Some(conv)) = sharable.get_tx(conv.cc_ref[0]) {
                                    algebraic_conversion(cn, &conv, &cycle_count)
                                }
                            }
                        }
                        4 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                value_to_value_with_interpolation(cn, cc_val.clone(), &cycle_count)
                            }
                            CcVal::Uint(_) => (),
                        },
                        5 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_to_value_without_interpolation(
                                cn,
                                cc_val.clone(),
                                &cycle_count,
                            ),
                            CcVal::Uint(_) => (),
                        },
                        6 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                value_range_to_value_table(cn, cc_val.clone(), &cycle_count)
                            }
                            CcVal::Uint(_) => (),
                        },
                        7 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                value_to_text(cn, cc_val, &conv.cc_ref, &cycle_count, sharable)
                            }
                            CcVal::Uint(_) => (),
                        },
                        8 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_range_to_text(
                                cn,
                                cc_val,
                                &conv.cc_ref,
                                &cycle_count,
                                sharable,
                            ),
                            CcVal::Uint(_) => (),
                        },
                        9 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                text_to_value(cn, cc_val, &conv.cc_ref, &cycle_count, sharable)
                            }
                            CcVal::Uint(_) => (),
                        },
                        10 => text_to_text(cn, &conv.cc_ref, &cycle_count, sharable),
                        11 => match &conv.cc_val {
                            CcVal::Real(_) => (),
                            CcVal::Uint(cc_val) => bitfield_text_table(
                                cn,
                                cc_val,
                                &conv.cc_ref,
                                &cycle_count,
                                sharable,
                            ),
                        },
                        0 => (),
                        _ => warn!(
                            "conversion type not recognised for channel {} not possible, type {}",
                            cn.unique_name, conv.cc_type,
                        ),
                    }
                }
            })
    }
}

/// Generic function calculating linear expression
#[inline]
fn linear_conversion_calculation<T: ToPrimitive>(
    array: &Vec<T>,
    p1: f64,
    p2: f64,
    cycle_count: &usize,
) -> Vec<f64> {
    let mut new_array = vec![0f64; *cycle_count];
    new_array
        .iter_mut()
        .zip(array)
        .for_each(|(new_array, a)| *new_array = a.to_f64().unwrap_or_default() * p2 + p1);
    new_array
}

/// Apply linear conversion to get physical data
fn linear_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && num::abs(p2 - 1.0) < 1e-12) {
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int8(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int16(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::UInt16(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Float16(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int24(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::UInt24(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int32(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::UInt32(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Float32(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int48(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::UInt48(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Int64(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::UInt64(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Float64(a) => {
                cn.data = ChannelData::Float64(linear_conversion_calculation(
                    a,
                    p1,
                    p2,
                    &(*cycle_count as usize),
                ));
            }
            ChannelData::Complex16(a) => {
                cn.data = ChannelData::Complex64(ArrowComplex(linear_conversion_calculation(
                    &a.0,
                    p1,
                    p2,
                    &a.0.len(),
                )))
            }
            ChannelData::Complex32(a) => {
                cn.data = ChannelData::Complex64(ArrowComplex(linear_conversion_calculation(
                    &a.0,
                    p1,
                    p2,
                    &a.0.len(),
                )))
            }
            ChannelData::Complex64(a) => {
                cn.data = ChannelData::Complex64(ArrowComplex(linear_conversion_calculation(
                    &a.0,
                    p1,
                    p2,
                    &a.0.len(),
                )))
            }
            ChannelData::ArrayDUInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDUInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDFloat16(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt24(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDUInt24(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDUInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDFloat32(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt48(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDUInt48(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDUInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDFloat64(a) => {
                cn.data = ChannelData::ArrayDFloat64((
                    linear_conversion_calculation(&a.0, p1, p2, &a.0.len()),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDComplex16(a) => {
                cn.data = ChannelData::ArrayDComplex64((
                    ArrowComplex(linear_conversion_calculation(&a.0 .0, p1, p2, &a.0.len())),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDComplex32(a) => {
                cn.data = ChannelData::ArrayDComplex64((
                    ArrowComplex(linear_conversion_calculation(&a.0 .0, p1, p2, &a.0.len())),
                    a.1.clone(),
                ))
            }
            ChannelData::ArrayDComplex64(a) => {
                cn.data = ChannelData::ArrayDComplex64((
                    ArrowComplex(linear_conversion_calculation(&a.0 .0, p1, p2, &a.0.len())),
                    a.1.clone(),
                ))
            }
            _ => warn!(
                "linear conversion of channel {} not possible, channel does not contain primitives",
                cn.unique_name
            ),
        }
    }
}

/// Generic function calculating rational expression
#[inline]
fn rational_conversion_calculation<T: ToPrimitive>(
    array: &Vec<T>,
    cc_val: &[f64],
    cycle_count: &usize,
) -> Vec<f64> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_array, a)| {
        let m = a.to_f64().unwrap_or_default();
        let m_2 = f64::powi(m, 2);
        *new_array = (m_2 * p1 + m * p2 + p3) / (m_2 * p4 + m * p5 + p6)
    });
    new_array
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn4, cc_val: &[f64], cycle_count: &u64) {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(rational_conversion_calculation(
                a,
                cc_val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Complex16(a) => {
            cn.data = ChannelData::Complex64(ArrowComplex(rational_conversion_calculation(
                &a.0,
                cc_val,
                &a.0.len(),
            )))
        }
        ChannelData::Complex32(a) => {
            cn.data = ChannelData::Complex64(ArrowComplex(rational_conversion_calculation(
                &a.0,
                cc_val,
                &a.0.len(),
            )))
        }
        ChannelData::Complex64(a) => {
            cn.data = ChannelData::Complex64(ArrowComplex(rational_conversion_calculation(
                &a.0,
                cc_val,
                &a.0.len(),
            )))
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDFloat16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDUInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDUInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                rational_conversion_calculation(&a.0, cc_val, &a.0.len()),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDComplex16(a) => {
            cn.data = ChannelData::ArrayDComplex64((
                ArrowComplex(rational_conversion_calculation(&a.0 .0, cc_val, &a.0.len())),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDComplex32(a) => {
            cn.data = ChannelData::ArrayDComplex64((
                ArrowComplex(rational_conversion_calculation(&a.0 .0, cc_val, &a.0.len())),
                a.1.clone(),
            ))
        }
        ChannelData::ArrayDComplex64(a) => {
            cn.data = ChannelData::ArrayDComplex64((
                ArrowComplex(rational_conversion_calculation(&a.0 .0, cc_val, &a.0.len())),
                a.1.clone(),
            ))
        }
        _ => warn!(
            "rational conversion of channel {} not possible, channel does not contain primitives",
            cn.unique_name
        ),
    }
}

/// Generic function calculating algebraic expression
#[inline]
fn alegbraic_conversion_calculation<T: ToPrimitive>(
    compiled: &Instruction,
    slab: &Slab,
    array: &Vec<T>,
    cycle_count: &usize,
) -> Vec<f64> {
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_a, a)| {
        let mut map = BTreeMap::new();
        let value = a.to_f64().unwrap_or_default();
        map.insert("X".to_string(), value);
        let val = compiled.eval(slab, &mut map);
        *new_a = match val {
            Ok(val) => val,
            Err(err) => {
                warn!(
                    "could not compute the value {:?} with expression {:?}, error {}",
                    value, compiled, err
                );
                value
            }
        }
    });
    new_array
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn4, formulae: &str, cycle_count: &u64) {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let compiled = parser.parse(formulae, &mut slab.ps);
    match compiled {
        Ok(c) => {
            let compiled = c.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
            match &mut cn.data {
                ChannelData::UInt8(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int8(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int16(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::UInt16(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Float16(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int24(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::UInt24(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int32(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::UInt32(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Float32(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int48(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::UInt48(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Int64(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::UInt64(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Float64(a) => {
                    cn.data = ChannelData::Float64(alegbraic_conversion_calculation(
                        &compiled,
                        &slab,
                        a,
                        &(*cycle_count as usize),
                    ));
                }
                ChannelData::Complex16(a) => cn.data = ChannelData::Complex64(ArrowComplex(alegbraic_conversion_calculation(
                    &compiled, &slab, &a.0, &a.0.len()
                ))),
                ChannelData::Complex32(a) => cn.data = ChannelData::Complex64(ArrowComplex(alegbraic_conversion_calculation(
                    &compiled, &slab, &a.0, &a.0.len()
                ))),
                ChannelData::Complex64(a) => cn.data = ChannelData::Complex64(ArrowComplex(alegbraic_conversion_calculation(
                    &compiled, &slab, &a.0, &a.0.len()
                ))),
                ChannelData::ArrayDInt8(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt8(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDInt16(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt16(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDFloat16(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDInt24(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt24(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDInt32(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt32(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDFloat32(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDInt48(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt48(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDInt64(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDUInt64(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDFloat64(a) => {
                    cn.data = ChannelData::ArrayDFloat64((
                        alegbraic_conversion_calculation(&compiled, &slab, &a.0, &a.0.len()),
                        a.1.clone(),
                    ));
                }
                ChannelData::ArrayDComplex16(a) => {
                    cn.data = ChannelData::ArrayDComplex64((
                        ArrowComplex(alegbraic_conversion_calculation(&compiled, &slab, &a.0.0, &a.0.len())),
                        a.1.clone(),
                    ))
                },
                ChannelData::ArrayDComplex32(a) => {
                    cn.data = ChannelData::ArrayDComplex64((
                        ArrowComplex(alegbraic_conversion_calculation(&compiled, &slab, &a.0.0, &a.0.len())),
                        a.1.clone(),
                    ))
                },
                ChannelData::ArrayDComplex64(a) => {
                    cn.data = ChannelData::ArrayDComplex64((
                        ArrowComplex(alegbraic_conversion_calculation(&compiled, &slab, &a.0.0, &a.0.len())),
                        a.1.clone(),
                    ))
                },
                _=> warn!(
                    "algebraic conversion of channel {} not possible, channel does not contain primitives",
                    cn.unique_name
                )
            }
        }
        Err(err) => {
            warn!(
                "could not compile algebraic conversion expression {}, {}, channel {} not converted",
                formulae, err, cn.unique_name
            )
        }
    }
}

/// Generic function calculating value to value interpolation
#[inline]
fn value_to_value_with_interpolation_calculation<T: ToPrimitive>(
    a: &Vec<T>,
    val: Vec<(&f64, &f64)>,
    cycle_count: &usize,
) -> Vec<f64> {
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
        let a64 = a.to_f64().unwrap_or_default();
        *new_array = match val
            .binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap_or(Ordering::Equal))
        {
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
    new_array
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(value_to_value_with_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_with_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        _ => warn!(
            "value to value with interpolation conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        )
    }
}

/// Generic function calculating value to value without interpolation
#[inline]
fn value_to_value_without_interpolation_calculation<T: ToPrimitive>(
    a: &Vec<T>,
    val: Vec<(&f64, &f64)>,
    cycle_count: &usize,
) -> Vec<f64> {
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
        let a64 = a.to_f64().unwrap_or_default();
        *new_array = match val
            .binary_search_by(|&(xi, _)| xi.partial_cmp(&a64).unwrap_or(Ordering::Equal))
        {
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
    new_array
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(value_to_value_without_interpolation_calculation(
                a,
                val,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_to_value_without_interpolation_calculation(&a.0, val, &a.0.len()),
                a.1.clone(),
            ));
        }
        _ => warn!(
            "value to value without interpolation conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        )
    }
}

/// Generic function calculating value range to value table without interpolation
#[inline]
fn value_range_to_value_table_calculation<T: ToPrimitive>(
    a: &Vec<T>,
    val: &[(f64, f64, f64)],
    default_value: &f64,
    cycle_count: &usize,
) -> Vec<f64> {
    let mut new_array = vec![0f64; *cycle_count];
    new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
        let a64 = a.to_f64().unwrap_or_default();
        *new_array = match val
            .binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a64).unwrap_or(Ordering::Equal))
        {
            Ok(idx) => val[idx].2,
            Err(0) => *default_value,
            Err(idx) if (idx >= val.len() && a64 <= val[idx - 1].1) => val[idx - 1].2,
            Err(idx) => {
                if a64 <= val[idx].1 {
                    val[idx].2
                } else {
                    *default_value
                }
            }
        };
    });
    new_array
}

/// Apply value range to value table without interpolation conversion to get physical data
fn value_range_to_value_table(cn: &mut Cn4, cc_val: Vec<f64>, cycle_count: &u64) {
    let mut val: Vec<(f64, f64, f64)> = Vec::new();
    for (a, b, c) in cc_val.iter().tuples::<(_, _, _)>() {
        val.push((*a, *b, *c));
    }
    let default_value = cc_val[cc_val.len() - 1];
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(value_range_to_value_table_calculation(
                a,
                &val,
                &default_value,
                &(*cycle_count as usize),
            ));
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat16(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt24(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt48(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64((
                value_range_to_value_table_calculation(&a.0, &val, &default_value, &a.0.len()),
                a.1.clone(),
            ));
        }
        _ => warn!(
            "value range to value conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        )
    }
}

/// value can be txt, scaling text or null
#[derive(Debug)]
enum TextOrScaleConversion {
    Txt(String),
    Scale(Box<ConversionFunction>),
    Nil,
}

/// Default value can be txt, scaling text or null
#[derive(Debug)]
enum DefaultTextOrScaleConversion {
    DefaultTxt(String),
    DefaultScale(Box<ConversionFunction>),
    Nil,
}

/// Generic function calculating integer value range to text
#[inline]
fn value_to_text_calculation_int<T: Sized + Display + ToPrimitive>(
    a: &Vec<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    def: &DefaultTextOrScaleConversion,
    cycle_count: &usize,
    sharable: &SharableBlocks,
) -> Vec<String> {
    // table applicable only to integers, no canonization
    let mut table_int: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
    for (ind, val) in cc_val.iter().enumerate() {
        let val_i64 = (*val).round() as i64;
        if let Ok(Some(txt)) = sharable.get_tx(cc_ref[ind]) {
            table_int.insert(val_i64, TextOrScaleConversion::Txt(txt));
        } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
            let conv = conversion_function(cc, sharable);
            table_int.insert(val_i64, TextOrScaleConversion::Scale(Box::new(conv)));
        } else {
            table_int.insert(val_i64, TextOrScaleConversion::Nil);
        }
    }
    let mut new_array = vec![String::new(); *cycle_count];
    new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
        let ref_val = a.to_i64().unwrap_or_default();
        if let Some(tosc) = table_int.get(&ref_val) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    *new_array = txt.clone();
                }
                TextOrScaleConversion::Scale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    *new_array = txt.clone();
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        }
    });
    new_array
}

/// Generic function calculating float value range to text
#[inline]
fn value_to_text_calculation_f32(
    a: &Vec<f32>,
    cc_val: &[f64],
    cc_ref: &[i64],
    canonization_value: f64,
    def: &DefaultTextOrScaleConversion,
    cycle_count: &usize,
    sharable: &SharableBlocks,
) -> Vec<String> {
    // table for floating point comparison
    let mut table_float: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
    for (ind, val) in cc_val.iter().enumerate() {
        let ref_val = (*val * canonization_value).round() as i64; // Canonization
        if let Ok(Some(txt)) = sharable.get_tx(cc_ref[ind]) {
            table_float.insert(ref_val, TextOrScaleConversion::Txt(txt));
        } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
            let conv = conversion_function(cc, sharable);
            table_float.insert(ref_val, TextOrScaleConversion::Scale(Box::new(conv)));
        } else {
            table_float.insert(ref_val, TextOrScaleConversion::Nil);
        }
    }
    let mut new_array = vec![String::new(); *cycle_count];
    new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
        let ref_val = (*a * canonization_value as f32)
            .round()
            .to_i64()
            .unwrap_or_default();
        if let Some(tosc) = table_float.get(&ref_val) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    *new_array = txt.clone();
                }
                TextOrScaleConversion::Scale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    *new_array = txt.clone();
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        }
    });
    new_array
}

/// Apply value to text or scale conversion to get physical data
fn value_to_text(
    cn: &mut Cn4,
    cc_val: &[f64],
    cc_ref: &[i64],
    cycle_count: &u64,
    sharable: &SharableBlocks,
) {
    let def: DefaultTextOrScaleConversion;
    if let Ok(Some(txt)) = sharable.get_tx(cc_ref[cc_val.len()]) {
        def = DefaultTextOrScaleConversion::DefaultTxt(txt);
    } else if let Some(cc) = sharable.cc.get(&cc_ref[cc_val.len()]) {
        let conv = conversion_function(cc, sharable);
        def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
    } else {
        def = DefaultTextOrScaleConversion::Nil;
    }
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_f32(
                a,
                cc_val,
                cc_ref,
                128.0f64,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_f32(
                a,
                cc_val,
                cc_ref,
                1048576.0f64,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::StringUTF8(value_to_text_calculation_int(
                a,
                cc_val,
                cc_ref,
                &def,
                &(*cycle_count as usize),
                sharable,
            ));
        }
        ChannelData::Float64(a) => {
            // table for floating point comparison
            let mut table_float: HashMap<i64, TextOrScaleConversion> =
                HashMap::with_capacity(cc_val.len());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = (*val * 1024.0 * 1024.0).round() as i64; // Canonization
                if let Ok(Some(txt)) = sharable.get_tx(cc_ref[ind]) {
                    table_float.insert(ref_val, TextOrScaleConversion::Txt(txt));
                } else if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
                    let conv = conversion_function(cc, sharable);
                    table_float.insert(ref_val, TextOrScaleConversion::Scale(Box::new(conv)));
                } else {
                    table_float.insert(ref_val, TextOrScaleConversion::Nil);
                }
            }
            let mut new_array = vec![String::new(); *cycle_count as usize];
            new_array.iter_mut().zip(a).for_each(|(new_array, a)| {
                let ref_val = (*a * 1024.0 * 1024.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            *new_array = txt.clone();
                        }
                        TextOrScaleConversion::Scale(conv) => {
                            *new_array = conv.eval_to_txt(*a);
                        }
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            *new_array = txt.clone();
                        }
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            *new_array = conv.eval_to_txt(*a);
                        }
                        _ => {
                            *new_array = a.to_string();
                        }
                    }
                }
            });
            cn.data = ChannelData::StringUTF8(new_array);
        }
        _ => warn!(
            "value to text conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
}

/// enum of conversion functions for unique value
#[derive(Debug)]
enum ConversionFunction {
    Identity,
    Linear(f64, f64),
    Rational(f64, f64, f64, f64, f64, f64),
    Algebraic(Instruction, Box<Slab>),
}

/// conversion function of single value (not arrays)
fn conversion_function(cc: &Cc4Block, sharable: &SharableBlocks) -> ConversionFunction {
    match &cc.cc_val {
        CcVal::Real(cc_val) => match cc.cc_type {
            0 => ConversionFunction::Identity,
            1 => ConversionFunction::Linear(cc_val[0], cc_val[1]),
            2 => ConversionFunction::Rational(
                cc_val[0], cc_val[1], cc_val[2], cc_val[3], cc_val[4], cc_val[5],
            ),
            3 => {
                if !&cc.cc_ref.is_empty() {
                    if let Ok(Some(formulae)) = sharable.get_tx(cc.cc_ref[0]) {
                        let parser = fasteval::Parser::new();
                        let mut slab = fasteval::Slab::new();
                        let compiled = parser.parse(&formulae, &mut slab.ps);
                        match compiled {
                            Ok(c) => ConversionFunction::Algebraic(
                                c.from(&slab.ps).compile(&slab.ps, &mut slab.cs),
                                Box::new(slab),
                            ),
                            Err(e) => {
                                warn!("Error parsing formulae {}, error {}", formulae, e);
                                ConversionFunction::Identity
                            }
                        }
                    } else {
                        ConversionFunction::Identity
                    }
                } else {
                    ConversionFunction::Identity
                }
            }
            _ => ConversionFunction::Identity,
        },
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
                ((a_2 * p1 + a * p2 + p3) / (a_2 * p4 + a * p5 + p6)).to_string()
            }
            ConversionFunction::Algebraic(compiled, slab) => {
                let mut map: BTreeMap<String, f64> = BTreeMap::new();
                map.insert("X".to_string(), a);
                let result = compiled.eval(slab, &mut map);
                match result {
                    Ok(res) => res.to_string(),
                    Err(e) => {
                        warn!(
                            "could not evaluate algebraic expression for {}, error {}",
                            a, e
                        );
                        a.to_string()
                    }
                }
            }
        }
    }
}

/// keys range struct
#[derive(Debug)]
struct KeyRange {
    min: f64,
    max: f64,
}

/// Generic function calculating value range to text
#[inline]
fn value_range_to_text_calculation<T: Sized + Display + ToPrimitive>(
    array: &Vec<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    cycle_count: usize,
    sharable: &SharableBlocks,
) -> Vec<String> {
    let n_keys = cc_val.len() / 2;
    let mut keys: Vec<KeyRange> = Vec::with_capacity(n_keys);
    for (key_min, key_max) in cc_val.iter().tuples() {
        let key: KeyRange = KeyRange {
            min: *key_min,
            max: *key_max,
        };
        keys.push(key);
    }
    let mut txt: Vec<TextOrScaleConversion> = Vec::with_capacity(n_keys);
    for pointer in cc_ref.iter() {
        if let Ok(Some(t)) = sharable.get_tx(*pointer) {
            txt.push(TextOrScaleConversion::Txt(t));
        } else if let Some(cc) = sharable.cc.get(pointer) {
            let conv = conversion_function(cc, sharable);
            txt.push(TextOrScaleConversion::Scale(Box::new(conv)));
        } else {
            txt.push(TextOrScaleConversion::Nil);
        }
    }
    let def: DefaultTextOrScaleConversion;
    if let Ok(Some(t)) = sharable.get_tx(cc_ref[n_keys]) {
        def = DefaultTextOrScaleConversion::DefaultTxt(t);
    } else if let Some(cc) = sharable.cc.get(&cc_ref[n_keys]) {
        let conv = conversion_function(cc, sharable);
        def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
    } else {
        def = DefaultTextOrScaleConversion::Nil;
    }
    let mut new_array = vec![String::new(); cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_array, a)| {
        let matched_key = keys.iter().enumerate().find(|&x| {
            x.1.min <= a.to_f64().unwrap_or_default() && a.to_f64().unwrap_or_default() <= x.1.max
        });
        if let Some(key) = matched_key {
            match &txt[key.0] {
                TextOrScaleConversion::Txt(txt) => {
                    *new_array = txt.clone();
                }
                TextOrScaleConversion::Scale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    *new_array = txt.clone();
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    *new_array = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                }
                _ => {
                    *new_array = a.to_string();
                }
            }
        }
    });
    new_array
}

/// Apply value range to text or scale conversion to get physical data
fn value_range_to_text(
    cn: &mut Cn4,
    cc_val: &[f64],
    cc_ref: &[i64],
    cycle_count: &u64,
    sharable: &SharableBlocks,
) {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Float16(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Int24(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt24(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Int48(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt48(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::StringUTF8(value_range_to_text_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        _ => warn!(
            "value range to text conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
}

/// Generic function calculating text to value
#[inline]
fn text_to_value_calculation(
    array: &Vec<String>,
    cc_val: &[f64],
    cc_ref: &[i64],
    cycle_count: usize,
    sharable: &SharableBlocks,
) -> Vec<f64> {
    let mut table: HashMap<String, f64> = HashMap::with_capacity(cc_ref.len());
    for (ind, ccref) in cc_ref.iter().enumerate() {
        if let Ok(Some(txt)) = sharable.get_tx(*ccref) {
            table.insert(txt, cc_val[ind]);
        }
    }
    let default = cc_val[cc_val.len() - 1];
    let mut new_array = vec![0f64; cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_a, a)| {
        if let Some(val) = table.get(a) {
            *new_a = *val;
        } else {
            *new_a = default;
        }
    });
    new_array
}

/// Apply text to value conversion to get physical data
fn text_to_value(
    cn: &mut Cn4,
    cc_val: &[f64],
    cc_ref: &[i64],
    cycle_count: &u64,
    sharable: &SharableBlocks,
) {
    match &mut cn.data {
        ChannelData::StringSBC(a) => {
            cn.data = ChannelData::Float64(text_to_value_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::StringUTF8(a) => {
            cn.data = ChannelData::Float64(text_to_value_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::StringUTF16(a) => {
            cn.data = ChannelData::Float64(text_to_value_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        _ => warn!(
            "text conversion into value of channel {} not possible, channel does not contain string",
            cn.unique_name
        ),
    }
}

/// Generic function calculating text to value
#[inline]
fn text_to_text_calculation(
    array: &Vec<String>,
    cc_ref: &[i64],
    cycle_count: usize,
    sharable: &SharableBlocks,
) -> Vec<String> {
    let pairs: Vec<(&i64, &i64)> = cc_ref.iter().tuples().collect();
    let mut table: HashMap<String, Option<String>> = HashMap::with_capacity(cc_ref.len());
    for ccref in pairs.iter() {
        if let Ok(Some(key)) = sharable.get_tx(*ccref.0) {
            if let Ok(Some(txt)) = sharable.get_tx(*ccref.1) {
                table.insert(key, Some(txt));
            } else {
                table.insert(key, None);
            }
        }
    }
    let mut default: Option<String> = None;
    if let Ok(Some(txt)) = sharable.get_tx(cc_ref[cc_ref.len() - 1]) {
        default = Some(txt);
    }
    let mut new_array = vec![String::new(); cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_a, a)| {
        if let Some(val) = table.get(a) {
            if let Some(txt) = val.clone() {
                *new_a = txt;
            } else {
                *new_a = a.clone();
            }
        } else if let Some(tx) = default.clone() {
            *new_a = tx;
        } else {
            *new_a = a.clone();
        }
    });
    new_array
}

/// Apply text to text conversion to get physical data
fn text_to_text(cn: &mut Cn4, cc_ref: &[i64], cycle_count: &u64, sharable: &SharableBlocks) {
    match &mut cn.data {
        ChannelData::StringSBC(a) => {
            cn.data = ChannelData::StringUTF8(text_to_text_calculation(
                a,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::StringUTF8(a) => {
            cn.data = ChannelData::StringUTF8(text_to_text_calculation(
                a,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::StringUTF16(a) => {
            cn.data = ChannelData::StringUTF8(text_to_text_calculation(
                a,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        _ => warn!(
            "text conversion into text of channel {} not possible, channel does not contain string",
            cn.unique_name
        ),
    }
}

enum ValueOrValueRangeToText {
    ValueToText(
        HashMap<i64, TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
    ),
    ValueRangeToText(
        Vec<TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
        Vec<KeyRange>,
    ),
}

/// Generic function calculating text to value
#[inline]
fn bitfield_text_table_calculation<T: Integer + ToPrimitive>(
    array: &Vec<T>,
    cc_val: &[u64],
    cc_ref: &[i64],
    cycle_count: usize,
    sharable: &SharableBlocks,
) -> Vec<String> {
    let mut table: Vec<(ValueOrValueRangeToText, Option<String>)> =
        Vec::with_capacity(cc_ref.len());
    for pointer in cc_ref.iter() {
        if let Some(cc) = sharable.cc.get(pointer) {
            let name: Option<String>;
            if cc.cc_tx_name != 0 {
                if let Ok(Some(n)) = sharable.get_tx(cc.cc_tx_name) {
                    name = Some(n);
                } else {
                    name = None
                }
            } else {
                name = None
            }
            if cc.cc_type == 7 {
                match &cc.cc_val {
                    CcVal::Real(cc_val) => {
                        let mut table_int: HashMap<i64, TextOrScaleConversion> =
                            HashMap::with_capacity(cc_val.len());
                        for (ind, val) in cc_val.iter().enumerate() {
                            let val_i64 = (*val).round() as i64;
                            if let Ok(Some(txt)) = sharable.get_tx(cc.cc_ref[ind]) {
                                table_int.insert(val_i64, TextOrScaleConversion::Txt(txt));
                            } else if let Some(cc) = sharable.cc.get(&cc.cc_ref[ind]) {
                                let conv = conversion_function(cc, sharable);
                                table_int
                                    .insert(val_i64, TextOrScaleConversion::Scale(Box::new(conv)));
                            } else {
                                table_int.insert(val_i64, TextOrScaleConversion::Nil);
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        if let Ok(Some(txt)) = sharable.get_tx(cc.cc_ref[cc_val.len()]) {
                            def = DefaultTextOrScaleConversion::DefaultTxt(txt);
                        } else if let Some(cc) = sharable.cc.get(&cc.cc_ref[cc_val.len()]) {
                            let conv = conversion_function(cc, sharable);
                            def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
                        } else {
                            def = DefaultTextOrScaleConversion::Nil;
                        }
                        table.push((ValueOrValueRangeToText::ValueToText(table_int, def), name));
                    }
                    CcVal::Uint(_) => (),
                }
            } else if cc.cc_type == 8 {
                match &cc.cc_val {
                    CcVal::Real(cc_val) => {
                        let n_keys = cc_val.len() / 2;
                        let mut keys: Vec<KeyRange> = Vec::with_capacity(n_keys);
                        for (key_min, key_max) in cc_val.iter().tuples() {
                            let key: KeyRange = KeyRange {
                                min: *key_min,
                                max: *key_max,
                            };
                            keys.push(key);
                        }
                        let mut txt: Vec<TextOrScaleConversion> = Vec::with_capacity(n_keys);
                        for pointer in cc.cc_ref.iter() {
                            if let Ok(Some(t)) = sharable.get_tx(*pointer) {
                                txt.push(TextOrScaleConversion::Txt(t));
                            } else if let Some(ccc) = sharable.cc.get(pointer) {
                                let conv = conversion_function(ccc, sharable);
                                txt.push(TextOrScaleConversion::Scale(Box::new(conv)));
                            } else {
                                txt.push(TextOrScaleConversion::Nil);
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        if let Ok(Some(t)) = sharable.get_tx(cc.cc_ref[n_keys]) {
                            def = DefaultTextOrScaleConversion::DefaultTxt(t);
                        } else if let Some(ccc) = sharable.cc.get(&cc.cc_ref[n_keys]) {
                            let conv = conversion_function(ccc, sharable);
                            def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
                        } else {
                            def = DefaultTextOrScaleConversion::Nil;
                        }
                        table.push((
                            ValueOrValueRangeToText::ValueRangeToText(txt, def, keys),
                            name,
                        ));
                    }
                    CcVal::Uint(_) => (),
                }
            }
        }
    }
    let mut new_array = vec![String::new(); cycle_count];
    new_array.iter_mut().zip(array).for_each(|(new_a, a)| {
        for (ind, val) in cc_val.iter().enumerate() {
            match &table[ind] {
                (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                    let ref_val =
                        a.to_i64().unwrap_or_default() & (val.to_i64().unwrap_or_default());
                    if let Some(tosc) = table_int.get(&ref_val) {
                        match tosc {
                            TextOrScaleConversion::Txt(txt) => {
                                if let Some(n) = name {
                                    *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                } else {
                                    *new_a = format!("{} | {}", new_a, txt.clone());
                                }
                            }
                            TextOrScaleConversion::Scale(conv) => {
                                if let Some(n) = name {
                                    *new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                } else {
                                    *new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                }
                            }
                            _ => {
                                *new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    } else {
                        match &def {
                            DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                *new_a = txt.clone();
                            }
                            DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                *new_a = conv.eval_to_txt(a.to_f64().unwrap_or_default());
                            }
                            _ => {
                                *new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    }
                }
                (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                    let matched_key = keys.iter().enumerate().find(|&x| {
                        x.1.min <= a.to_f64().unwrap_or_default()
                            && a.to_f64().unwrap_or_default() <= x.1.max
                    });
                    if let Some(key) = matched_key {
                        match &txt[key.0] {
                            TextOrScaleConversion::Txt(txt) => {
                                if let Some(n) = name {
                                    *new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                } else {
                                    *new_a = format!("{} | {}", new_a, txt.clone());
                                }
                            }
                            TextOrScaleConversion::Scale(conv) => {
                                if let Some(n) = name {
                                    *new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                } else {
                                    *new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                }
                            }
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
                            }
                            DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                if let Some(n) = name {
                                    *new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                } else {
                                    *new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a.to_f64().unwrap_or_default())
                                    );
                                }
                            }
                            _ => {
                                *new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    }
                }
            }
        }
    });
    new_array
}

fn bitfield_text_table(
    cn: &mut Cn4,
    cc_val: &[u64],
    cc_ref: &[i64],
    cycle_count: &u64,
    sharable: &SharableBlocks,
) {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::StringUTF8(bitfield_text_table_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::StringUTF8(bitfield_text_table_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::StringUTF8(bitfield_text_table_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::StringUTF8(bitfield_text_table_calculation(
                a,
                cc_val,
                cc_ref,
                *cycle_count as usize,
                sharable,
            ));
        }
        _ => {
            warn!(
                "bitfield conversion into text of channel {} not possible, channel does not contain unsigned integer",
                cn.unique_name
            )
        }
    }
}
