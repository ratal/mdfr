//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result, bail};
use arrow::array::{ArrayBuilder, Float64Builder, LargeStringBuilder, PrimitiveBuilder};
use arrow::datatypes::{ArrowPrimitiveType, Float32Type, Float64Type};
use itertools::Itertools;
use log::warn;
use num::abs;
use num::cast::AsPrimitive;
use num::{NumCast, ToPrimitive};

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};

use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::tensor_arrow::TensorArrow;
use crate::mdfinfo::mdfinfo4::{Cc4Block, CcVal, Cn4, Dg4, SharableBlocks};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use rayon::prelude::*;

use crate::data_holder::complex_arrow::ComplexArrow;

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg4, sharable: &SharableBlocks) -> Result<(), Error> {
    for channel_group in dg.cg.values_mut() {
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| !cn.data.is_empty())
            .try_for_each(|(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                // Could be empty if only initialised
                if let Some(conv) = sharable.cc.get(&cn.block.cn_cc_conversion) {
                    match conv.cc_type {
                        1 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                linear_conversion(cn, cc_val).with_context(|| {
                                    format!("linear conversion failed for {}", cn.unique_name)
                                })?
                            }
                            CcVal::Uint(_) => (),
                        },
                        2 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                rational_conversion(cn, cc_val).with_context(|| {
                                    format!("rational conversion failed for {}", cn.unique_name)
                                })?
                            }
                            CcVal::Uint(_) => (),
                        },
                        3 => {
                            if !&conv.cc_ref.is_empty()
                                && let Ok(Some(conv)) = sharable.get_tx(conv.cc_ref[0])
                            {
                                algebraic_conversion(cn, &conv).with_context(|| {
                                    format!("algebraic conversion failed for {}", cn.unique_name)
                                })?
                            }
                        }
                        4 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_to_value_with_interpolation(
                                cn,
                                cc_val.clone(),
                            )
                            .with_context(|| {
                                format!(
                                    "value to value conversion with interpolation failed for {}",
                                    cn.unique_name
                                )
                            })?,
                            CcVal::Uint(_) => (),
                        },
                        5 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_to_value_without_interpolation(
                                cn,
                                cc_val.clone(),
                            )
                            .with_context(|| {
                                format!(
                                    "value to value conversion without interpolation failed for {}",
                                    cn.unique_name
                                )
                            })?,
                            CcVal::Uint(_) => (),
                        },
                        6 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_range_to_value_table(cn, cc_val.clone())
                                .with_context(|| {
                                format!(
                                    "value range to value table conversion failed for {}",
                                    cn.unique_name
                                )
                            })?,
                            CcVal::Uint(_) => (),
                        },
                        7 => match &conv.cc_val {
                            CcVal::Real(cc_val) => value_to_text(
                                cn,
                                cc_val,
                                &conv.cc_ref,
                                sharable,
                            )
                            .with_context(|| {
                                format!("value to text conversion failed for {}", cn.unique_name)
                            })?,
                            CcVal::Uint(_) => (),
                        },
                        8 => match &conv.cc_val {
                            CcVal::Real(cc_val) => {
                                value_range_to_text(cn, cc_val, &conv.cc_ref, sharable)
                                    .with_context(|| {
                                        format!(
                                            "value range to text conversion failed for {}",
                                            cn.unique_name
                                        )
                                    })?
                            }
                            CcVal::Uint(_) => (),
                        },
                        9 => match &conv.cc_val {
                            CcVal::Real(cc_val) => text_to_value(
                                cn,
                                cc_val,
                                &conv.cc_ref,
                                sharable,
                            )
                            .with_context(|| {
                                format!("text to value conversion failed for {}", cn.unique_name)
                            })?,
                            CcVal::Uint(_) => (),
                        },
                        10 => text_to_text(cn, &conv.cc_ref, sharable).with_context(|| {
                            format!("text to text conversion failed for {}", cn.unique_name)
                        })?,
                        11 => match &conv.cc_val {
                            CcVal::Real(_) => (),
                            CcVal::Uint(cc_val) => {
                                bitfield_text_table(cn, cc_val, &conv.cc_ref, sharable)
                                    .with_context(|| {
                                        format!(
                                            "bitfield text table conversion failed for {}",
                                            cn.unique_name
                                        )
                                    })?
                            }
                        },
                        0 => (),
                        _ => bail!(
                            "conversion type not recognised for channel {} not possible, type {}",
                            cn.unique_name,
                            conv.cc_type,
                        ),
                    }
                }
                Ok(())
            })?
    }
    Ok(())
}

/// Generic function calculating linear expression
#[inline]
pub fn linear_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    p1: f64,
    p2: f64,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values_slice();
    let converted: Vec<f64> = values.iter().map(|v| (*v).as_() * p2 + p1).collect();
    Ok(Float64Builder::new_from_buffer(converted.into(), None))
}

/// Apply linear conversion to get physical data
fn linear_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && abs(p2 - 1.0) < 1e-12) {
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u8 channel")?,
                );
            }
            ChannelData::Int8(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of i8 channel")?,
                );
            }
            ChannelData::Int16(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of i16 channel")?,
                );
            }
            ChannelData::UInt16(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u16 channel")?,
                );
            }
            ChannelData::Int32(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of i32 channel")?,
                );
            }
            ChannelData::UInt32(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u32 channel")?,
                );
            }
            ChannelData::Float32(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of f32 channel")?,
                );
            }
            ChannelData::Int64(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u16 channel")?,
                );
            }
            ChannelData::UInt64(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u64 channel")?,
                );
            }
            ChannelData::Float64(a) => {
                // In-place conversion for f64 - avoids all allocations
                a.values_slice_mut().iter_mut().for_each(|x| {
                    *x = *x * p2 + p1;
                });
            }
            ChannelData::Complex32(a) => {
                cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of complex f32 channel")?,
                    a.nulls(),
                ));
            }
            ChannelData::Complex64(a) => {
                cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of complex f64 channel")?,
                    a.nulls(),
                ));
            }
            ChannelData::ArrayDUInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor u8 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDInt8(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor i8 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor i16 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDUInt16(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor u16 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor i32 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDUInt32(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor u16 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDFloat32(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor f32 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor i64 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDUInt64(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor u64 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            ChannelData::ArrayDFloat64(a) => {
                cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                    linear_calculation(a.values(), p1, p2)
                        .context("failed linear conversion of tensor f64 channel")?,
                    a.nulls(),
                    a.shape().clone(),
                    a.order().clone(),
                ))
            }
            _ => warn!(
                "linear conversion of channel {} not possible, channel does not contain primitives",
                cn.unique_name
            ),
        }
    }
    Ok(())
}

/// Generic function calculating rational expression for a primitive
#[inline]
pub fn rational_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveBuilder<T>,
    cc_val: &[f64],
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let values = array.values_slice();
    let converted: Vec<f64> = values
        .iter()
        .map(|v| {
            let x: f64 = (*v).as_();
            (x * x * p1 + x * p2 + p3) / (x * x * p4 + x * p5 + p6)
        })
        .collect();
    Ok(Float64Builder::new_from_buffer(converted.into(), None))
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of u8 channel")?,
            );
        }
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of i8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed rational conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            let p1 = cc_val[0];
            let p2 = cc_val[1];
            let p3 = cc_val[2];
            let p4 = cc_val[3];
            let p5 = cc_val[4];
            let p6 = cc_val[5];
            a.values_slice_mut().iter_mut().for_each(|x| {
                let v = *x;
                *x = (v * v * p1 + v * p2 + p3) / (v * v * p4 + v * p5 + p6);
            });
        }
        ChannelData::Complex32(a) => {
            cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of complex f32 channel")?,
                a.nulls(),
            ));
        }
        ChannelData::Complex64(a) => {
            cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of complex f64 channel")?,
                a.nulls(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of u8 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of i8 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of i16 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of u16 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of i32 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of u32 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of f32 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of i64 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of u64 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                rational_calculation(a.values(), cc_val)
                    .context("failed rational conversion of f64 tensor channel")?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ))
        }
        _ => warn!(
            "rational conversion of channel {} not possible, channel does not contain primitives",
            cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating algebraic expression for a primitive
#[inline]
fn alegbraic_conversion_primitive<T: ArrowPrimitiveType>(
    compiled: &Instruction,
    slab: &Slab,
    array: &PrimitiveBuilder<T>,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values_slice();
    let mut new_array = vec![0f64; values.len()];
    let mut map = BTreeMap::new();
    new_array.iter_mut().zip(values).for_each(|(new_a, v)| {
        let a: f64 = (*v).as_();
        map.clear();
        map.insert("X".to_string(), a);
        let val = compiled.eval(slab, &mut map);
        *new_a = match val {
            Ok(val) => val,
            Err(err) => {
                warn!(
                    "could not compute the value {a:?} with expression {compiled:?}, error {err}"
                );
                a
            }
        }
    });
    Ok(PrimitiveBuilder::new_from_buffer(new_array.into(), None))
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn4, formulae: &str) -> Result<(), Error> {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let compiled = parser.parse(formulae, &mut slab.ps);
    match compiled {
        Ok(c) => {
            let compiled = c.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
            match &mut cn.data {
                ChannelData::UInt8(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of u8 channel")?,
                    );
                }
                ChannelData::Int8(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of i8 channel")?,
                    );
                }
                ChannelData::Int16(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of i16 channel")?,
                    );
                }
                ChannelData::UInt16(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of u16 channel")?,
                    );
                }
                ChannelData::Int32(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of i32 channel")?,
                    );
                }
                ChannelData::UInt32(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of u32 channel")?,
                    );
                }
                ChannelData::Float32(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of f32 channel")?,
                    );
                }
                ChannelData::Int64(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of i64 channel")?,
                    );
                }
                ChannelData::UInt64(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of u64 channel")?,
                    );
                }
                ChannelData::Float64(a) => {
                    cn.data = ChannelData::Float64(
                        alegbraic_conversion_primitive(&compiled, &slab, a)
                            .context("failed algebraic conversion of f64 channel")?,
                    );
                }
                ChannelData::Complex32(a) => {
                    cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of complex f32 channel")?,
                        a.nulls(),
                    ));
                }
                ChannelData::Complex64(a) => {
                    cn.data = ChannelData::Complex64(ComplexArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of complex f64 channel")?,
                        a.nulls(),
                    ));
                }
                ChannelData::ArrayDInt8(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor i8 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDUInt8(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor u8 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDInt16(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor i16 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDUInt16(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor u16 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDInt32(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor i32 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDUInt32(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor u32 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDFloat32(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor f32 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDInt64(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor i64 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDUInt64(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor u64 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                ChannelData::ArrayDFloat64(a) => {
                    cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                        alegbraic_conversion_primitive(&compiled, &slab, a.values())
                            .context("failed algebraic conversion of tensor f64 channel")?,
                        a.nulls(),
                        a.shape().clone(),
                        a.order().clone(),
                    ));
                }
                _ => warn!(
                    "algebraic conversion of channel {} not possible, channel does not contain primitives",
                    cn.unique_name
                ),
            }
        }
        Err(err) => {
            warn!(
                "could not compile algebraic conversion expression {}, {}, channel {} not converted",
                formulae, err, cn.unique_name
            )
        }
    }
    Ok(())
}

/// Generic function calculating value to value interpolation for a primitive
#[inline]
fn value_to_value_with_interpolation_primitive<T: ArrowPrimitiveType>(
    array: &PrimitiveBuilder<T>,
    val: Vec<(&f64, &f64)>,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values_slice();
    let mut new_array = vec![0f64; values.len()];
    new_array
        .iter_mut()
        .zip(values)
        .for_each(|(new_a, v)| {
            let a: f64 = (*v).as_();
            *new_a = match val
                .binary_search_by(|&(xi, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal))
            {
                Ok(idx) => *val[idx].1,
                Err(0) => *val[0].1,
                Err(idx) if idx >= val.len() => *val[idx - 1].1,
                Err(idx) => {
                    let (x0, y0) = val[idx - 1];
                    let (x1, y1) = val[idx];
                    (y0 * (x1 - a) + y1 * (a - x0)) / (x1 - x0)
                }
            };
        });
    Ok(PrimitiveBuilder::new_from_buffer(
        new_array.into(),
        array
            .validity_slice()
            .map(|validity| validity.to_vec().into()),
    ))
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_primitive(a, val)
                    .context("failed value to value with interpolation conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_primitive(a, val)
                    .context("failed value to value with interpolation conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of i16 channel",
                )?);
        }
        ChannelData::UInt16(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of u16 channel",
                )?);
        }
        ChannelData::Int32(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of i32 channel",
                )?);
        }
        ChannelData::UInt32(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of u32 channel",
                )?);
        }
        ChannelData::Float32(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of f32 channel",
                )?);
        }
        ChannelData::Int64(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of i64 channel",
                )?);
        }
        ChannelData::UInt64(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of u64 channel",
                )?);
        }
        ChannelData::Float64(a) => {
            cn.data =
                ChannelData::Float64(value_to_value_with_interpolation_primitive(a, val).context(
                    "failed value to value with interpolation conversion of f64 channel",
                )?);
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor i8 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor u8 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor i16 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val
            ).context("failed value to value with interpolation conversion of tensor tensor u16 channel")?, a.nulls(), a.shape().clone(), a.order().clone(),));
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor i32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor u32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor f32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor u64 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_with_interpolation_primitive(a.values(), val).context(
                    "failed value to value with interpolation conversion of tensor f64 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        _ => warn!(
            "value to value with interpolation conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating value to value without interpolation for a primitive
#[inline]
fn value_to_value_without_interpolation_primitive<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    val: Vec<(&f64, &f64)>,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values_slice();
    let mut new_array = vec![0f64; values.len()];
    new_array
        .iter_mut()
        .zip(values)
        .for_each(|(new_a, v)| {
            let a: f64 = (*v).as_();
            *new_a = match val
                .binary_search_by(|&(xi, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal))
            {
                Ok(idx) => *val[idx].1,
                Err(0) => *val[0].1,
                Err(idx) if idx >= val.len() => *val[idx - 1].1,
                Err(idx) => {
                    let (x0, y0) = val[idx - 1];
                    let (x1, y1) = val[idx];
                    if (a - x0) > (x1 - a) { *y1 } else { *y0 }
                }
            };
        });
    Ok(PrimitiveBuilder::new_from_buffer(
        new_array.into(),
        array
            .validity_slice()
            .map(|validity| validity.to_vec().into()),
    ))
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of i8 channel",
                )?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of u8 channel",
                )?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of i16 channel",
                )?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of u16 channel",
                )?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of i32 channel",
                )?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of u32 channel",
                )?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of f32 channel",
                )?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of i64 channel",
                )?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of u64 channel",
                )?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_primitive(a, val).context(
                    "failed value to value without interpolation conversion of f64 channel",
                )?,
            );
        }
        ChannelData::ArrayDInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor i8 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt8(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor u8 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor i16 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt16(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor u16 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor i32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor u32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDFloat32(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor f32 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor i64 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDUInt64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor u64 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        ChannelData::ArrayDFloat64(a) => {
            cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_primitive(
                value_to_value_without_interpolation_primitive(a.values(), val).context(
                    "failed value to value without interpolation conversion of tensor f64 channel",
                )?,
                a.nulls(),
                a.shape().clone(),
                a.order().clone(),
            ));
        }
        _ => warn!(
            "value to value without interpolation conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating value range to value table without interpolation
#[inline]
fn value_range_to_value_table_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveBuilder<T>,
    val: &[(f64, f64, f64)],
    default_value: &f64,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values_slice();
    let mut new_array = vec![0f64; values.len()];
    new_array
        .iter_mut()
        .zip(values)
        .for_each(|(new_a, v)| {
            let a: f64 = (*v).as_();
            *new_a = match val
                .binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal))
            {
                Ok(idx) => val[idx].2,
                Err(0) => *default_value,
                Err(idx) if (idx >= val.len() && a <= val[idx - 1].1) => val[idx - 1].2,
                Err(idx) => {
                    if a <= val[idx].1 {
                        val[idx].2
                    } else {
                        *default_value
                    }
                }
            };
        });
    Ok(PrimitiveBuilder::new_from_buffer(
        new_array.into(),
        array
            .validity_slice()
            .map(|validity| validity.to_vec().into()),
    ))
}

/// Apply value range to value table without interpolation conversion to get physical data
fn value_range_to_value_table(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let mut val: Vec<(f64, f64, f64)> = Vec::new();
    for (a, b, c) in cc_val.iter().tuples::<(_, _, _)>() {
        val.push((*a, *b, *c));
    }
    let default_value = cc_val[cc_val.len() - 1];
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value)
                    .context("failed value range to value table conversion of f64 channel")?,
            );
        }
        _ => warn!(
            "value range to value conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
    Ok(())
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
fn value_to_text_calculation_int<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    def: &DefaultTextOrScaleConversion,
    sharable: &SharableBlocks,
) -> Result<LargeStringBuilder, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64> + AsPrimitive<i64>,
{
    // table applicable only to integers, no canonization
    let mut table_int: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
    for (ind, val) in cc_val.iter().enumerate() {
        let val_i64 = (*val).round() as i64;
        match sharable.get_tx(cc_ref[ind]) {
            Ok(Some(txt)) => {
                table_int.insert(val_i64, TextOrScaleConversion::Txt(txt));
            }
            _ => {
                if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
                    let conv = conversion_function(cc, sharable);
                    table_int.insert(val_i64, TextOrScaleConversion::Scale(Box::new(conv)));
                } else {
                    table_int.insert(val_i64, TextOrScaleConversion::Nil);
                }
            }
        }
    }
    let values = array.values_slice();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    values.iter().for_each(|v| {
        let a_f64: f64 = (*v).as_();
        let a_i64: i64 = (*v).as_();
        if let Some(tosc) = table_int.get(&a_i64) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.clone());
                }
                TextOrScaleConversion::Scale(conv) => {
                    new_array.append_value(conv.eval_to_txt(a_f64));
                }
                _ => {
                    new_array.append_value(a_f64.to_string());
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.clone());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    new_array.append_value(conv.eval_to_txt(a_f64));
                }
                _ => {
                    new_array.append_value(a_f64.to_string());
                }
            }
        }
    });
    if let Some(validity) = array.validity_slice_mut() {
        let _ = new_array.validity_slice_mut().insert(validity);
    }
    Ok(new_array)
}

/// Generic function calculating float value range to text
#[inline]
fn value_to_text_calculation_f32(
    a: &PrimitiveBuilder<Float32Type>,
    cc_val: &[f64],
    cc_ref: &[i64],
    canonization_value: f64,
    def: &DefaultTextOrScaleConversion,
    sharable: &SharableBlocks,
) -> LargeStringBuilder {
    // table for floating point comparison
    let mut table_float: HashMap<i64, TextOrScaleConversion> = HashMap::with_capacity(cc_val.len());
    for (ind, val) in cc_val.iter().enumerate() {
        let ref_val = (*val * canonization_value).round() as i64; // Canonization
        match sharable.get_tx(cc_ref[ind]) {
            Ok(Some(txt)) => {
                table_float.insert(ref_val, TextOrScaleConversion::Txt(txt));
            }
            _ => {
                if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
                    let conv = conversion_function(cc, sharable);
                    table_float.insert(ref_val, TextOrScaleConversion::Scale(Box::new(conv)));
                } else {
                    table_float.insert(ref_val, TextOrScaleConversion::Nil);
                }
            }
        }
    }
    let mut new_array = LargeStringBuilder::with_capacity(a.len(), 32);
    a.values_slice().iter().for_each(|a| {
        let ref_val = (a * canonization_value as f32)
            .round()
            .to_i64()
            .unwrap_or_default();
        if let Some(tosc) = table_float.get(&ref_val) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.clone());
                }
                TextOrScaleConversion::Scale(conv) => {
                    new_array.append_value(conv.eval_to_txt(a.to_f64().unwrap_or_default()));
                }
                _ => {
                    new_array.append_value(a.to_string());
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.clone());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    new_array.append_value(conv.eval_to_txt(a.to_f64().unwrap_or_default()));
                }
                _ => {
                    new_array.append_value(a.to_string());
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
    sharable: &SharableBlocks,
) -> Result<(), Error> {
    let def: DefaultTextOrScaleConversion;
    match sharable.get_tx(cc_ref[cc_val.len()]) {
        Ok(Some(txt)) => {
            def = DefaultTextOrScaleConversion::DefaultTxt(txt);
        }
        _ => {
            if let Some(cc) = sharable.cc.get(&cc_ref[cc_val.len()]) {
                let conv = conversion_function(cc, sharable);
                def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
            } else {
                def = DefaultTextOrScaleConversion::Nil;
            }
        }
    }
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of i8 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(value_to_text_calculation_f32(
                a,
                cc_val,
                cc_ref,
                1048576.0f64,
                &def,
                sharable,
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation_int(a, cc_val, cc_ref, &def, sharable)
                    .context("failed value to text conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            // table for floating point comparison
            let mut table_float: HashMap<i64, TextOrScaleConversion> =
                HashMap::with_capacity(cc_val.len());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = (*val * 1024.0 * 1024.0).round() as i64; // Canonization
                match sharable.get_tx(cc_ref[ind]) {
                    Ok(Some(txt)) => {
                        table_float.insert(ref_val, TextOrScaleConversion::Txt(txt));
                    }
                    _ => {
                        if let Some(cc) = sharable.cc.get(&cc_ref[ind]) {
                            let conv = conversion_function(cc, sharable);
                            table_float
                                .insert(ref_val, TextOrScaleConversion::Scale(Box::new(conv)));
                        } else {
                            table_float.insert(ref_val, TextOrScaleConversion::Nil);
                        }
                    }
                }
            }
            let mut new_array = LargeStringBuilder::with_capacity(a.len(), 32);
            a.values_slice().iter().for_each(|a| {
                let ref_val = (a * 1024.0 * 1024.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            new_array.append_value(txt.clone());
                        }
                        TextOrScaleConversion::Scale(conv) => {
                            new_array.append_value(conv.eval_to_txt(*a));
                        }
                        _ => {
                            new_array.append_value(a.to_string());
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            new_array.append_value(txt.clone());
                        }
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            new_array.append_value(conv.eval_to_txt(*a));
                        }
                        _ => {
                            new_array.append_value(a.to_string());
                        }
                    }
                }
            });
            cn.data = ChannelData::Utf8(new_array);
        }
        _ => warn!(
            "value to text conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
    Ok(())
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
                    match sharable.get_tx(cc.cc_ref[0]) {
                        Ok(Some(formulae)) => {
                            let parser = fasteval::Parser::new();
                            let mut slab = fasteval::Slab::new();
                            let compiled = parser.parse(&formulae, &mut slab.ps);
                            match compiled {
                                Ok(c) => ConversionFunction::Algebraic(
                                    c.from(&slab.ps).compile(&slab.ps, &mut slab.cs),
                                    Box::new(slab),
                                ),
                                Err(e) => {
                                    warn!("Error parsing formulae {formulae}, error {e}");
                                    ConversionFunction::Identity
                                }
                            }
                        }
                        _ => ConversionFunction::Identity,
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
                        warn!("could not evaluate algebraic expression for {a}, error {e}");
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
fn value_range_to_text_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveBuilder<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> LargeStringBuilder
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
{
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
        match sharable.get_tx(*pointer) {
            Ok(Some(t)) => {
                txt.push(TextOrScaleConversion::Txt(t));
            }
            _ => {
                if let Some(cc) = sharable.cc.get(pointer) {
                    let conv = conversion_function(cc, sharable);
                    txt.push(TextOrScaleConversion::Scale(Box::new(conv)));
                } else {
                    txt.push(TextOrScaleConversion::Nil);
                }
            }
        }
    }
    let def: DefaultTextOrScaleConversion;
    match sharable.get_tx(cc_ref[n_keys]) {
        Ok(Some(t)) => {
            def = DefaultTextOrScaleConversion::DefaultTxt(t);
        }
        _ => {
            if let Some(cc) = sharable.cc.get(&cc_ref[n_keys]) {
                let conv = conversion_function(cc, sharable);
                def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
            } else {
                def = DefaultTextOrScaleConversion::Nil;
            }
        }
    }
    let values = array.values_slice();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    values.iter().for_each(|v| {
        let a: &f64 = &(*v).as_();
        let matched_key = keys
            .iter()
            .enumerate()
            .find(|&x| (&x.1.min <= a) && (a <= &x.1.max));
        if let Some(key) = matched_key {
            match &txt[key.0] {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.clone());
                }
                TextOrScaleConversion::Scale(conv) => {
                    new_array.append_value(conv.eval_to_txt(*a));
                }
                _ => {
                    new_array.append_value(a.to_string());
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.clone());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    new_array.append_value(conv.eval_to_txt(*a));
                }
                _ => {
                    new_array.append_value(a.to_string());
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
    sharable: &SharableBlocks,
) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::UInt8(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::Int16(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::UInt16(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::Int32(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::UInt32(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::Float32(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::Int64(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::UInt64(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        ChannelData::Float64(a) => {
            cn.data =
                ChannelData::Utf8(value_range_to_text_calculation(a, cc_val, cc_ref, sharable));
        }
        _ => warn!(
            "value range to text conversion of channel {} not possible, channel does not contain primitive",
            cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating text to value
#[inline]
fn text_to_value_calculation(
    array: &LargeStringBuilder,
    cc_val: &[f64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> PrimitiveBuilder<Float64Type> {
    let mut table: HashMap<String, f64> = HashMap::with_capacity(cc_ref.len());
    for (ind, ccref) in cc_ref.iter().enumerate() {
        if let Ok(Some(txt)) = sharable.get_tx(*ccref) {
            table.insert(txt, cc_val[ind]);
        }
    }
    let default = cc_val[cc_val.len() - 1];
    let mut new_array = vec![0f64; array.len()];
    new_array
        .iter_mut()
        .zip(array.finish_cloned().iter())
        .for_each(|(new_a, a)| {
            if let Some(val) = table.get(a.unwrap_or_default()) {
                *new_a = *val;
            } else {
                *new_a = default;
            }
        });
    PrimitiveBuilder::new_from_buffer(
        new_array.into(),
        array
            .validity_slice()
            .map(|validity| validity.to_vec().into()),
    )
}

/// Apply text to value conversion to get physical data
fn text_to_value(
    cn: &mut Cn4,
    cc_val: &[f64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Utf8(a) => {
            cn.data = ChannelData::Float64(text_to_value_calculation(a, cc_val, cc_ref, sharable));
        }
        _ => warn!(
            "text conversion into value of channel {} not possible, channel does not contain string",
            cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating text to value
#[inline]
fn text_to_text_calculation(
    array: &LargeStringBuilder,
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> LargeStringBuilder {
    let pairs: Vec<(&i64, &i64)> = cc_ref.iter().tuples().collect();
    let mut table: HashMap<String, Option<String>> = HashMap::with_capacity(cc_ref.len());
    for ccref in pairs.iter() {
        if let Ok(Some(key)) = sharable.get_tx(*ccref.0) {
            match sharable.get_tx(*ccref.1) {
                Ok(Some(txt)) => {
                    table.insert(key, Some(txt));
                }
                _ => {
                    table.insert(key, None);
                }
            }
        }
    }
    let mut default: Option<String> = None;
    if let Ok(Some(txt)) = sharable.get_tx(cc_ref[cc_ref.len() - 1]) {
        default = Some(txt);
    }
    let mut new_array = LargeStringBuilder::with_capacity(array.len(), 32);
    array.finish_cloned().iter().for_each(|a| {
        if let Some(val) = table.get(a.unwrap_or_default()) {
            if let Some(txt) = val.clone() {
                new_array.append_value(txt);
            } else {
                new_array.append_value(a.unwrap_or_default());
            }
        } else if let Some(tx) = default.clone() {
            new_array.append_value(tx);
        } else {
            new_array.append_value(a.unwrap_or_default());
        }
    });
    new_array
}

/// Apply text to text conversion to get physical data
fn text_to_text(cn: &mut Cn4, cc_ref: &[i64], sharable: &SharableBlocks) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Utf8(a) => {
            cn.data = ChannelData::Utf8(text_to_text_calculation(a, cc_ref, sharable));
        }
        _ => warn!(
            "text conversion into text of channel {} not possible, channel does not contain string",
            cn.unique_name
        ),
    }
    Ok(())
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
fn bitfield_text_table_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveBuilder<T>,
    cc_val: &[u64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> Result<LargeStringBuilder>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64> + AsPrimitive<i64>,
{
    let mut table: Vec<(ValueOrValueRangeToText, Option<String>)> =
        Vec::with_capacity(cc_ref.len());
    for pointer in cc_ref.iter() {
        if let Some(cc) = sharable.cc.get(pointer) {
            let name: Option<String>;
            if cc.cc_tx_name != 0 {
                match sharable.get_tx(cc.cc_tx_name) {
                    Ok(Some(n)) => {
                        name = Some(n);
                    }
                    _ => name = None,
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
                            match sharable.get_tx(cc.cc_ref[ind]) {
                                Ok(Some(txt)) => {
                                    table_int.insert(val_i64, TextOrScaleConversion::Txt(txt));
                                }
                                _ => {
                                    if let Some(cc) = sharable.cc.get(&cc.cc_ref[ind]) {
                                        let conv = conversion_function(cc, sharable);
                                        table_int.insert(
                                            val_i64,
                                            TextOrScaleConversion::Scale(Box::new(conv)),
                                        );
                                    } else {
                                        table_int.insert(val_i64, TextOrScaleConversion::Nil);
                                    }
                                }
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        match sharable.get_tx(cc.cc_ref[cc_val.len()]) {
                            Ok(Some(txt)) => {
                                def = DefaultTextOrScaleConversion::DefaultTxt(txt);
                            }
                            _ => {
                                if let Some(cc) = sharable.cc.get(&cc.cc_ref[cc_val.len()]) {
                                    let conv = conversion_function(cc, sharable);
                                    def =
                                        DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
                                } else {
                                    def = DefaultTextOrScaleConversion::Nil;
                                }
                            }
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
                            match sharable.get_tx(*pointer) {
                                Ok(Some(t)) => {
                                    txt.push(TextOrScaleConversion::Txt(t));
                                }
                                _ => {
                                    if let Some(ccc) = sharable.cc.get(pointer) {
                                        let conv = conversion_function(ccc, sharable);
                                        txt.push(TextOrScaleConversion::Scale(Box::new(conv)));
                                    } else {
                                        txt.push(TextOrScaleConversion::Nil);
                                    }
                                }
                            }
                        }
                        let def: DefaultTextOrScaleConversion;
                        match sharable.get_tx(cc.cc_ref[n_keys]) {
                            Ok(Some(t)) => {
                                def = DefaultTextOrScaleConversion::DefaultTxt(t);
                            }
                            _ => {
                                if let Some(ccc) = sharable.cc.get(&cc.cc_ref[n_keys]) {
                                    let conv = conversion_function(ccc, sharable);
                                    def =
                                        DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
                                } else {
                                    def = DefaultTextOrScaleConversion::Nil;
                                }
                            }
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

    let values = array.values_slice();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    values.iter().for_each(|v| {
        let a_f64: f64 = (*v).as_();
        let a_i64: i64 = (*v).as_();
        let mut new_a = String::new();
        for (ind, val) in cc_val.iter().enumerate() {
            match &table[ind] {
                (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                    let ref_val = a_i64 & (val.to_i64().unwrap_or_default());
                    if let Some(tosc) = table_int.get(&ref_val) {
                        match tosc {
                            TextOrScaleConversion::Txt(txt) => {
                                if let Some(n) = name {
                                    new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                } else {
                                    new_a = format!("{} | {}", new_a, txt.clone());
                                }
                            }
                            TextOrScaleConversion::Scale(conv) => {
                                if let Some(n) = name {
                                    new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a_f64)
                                    );
                                } else {
                                    new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a_f64)
                                    );
                                }
                            }
                            _ => {
                                new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    } else {
                        match &def {
                            DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                new_a.clone_from(txt);
                            }
                            DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                new_a = conv.eval_to_txt(a_f64);
                            }
                            _ => {
                                new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    }
                }
                (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                    let matched_key = keys.iter().enumerate().find(|&x| {
                        (x.1.min <= a_f64) && (a_f64 <= x.1.max)
                    });
                    if let Some(key) = matched_key {
                        match &txt[key.0] {
                            TextOrScaleConversion::Txt(txt) => {
                                if let Some(n) = name {
                                    new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                } else {
                                    new_a = format!("{} | {}", new_a, txt.clone());
                                }
                            }
                            TextOrScaleConversion::Scale(conv) => {
                                if let Some(n) = name {
                                    new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a_f64)
                                    );
                                } else {
                                    new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a_f64)
                                    );
                                }
                            }
                            _ => {
                                new_array.append_value(format!("{} | {}", new_a, "nothing"));
                            }
                        }
                    } else {
                        match &def {
                            DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                if let Some(n) = name {
                                    new_a = format!("{} | {} = {}", new_a, n, txt.clone());
                                } else {
                                    new_a = format!("{} | {}", new_a, txt.clone());
                                }
                            }
                            DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                if let Some(n) = name {
                                    new_a = format!(
                                        "{} | {} = {}",
                                        new_a,
                                        n,
                                        conv.eval_to_txt(a_f64)
                                    );
                                } else {
                                    new_a = format!(
                                        "{} | {}",
                                        new_a,
                                        conv.eval_to_txt(a_f64)
                                    );
                                }
                            }
                            _ => {
                                new_a = format!("{} | {}", new_a, "nothing");
                            }
                        }
                    }
                }
            }
        }
        new_array.append_value(new_a);
    });
    Ok(new_array)
}

fn bitfield_text_table(
    cn: &mut Cn4,
    cc_val: &[u64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of u8 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of u16 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of u32 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of u64 channel")?,
            );
        }
        _ => {
            warn!(
                "bitfield conversion into text of channel {} not possible, channel does not contain unsigned integer",
                cn.unique_name
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Builder, Int32Builder, UInt16Builder};

    #[test]
    fn test_linear_calculation() {
        // v * p2 + p1 where p1=10.0, p2=2.0
        let p1 = 10.0;
        let p2 = 2.0;

        // Int32
        let mut builder = Int32Builder::new();
        builder.append_value(0);
        builder.append_value(1);
        builder.append_value(5);
        builder.append_value(-3);
        let result = linear_calculation(&mut builder, p1, p2).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 10.0).abs() < 1e-12); // 0*2+10
        assert!((values[1] - 12.0).abs() < 1e-12); // 1*2+10
        assert!((values[2] - 20.0).abs() < 1e-12); // 5*2+10
        assert!((values[3] - 4.0).abs() < 1e-12); // -3*2+10

        // Float64
        let mut builder = Float64Builder::new();
        builder.append_value(1.5);
        builder.append_value(2.5);
        let result = linear_calculation(&mut builder, p1, p2).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 13.0).abs() < 1e-12); // 1.5*2+10
        assert!((values[1] - 15.0).abs() < 1e-12); // 2.5*2+10

        // UInt16
        let mut builder = UInt16Builder::new();
        builder.append_value(100);
        builder.append_value(0);
        let result = linear_calculation(&mut builder, p1, p2).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 210.0).abs() < 1e-12); // 100*2+10
        assert!((values[1] - 10.0).abs() < 1e-12); // 0*2+10
    }

    #[test]
    fn test_rational_calculation() {
        // (x²*p1 + x*p2 + p3) / (x²*p4 + x*p5 + p6)
        // Simple linear: p1=0, p2=2, p3=1, p4=0, p5=0, p6=1 → (2x+1)/1
        let cc_val = vec![0.0, 2.0, 1.0, 0.0, 0.0, 1.0];

        let mut builder = Int32Builder::new();
        builder.append_value(0);
        builder.append_value(1);
        builder.append_value(5);
        let result = rational_calculation(&builder, &cc_val).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 1.0).abs() < 1e-12); // (0+0+1)/1
        assert!((values[1] - 3.0).abs() < 1e-12); // (0+2+1)/1
        assert!((values[2] - 11.0).abs() < 1e-12); // (0+10+1)/1

        // Quadratic: p1=1, p2=0, p3=0, p4=0, p5=0, p6=1 → x²
        let cc_val = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut builder = Float64Builder::new();
        builder.append_value(3.0);
        builder.append_value(4.0);
        let result = rational_calculation(&builder, &cc_val).unwrap();
        let values = result.values_slice();
        assert!((values[0] - 9.0).abs() < 1e-12); // 3²
        assert!((values[1] - 16.0).abs() < 1e-12); // 4²
    }

    #[test]
    fn test_value_to_value_with_interpolation_primitive() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let keys = [0.0, 0.0, 10.0, 100.0, 20.0, 200.0];
        let val: Vec<(&f64, &f64)> = keys.iter().tuples().collect();

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact → 0
        builder.append_value(5.0); // interpolate → 50
        builder.append_value(10.0); // exact → 100
        builder.append_value(15.0); // interpolate → 150
        builder.append_value(-5.0); // below first → 0
        builder.append_value(25.0); // above last → 200

        let result = value_to_value_with_interpolation_primitive(&builder, val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12);
        assert!((values[1] - 50.0).abs() < 1e-12);
        assert!((values[2] - 100.0).abs() < 1e-12);
        assert!((values[3] - 150.0).abs() < 1e-12);
        assert!((values[4] - 0.0).abs() < 1e-12);
        assert!((values[5] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_to_value_without_interpolation_primitive() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let keys = [0.0, 0.0, 10.0, 100.0, 20.0, 200.0];
        let val: Vec<(&f64, &f64)> = keys.iter().tuples().collect();

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact → 0
        builder.append_value(3.0); // nearer to x=0 (dist=3) than x=10 (dist=7) → 0
        builder.append_value(7.0); // nearer to x=10 (dist=3) than x=0 (dist=7) → 100
        builder.append_value(10.0); // exact → 100
        builder.append_value(-5.0); // below first → 0
        builder.append_value(25.0); // above last → 200

        let result = value_to_value_without_interpolation_primitive(&mut builder, val).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 6);
        assert!((values[0] - 0.0).abs() < 1e-12);
        assert!((values[1] - 0.0).abs() < 1e-12);
        assert!((values[2] - 100.0).abs() < 1e-12);
        assert!((values[3] - 100.0).abs() < 1e-12);
        assert!((values[4] - 0.0).abs() < 1e-12);
        assert!((values[5] - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_value_range_to_value_table() {
        // Ranges: (key_min, key_max, value)
        let val = vec![
            (0.0, 10.0, 100.0),
            (10.0, 20.0, 200.0),
            (20.0, 30.0, 300.0),
        ];
        let default = -1.0;

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact match on key_min=0 → 100
        builder.append_value(10.0); // exact match on key_min=10 → 200
        builder.append_value(20.0); // exact match on key_min=20 → 300
        builder.append_value(-5.0); // below all ranges → default
        builder.append_value(25.0); // above last key_min but within last upper bound → 300

        let result = value_range_to_value_table_calculation(&builder, &val, &default).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 100.0).abs() < 1e-12);
        assert!((values[1] - 200.0).abs() < 1e-12);
        assert!((values[2] - 300.0).abs() < 1e-12);
        assert!((values[3] - (-1.0)).abs() < 1e-12);
        assert!((values[4] - 300.0).abs() < 1e-12);
    }

    #[test]
    fn test_algebraic_conversion_primitive() {
        // Expression: X * 2 + 1
        let parser = fasteval::Parser::new();
        let mut slab = fasteval::Slab::new();
        let compiled = parser.parse("X * 2 + 1", &mut slab.ps).unwrap();
        let compiled = compiled.from(&slab.ps).compile(&slab.ps, &mut slab.cs);

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // → 1
        builder.append_value(1.0); // → 3
        builder.append_value(5.0); // → 11
        builder.append_value(-3.0); // → -5

        let result = alegbraic_conversion_primitive(&compiled, &slab, &builder).unwrap();
        let values = result.values_slice();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.0).abs() < 1e-12);
        assert!((values[1] - 3.0).abs() < 1e-12);
        assert!((values[2] - 11.0).abs() < 1e-12);
        assert!((values[3] - (-5.0)).abs() < 1e-12);
    }
}
