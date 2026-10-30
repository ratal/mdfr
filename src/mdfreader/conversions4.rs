//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result, bail};
use arrow::array::{Array, LargeStringArray, LargeStringBuilder, PrimitiveArray};
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

/// convert a single channel into physical values as required by CCBlock content
pub fn convert_channel(cn: &mut Cn4, sharable: &SharableBlocks) -> Result<(), Error> {
    if cn.is_converted {
        return Ok(());
    }
    if cn.data.is_empty() {
        cn.is_converted = true;
        return Ok(());
    }

    if let Some(conv) = sharable.cc.get(&cn.block.cn_cc_conversion) {
        match conv.cc_type {
            1 => match &conv.cc_val {
                CcVal::Real(cc_val) => linear_conversion(cn, cc_val)
                    .with_context(|| format!("linear conversion failed for {}", cn.unique_name))?,
                CcVal::Uint(_) => (),
            },
            2 => match &conv.cc_val {
                CcVal::Real(cc_val) => rational_conversion(cn, cc_val).with_context(|| {
                    format!("rational conversion failed for {}", cn.unique_name)
                })?,
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
                CcVal::Real(cc_val) => value_to_value_with_interpolation(cn, cc_val.clone())
                    .with_context(|| {
                        format!(
                            "value to value conversion with interpolation failed for {}",
                            cn.unique_name
                        )
                    })?,
                CcVal::Uint(_) => (),
            },
            5 => match &conv.cc_val {
                CcVal::Real(cc_val) => value_to_value_without_interpolation(cn, cc_val.clone())
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
                CcVal::Real(cc_val) => value_to_text(cn, cc_val, &conv.cc_ref, sharable)
                    .with_context(|| {
                        format!("value to text conversion failed for {}", cn.unique_name)
                    })?,
                CcVal::Uint(_) => (),
            },
            8 => match &conv.cc_val {
                CcVal::Real(cc_val) => value_range_to_text(cn, cc_val, &conv.cc_ref, sharable)
                    .with_context(|| {
                        format!(
                            "value range to text conversion failed for {}",
                            cn.unique_name
                        )
                    })?,
                CcVal::Uint(_) => (),
            },
            9 => match &conv.cc_val {
                CcVal::Real(cc_val) => text_to_value(cn, cc_val, &conv.cc_ref, sharable)
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
                CcVal::Uint(cc_val) => bitfield_text_table(cn, cc_val, &conv.cc_ref, sharable)
                    .with_context(|| {
                        format!(
                            "bitfield text table conversion failed for {}",
                            cn.unique_name
                        )
                    })?,
            },
            0 => (),
            _ => bail!(
                "conversion type not recognised for channel {} not possible, type {}",
                cn.unique_name,
                conv.cc_type,
            ),
        }
    }
    cn.is_converted = true;
    // as data are converted, CCBlock is not anymore applicable
    cn.block.cn_cc_conversion = 0;
    Ok(())
}

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg4, sharable: &SharableBlocks) -> Result<(), Error> {
    dg.cg
        .par_iter_mut()
        .try_for_each(|(_, channel_group)| -> Result<(), Error> {
            channel_group.cn.par_iter_mut().try_for_each(
                |(_rec_pos, cn): (&i32, &mut Cn4)| -> Result<(), Error> {
                    convert_channel(cn, sharable)
                },
            )?;
            Ok(())
        })?;
    Ok(())
}

/// Dispatches a numeric conversion over ChannelData variants, emitting a complete match block.
///
/// Variant A (with Complex): scalar(a) => expr, f64(a) => block, complex(a) => expr, arrayd(a) => expr
/// Variant B (no Complex):   scalar(a) => expr, f64(a) => block, arrayd(a) => expr
///
/// The `scalar` arm covers Int8/UInt8/Int16/UInt16/Int32/UInt32/Float32/Int64/UInt64 → Float64.
/// The `complex` arm covers Complex32/64 → Complex64 (macro appends `.nulls()`).
/// The `arrayd` arm covers all 10 ArrayD* variants → ArrayDFloat64 (macro appends nulls/shape/order).
/// The `f64` block is emitted verbatim — caller must handle the assignment itself.
macro_rules! apply_conversion_f64 {
    // Variant A: with Complex arms
    (
        $cn:expr,
        scalar($a:ident) => $scalar_body:expr,
        f64($af:ident) => $f64_block:block,
        complex($ac:ident) => $complex_body:expr,
        arrayd($ad:ident) => $arrayd_body:expr $(,)?
    ) => {
        match &mut $cn.data {
            ChannelData::Int8($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt8($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int16($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt16($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Float32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int64($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt64($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Float64($af) => $f64_block,
            ChannelData::Complex32($ac) => {
                $cn.data = ChannelData::Complex64(ComplexArrow::new_from_array(
                    $complex_body,
                    $ac.nulls().cloned(),
                    $ac.shape().clone(),
                    $ac.order().clone(),
                ));
            }
            ChannelData::Complex64($ac) => {
                $cn.data = ChannelData::Complex64(ComplexArrow::new_from_array(
                    $complex_body,
                    $ac.nulls().cloned(),
                    $ac.shape().clone(),
                    $ac.order().clone(),
                ));
            }
            ChannelData::ArrayDInt8($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt8($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt16($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt16($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDFloat32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDFloat64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            _ => warn!(
                "conversion of channel {} not possible, channel does not contain primitives",
                $cn.unique_name
            ),
        }
    };

    // Variant B: no Complex arms (value-to-value conversions)
    (
        $cn:expr,
        scalar($a:ident) => $scalar_body:expr,
        f64($af:ident) => $f64_block:block,
        arrayd($ad:ident) => $arrayd_body:expr $(,)?
    ) => {
        match &mut $cn.data {
            ChannelData::Int8($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt8($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int16($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt16($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Float32($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Int64($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::UInt64($a) => {
                $cn.data = ChannelData::Float64($scalar_body);
            }
            ChannelData::Float64($af) => $f64_block,
            ChannelData::ArrayDInt8($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt8($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt16($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt16($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDFloat32($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDInt64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDUInt64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            ChannelData::ArrayDFloat64($ad) => {
                $cn.data = ChannelData::ArrayDFloat64(TensorArrow::new_from_array(
                    $arrayd_body,
                    $ad.nulls().cloned(),
                    $ad.shape().clone(),
                    $ad.order().clone(),
                ));
            }
            _ => warn!(
                "conversion of channel {} not possible, channel does not contain primitives",
                $cn.unique_name
            ),
        }
    };
}

/// Generic function calculating linear expression
#[inline]
pub fn linear_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    p1: f64,
    p2: f64,
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values();
    let converted: Vec<f64> = values.iter().map(|v| (*v).as_() * p2 + p1).collect();
    Ok(PrimitiveArray::new(converted.into(), array.nulls().cloned()))
}

/// Apply linear conversion to get physical data
pub fn linear_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    let (p1, p2) = (cc_val[0], cc_val[1]);
    if !(p1 == 0.0 && abs(p2 - 1.0) < 1e-12) {
        apply_conversion_f64!(
            cn,
            scalar(a) => linear_calculation(a, p1, p2).context("failed linear conversion")?,
            f64(a) => {
                let values = a.values();
                let converted: Vec<f64> = values.iter().map(|&x| x * p2 + p1).collect();
                *a = PrimitiveArray::new(converted.into(), a.nulls().cloned());
            },
            complex(a) => linear_calculation(a.values(), p1, p2).context("failed linear conversion of complex channel")?,
            arrayd(a) => linear_calculation(a.values(), p1, p2).context("failed linear conversion of tensor channel")?,
        );
    }
    Ok(())
}

/// Generic function calculating rational expression for a primitive
#[inline]
pub fn rational_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
) -> Result<PrimitiveArray<Float64Type>, Error>
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
    let values = array.values();
    let converted: Vec<f64> = values
        .iter()
        .map(|v| {
            let x: f64 = (*v).as_();
            (x * x * p1 + x * p2 + p3) / (x * x * p4 + x * p5 + p6)
        })
        .collect();
    Ok(PrimitiveArray::new(converted.into(), array.nulls().cloned()))
}

/// Apply rational conversion to get physical data
pub fn rational_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    apply_conversion_f64!(
        cn,
        scalar(a) => rational_calculation(a, cc_val).context("failed rational conversion")?,
        f64(a) => {
            let (p1, p2, p3, p4, p5, p6) = (cc_val[0], cc_val[1], cc_val[2], cc_val[3], cc_val[4], cc_val[5]);
            let values = a.values();
            let converted: Vec<f64> = values
                .iter()
                .map(|&v| {
                    (v * v * p1 + v * p2 + p3) / (v * v * p4 + v * p5 + p6)
                })
                .collect();
            *a = PrimitiveArray::new(converted.into(), a.nulls().cloned());
        },
        complex(a) => rational_calculation(a.values(), cc_val).context("failed rational conversion of complex channel")?,
        arrayd(a) => rational_calculation(a.values(), cc_val).context("failed rational conversion of tensor channel")?,
    );
    Ok(())
}

struct SingleVarNamespace {
    val: f64,
}

impl fasteval::EvalNamespace for SingleVarNamespace {
    #[inline(always)]
    fn lookup(&mut self, name: &str, _args: Vec<f64>, _keybuf: &mut String) -> Option<f64> {
        // ASAM MDF 4.3 §6.7.3 specifies 'X' (or 'x', or 'X1' for single-input CCBLOCKs)
        match name {
            "X" | "x" | "X1" => Some(self.val),
            _ => None,
        }
    }
}

/// Generic function calculating algebraic expression for a primitive
#[inline]
pub fn alegbraic_conversion_primitive<T: ArrowPrimitiveType>(
    compiled: &Instruction,
    slab: &Slab,
    array: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
    <T as ArrowPrimitiveType>::Native: Sync,
{
    let values = array.values();
    let mut new_array = vec![0f64; values.len()];
    if values.len() > 1024 {
        new_array
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(new_a, v)| {
                let a: f64 = (*v).as_();
                let mut ns = SingleVarNamespace { val: a };
                *new_a = match compiled.eval(slab, &mut ns) {
                    Ok(val) => val,
                    Err(err) => {
                        warn!(
                            "could not compute the value {a:?} with expression {compiled:?}, error {err}"
                        );
                        a
                    }
                };
            });
    } else {
        let mut ns = SingleVarNamespace { val: 0.0 };
        new_array.iter_mut().zip(values).for_each(|(new_a, v)| {
            let a: f64 = (*v).as_();
            ns.val = a;
            *new_a = match compiled.eval(slab, &mut ns) {
                Ok(val) => val,
                Err(err) => {
                    warn!(
                        "could not compute the value {a:?} with expression {compiled:?}, error {err}"
                    );
                    a
                }
            };
        });
    }
    Ok(PrimitiveArray::new(new_array.into(), array.nulls().cloned()))
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn4, formulae: &str) -> Result<(), Error> {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    match parser.parse(formulae, &mut slab.ps) {
        Ok(c) => {
            let compiled = c.from(&slab.ps).compile(&slab.ps, &mut slab.cs);
            apply_conversion_f64!(
                cn,
                scalar(a) => alegbraic_conversion_primitive(&compiled, &slab, a).context("failed algebraic conversion")?,
                f64(a) => {
                    *a = alegbraic_conversion_primitive(&compiled, &slab, a).context("failed algebraic conversion")?;
                },
                complex(a) => alegbraic_conversion_primitive(&compiled, &slab, a.values()).context("failed algebraic conversion of complex channel")?,
                arrayd(a) => alegbraic_conversion_primitive(&compiled, &slab, a.values()).context("failed algebraic conversion of tensor channel")?,
            );
        }
        Err(err) => warn!(
            "could not compile algebraic conversion expression {}, {}, channel {} not converted",
            formulae, err, cn.unique_name
        ),
    }
    Ok(())
}

/// Generic function calculating value to value interpolation for a primitive
#[inline]
pub fn value_to_value_with_interpolation_primitive<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    val: &[(&f64, &f64)],
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values();
    let mut new_array = vec![0f64; values.len()];
    new_array.iter_mut().zip(values).for_each(|(new_a, v)| {
        let a: f64 = (*v).as_();
        *new_a =
            match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal)) {
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
    Ok(PrimitiveArray::new(new_array.into(), array.nulls().cloned()))
}

/// Apply value to value with interpolation conversion to get physical data
pub fn value_to_value_with_interpolation(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    apply_conversion_f64!(
        cn,
        scalar(a) => value_to_value_with_interpolation_primitive(a, &val).context("failed value to value with interpolation conversion")?,
        f64(a) => {
            *a = value_to_value_with_interpolation_primitive(a, &val).context("failed value to value with interpolation conversion")?;
        },
        arrayd(a) => value_to_value_with_interpolation_primitive(a.values(), &val).context("failed value to value with interpolation conversion of tensor channel")?,
    );
    Ok(())
}

/// Generic function calculating value to value without interpolation for a primitive
#[inline]
pub fn value_to_value_without_interpolation_primitive<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    val: &[(&f64, &f64)],
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values();
    let mut new_array = vec![0f64; values.len()];
    new_array.iter_mut().zip(values).for_each(|(new_a, v)| {
        let a: f64 = (*v).as_();
        // MDF4 spec 6.17.7 (cc_type=5, value to value without interpolation):
        // - Exact match: return value[i]
        // - Below first key: return value[0]
        // - Above last key: return value[n-1]
        // - Between keys: return nearest neighbor; if equidistant, use lower key's value
        *new_a =
            match val.binary_search_by(|&(xi, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal)) {
                Ok(idx) => *val[idx].1,
                Err(0) => *val[0].1,
                Err(idx) if idx >= val.len() => *val[idx - 1].1,
                Err(idx) => {
                    let (x0, y0) = val[idx - 1];
                    let (x1, y1) = val[idx];
                    // spec: if (Int - key[i]) > (key[i+1] - Int) use upper, else lower
                    if (a - x0) > (x1 - a) { *y1 } else { *y0 }
                }
            };
    });
    Ok(PrimitiveArray::new(new_array.into(), array.nulls().cloned()))
}

/// Apply value to value without interpolation conversion to get physical data
pub fn value_to_value_without_interpolation(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    apply_conversion_f64!(
        cn,
        scalar(a) => value_to_value_without_interpolation_primitive(a, &val).context("failed value to value without interpolation conversion")?,
        f64(a) => {
            *a = value_to_value_without_interpolation_primitive(a, &val).context("failed value to value without interpolation conversion")?;
        },
        arrayd(a) => value_to_value_without_interpolation_primitive(a.values(), &val).context("failed value to value without interpolation conversion of tensor channel")?,
    );
    Ok(())
}

/// Generic function calculating value range to value table without interpolation
#[inline]
pub fn value_range_to_value_table_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    val: &[(f64, f64, f64)],
    default_value: &f64,
    inclusive_upper: bool,
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let values = array.values();
    let mut new_array = vec![0f64; values.len()];
    new_array.iter_mut().zip(values).for_each(|(new_a, v)| {
        let a: f64 = (*v).as_();
        // MDF4 spec 6.17.8 (cc_type=6, value range to value):
        // - For float types (cn_data_type > 3): key_min[i] ≤ Int < key_max[i] (exclusive upper)
        // - For integer types (cn_data_type ≤ 3): key_min[i] ≤ Int ≤ key_max[i] (both inclusive)
        // Ranges are sorted ascending and shall not overlap (key_max[i-1] ≤ key_min[i]).
        // For touching boundaries with float data, a == key_min[i] → range i wins (not i-1).
        // Binary search on min keys handles both cases naturally:
        //   Ok(idx): a == min[idx] → range idx includes a as its lower bound ✓
        //   Err(idx): min[idx-1] < a < min[idx] → check if a < max[idx-1] (exclusive upper)
        *new_a = match val
            .binary_search_by(|&(xi, _, _)| xi.partial_cmp(&a).unwrap_or(Ordering::Equal))
        {
            Ok(idx) => val[idx].2,
            Err(0) => *default_value, // below the minimum of all lower bounds
            Err(idx) => {
                // min[idx-1] < a < min[idx]: candidate is range idx-1
                // Float: upper exclusive (a < max[idx-1])
                // Integer: upper inclusive (a <= max[idx-1])
                let in_range = if inclusive_upper {
                    a <= val[idx - 1].1
                } else {
                    a < val[idx - 1].1
                };
                if in_range {
                    val[idx - 1].2
                } else {
                    *default_value
                }
            }
        };
    });
    Ok(PrimitiveArray::new(new_array.into(), array.nulls().cloned()))
}

/// Apply value range to value table without interpolation conversion to get physical data
pub fn value_range_to_value_table(cn: &mut Cn4, cc_val: Vec<f64>) -> Result<(), Error> {
    let mut val: Vec<(f64, f64, f64)> = Vec::new();
    for (a, b, c) in cc_val.iter().tuples::<(_, _, _)>() {
        val.push((*a, *b, *c));
    }
    let default_value = cc_val[cc_val.len() - 1];
    // MDF4 spec 6.17.8: integer data uses inclusive upper bound; float data uses exclusive upper.
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, false)
                    .context("failed value range to value table conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, true)
                    .context("failed value range to value table conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                value_range_to_value_table_calculation(a, &val, &default_value, false)
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
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    def: &DefaultTextOrScaleConversion,
    sharable: &SharableBlocks,
) -> Result<LargeStringArray, Error>
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
    let values = array.values();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    for (i, v) in values.iter().enumerate() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
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
    }
    Ok(new_array.finish())
}

/// Generic function calculating float value range to text
#[inline]
fn value_to_text_calculation_f32(
    a: &PrimitiveArray<Float32Type>,
    cc_val: &[f64],
    cc_ref: &[i64],
    canonization_value: f64,
    def: &DefaultTextOrScaleConversion,
    sharable: &SharableBlocks,
) -> LargeStringArray {
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
    for (i, val) in a.values().iter().enumerate() {
        if a.is_null(i) {
            new_array.append_null();
            continue;
        }
        let ref_val = (val * canonization_value as f32)
            .round()
            .to_i64()
            .unwrap_or_default();
        if let Some(tosc) = table_float.get(&ref_val) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.clone());
                }
                TextOrScaleConversion::Scale(conv) => {
                    new_array.append_value(conv.eval_to_txt(val.to_f64().unwrap_or_default()));
                }
                _ => {
                    new_array.append_value(val.to_string());
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.clone());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    new_array.append_value(conv.eval_to_txt(val.to_f64().unwrap_or_default()));
                }
                _ => {
                    new_array.append_value(val.to_string());
                }
            }
        }
    }
    new_array.finish()
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
            for (i, val) in a.values().iter().enumerate() {
                if a.is_null(i) {
                    new_array.append_null();
                    continue;
                }
                let ref_val = (val * 1024.0 * 1024.0).round() as i64;
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            new_array.append_value(txt.clone());
                        }
                        TextOrScaleConversion::Scale(conv) => {
                            new_array.append_value(conv.eval_to_txt(*val));
                        }
                        _ => {
                            new_array.append_value(val.to_string());
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            new_array.append_value(txt.clone());
                        }
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            new_array.append_value(conv.eval_to_txt(*val));
                        }
                        _ => {
                            new_array.append_value(val.to_string());
                        }
                    }
                }
            }
            cn.data = ChannelData::Utf8(new_array.finish());
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
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> LargeStringArray
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
    for pointer in cc_ref {
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
    let values = array.values();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    for (i, v) in values.iter().enumerate() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
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
    }
    new_array.finish()
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
    array: &LargeStringArray,
    cc_val: &[f64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> PrimitiveArray<Float64Type> {
    let mut table: HashMap<String, f64> = HashMap::with_capacity(cc_ref.len());
    for (ind, ccref) in cc_ref.iter().enumerate() {
        if let Ok(Some(txt)) = sharable.get_tx(*ccref) {
            table.insert(txt, cc_val[ind]);
        }
    }
    let default = cc_val[cc_val.len() - 1];
    let mut new_array = vec![0f64; array.len()];
    for (i, new_a) in new_array.iter_mut().enumerate() {
        if array.is_null(i) {
            *new_a = default;
            continue;
        }
        let s = array.value(i);
        if let Some(val) = table.get(s) {
            *new_a = *val;
        } else {
            *new_a = default;
        }
    }
    PrimitiveArray::new(new_array.into(), array.nulls().cloned())
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
    array: &LargeStringArray,
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> LargeStringArray {
    let pairs: Vec<(&i64, &i64)> = cc_ref.iter().tuples().collect();
    let mut table: HashMap<String, Option<String>> = HashMap::with_capacity(cc_ref.len());
    for ccref in &pairs {
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
    for i in 0..array.len() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
        let s = array.value(i);
        if let Some(val) = table.get(s) {
            if let Some(txt) = val.clone() {
                new_array.append_value(txt);
            } else {
                new_array.append_value(s);
            }
        } else if let Some(tx) = default.clone() {
            new_array.append_value(tx);
        } else {
            new_array.append_value(s);
        }
    }
    new_array.finish()
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
    array: &PrimitiveArray<T>,
    cc_val: &[u64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> Result<LargeStringArray>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64> + AsPrimitive<i64>,
{
    let mut table: Vec<(ValueOrValueRangeToText, Option<String>)> =
        Vec::with_capacity(cc_ref.len());
    for pointer in cc_ref {
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
                        for pointer in &cc.cc_ref {
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

    let values = array.values();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    values.iter().enumerate().for_each(|(i, v)| {
        if array.is_null(i) {
            new_array.append_null();
            return;
        }
        let a_f64: f64 = (*v).as_();
        let a_i64: i64 = a_f64.round() as i64;
        let mut new_a = String::new();
        for (ind, val) in cc_val.iter().enumerate() {
            match &table[ind] {
                (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                    let ref_val = a_i64 & (*val as i64);
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
                                    new_a =
                                        format!("{} | {} = {}", new_a, n, conv.eval_to_txt(a_f64));
                                } else {
                                    new_a = format!("{} | {}", new_a, conv.eval_to_txt(a_f64));
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
                    let matched_key = keys
                        .iter()
                        .enumerate()
                        .find(|&x| (x.1.min <= a_f64) && (a_f64 <= x.1.max));
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
                                    new_a =
                                        format!("{} | {} = {}", new_a, n, conv.eval_to_txt(a_f64));
                                } else {
                                    new_a = format!("{} | {}", new_a, conv.eval_to_txt(a_f64));
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
                                    new_a =
                                        format!("{} | {} = {}", new_a, n, conv.eval_to_txt(a_f64));
                                } else {
                                    new_a = format!("{} | {}", new_a, conv.eval_to_txt(a_f64));
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
    Ok(new_array.finish())
}

pub fn bitfield_text_table(
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
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of i8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of i16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of i32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of i64 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of f32 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Utf8(
                bitfield_text_table_calculation(a, cc_val, cc_ref, sharable)
                    .context("failed bitfield text table conversion of f64 channel")?,
            );
        }
        _ => {
            warn!(
                "bitfield conversion into text of channel {} not possible, channel type unsupported",
                cn.unique_name
            )
        }
    }
    Ok(())
}
