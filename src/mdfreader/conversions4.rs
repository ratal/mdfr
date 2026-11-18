//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result, bail};
use arrow::array::{Array, LargeStringArray, LargeStringBuilder, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Float32Type, Float64Type};
use itertools::Itertools;
use log::warn;
use num::NumCast;
use num::abs;
use num::cast::AsPrimitive;

use rustc_hash::FxHashMap;
use std::borrow::Cow;
use std::cmp::Ordering;

use crate::data_holder::channel_data::ChannelData;
use crate::data_holder::tensor_arrow::TensorArrow;
use crate::mdfinfo::mdfinfo4::{Ca4Block, Cc4Block, CcVal, Cn4, Cn4Block, Dg4, SharableBlocks};
use fasteval::{Compiler, Evaler, Instruction, Slab};
use rayon::prelude::*;

use crate::data_holder::complex_arrow::ComplexArrow;

/// Output MDF4 block signature of a channel after conversion.
#[derive(Debug, Clone, Copy)]
pub struct ConvertedOutputSignature {
    pub bit_count: u32,
    pub byte_count: u32,
    pub data_type: u8,
    pub is_vlsd: bool,
    pub is_array: bool,
    pub has_validity: bool,
}

pub(crate) fn channel_data_signature(
    data: &ChannelData,
    machine_endian: bool,
) -> ConvertedOutputSignature {
    ConvertedOutputSignature {
        bit_count: data.bit_count(),
        byte_count: data.byte_count(),
        data_type: data.data_type(machine_endian),
        is_vlsd: matches!(
            data,
            ChannelData::Utf8(_) | ChannelData::VariableSizeByteArray(_)
        ),
        is_array: matches!(
            data,
            ChannelData::ArrayDInt8(_)
                | ChannelData::ArrayDUInt8(_)
                | ChannelData::ArrayDInt16(_)
                | ChannelData::ArrayDUInt16(_)
                | ChannelData::ArrayDInt32(_)
                | ChannelData::ArrayDUInt32(_)
                | ChannelData::ArrayDFloat32(_)
                | ChannelData::ArrayDInt64(_)
                | ChannelData::ArrayDUInt64(_)
                | ChannelData::ArrayDFloat64(_)
        ),
        has_validity: data.validity().is_some(),
    }
}

fn will_noop_conversion(cn: &Cn4, sharable: &SharableBlocks) -> bool {
    if cn.is_converted || cn.data.is_empty() || cn.block.cn_cc_conversion == 0 {
        return true;
    }
    let Some(conv) = sharable.cc.get(&cn.block.cn_cc_conversion) else {
        return true;
    };
    match conv.cc_type {
        0 => true,
        1 => match &conv.cc_val {
            CcVal::Real(v) => v.len() >= 2 && v[0] == 0.0 && abs(v[1] - 1.0) < 1e-12,
            CcVal::Uint(_) => true,
        },
        2 | 4 | 5 | 6 | 7 | 8 | 9 => matches!(conv.cc_val, CcVal::Uint(_)),
        11 => matches!(conv.cc_val, CcVal::Real(_)),
        3 => conv.cc_ref.is_empty() || !matches!(sharable.get_tx(conv.cc_ref[0]), Ok(Some(_))),
        _ => false,
    }
}

/// Converts `cn.data` on the fly without mutating `cn` in place, borrowing `cn.data` when no conversion applies.
pub fn convert_channel_data_cow<'a>(
    cn: &'a Cn4,
    sharable: &SharableBlocks,
) -> Result<Cow<'a, ChannelData>, Error> {
    if will_noop_conversion(cn, sharable) {
        return Ok(Cow::Borrowed(&cn.data));
    }
    let mut tmp = Cn4 {
        unique_name: cn.unique_name.clone(),
        block: Cn4Block {
            cn_cc_conversion: cn.block.cn_cc_conversion,
            ..Default::default()
        },
        data: cn.data.clone(),
        is_converted: false,
        ..Default::default()
    };
    convert_channel(&mut tmp, sharable)?;
    Ok(Cow::Owned(tmp.data))
}

/// Computes the post-conversion MDF4 block signature for `cn` without converting numeric channels upfront.
pub fn converted_output_signature(
    cn: &Cn4,
    sharable: &SharableBlocks,
    machine_endian: bool,
) -> Result<ConvertedOutputSignature, Error> {
    let raw = &cn.data;
    if raw.is_empty() || will_noop_conversion(cn, sharable) {
        return Ok(channel_data_signature(raw, machine_endian));
    }
    let conv = sharable
        .cc
        .get(&cn.block.cn_cc_conversion)
        .expect("checked in will_noop_conversion");
    match conv.cc_type {
        1..=6 => {
            if conv.cc_type == 1
                && let CcVal::Real(v) = &conv.cc_val
                && v.len() < 2
            {
                bail!(
                    "malformed linear conversion: expected at least 2 parameters, got {}",
                    v.len()
                );
            }
            if conv.cc_type == 2
                && let CcVal::Real(v) = &conv.cc_val
                && v.len() < 6
            {
                bail!(
                    "malformed rational conversion: expected at least 6 parameters, got {}",
                    v.len()
                );
            }
            let has_validity = raw.validity().is_some();
            Ok(match raw {
                ChannelData::Int8(_)
                | ChannelData::UInt8(_)
                | ChannelData::Int16(_)
                | ChannelData::UInt16(_)
                | ChannelData::Int32(_)
                | ChannelData::UInt32(_)
                | ChannelData::Float32(_)
                | ChannelData::Int64(_)
                | ChannelData::UInt64(_)
                | ChannelData::Float64(_) => ConvertedOutputSignature {
                    bit_count: 64,
                    byte_count: 8,
                    data_type: if machine_endian { 5 } else { 4 },
                    is_vlsd: false,
                    is_array: false,
                    has_validity,
                },
                ChannelData::Complex32(_) | ChannelData::Complex64(_) => ConvertedOutputSignature {
                    bit_count: 128,
                    byte_count: 16,
                    data_type: if machine_endian { 16 } else { 15 },
                    is_vlsd: false,
                    is_array: false,
                    has_validity,
                },
                ChannelData::ArrayDInt8(_)
                | ChannelData::ArrayDUInt8(_)
                | ChannelData::ArrayDInt16(_)
                | ChannelData::ArrayDUInt16(_)
                | ChannelData::ArrayDInt32(_)
                | ChannelData::ArrayDUInt32(_)
                | ChannelData::ArrayDFloat32(_)
                | ChannelData::ArrayDInt64(_)
                | ChannelData::ArrayDUInt64(_)
                | ChannelData::ArrayDFloat64(_) => ConvertedOutputSignature {
                    bit_count: 64,
                    byte_count: 8,
                    data_type: if machine_endian { 5 } else { 4 },
                    is_vlsd: false,
                    is_array: true,
                    has_validity,
                },
                _ => channel_data_signature(raw, machine_endian),
            })
        }
        9 => {
            if matches!(raw, ChannelData::Utf8(_)) {
                Ok(ConvertedOutputSignature {
                    bit_count: 64,
                    byte_count: 8,
                    data_type: if machine_endian { 5 } else { 4 },
                    is_vlsd: false,
                    is_array: false,
                    has_validity: raw.validity().is_some(),
                })
            } else {
                Ok(channel_data_signature(raw, machine_endian))
            }
        }
        7 | 8 | 10 | 11 => {
            let converted = convert_channel_data_cow(cn, sharable)?;
            Ok(channel_data_signature(&converted, machine_endian))
        }
        _ => bail!(
            "conversion type not recognised for channel {} not possible, type {}",
            cn.unique_name,
            conv.cc_type,
        ),
    }
}

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
                CcVal::Real(cc_val) => {
                    value_to_value_with_interpolation(cn, cc_val).with_context(|| {
                        format!(
                            "value to value conversion with interpolation failed for {}",
                            cn.unique_name
                        )
                    })?
                }
                CcVal::Uint(_) => (),
            },
            5 => match &conv.cc_val {
                CcVal::Real(cc_val) => value_to_value_without_interpolation(cn, cc_val)
                    .with_context(|| {
                        format!(
                            "value to value conversion without interpolation failed for {}",
                            cn.unique_name
                        )
                    })?,
                CcVal::Uint(_) => (),
            },
            6 => match &conv.cc_val {
                CcVal::Real(cc_val) => {
                    value_range_to_value_table(cn, cc_val).with_context(|| {
                        format!(
                            "value range to value table conversion failed for {}",
                            cn.unique_name
                        )
                    })?
                }
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
                |(_rec_pos, cn): (&i64, &mut Cn4)| -> Result<(), Error> {
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
    Ok(PrimitiveArray::new(
        converted.into(),
        array.nulls().cloned(),
    ))
}

/// Apply linear conversion to get physical data
pub fn linear_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    if cc_val.len() < 2 {
        anyhow::bail!(
            "malformed linear conversion: expected at least 2 parameters, got {}",
            cc_val.len()
        );
    }
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
    if cc_val.len() < 6 {
        anyhow::bail!(
            "malformed rational conversion: expected at least 6 parameters, got {}",
            cc_val.len()
        );
    }
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
    Ok(PrimitiveArray::new(
        converted.into(),
        array.nulls().cloned(),
    ))
}

/// Apply rational conversion to get physical data
pub fn rational_conversion(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    if cc_val.len() < 6 {
        anyhow::bail!(
            "malformed rational conversion: expected at least 6 parameters, got {}",
            cc_val.len()
        );
    }
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

pub(crate) struct SingleVarNamespace {
    pub(crate) val: f64,
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
    Ok(PrimitiveArray::new(
        new_array.into(),
        array.nulls().cloned(),
    ))
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
    if val.is_empty() {
        let converted: Vec<f64> = values.iter().map(|v| (*v).as_()).collect();
        return Ok(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        ));
    }
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
    Ok(PrimitiveArray::new(
        new_array.into(),
        array.nulls().cloned(),
    ))
}

/// Apply value to value with interpolation conversion to get physical data
pub fn value_to_value_with_interpolation(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
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
    if val.is_empty() {
        let converted: Vec<f64> = values.iter().map(|v| (*v).as_()).collect();
        return Ok(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        ));
    }
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
    Ok(PrimitiveArray::new(
        new_array.into(),
        array.nulls().cloned(),
    ))
}

/// Apply value to value without interpolation conversion to get physical data
pub fn value_to_value_without_interpolation(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
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
    if val.is_empty() {
        let converted = vec![*default_value; values.len()];
        return Ok(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        ));
    }
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
        let idx = val.partition_point(|&(xi, _, _)| xi <= a);
        *new_a = if idx > 0 {
            let candidate = &val[idx - 1];
            let in_range = if inclusive_upper {
                a <= candidate.1
            } else {
                a < candidate.1 || (candidate.0 == candidate.1 && a == candidate.0)
            };
            if in_range {
                candidate.2
            } else {
                *default_value
            }
        } else {
            *default_value
        };
    });
    Ok(PrimitiveArray::new(
        new_array.into(),
        array.nulls().cloned(),
    ))
}

/// Apply value range to value table without interpolation conversion to get physical data
pub fn value_range_to_value_table(cn: &mut Cn4, cc_val: &[f64]) -> Result<(), Error> {
    let mut val: Vec<(f64, f64, f64)> = Vec::new();
    for (a, b, c) in cc_val.iter().tuples::<(_, _, _)>() {
        val.push((*a, *b, *c));
    }
    let default_value = cc_val
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("malformed value range to value table: empty cc_val"))?;
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

/// Canonical bit pattern for IEEE-754 f64 lookup keys, normalizing `-0.0` to `+0.0`.
#[inline]
fn canonical_f64_bits(val: f64) -> u64 {
    if val == 0.0 {
        0.0f64.to_bits()
    } else {
        val.to_bits()
    }
}

/// Resolves a CCBLOCK `cc_ref` link (text, nested CCBLOCK, or NIL).
fn resolve_text_or_scale(
    ptr: i64,
    sharable: &SharableBlocks,
    depth: usize,
) -> TextOrScaleConversion {
    if ptr == 0 {
        return TextOrScaleConversion::Nil;
    }
    match sharable.get_tx(ptr) {
        Ok(Some(txt)) => TextOrScaleConversion::Txt(txt),
        _ => {
            if let Some(cc) = sharable.cc.get(&ptr) {
                let conv = conversion_function_with_depth(cc, sharable, depth + 1);
                TextOrScaleConversion::Scale(Box::new(conv))
            } else {
                TextOrScaleConversion::Nil
            }
        }
    }
}

/// Resolves the default `cc_ref` link into `DefaultTextOrScaleConversion`.
fn resolve_default_text_or_scale(
    ptr: Option<i64>,
    sharable: &SharableBlocks,
    depth: usize,
) -> DefaultTextOrScaleConversion {
    match ptr {
        Some(p) if p != 0 => match sharable.get_tx(p) {
            Ok(Some(txt)) => DefaultTextOrScaleConversion::DefaultTxt(txt),
            _ => {
                if let Some(cc) = sharable.cc.get(&p) {
                    let conv = conversion_function_with_depth(cc, sharable, depth + 1);
                    DefaultTextOrScaleConversion::DefaultScale(Box::new(conv))
                } else {
                    DefaultTextOrScaleConversion::Nil
                }
            }
        },
        _ => DefaultTextOrScaleConversion::Nil,
    }
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
    let mut table_int: FxHashMap<i64, TextOrScaleConversion> =
        FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
    for (ind, val) in cc_val.iter().enumerate() {
        let val_i64 = (*val).round() as i64;
        let ptr = cc_ref.get(ind).copied().unwrap_or(0);
        table_int.insert(val_i64, resolve_text_or_scale(ptr, sharable, 0));
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
                    new_array.append_value(txt.as_str());
                }
                TextOrScaleConversion::Scale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(a_f64) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                TextOrScaleConversion::Nil => {
                    new_array.append_null();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.as_str());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(a_f64) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                DefaultTextOrScaleConversion::Nil => {
                    new_array.append_null();
                }
            }
        }
    }
    Ok(new_array.finish())
}

/// Generic function calculating float value to text
#[inline]
fn value_to_text_calculation_f32(
    a: &PrimitiveArray<Float32Type>,
    cc_val: &[f64],
    cc_ref: &[i64],
    def: &DefaultTextOrScaleConversion,
    sharable: &SharableBlocks,
) -> LargeStringArray {
    let mut table_float: FxHashMap<u64, TextOrScaleConversion> =
        FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
    for (ind, val) in cc_val.iter().enumerate() {
        // Key by the f32-widened f64 bit pattern so `f32 as f64` matches `cc_val` entries
        let ref_val = canonical_f64_bits((*val as f32) as f64);
        let ptr = cc_ref.get(ind).copied().unwrap_or(0);
        table_float.insert(ref_val, resolve_text_or_scale(ptr, sharable, 0));
    }
    let mut new_array = LargeStringBuilder::with_capacity(a.len(), 32);
    for (i, val) in a.values().iter().enumerate() {
        if a.is_null(i) {
            new_array.append_null();
            continue;
        }
        let val_f64 = *val as f64;
        let ref_val = canonical_f64_bits(val_f64);
        if let Some(tosc) = table_float.get(&ref_val) {
            match tosc {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.as_str());
                }
                TextOrScaleConversion::Scale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(val_f64) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                TextOrScaleConversion::Nil => {
                    new_array.append_null();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.as_str());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(val_f64) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                DefaultTextOrScaleConversion::Nil => {
                    new_array.append_null();
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
    let def = resolve_default_text_or_scale(cc_ref.get(cc_val.len()).copied(), sharable, 0);
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
                    .context("failed value to text conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(value_to_text_calculation_f32(
                a, cc_val, cc_ref, &def, sharable,
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
            let mut table_float: FxHashMap<u64, TextOrScaleConversion> =
                FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
            for (ind, val) in cc_val.iter().enumerate() {
                let ref_val = canonical_f64_bits(*val);
                let ptr = cc_ref.get(ind).copied().unwrap_or(0);
                table_float.insert(ref_val, resolve_text_or_scale(ptr, sharable, 0));
            }
            let mut new_array = LargeStringBuilder::with_capacity(a.len(), 32);
            for (i, val) in a.values().iter().enumerate() {
                if a.is_null(i) {
                    new_array.append_null();
                    continue;
                }
                let ref_val = canonical_f64_bits(*val);
                if let Some(tosc) = table_float.get(&ref_val) {
                    match tosc {
                        TextOrScaleConversion::Txt(txt) => {
                            new_array.append_value(txt.as_str());
                        }
                        TextOrScaleConversion::Scale(conv) => {
                            if let Some(s) = conv.eval_to_opt_txt(*val) {
                                new_array.append_value(s);
                            } else {
                                new_array.append_null();
                            }
                        }
                        TextOrScaleConversion::Nil => {
                            new_array.append_null();
                        }
                    }
                } else {
                    match &def {
                        DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                            new_array.append_value(txt.as_str());
                        }
                        DefaultTextOrScaleConversion::DefaultScale(conv) => {
                            if let Some(s) = conv.eval_to_opt_txt(*val) {
                                new_array.append_value(s);
                            } else {
                                new_array.append_null();
                            }
                        }
                        DefaultTextOrScaleConversion::Nil => {
                            new_array.append_null();
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

/// Scalar conversion evaluator supporting nested / chained MDF4 CCBLOCKs
#[derive(Debug)]
enum ConversionFunction {
    Identity,
    Linear(f64, f64),
    Rational(f64, f64, f64, f64, f64, f64),
    Algebraic(Instruction, Box<Slab>),
    ValueToValueInterp(Vec<f64>, Vec<f64>),
    ValueToValue(Vec<f64>, Vec<f64>),
    ValueRangeToValue(Vec<KeyRange>, Vec<f64>, f64),
    ValueToText(
        FxHashMap<i64, TextOrScaleConversion>,
        FxHashMap<u64, TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
    ),
    ValueRangeToText(
        Vec<KeyRange>,
        Vec<TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
    ),
}

const MAX_CONVERSION_DEPTH: usize = 8;

/// Builds a single-value [`ConversionFunction`] from a `Cc4Block` with recursion guard.
fn conversion_function(cc: &Cc4Block, sharable: &SharableBlocks) -> ConversionFunction {
    conversion_function_with_depth(cc, sharable, 0)
}

fn conversion_function_with_depth(
    cc: &Cc4Block,
    sharable: &SharableBlocks,
    depth: usize,
) -> ConversionFunction {
    if depth >= MAX_CONVERSION_DEPTH {
        return ConversionFunction::Identity;
    }
    match &cc.cc_val {
        CcVal::Real(cc_val) => match cc.cc_type {
            0 => ConversionFunction::Identity,
            1 => {
                if cc_val.len() >= 2 {
                    ConversionFunction::Linear(cc_val[0], cc_val[1])
                } else {
                    ConversionFunction::Identity
                }
            }
            2 => {
                if cc_val.len() >= 6 {
                    ConversionFunction::Rational(
                        cc_val[0], cc_val[1], cc_val[2], cc_val[3], cc_val[4], cc_val[5],
                    )
                } else {
                    ConversionFunction::Identity
                }
            }
            3 => {
                if let Some(&tx_ptr) = cc.cc_ref.first() {
                    match sharable.get_tx(tx_ptr) {
                        Ok(Some(formulae)) => {
                            let parser = fasteval::Parser::new();
                            let mut slab = fasteval::Slab::new();
                            match parser.parse(&formulae, &mut slab.ps) {
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
            4 | 5 => {
                let mut keys = Vec::with_capacity(cc_val.len() / 2);
                let mut vals = Vec::with_capacity(cc_val.len() / 2);
                for pair in cc_val.as_chunks::<2>().0 {
                    keys.push(pair[0]);
                    vals.push(pair[1]);
                }
                if cc.cc_type == 4 {
                    ConversionFunction::ValueToValueInterp(keys, vals)
                } else {
                    ConversionFunction::ValueToValue(keys, vals)
                }
            }
            6 => {
                let n_ranges = cc_val.len().saturating_sub(1) / 3;
                let mut ranges = Vec::with_capacity(n_ranges);
                let mut vals = Vec::with_capacity(n_ranges);
                for chunk in cc_val.as_chunks::<3>().0.iter().take(n_ranges) {
                    ranges.push(KeyRange {
                        min: chunk[0],
                        max: chunk[1],
                    });
                    vals.push(chunk[2]);
                }
                let def = cc_val.last().copied().unwrap_or(0.0);
                ConversionFunction::ValueRangeToValue(ranges, vals, def)
            }
            7 => {
                let mut table_int =
                    FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
                let mut table_bits =
                    FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
                for (ind, &v) in cc_val.iter().enumerate() {
                    let ptr = cc.cc_ref.get(ind).copied().unwrap_or(0);
                    table_int.insert(
                        v.round() as i64,
                        resolve_text_or_scale(ptr, sharable, depth),
                    );
                    let ptr2 = cc.cc_ref.get(ind).copied().unwrap_or(0);
                    table_bits.insert(
                        canonical_f64_bits(v),
                        resolve_text_or_scale(ptr2, sharable, depth),
                    );
                }
                let def = resolve_default_text_or_scale(
                    cc.cc_ref.get(cc_val.len()).copied(),
                    sharable,
                    depth,
                );
                ConversionFunction::ValueToText(table_int, table_bits, def)
            }
            8 => {
                let n_keys = cc_val.len() / 2;
                let mut keys = Vec::with_capacity(n_keys);
                let mut entries = Vec::with_capacity(n_keys);
                for (ind, pair) in cc_val.as_chunks::<2>().0.iter().enumerate() {
                    keys.push(KeyRange {
                        min: pair[0],
                        max: pair[1],
                    });
                    let ptr = cc.cc_ref.get(ind).copied().unwrap_or(0);
                    entries.push(resolve_text_or_scale(ptr, sharable, depth));
                }
                let def =
                    resolve_default_text_or_scale(cc.cc_ref.get(n_keys).copied(), sharable, depth);
                ConversionFunction::ValueRangeToText(keys, entries, def)
            }
            _ => ConversionFunction::Identity,
        },
        CcVal::Uint(_) => ConversionFunction::Identity,
    }
}

/// Scalar conversion evaluation helpers (used for nested/partial rules and CABLOCK axes).
impl ConversionFunction {
    pub(crate) fn eval_to_f64(&self, a: f64) -> f64 {
        match self {
            ConversionFunction::Identity => a,
            ConversionFunction::Linear(p1, p2) => a * p2 + p1,
            ConversionFunction::Rational(p1, p2, p3, p4, p5, p6) => {
                let a_2 = a * a;
                (a_2 * p1 + a * p2 + p3) / (a_2 * p4 + a * p5 + p6)
            }
            ConversionFunction::Algebraic(compiled, slab) => {
                let mut ns = SingleVarNamespace { val: a };
                match compiled.eval(slab, &mut ns) {
                    Ok(res) => res,
                    Err(e) => {
                        warn!("could not evaluate algebraic expression for {a}, error {e}");
                        a
                    }
                }
            }
            ConversionFunction::ValueToValueInterp(keys, vals) => {
                if keys.is_empty() {
                    return a;
                }
                let idx = keys.partition_point(|&k| k <= a);
                if idx == 0 {
                    vals[0]
                } else if idx >= keys.len() {
                    vals[vals.len() - 1]
                } else {
                    let k0 = keys[idx - 1];
                    let k1 = keys[idx];
                    let v0 = vals[idx - 1];
                    let v1 = vals[idx];
                    if (k1 - k0).abs() < f64::EPSILON {
                        v0
                    } else {
                        v0 + (a - k0) * (v1 - v0) / (k1 - k0)
                    }
                }
            }
            ConversionFunction::ValueToValue(keys, vals) => {
                if keys.is_empty() {
                    return a;
                }
                let idx = keys.partition_point(|&k| k < a);
                if idx == 0 {
                    vals[0]
                } else if idx >= keys.len() {
                    vals[vals.len() - 1]
                } else if (keys[idx] - a).abs() < (a - keys[idx - 1]).abs() {
                    vals[idx]
                } else {
                    vals[idx - 1]
                }
            }
            ConversionFunction::ValueRangeToValue(ranges, vals, def) => {
                let idx = ranges.partition_point(|r| r.min <= a);
                if idx > 0 {
                    let r = &ranges[idx - 1];
                    if a < r.max || (r.min == r.max && a == r.min) {
                        return vals[idx - 1];
                    }
                }
                *def
            }
            ConversionFunction::ValueToText(..) | ConversionFunction::ValueRangeToText(..) => a,
        }
    }

    fn eval_to_opt_txt(&self, a: f64) -> Option<String> {
        match self {
            ConversionFunction::ValueToText(table_int, table_bits, def) => {
                let item = table_bits
                    .get(&canonical_f64_bits(a))
                    .or_else(|| table_int.get(&(a.round() as i64)));
                if let Some(entry) = item {
                    match entry {
                        TextOrScaleConversion::Txt(t) => Some(t.clone()),
                        TextOrScaleConversion::Scale(sub) => sub.eval_to_opt_txt(a),
                        TextOrScaleConversion::Nil => None,
                    }
                } else {
                    match def {
                        DefaultTextOrScaleConversion::DefaultTxt(t) => Some(t.clone()),
                        DefaultTextOrScaleConversion::DefaultScale(sub) => sub.eval_to_opt_txt(a),
                        DefaultTextOrScaleConversion::Nil => None,
                    }
                }
            }
            ConversionFunction::ValueRangeToText(keys, entries, def) => {
                let idx = keys.partition_point(|k| k.min <= a);
                let matched = if idx > 0 {
                    let cand = &keys[idx - 1];
                    if a < cand.max || (cand.min == cand.max && a == cand.min) {
                        Some(idx - 1)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(k_idx) = matched {
                    match &entries[k_idx] {
                        TextOrScaleConversion::Txt(t) => Some(t.clone()),
                        TextOrScaleConversion::Scale(sub) => sub.eval_to_opt_txt(a),
                        TextOrScaleConversion::Nil => None,
                    }
                } else {
                    match def {
                        DefaultTextOrScaleConversion::DefaultTxt(t) => Some(t.clone()),
                        DefaultTextOrScaleConversion::DefaultScale(sub) => sub.eval_to_opt_txt(a),
                        DefaultTextOrScaleConversion::Nil => None,
                    }
                }
            }
            _ => Some(self.eval_to_f64(a).to_string()),
        }
    }

    fn eval_to_txt(&self, a: f64) -> String {
        self.eval_to_opt_txt(a).unwrap_or_default()
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
    inclusive_upper: bool,
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
    if let Some(def_ref) = cc_ref.get(n_keys) {
        match sharable.get_tx(*def_ref) {
            Ok(Some(t)) => {
                def = DefaultTextOrScaleConversion::DefaultTxt(t);
            }
            _ => {
                if let Some(cc) = sharable.cc.get(def_ref) {
                    let conv = conversion_function(cc, sharable);
                    def = DefaultTextOrScaleConversion::DefaultScale(Box::new(conv));
                } else {
                    def = DefaultTextOrScaleConversion::Nil;
                }
            }
        }
    } else {
        def = DefaultTextOrScaleConversion::Nil;
    }
    let values = array.values();
    let mut new_array = LargeStringBuilder::with_capacity(values.len(), 32);
    for (i, v) in values.iter().enumerate() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
        let a = (*v).as_();
        let idx = keys.partition_point(|k| k.min <= a);
        let matched_key = if idx > 0 {
            let candidate = &keys[idx - 1];
            let in_range = if inclusive_upper {
                a <= candidate.max
            } else {
                a < candidate.max || (candidate.min == candidate.max && a == candidate.min)
            };
            if in_range { Some(idx - 1) } else { None }
        } else {
            None
        };
        if let Some(key_idx) = matched_key {
            match &txt[key_idx] {
                TextOrScaleConversion::Txt(txt) => {
                    new_array.append_value(txt.as_str());
                }
                TextOrScaleConversion::Scale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(a) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                TextOrScaleConversion::Nil => {
                    new_array.append_null();
                }
            }
        } else {
            match &def {
                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                    new_array.append_value(txt.as_str());
                }
                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                    if let Some(s) = conv.eval_to_opt_txt(a) {
                        new_array.append_value(s);
                    } else {
                        new_array.append_null();
                    }
                }
                DefaultTextOrScaleConversion::Nil => {
                    new_array.append_null();
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
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, false,
            ));
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, true,
            ));
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Utf8(value_range_to_text_calculation(
                a, cc_val, cc_ref, sharable, false,
            ));
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
    let mut table: FxHashMap<String, f64> =
        FxHashMap::with_capacity_and_hasher(cc_ref.len(), Default::default());
    for (ind, ccref) in cc_ref.iter().enumerate() {
        if let (Ok(Some(txt)), Some(&val)) = (sharable.get_tx(*ccref), cc_val.get(ind)) {
            table.insert(txt, val);
        }
    }
    let default = cc_val
        .get(cc_ref.len())
        .copied()
        .unwrap_or_else(|| cc_val.last().copied().unwrap_or(0.0));
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
    let mut table: FxHashMap<String, Option<String>> =
        FxHashMap::with_capacity_and_hasher(cc_ref.len(), Default::default());
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
    if !cc_ref.len().is_multiple_of(2)
        && let Some(&def_ptr) = cc_ref.last()
        && let Ok(Some(txt)) = sharable.get_tx(def_ptr)
    {
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
        } else if let Some(tx) = default.as_deref() {
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
        FxHashMap<u64, TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
    ),
    ValueRangeToText(
        Vec<TextOrScaleConversion>,
        DefaultTextOrScaleConversion,
        Vec<KeyRange>,
    ),
}

/// Lossless 64-bit mask extraction trait for integers, with rounding for floats.
trait ToMaskU64: Copy {
    fn to_mask_u64(self) -> u64;
}
impl ToMaskU64 for u8 {
    fn to_mask_u64(self) -> u64 {
        self as u64
    }
}
impl ToMaskU64 for u16 {
    fn to_mask_u64(self) -> u64 {
        self as u64
    }
}
impl ToMaskU64 for u32 {
    fn to_mask_u64(self) -> u64 {
        self as u64
    }
}
impl ToMaskU64 for u64 {
    fn to_mask_u64(self) -> u64 {
        self
    }
}
impl ToMaskU64 for i8 {
    fn to_mask_u64(self) -> u64 {
        (self as i64) as u64
    }
}
impl ToMaskU64 for i16 {
    fn to_mask_u64(self) -> u64 {
        (self as i64) as u64
    }
}
impl ToMaskU64 for i32 {
    fn to_mask_u64(self) -> u64 {
        (self as i64) as u64
    }
}
impl ToMaskU64 for i64 {
    fn to_mask_u64(self) -> u64 {
        self as u64
    }
}
impl ToMaskU64 for f32 {
    fn to_mask_u64(self) -> u64 {
        (self.round() as i64) as u64
    }
}
impl ToMaskU64 for f64 {
    fn to_mask_u64(self) -> u64 {
        (self.round() as i64) as u64
    }
}

#[inline]
fn cc_val_f64_to_u64_key(v: f64) -> u64 {
    let r = v.round();
    if r < 0.0 { (r as i64) as u64 } else { r as u64 }
}

/// Generic function calculating text to value
#[inline]
fn append_bitfield_part(out: &mut String, name: Option<&String>, part: &str) {
    if !out.is_empty() {
        out.push_str(" | ");
    }
    if let Some(n) = name {
        out.push_str(n);
        out.push_str(" = ");
    }
    out.push_str(part);
}

#[inline]
fn bitfield_text_table_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: &[u64],
    cc_ref: &[i64],
    sharable: &SharableBlocks,
) -> Result<LargeStringArray>
where
    <T as ArrowPrimitiveType>::Native: ToMaskU64,
{
    let mut table: Vec<(ValueOrValueRangeToText, Option<String>)> =
        Vec::with_capacity(cc_ref.len());
    for pointer in cc_ref {
        if let Some(cc) = sharable.cc.get(pointer) {
            let name: Option<String> = if cc.cc_tx_name != 0 {
                match sharable.get_tx(cc.cc_tx_name) {
                    Ok(Some(n)) => Some(n),
                    _ => None,
                }
            } else {
                None
            };
            if cc.cc_type == 7 {
                match &cc.cc_val {
                    CcVal::Real(cc_val) => {
                        let mut table_int: FxHashMap<u64, TextOrScaleConversion> =
                            FxHashMap::with_capacity_and_hasher(cc_val.len(), Default::default());
                        for (ind, val) in cc_val.iter().enumerate() {
                            let val_u64 = cc_val_f64_to_u64_key(*val);
                            let ref_ptr = cc.cc_ref.get(ind).copied().unwrap_or(0);
                            table_int.insert(val_u64, resolve_text_or_scale(ref_ptr, sharable, 0));
                        }
                        let def = resolve_default_text_or_scale(
                            cc.cc_ref.get(cc_val.len()).copied(),
                            sharable,
                            0,
                        );
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
                            let key = KeyRange {
                                min: *key_min,
                                max: *key_max,
                            };
                            keys.push(key);
                        }
                        let mut txt: Vec<TextOrScaleConversion> = Vec::with_capacity(n_keys);
                        for pointer in cc.cc_ref.iter().take(n_keys) {
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
                        let def: DefaultTextOrScaleConversion = if let Some(&def_ptr) =
                            cc.cc_ref.get(n_keys)
                        {
                            match sharable.get_tx(def_ptr) {
                                Ok(Some(t)) => DefaultTextOrScaleConversion::DefaultTxt(t),
                                _ => {
                                    if let Some(ccc) = sharable.cc.get(&def_ptr) {
                                        let conv = conversion_function(ccc, sharable);
                                        DefaultTextOrScaleConversion::DefaultScale(Box::new(conv))
                                    } else {
                                        DefaultTextOrScaleConversion::Nil
                                    }
                                }
                            }
                        } else {
                            DefaultTextOrScaleConversion::Nil
                        };
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
        let a_u64 = (*v).to_mask_u64();
        let mut new_a = String::new();
        for (ind, val) in cc_val.iter().enumerate() {
            if let Some(entry) = table.get(ind) {
                let ref_val = a_u64 & *val;
                let masked_f64 = ref_val as f64;
                match entry {
                    (ValueOrValueRangeToText::ValueToText(table_int, def), name) => {
                        if let Some(tosc) = table_int.get(&ref_val) {
                            match tosc {
                                TextOrScaleConversion::Txt(txt) => {
                                    append_bitfield_part(&mut new_a, name.as_ref(), txt);
                                }
                                TextOrScaleConversion::Scale(conv) => {
                                    append_bitfield_part(
                                        &mut new_a,
                                        name.as_ref(),
                                        &conv.eval_to_txt(masked_f64),
                                    );
                                }
                                TextOrScaleConversion::Nil => {}
                            }
                        } else {
                            match &def {
                                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                    append_bitfield_part(&mut new_a, name.as_ref(), txt);
                                }
                                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                    append_bitfield_part(
                                        &mut new_a,
                                        name.as_ref(),
                                        &conv.eval_to_txt(masked_f64),
                                    );
                                }
                                DefaultTextOrScaleConversion::Nil => {}
                            }
                        }
                    }
                    (ValueOrValueRangeToText::ValueRangeToText(txt, def, keys), name) => {
                        let idx = keys.partition_point(|k| k.min <= masked_f64);
                        let matched_key = if idx > 0 {
                            let candidate = &keys[idx - 1];
                            if masked_f64 <= candidate.max {
                                Some(idx - 1)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if let Some(key_idx) = matched_key {
                            if let Some(conv_item) = txt.get(key_idx) {
                                match conv_item {
                                    TextOrScaleConversion::Txt(txt) => {
                                        append_bitfield_part(&mut new_a, name.as_ref(), txt);
                                    }
                                    TextOrScaleConversion::Scale(conv) => {
                                        append_bitfield_part(
                                            &mut new_a,
                                            name.as_ref(),
                                            &conv.eval_to_txt(masked_f64),
                                        );
                                    }
                                    TextOrScaleConversion::Nil => {}
                                }
                            }
                        } else {
                            match &def {
                                DefaultTextOrScaleConversion::DefaultTxt(txt) => {
                                    append_bitfield_part(&mut new_a, name.as_ref(), txt);
                                }
                                DefaultTextOrScaleConversion::DefaultScale(conv) => {
                                    append_bitfield_part(
                                        &mut new_a,
                                        name.as_ref(),
                                        &conv.eval_to_txt(masked_f64),
                                    );
                                }
                                DefaultTextOrScaleConversion::Nil => {}
                            }
                        }
                    }
                }
            }
        }
        new_array.append_value(&new_a);
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

/// Converts raw `CABLOCK` axis values for dimension `axis_idx` using `ca_cc_axis_conversion[axis_idx]`.
///
/// Per ASAM MDF 4.3 §6.20.1, `ca_cc_axis_conversion` overrules any conversion rule on the
/// referenced axis channel, including treating a NIL pointer (`0`) as an identity (1:1) conversion.
pub fn convert_ca_axis_values(
    ca: &Ca4Block,
    axis_idx: usize,
    raw_axis_vals: &[f64],
    sharable: &SharableBlocks,
) -> Vec<f64> {
    let cc_ptr = ca
        .ca_cc_axis_conversion
        .as_ref()
        .and_then(|ptrs| ptrs.get(axis_idx).copied())
        .unwrap_or(0);
    if cc_ptr == 0 {
        return raw_axis_vals.to_vec();
    }
    if let Some(cc) = sharable.cc.get(&cc_ptr) {
        let conv = conversion_function(cc, sharable);
        raw_axis_vals.iter().map(|&v| conv.eval_to_f64(v)).collect()
    } else {
        raw_axis_vals.to_vec()
    }
}

/// Extracts and converts fixed axis values for dimension `axis_idx` from `ca_axis_value`.
pub fn extract_ca_fixed_axis(
    ca: &Ca4Block,
    axis_idx: usize,
    sharable: &SharableBlocks,
) -> Option<Vec<f64>> {
    let all_vals = ca.ca_axis_value.as_ref()?;
    let ndim = ca.ca_ndim as usize;
    if axis_idx >= ndim || axis_idx >= ca.ca_dim_size.len() {
        return None;
    }
    let sum_dims: usize = ca.ca_dim_size.iter().take(ndim).map(|&d| d as usize).sum();
    let is_interval_boundaries =
        (ca.ca_type == 3 || ca.ca_type == 4) && all_vals.len() >= sum_dims + ndim;
    let extra = if is_interval_boundaries { 1 } else { 0 };
    let offset: usize = ca
        .ca_dim_size
        .iter()
        .take(axis_idx)
        .map(|&d| d as usize + extra)
        .sum();
    let len = ca.ca_dim_size[axis_idx] as usize + extra;
    let slice = all_vals.get(offset..offset + len)?;
    Some(convert_ca_axis_values(ca, axis_idx, slice, sharable))
}

/// Evaluates a D-dimensional `CABLOCK` lookup table (`ca_type == 2`) via multilinear interpolation.
///
/// Out-of-bounds coordinates are clamped to endpoint axis values. Supports both row-major and
/// column-major (`CA_F_INVERSE`) table layouts.
pub fn eval_ca_lookup(
    axes: &[Vec<f64>],
    table_values: &[f64],
    query: &[f64],
    row_major: bool,
) -> Result<f64> {
    let ndim = axes.len();
    if ndim == 0 || query.len() != ndim {
        bail!(
            "CA lookup dimension mismatch: {} axes vs {} coordinates",
            ndim,
            query.len()
        );
    }
    if ndim > 16 {
        bail!("CA lookup with {ndim} dimensions exceeds 16-dimension limit");
    }
    if axes.iter().any(|a| a.is_empty()) {
        bail!("CA lookup axis cannot be empty");
    }
    let expected_len: usize = axes.iter().map(|a| a.len()).product();
    if table_values.len() < expected_len {
        bail!(
            "CA lookup table has {} values, expected at least {}",
            table_values.len(),
            expected_len
        );
    }

    let mut idx_lo = Vec::with_capacity(ndim);
    let mut idx_hi = Vec::with_capacity(ndim);
    let mut frac = Vec::with_capacity(ndim);
    for (axis, &coord) in axes.iter().zip(query.iter()) {
        if axis.len() == 1 || coord <= axis[0] {
            idx_lo.push(0);
            idx_hi.push(0);
            frac.push(0.0);
        } else if coord >= axis[axis.len() - 1] {
            let last = axis.len() - 1;
            idx_lo.push(last);
            idx_hi.push(last);
            frac.push(0.0);
        } else {
            let p = axis.partition_point(|&v| v <= coord);
            let i0 = p.saturating_sub(1);
            let i1 = p.min(axis.len() - 1);
            let span = axis[i1] - axis[i0];
            let t = if span.abs() < f64::EPSILON {
                0.0
            } else {
                (coord - axis[i0]) / span
            };
            idx_lo.push(i0);
            idx_hi.push(i1);
            frac.push(t);
        }
    }

    let mut strides = vec![1usize; ndim];
    if row_major {
        for d in (0..ndim.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * axes[d + 1].len();
        }
    } else {
        for d in 1..ndim {
            strides[d] = strides[d - 1] * axes[d - 1].len();
        }
    }

    let mut acc = 0.0;
    for corner in 0..(1usize << ndim) {
        let mut weight = 1.0;
        let mut flat_idx = 0;
        for d in 0..ndim {
            if (corner & (1 << d)) == 0 {
                weight *= 1.0 - frac[d];
                flat_idx += idx_lo[d] * strides[d];
            } else {
                weight *= frac[d];
                flat_idx += idx_hi[d] * strides[d];
            }
        }
        acc += weight * table_values[flat_idx];
    }
    Ok(acc)
}

/// Evaluates a 1D `IntervalAxis` / `ClassificationResult` (`ca_type == 3 | 4`) bin value.
///
/// Respects MDF4 `CA_F_LEFT_OPEN` (`left_open`): `true` uses `(lo, hi]`, `false` uses `[lo, hi)`.
pub fn eval_ca_classification(
    boundaries: &[f64],
    class_counts: &[f64],
    value: f64,
    left_open: bool,
) -> Option<f64> {
    if boundaries.len() < class_counts.len() + 1 {
        return None;
    }
    for (i, &count) in class_counts.iter().enumerate() {
        let lo = boundaries[i];
        let hi = boundaries[i + 1];
        let in_bin = if left_open {
            value > lo && value <= hi
        } else {
            value >= lo && value < hi
        };
        if in_bin {
            return Some(count);
        }
    }
    None
}
