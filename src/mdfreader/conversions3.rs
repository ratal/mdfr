//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result, bail};
use arrow::array::{Array, LargeStringArray, LargeStringBuilder, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Float64Type};
use itertools::Itertools;
use num::NumCast;
use num::abs;
use num::cast::AsPrimitive;
use rustc_hash::FxHashMap;
use std::borrow::Cow;

use crate::data_holder::channel_data::ChannelData;
use crate::mdfinfo::mdfinfo3::{Cn3, Conversion, Dg3, SharableBlocks3};
use crate::mdfreader::conversions4::{
    ConvertedOutputSignature, alegbraic_conversion_primitive, channel_data_signature,
    linear_calculation, rational_calculation, value_to_value_with_interpolation_primitive,
    value_to_value_without_interpolation_primitive,
};
use fasteval::Compiler;
use log::warn;
use rayon::prelude::*;

/// Dispatches a conversion closure over all 10 numeric scalar [`ChannelData`] variants.
macro_rules! apply_scalar_cn3 {
    ($cn:expr, $conv_name:literal, |$a:ident| $body:expr) => {
        match &mut $cn.data {
            ChannelData::Int8($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of i8 channel"))?,
                );
            }
            ChannelData::UInt8($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of u8 channel"))?,
                );
            }
            ChannelData::Int16($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of i16 channel"))?,
                );
            }
            ChannelData::UInt16($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of u16 channel"))?,
                );
            }
            ChannelData::Int32($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of i32 channel"))?,
                );
            }
            ChannelData::UInt32($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of u32 channel"))?,
                );
            }
            ChannelData::Float32($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of f32 channel"))?,
                );
            }
            ChannelData::Int64($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of i64 channel"))?,
                );
            }
            ChannelData::UInt64($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of u64 channel"))?,
                );
            }
            ChannelData::Float64($a) => {
                $cn.data = ChannelData::Float64(
                    $body.with_context(|| concat!("failed ", $conv_name, " of f64 channel"))?,
                );
            }
            _ => warn!(
                "not possible to apply {} to the data type of channel {}",
                $conv_name, $cn.unique_name,
            ),
        }
    };
    ($cn:expr, $conv_name:literal, opt |$a:ident| $body:expr) => {
        match &mut $cn.data {
            ChannelData::Int8($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of i8 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::UInt8($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of u8 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::Int16($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of i16 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::UInt16($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of u16 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::Int32($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of i32 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::UInt32($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of u32 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::Float32($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of f32 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::Int64($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of i64 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::UInt64($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of u64 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            ChannelData::Float64($a) => {
                if let Some(new_array) =
                    $body.with_context(|| concat!("failed ", $conv_name, " of f64 channel"))?
                {
                    $cn.data = ChannelData::Float64(new_array);
                }
            }
            _ => warn!(
                "not possible to apply {} to the data type of channel {}",
                $conv_name, $cn.unique_name,
            ),
        }
    };
    ($cn:expr, $conv_name:literal, utf8 |$a:ident, $is_float:ident| $body:expr) => {
        match &mut $cn.data {
            ChannelData::Int8($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of i8 channel"))?,
                );
            }
            ChannelData::UInt8($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of u8 channel"))?,
                );
            }
            ChannelData::Int16($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of i16 channel"))?,
                );
            }
            ChannelData::UInt16($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of u16 channel"))?,
                );
            }
            ChannelData::Int32($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of i32 channel"))?,
                );
            }
            ChannelData::UInt32($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of u32 channel"))?,
                );
            }
            ChannelData::Float32($a) => {
                let $is_float = true;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of f32 channel"))?,
                );
            }
            ChannelData::Int64($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of i64 channel"))?,
                );
            }
            ChannelData::UInt64($a) => {
                let $is_float = false;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of u64 channel"))?,
                );
            }
            ChannelData::Float64($a) => {
                let $is_float = true;
                $cn.data = ChannelData::Utf8(
                    $body.with_context(|| concat!($conv_name, " of f64 channel"))?,
                );
            }
            _ => warn!(
                "not possible to apply {} to the data type of channel {}",
                $conv_name, $cn.unique_name,
            ),
        }
    };
}

fn will_noop_conversion_3(cn: &Cn3, sharable: &SharableBlocks3) -> bool {
    if cn.is_converted || cn.data.is_empty() {
        return true;
    }
    let Some((_block, conv)) = sharable.cc.get(&cn.block1.cn_cc_conversion) else {
        return true;
    };
    match conv {
        Conversion::Identity => true,
        Conversion::Linear(cc_val) => {
            cc_val.len() >= 2 && cc_val[0] == 0.0 && abs(cc_val[1] - 1.0) < 1e-12
        }
        Conversion::Exponential(cc_val) | Conversion::Logarithmic(cc_val) => {
            cc_val.len() < 7 || (cc_val[3] != 0.0 && cc_val[0] != 0.0)
        }
        Conversion::Formula(formula) => {
            let parser = fasteval::Parser::new();
            let mut slab = fasteval::Slab::new();
            parser.parse(formula, &mut slab.ps).is_err()
        }
        Conversion::TextTable(cc_val_ref) => cc_val_ref.is_empty(),
        _ => false,
    }
}

/// Converts `cn.data` on the fly without mutating `cn` in place, borrowing `cn.data` when no conversion applies.
pub fn convert_channel_data_cow_3<'a>(
    cn: &'a Cn3,
    sharable: &SharableBlocks3,
) -> Result<Cow<'a, ChannelData>, Error> {
    if will_noop_conversion_3(cn, sharable) {
        return Ok(Cow::Borrowed(&cn.data));
    }
    let mut tmp = Cn3 {
        unique_name: cn.unique_name.clone(),
        block1: cn.block1.clone(),
        data: cn.data.clone(),
        is_converted: false,
        ..Default::default()
    };
    convert_channel(&mut tmp, sharable)?;
    Ok(Cow::Owned(tmp.data))
}

/// Computes the post-conversion MDF4 block signature for an MDF3 channel without converting numeric channels upfront.
pub fn converted_output_signature_3(
    cn: &Cn3,
    sharable: &SharableBlocks3,
    machine_endian: bool,
) -> Result<ConvertedOutputSignature, Error> {
    let raw = &cn.data;
    if raw.is_empty() || will_noop_conversion_3(cn, sharable) {
        return Ok(channel_data_signature(raw, machine_endian));
    }
    let (_block, conv) = sharable
        .cc
        .get(&cn.block1.cn_cc_conversion)
        .expect("checked in will_noop_conversion_3");
    match conv {
        Conversion::Linear(cc_val) => {
            if cc_val.len() < 2 {
                bail!(
                    "malformed linear conversion: expected at least 2 parameters, got {}",
                    cc_val.len()
                );
            }
            numeric_scalar_signature(raw, machine_endian)
        }
        Conversion::Rational(cc_val) => {
            if cc_val.len() < 6 {
                bail!(
                    "malformed rational conversion: expected at least 6 parameters, got {}",
                    cc_val.len()
                );
            }
            numeric_scalar_signature(raw, machine_endian)
        }
        Conversion::Polynomial(cc_val) => {
            if cc_val.len() < 6 {
                bail!(
                    "malformed polynomial conversion: expected at least 6 parameters, got {}",
                    cc_val.len()
                );
            }
            numeric_scalar_signature(raw, machine_endian)
        }
        Conversion::TabularInterpolation(_)
        | Conversion::Tabular(_)
        | Conversion::Formula(_)
        | Conversion::Exponential(_)
        | Conversion::Logarithmic(_) => numeric_scalar_signature(raw, machine_endian),
        Conversion::Identity => Ok(channel_data_signature(raw, machine_endian)),
        Conversion::TextTable(_) | Conversion::TextRangeTable(_) => {
            let converted = convert_channel_data_cow_3(cn, sharable)?;
            Ok(channel_data_signature(&converted, machine_endian))
        }
    }
}

#[inline]
fn numeric_scalar_signature(
    raw: &ChannelData,
    machine_endian: bool,
) -> Result<ConvertedOutputSignature, Error> {
    let is_numeric_scalar = matches!(
        raw,
        ChannelData::Int8(_)
            | ChannelData::UInt8(_)
            | ChannelData::Int16(_)
            | ChannelData::UInt16(_)
            | ChannelData::Int32(_)
            | ChannelData::UInt32(_)
            | ChannelData::Float32(_)
            | ChannelData::Int64(_)
            | ChannelData::UInt64(_)
            | ChannelData::Float64(_)
    );
    if is_numeric_scalar {
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

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_channel(cn: &mut Cn3, sharable: &SharableBlocks3) -> Result<(), Error> {
    if cn.is_converted {
        return Ok(());
    }
    if cn.data.is_empty() {
        cn.is_converted = true;
        return Ok(());
    }

    // Could be empty if only initialised
    if let Some((_block, conv)) = sharable.cc.get(&cn.block1.cn_cc_conversion) {
        match conv {
            Conversion::Linear(cc_val) => linear_conversion(cn, cc_val)
                .with_context(|| format!("linear conversion failed for {}", cn.unique_name))?,
            Conversion::TabularInterpolation(cc_val) => {
                value_to_value_with_interpolation_slice(cn, cc_val).with_context(|| {
                    format!(
                        "value to value with interpolation conversion failed for {}",
                        cn.unique_name
                    )
                })?
            }
            Conversion::Tabular(cc_val) => value_to_value_without_interpolation_slice(cn, cc_val)
                .with_context(|| {
                format!(
                    "value to value without interpolation conversion failed for {}",
                    cn.unique_name
                )
            })?,
            Conversion::Rational(cc_val) => rational_conversion(cn, cc_val)
                .with_context(|| format!("rational conversion failed for {}", cn.unique_name))?,
            Conversion::Formula(formula) => algebraic_conversion(cn, formula)
                .with_context(|| format!("algebraic conversion failed for {}", cn.unique_name))?,
            Conversion::Identity => {}
            Conversion::Polynomial(cc_val) => polynomial_conversion(cn, cc_val)
                .with_context(|| format!("polynomial conversion failed for {}", cn.unique_name))?,
            Conversion::Exponential(cc_val) => exponential_conversion(cn, cc_val)
                .with_context(|| format!("exponential conversion failed for {}", cn.unique_name))?,
            Conversion::Logarithmic(cc_val) => logarithmic_conversion(cn, cc_val)
                .with_context(|| format!("logarithmic conversion failed for {}", cn.unique_name))?,
            Conversion::TextTable(cc_val_ref) => {
                if !cc_val_ref.is_empty() {
                    value_to_text(cn, cc_val_ref).with_context(|| {
                        format!("value to text conversion failed for {}", cn.unique_name)
                    })?
                }
            }
            Conversion::TextRangeTable(cc_val_ref) => value_range_to_text(cn, cc_val_ref)
                .with_context(|| {
                    format!("text range table conversion failed for {}", cn.unique_name)
                })?,
        }
        // as data are converted, CCBlock is not anymore applicable
        cn.block1.cn_cc_conversion = 0;
        cn.is_converted = true;
    }
    Ok(())
}

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg3, sharable: &SharableBlocks3) -> Result<(), Error> {
    dg.cg
        .par_iter_mut()
        .try_for_each(|(_, channel_group)| -> Result<(), Error> {
            channel_group.cn.par_iter_mut().try_for_each(
                |(_rec_pos, cn): (&u32, &mut Cn3)| -> Result<(), Error> {
                    convert_channel(cn, sharable)
                },
            )?;
            Ok(())
        })?;
    Ok(())
}

/// Apply linear conversion to get physical data
pub fn linear_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    if cc_val.len() < 2 {
        bail!(
            "malformed linear conversion: expected at least 2 parameters, got {}",
            cc_val.len()
        );
    }
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && abs(p2 - 1.0) < 1e-12) {
        apply_scalar_cn3!(cn, "linear conversion", |a| linear_calculation(a, p1, p2));
    }
    Ok(())
}

/// Apply rational conversion to get physical data
pub fn rational_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    if cc_val.len() < 6 {
        bail!(
            "malformed rational conversion: expected at least 6 parameters, got {}",
            cc_val.len()
        );
    }
    apply_scalar_cn3!(cn, "rational conversion", |a| rational_calculation(
        a, cc_val
    ));
    Ok(())
}

/// Generic function calculating polynomial conversion
#[inline]
fn polynomial_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    if cc_val.len() < 6 {
        bail!(
            "malformed polynomial conversion: expected at least 6 parameters, got {}",
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
            (p2 - (p4 * (x - p5 - p6))) / (p3 * (x - p5 - p6) - p1)
        })
        .collect();
    Ok(PrimitiveArray::new(
        converted.into(),
        array.nulls().cloned(),
    ))
}

/// Apply polynomial conversion to get physical data
pub fn polynomial_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    if cc_val.len() < 6 {
        bail!(
            "malformed polynomial conversion: expected at least 6 parameters, got {}",
            cc_val.len()
        );
    }
    apply_scalar_cn3!(cn, "polynomial conversion", |a| polynomial_calculation(
        a, cc_val
    ));
    Ok(())
}

/// Generic function calculating exponential conversion
#[inline]
fn exponential_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
) -> Result<Option<PrimitiveArray<Float64Type>>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    if cc_val.len() < 7 {
        return Ok(None);
    }
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];
    let values = array.values();
    if p4 == 0.0 {
        let converted: Vec<f64> = values
            .iter()
            .map(|v| {
                let x: f64 = (*v).as_();
                (((x - p7) * p6 - p3) / p1).exp() / p2
            })
            .collect();
        Ok(Some(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        )))
    } else if p1 == 0.0 {
        let converted: Vec<f64> = values
            .iter()
            .map(|v| {
                let x: f64 = (*v).as_();
                ((p3 / (x - p7) - p6) / p4).exp() / p5
            })
            .collect();
        Ok(Some(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        )))
    } else {
        Ok(None)
    }
}

/// Apply exponential conversion to get physical data
pub fn exponential_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    apply_scalar_cn3!(
        cn,
        "exponential conversion",
        opt | a | exponential_calculation(a, cc_val)
    );
    Ok(())
}

/// Generic function calculating value logarithmic conversion
#[inline]
fn logarithmic_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: &[f64],
) -> Result<Option<PrimitiveArray<Float64Type>>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    if cc_val.len() < 7 {
        return Ok(None);
    }
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];
    let values = array.values();
    if p4 == 0.0 {
        let converted: Vec<f64> = values
            .iter()
            .map(|v| {
                let x: f64 = (*v).as_();
                (((x - p7) * p6 - p3) / p1).ln() / p2
            })
            .collect();
        Ok(Some(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        )))
    } else if p1 == 0.0 {
        let converted: Vec<f64> = values
            .iter()
            .map(|v| {
                let x: f64 = (*v).as_();
                ((p3 / (x - p7) - p6) / p4).ln() / p5
            })
            .collect();
        Ok(Some(PrimitiveArray::new(
            converted.into(),
            array.nulls().cloned(),
        )))
    } else {
        Ok(None)
    }
}

/// Apply logarithmic conversion to get physical data
pub fn logarithmic_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    apply_scalar_cn3!(
        cn,
        "logarithmic conversion",
        opt | a | logarithmic_calculation(a, cc_val)
    );
    Ok(())
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn3, formulae: &str) -> Result<(), Error> {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let compiled_instruction = parser.parse(formulae, &mut slab.ps);
    if let Ok(compiled_instruct) = compiled_instruction {
        let compiled = compiled_instruct
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs);
        apply_scalar_cn3!(cn, "algebraic conversion", |a| {
            alegbraic_conversion_primitive(&compiled, &slab, a)
        });
    } else if let Err(error_message) = compiled_instruction {
        // could not parse the formulae, probably some function or syntax not yet implementated by fasteval
        warn!(
            "{}\n Could not parse formulae {} for channel {}",
            error_message, formulae, cn.unique_name
        );
    }
    Ok(())
}

/// Generic function calculating value to value with interpolation conversion
#[inline]
pub fn value_to_value_with_interpolation_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: Vec<f64>,
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    value_to_value_with_interpolation_primitive(array, &val)
}

fn value_to_value_with_interpolation_slice(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    apply_scalar_cn3!(cn, "value to value with interpolation conversion", |a| {
        value_to_value_with_interpolation_primitive(a, &val)
    });
    Ok(())
}

/// Apply value to value with interpolation conversion to get physical data
pub fn value_to_value_with_interpolation(cn: &mut Cn3, cc_val: Vec<f64>) -> Result<(), Error> {
    value_to_value_with_interpolation_slice(cn, &cc_val)
}

/// Generic function calculating value to value without interpolation conversion
#[inline]
pub fn value_to_value_without_interpolation_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val: Vec<f64>,
) -> Result<PrimitiveArray<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    value_to_value_without_interpolation_primitive(array, &val)
}

fn value_to_value_without_interpolation_slice(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    apply_scalar_cn3!(cn, "value to value without interpolation conversion", |a| {
        value_to_value_without_interpolation_primitive(a, &val)
    });
    Ok(())
}

/// Apply value to value without interpolation conversion to get physical data
pub fn value_to_value_without_interpolation(cn: &mut Cn3, cc_val: Vec<f64>) -> Result<(), Error> {
    value_to_value_without_interpolation_slice(cn, &cc_val)
}

/// Generic function calculating value to text expression
#[inline]
pub fn value_to_text_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val_ref: &[(f64, String)],
) -> Result<LargeStringArray, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let table: FxHashMap<i64, &str> = cc_val_ref
        .iter()
        .map(|(val, s)| (val.round() as i64, s.as_str()))
        .collect();
    let default_val = "";
    let mut new_array = LargeStringBuilder::with_capacity(array.len(), 32);
    for (i, val) in array.values().iter().enumerate() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
        let val_f64: f64 = (*val).as_();
        let val_i64 = val_f64.round() as i64;
        if let Some(&txt) = table.get(&val_i64) {
            new_array.append_value(txt);
        } else {
            new_array.append_value(default_val);
        }
    }
    Ok(new_array.finish())
}

/// Apply value to text or scale conversion to get physical data
pub fn value_to_text(cn: &mut Cn3, cc_val_ref: &[(f64, String)]) -> Result<(), Error> {
    apply_scalar_cn3!(
        cn,
        "value to text conversion",
        utf8 | a,
        _is_float | { value_to_text_calculation(a, cc_val_ref) }
    );
    Ok(())
}

/// Generic function calculating value range to text expression
#[inline]
pub fn value_range_to_text_calculation<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
    inclusive_upper: bool,
) -> Result<LargeStringArray, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let mut new_array = LargeStringBuilder::with_capacity(array.len(), 32);
    for (i, val) in array.values().iter().enumerate() {
        if array.is_null(i) {
            new_array.append_null();
            continue;
        }
        let a: f64 = (*val).as_();
        let idx = cc_val_ref.0.partition_point(|x| x.0 <= a);
        let matched = if idx > 0 {
            let candidate = &cc_val_ref.0[idx - 1];
            let in_range = if inclusive_upper {
                a <= candidate.1
            } else {
                a < candidate.1 || (candidate.0 == candidate.1 && a == candidate.0)
            };
            if in_range { Some(candidate) } else { None }
        } else {
            None
        };
        if let Some(key) = matched {
            new_array.append_value(&key.2);
        } else {
            new_array.append_value(&cc_val_ref.1);
        }
    }
    Ok(new_array.finish())
}

/// Apply value range to text or scale conversion to get physical data
pub fn value_range_to_text(
    cn: &mut Cn3,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
) -> Result<(), Error> {
    apply_scalar_cn3!(
        cn,
        "value range to text conversion",
        utf8 | a,
        is_float | { value_range_to_text_calculation(a, cc_val_ref, !is_float) }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mdfinfo::mdfinfo3::Cc3Block;
    use arrow::array::UInt16Array;

    #[test]
    fn truncated_linear_and_polynomial_return_err() {
        let mut cn = Cn3 {
            data: ChannelData::UInt16(UInt16Array::from(vec![1u16, 2u16])),
            ..Default::default()
        };
        assert!(linear_conversion(&mut cn, &[1.0]).is_err());
        assert!(polynomial_conversion(&mut cn, &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn unparseable_formula3_signature_matches_raw_type() {
        let mut sharable = SharableBlocks3::default();
        sharable.cc.insert(
            10,
            (Cc3Block::default(), Conversion::Formula("bad((expr".into())),
        );
        let mut cn = Cn3 {
            data: ChannelData::UInt16(UInt16Array::from(vec![5u16, 6u16])),
            ..Default::default()
        };
        cn.block1.cn_cc_conversion = 10;
        let sig = converted_output_signature_3(&cn, &sharable, false).unwrap();
        assert_eq!(sig.bit_count, 16);
        assert_eq!(sig.byte_count, 2);
    }
}
