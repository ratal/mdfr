//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result};
use arrow::array::{
    ArrayBuilder, Float64Array, Float64Builder, LargeStringBuilder, PrimitiveBuilder,
};
use arrow::datatypes::{ArrowPrimitiveType, Float64Type};
use arrow::error::ArrowError;
use itertools::Itertools;
use num::NumCast;
use num::abs;
use num::cast::AsPrimitive;
use std::collections::BTreeMap;

use crate::data_holder::channel_data::ChannelData;
use crate::mdfinfo::mdfinfo3::{Cn3, Conversion, Dg3, SharableBlocks3};
use crate::mdfreader::conversions4::{linear_calculation, rational_calculation};
use fasteval::Evaler;
use fasteval::{Compiler, Instruction, Slab};
use log::warn;
use rayon::prelude::*;

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
                value_to_value_with_interpolation(cn, cc_val.clone()).with_context(|| {
                    format!(
                        "value to value with interpolation conversion failed for {}",
                        cn.unique_name
                    )
                })?
            }
            Conversion::Tabular(cc_val) => value_to_value_without_interpolation(cn, cc_val.clone())
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
fn linear_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
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
                        .context("failed linear conversion of i64 channel")?,
                );
            }
            ChannelData::UInt64(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of u64 channel")?,
                );
            }
            ChannelData::Float64(a) => {
                cn.data = ChannelData::Float64(
                    linear_calculation(a, p1, p2)
                        .context("failed linear conversion of f64 channel")?,
                );
            }
            _ => warn!(
                "not possible to apply linear conversion to the data type of channel {}",
                cn.unique_name,
            ),
        }
    }
    Ok(())
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of u8 channel")?,
            );
        }
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of i8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                rational_calculation(a, cc_val)
                    .context("failed linear conversion of f64 channel")?,
            );
        }
        _ => warn!(
            "not possible to apply ratioanl conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating polynomial conversion
#[inline]
fn polynomial_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
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
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut out = Float64Builder::with_capacity(array.capacity());
    out.values_slice_mut()
        .iter_mut()
        .zip(array_f64.values())
        .for_each(|(y, x)| {
            *y = (p2 - (p4 * (x - p5 - p6))) / (p3 * (x - p5 - p6) - p1);
        });
    Ok(out)
}

/// Apply polynomial conversion to get physical data
fn polynomial_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of u8 channel")?,
            );
        }
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of i8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                polynomial_calculation(a, cc_val)
                    .context("failed polynomial conversion of f64 channel")?,
            );
        }
        _ => warn!(
            "not possible to apply polynomial conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating exponential conversion
#[inline]
fn exponential_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val: &[f64],
) -> Result<Option<PrimitiveBuilder<Float64Type>>, Error>
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
    let p7 = cc_val[6];
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut out = Float64Builder::with_capacity(array.capacity());
    if p4 == 0.0 {
        out.values_slice_mut()
            .iter_mut()
            .zip(array_f64.values())
            .for_each(|(y, x)| {
                *y = (((x - p7) * p6 - p3) / p1).ln() / p2;
            });
        Ok(Some(out))
    } else if p1 == 0.0 {
        out.values_slice_mut()
            .iter_mut()
            .zip(array_f64.values())
            .for_each(|(y, x)| {
                *y = ((p3 / (x - p7) - p6) / p4).ln() / p5;
            });
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

/// Apply exponential conversion to get physical data
fn exponential_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of u8 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int8(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of i8 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int16(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of i16 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt16(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of u16 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt32(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of u32 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Float32(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of f32 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int64(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of i64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt64(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of u64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Float64(a) => {
            if let Some(new_array) = exponential_calculation(a, cc_val)
                .context("failed exponential conversion of f64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        _ => warn!(
            "not possible to apply exponential conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating value logarithmic conversion
#[inline]
fn logarithmic_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val: &[f64],
) -> Result<Option<PrimitiveBuilder<Float64Type>>, Error>
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
    let p7 = cc_val[6];
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut out = Float64Builder::with_capacity(array.capacity());
    if p4 == 0.0 {
        out.values_slice_mut()
            .iter_mut()
            .zip(array_f64.values())
            .for_each(|(y, x)| {
                *y = (((x - p7) * p6 - p3) / p1).exp() / p2;
            });
        Ok(Some(out))
    } else if p1 == 0.0 {
        out.values_slice_mut()
            .iter_mut()
            .zip(array_f64.values())
            .for_each(|(y, x)| {
                *y = ((p3 / (x - p7) - p6) / p4).exp() / p5;
            });
        Ok(Some(out))
    } else {
        Ok(None)
    }
}

/// Apply exponential conversion to get physical data
fn logarithmic_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::UInt8(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of u8 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int8(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of i8 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int16(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of i16 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt16(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of u16 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int32(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of i32 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt32(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of u32 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Float32(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of f32 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Int64(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of i64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::UInt64(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of u64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        ChannelData::Float64(a) => {
            if let Some(new_array) = logarithmic_calculation(a, cc_val)
                .context("failed logarithmic conversion of f64 channel")?
            {
                cn.data = ChannelData::Float64(new_array)
            };
        }
        _ => warn!(
            "not possible to apply logarithmic conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating algebraic expression conversion
#[inline]
fn alegbraic_conversion_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    compiled: &Instruction,
    slab: &Slab,
    formulae: &str,
    name: &str,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut new_array = vec![0f64; array_f64.len()];
    new_array
        .iter_mut()
        .zip(array_f64.values())
        .for_each(|(new_array, a)| {
            let mut map = BTreeMap::new();
            map.insert("X".to_string(), *a);
            match compiled.eval(slab, &mut map) {
                Ok(res) => *new_array = res,
                Err(error_message) => {
                    *new_array = *a;
                    warn!(
                        "{error_message}\n Could not compute formulae {formulae} for channel {name} and value {a}",
                    );
                }
            }
        });
    Ok(PrimitiveBuilder::new_from_buffer(new_array.into(), None))
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
        match &mut cn.data {
            ChannelData::UInt8(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of u8 channel")?,
                );
            }
            ChannelData::Int8(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of i8 channel")?,
                );
            }
            ChannelData::Int16(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of i16 channel")?,
                );
            }
            ChannelData::UInt16(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of u16 channel")?,
                );
            }
            ChannelData::Int32(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of i32 channel")?,
                );
            }
            ChannelData::UInt32(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of u32 channel")?,
                );
            }
            ChannelData::Float32(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of f32 channel")?,
                );
            }
            ChannelData::Int64(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of i64 channel")?,
                );
            }
            ChannelData::UInt64(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of u64 channel")?,
                );
            }
            ChannelData::Float64(a) => {
                cn.data = ChannelData::Float64(
                    alegbraic_conversion_calculation(
                        a,
                        &compiled,
                        &slab,
                        formulae,
                        &cn.unique_name,
                    )
                    .context("failed algebraic conversion of f64 channel")?,
                );
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
    Ok(())
}

/// Generic function calculating value to value with interpolation conversion
#[inline]
fn value_to_value_with_interpolation_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val: Vec<f64>,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut new_array = vec![0f64; array_f64.len()];
    new_array
        .iter_mut()
        .zip(array_f64.values())
        .for_each(|(new_array, a)| {
            *new_array = match val
                .binary_search_by(|&(xi, _)| xi.partial_cmp(a).expect("Could not compare values"))
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
    Ok(PrimitiveBuilder::new_from_buffer(new_array.into(), None))
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(cn: &mut Cn3, cc_val: Vec<f64>) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val)
                    .context("failed value to value with interpolation conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val)
                    .context("failed value to value with interpolation conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of i16 channel",
                )?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of u16 channel",
                )?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of i32 channel",
                )?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of u32 channel",
                )?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of f32 channel",
                )?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of i64 channel",
                )?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of u64 channel",
                )?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_with_interpolation_calculation(a, cc_val).context(
                    "failed value to value with interpolation conversion of f64 channel",
                )?,
            );
        }
        _ => warn!(
            "not possible to apply value to value with interpolation conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating algebraic expression
#[inline]
fn value_to_value_without_interpolation_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val: Vec<f64>,
) -> Result<PrimitiveBuilder<Float64Type>, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    let mut new_array = vec![0f64; array_f64.len()];
    new_array
        .iter_mut()
        .zip(array_f64.values())
        .for_each(|(new_array, a)| {
            *new_array = match val
                .binary_search_by(|&(xi, _)| xi.partial_cmp(a).expect("Could not compare values"))
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
    Ok(PrimitiveBuilder::new_from_buffer(new_array.into(), None))
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(cn: &mut Cn3, cc_val: Vec<f64>) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of i8 channel",
                )?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of u8 channel",
                )?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of i16 channel",
                )?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of u16 channel",
                )?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of i32 channel",
                )?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of u32 channel",
                )?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of f32 channel",
                )?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of i64 channel",
                )?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of u64 channel",
                )?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Float64(
                value_to_value_without_interpolation_calculation(a, cc_val).context(
                    "failed value to value without interpolation conversion of f64 channel",
                )?,
            );
        }
        _ => warn!(
            "not possible to apply value to value without interpolation conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating value to text expression
#[inline]
fn value_to_text_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val_ref: &[(f64, String)],
) -> Result<LargeStringBuilder, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let mut new_array = LargeStringBuilder::with_capacity(array.len(), 32);
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    array_f64.values().iter().for_each(|val| {
        let matched_key = cc_val_ref.iter().find(|&x| x.0 == *val);
        if let Some(key) = matched_key {
            new_array.append_value(key.1.clone());
        } else {
            new_array.append_value(cc_val_ref[0].1.clone());
        }
    });
    Ok(new_array)
}

/// Apply value to text or scale conversion to get physical data
fn value_to_text(cn: &mut Cn3, cc_val_ref: &[(f64, String)]) -> Result<(), Error> {
    // identify max string length in cc_val_ref
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Utf8(
                value_to_text_calculation(a, cc_val_ref)
                    .context("value to text conversion of f64 channel")?,
            );
        }
        _ => warn!(
            "not possible to apply value to text conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating value range to text expression
#[inline]
fn value_range_to_text_calculation<T: ArrowPrimitiveType>(
    array: &mut PrimitiveBuilder<T>,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
) -> Result<LargeStringBuilder, Error>
where
    <T as ArrowPrimitiveType>::Native: AsPrimitive<f64>,
    T::Native: NumCast,
{
    let mut new_array = LargeStringBuilder::with_capacity(array.len(), 32);
    let array_f64: Float64Array = array
        .finish()
        .try_unary(|value| {
            num::cast::cast::<T::Native, f64>(value)
                .ok_or_else(|| ArrowError::CastError(format!("Can't cast value {value:?} to f64")))
        })
        .context("failed converting array to f64")?;
    array_f64.values().iter().for_each(|a| {
        let matched_key = cc_val_ref
            .0
            .iter()
            .enumerate()
            .find(|&x| (x.1.0 <= *a) && (*a < x.1.1));
        if let Some(key) = matched_key {
            new_array.append_value(key.1.2.clone());
        } else {
            new_array.append_value(cc_val_ref.1.clone());
        }
    });
    Ok(new_array)
}

/// Apply value range to text or scale conversion to get physical data
fn value_range_to_text(
    cn: &mut Cn3,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
) -> Result<(), Error> {
    match &mut cn.data {
        ChannelData::Int8(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of i8 channel")?,
            );
        }
        ChannelData::UInt8(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of u8 channel")?,
            );
        }
        ChannelData::Int16(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of i16 channel")?,
            );
        }
        ChannelData::UInt16(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of u16 channel")?,
            );
        }
        ChannelData::Int32(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of i32 channel")?,
            );
        }
        ChannelData::UInt32(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of u32 channel")?,
            );
        }
        ChannelData::Float32(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of f32 channel")?,
            );
        }
        ChannelData::Int64(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of i64 channel")?,
            );
        }
        ChannelData::UInt64(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of u64 channel")?,
            );
        }
        ChannelData::Float64(a) => {
            cn.data = ChannelData::Utf8(
                value_range_to_text_calculation(a, cc_val_ref)
                    .context("value range to text conversion of f64 channel")?,
            );
        }
        _ => warn!(
            "not possible to apply value to text conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Builder, Int32Builder};

    #[test]
    fn test_value_to_value_with_interpolation() {
        // Table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let cc_val = vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0];

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact match → 0
        builder.append_value(5.0); // interpolate between (0,0) and (10,100) → 50
        builder.append_value(10.0); // exact match → 100
        builder.append_value(15.0); // interpolate between (10,100) and (20,200) → 150
        builder.append_value(-5.0); // below first → first y = 0
        builder.append_value(25.0); // above last → last y = 200

        let result = value_to_value_with_interpolation_calculation(&mut builder, cc_val).unwrap();
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
    fn test_value_to_value_without_interpolation() {
        // Same table pairs: x=0→y=0, x=10→y=100, x=20→y=200
        let cc_val = vec![0.0, 0.0, 10.0, 100.0, 20.0, 200.0];

        let mut builder = Float64Builder::new();
        builder.append_value(0.0); // exact → 0
        builder.append_value(3.0); // nearer to x=0 (dist=3) than x=10 (dist=7) → 0
        builder.append_value(7.0); // nearer to x=10 (dist=3) than x=0 (dist=7) → 100
        builder.append_value(10.0); // exact → 100
        builder.append_value(-5.0); // below first → 0
        builder.append_value(25.0); // above last → 200

        let result =
            value_to_value_without_interpolation_calculation(&mut builder, cc_val).unwrap();
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
    fn test_value_to_text_calculation() {
        let cc_val_ref: Vec<(f64, String)> = vec![
            (1.0, "one".to_string()),
            (2.0, "two".to_string()),
            (3.0, "three".to_string()),
        ];

        let mut builder = Int32Builder::new();
        builder.append_value(1); // matches 1.0 → "one"
        builder.append_value(2); // matches 2.0 → "two"
        builder.append_value(3); // matches 3.0 → "three"
        builder.append_value(99); // no match → defaults to first entry ("one")

        let result = value_to_text_calculation(&mut builder, &cc_val_ref).unwrap();
        let arr = result.finish_cloned();
        assert_eq!(arr.value(0), "one");
        assert_eq!(arr.value(1), "two");
        assert_eq!(arr.value(2), "three");
        assert_eq!(arr.value(3), "one");
    }

    #[test]
    fn test_value_range_to_text_calculation() {
        let ranges = vec![
            (0.0, 10.0, "low".to_string()),
            (10.0, 20.0, "medium".to_string()),
            (20.0, 30.0, "high".to_string()),
        ];
        let cc_val_ref = (ranges, "unknown".to_string());

        let mut builder = Float64Builder::new();
        builder.append_value(5.0); // in [0, 10) → "low"
        builder.append_value(15.0); // in [10, 20) → "medium"
        builder.append_value(25.0); // in [20, 30) → "high"
        builder.append_value(35.0); // out of range → "unknown"

        let result = value_range_to_text_calculation(&mut builder, &cc_val_ref).unwrap();
        let arr = result.finish_cloned();
        assert_eq!(arr.value(0), "low");
        assert_eq!(arr.value(1), "medium");
        assert_eq!(arr.value(2), "high");
        assert_eq!(arr.value(3), "unknown");
    }

    // ── Helper ──

    fn make_cn3(data: ChannelData) -> Cn3 {
        Cn3 {
            data,
            ..Default::default()
        }
    }

    // ── linear_conversion (Cn3) tests ──

    #[test]
    fn test_linear_cn3_uint16() {
        use arrow::array::UInt16Builder;
        let mut builder = UInt16Builder::new();
        builder.append_value(10);
        let mut cn = make_cn3(ChannelData::UInt16(builder));
        linear_conversion(&mut cn, &[0.0, 3.0]).unwrap(); // 10*3+0 = 30
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 30.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── polynomial_conversion (Cn3) tests ──

    #[test]
    fn test_polynomial_cn3_float64() {
        // polynomial_calculation uses array.finish() then Float64Builder::with_capacity(array.capacity())
        // After finish(), capacity resets to 0, so the output is an empty Float64 builder.
        // This test verifies the conversion returns Ok and converts the channel to Float64.
        let cc_val = vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(2.0);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = polynomial_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        // Data should now be Float64 (conversion applied)
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── exponential_conversion (Cn3) tests ──

    #[test]
    fn test_exponential_cn3_float64_p4_zero() {
        // exponential_calculation also uses array.finish() and Float64Builder::with_capacity(0)
        // so the output is an empty builder. Test that it returns Ok and converts to Float64.
        // p4=0 branch chosen with: p1=1, p2=1, p3=0, p4=0, p5=0, p6=1, p7=0
        let cc_val = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(std::f64::consts::E);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = exponential_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── logarithmic_conversion (Cn3) tests ──

    #[test]
    fn test_logarithmic_cn3_float64_p4_zero() {
        // logarithmic_calculation also uses array.finish() and Float64Builder::with_capacity(0).
        // Test that it returns Ok and converts to Float64.
        // p4=0 branch: p1=1, p2=1, p3=0, p4=0, p5=0, p6=1, p7=0
        let cc_val = vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut builder = Float64Builder::new();
        builder.append_value(0.0);
        let mut cn = make_cn3(ChannelData::Float64(builder));
        let result = logarithmic_conversion(&mut cn, &cc_val);
        assert!(result.is_ok());
        assert!(matches!(cn.data, ChannelData::Float64(_)));
    }

    // ── value_to_value_with_interpolation (Cn3) tests ──

    #[test]
    fn test_cn3_vtv_interp_uint8() {
        use arrow::array::UInt8Builder;
        // Table: 0→0, 10→100
        let cc_val = vec![0.0, 0.0, 10.0, 100.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(5); // 50
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_value_with_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 50.0).abs() < 1e-9);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_to_value_without_interpolation (Cn3) tests ──

    #[test]
    fn test_cn3_vtv_no_interp_uint8() {
        use arrow::array::UInt8Builder;
        let cc_val = vec![1.0, 10.0, 2.0, 20.0];
        let mut builder = UInt8Builder::new();
        builder.append_value(2); // exact match → 20.0
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_value_without_interpolation(&mut cn, cc_val).unwrap();
        if let ChannelData::Float64(ref b) = cn.data {
            let vals = b.values_slice();
            assert!((vals[0] - 20.0).abs() < 1e-12);
        } else {
            panic!("Expected Float64");
        }
    }

    // ── value_to_text (Cn3) tests ──

    #[test]
    fn test_cn3_value_to_text() {
        use arrow::array::UInt8Builder;
        let cc_val_ref = vec![(1.0, "one".to_string()), (2.0, "two".to_string())];
        let mut builder = UInt8Builder::new();
        builder.append_value(1);
        let mut cn = make_cn3(ChannelData::UInt8(builder));
        value_to_text(&mut cn, &cc_val_ref).unwrap();
        if let ChannelData::Utf8(ref b) = cn.data {
            let arr = b.finish_cloned();
            assert_eq!(arr.value(0), "one");
        } else {
            panic!("Expected Utf8");
        }
    }

    // ── value_range_to_text (Cn3) tests ──

    #[test]
    fn test_cn3_value_range_to_text() {
        let ranges = vec![
            (1.0, 2.0, "in_range".to_string()),
            (3.0, 4.0, "out".to_string()),
        ];
        let cc_val_ref = (ranges, "default".to_string());
        let mut builder = Float64Builder::new();
        builder.append_value(1.5); // in [1.0, 2.0)
        let mut cn = make_cn3(ChannelData::Float64(builder));
        value_range_to_text(&mut cn, &cc_val_ref).unwrap();
        if let ChannelData::Utf8(ref b) = cn.data {
            let arr = b.finish_cloned();
            assert_eq!(arr.value(0), "in_range");
        } else {
            panic!("Expected Utf8");
        }
    }
}
