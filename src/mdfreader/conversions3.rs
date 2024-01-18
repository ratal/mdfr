//! this modules implements functions to convert arrays into physical arrays using CCBlock
use anyhow::{Context, Error, Result};
use arrow2::array::{Array, MutableArray, MutableUtf8ValuesArray, PrimitiveArray};
use arrow2::compute::arity_assign;
use arrow2::compute::cast::primitive_as_primitive;
use arrow2::datatypes::DataType;
use arrow2::types::NativeType;
use itertools::Itertools;
use num_traits::cast::AsPrimitive;
use std::collections::BTreeMap;
use std::fmt::Display;

use crate::mdfinfo::mdfinfo3::{Cn3, Conversion, Dg3, SharableBlocks3};
use fasteval::Evaler;
use fasteval::{Compiler, Instruction, Slab};
use log::warn;
use rayon::prelude::*;

/// convert all channel arrays into physical values as required by CCBlock content
pub fn convert_all_channels(dg: &mut Dg3, sharable: &SharableBlocks3) -> Result<(), Error> {
    for channel_group in dg.cg.values_mut() {
        let cycle_count = channel_group.block.cg_cycle_count;
        channel_group
            .cn
            .par_iter_mut()
            .filter(|(_cn_record_position, cn)| !cn.data.is_empty())
            .try_for_each(|(_rec_pos, cn): (&u32, &mut Cn3)| -> Result<(), Error> {
                // Could be empty if only initialised
                if let Some((_block, conv)) = sharable.cc.get(&cn.block1.cn_cc_conversion) {
                    match conv {
                        Conversion::Linear(cc_val) => linear_conversion(cn, cc_val)
                            .with_context(|| {
                            format!("linear conversion failed for {}", cn.unique_name)
                        })?,
                        Conversion::TabularInterpolation(cc_val) => {
                            value_to_value_with_interpolation(cn, cc_val.clone(), &cycle_count).with_context(|| {
                                format!("value to value with interpolation conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Tabular(cc_val) => {
                            value_to_value_without_interpolation(cn, cc_val.clone(), &cycle_count).with_context(|| {
                                format!("value to value without interpolation conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Rational(cc_val) => {
                            rational_conversion(cn, cc_val).with_context(|| {
                                format!("rational conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Formula(formula) => {
                            algebraic_conversion(cn, formula, &cycle_count).with_context(|| {
                                format!("algebraic conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Identity => {}
                        Conversion::Polynomial(cc_val) => {
                            polynomial_conversion(cn, cc_val).with_context(|| {
                                format!("polynomial conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Exponential(cc_val) => {
                            exponential_conversion(cn, cc_val).with_context(|| {
                                format!("exponential conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::Logarithmic(cc_val) => {
                            logarithmic_conversion(cn, cc_val).with_context(|| {
                                format!("logarithmic conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::TextTable(cc_val_ref) => {
                            value_to_text(cn, cc_val_ref, &cycle_count).with_context(|| {
                                format!("value to text conversion failed for {}", cn.unique_name)
                            })?
                        }
                        Conversion::TextRangeTable(cc_val_ref) => {
                            value_range_to_text(cn, cc_val_ref, &cycle_count).with_context(|| {
                                format!("text range table conversion failed for {}", cn.unique_name)
                            })?
                        }
                    }
                }
                Ok(())
            })?
    }
    Ok(())
}

/// Generic function calculating exponential conversion
#[inline]
fn linear_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    p1: f64,
    p2: f64,
) -> Result<Box<dyn Array>, Error> {
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("Linear conversion could not downcast to primitive array")?;
    let mut array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    arity_assign::unary(&mut array_f64, |x| x * p2 + p1);
    Ok(array_f64.to_boxed())
}

/// Apply linear conversion to get physical data
fn linear_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    if !(p1 == 0.0 && num::abs(p2 - 1.0) < 1e-12) {
        match cn.data.data_type() {
            DataType::UInt8 => {
                cn.data = linear_calculation::<u8>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of u8 channel")?;
            }
            DataType::UInt16 => {
                cn.data = linear_calculation::<u16>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of u16 channel")?;
            }
            DataType::UInt32 => {
                cn.data = linear_calculation::<u32>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of u32 channel")?;
            }
            DataType::UInt64 => {
                cn.data = linear_calculation::<u64>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of u64 channel")?;
            }
            DataType::Int8 => {
                cn.data = linear_calculation::<i8>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of i8 channel")?;
            }
            DataType::Int16 => {
                cn.data = linear_calculation::<i16>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of i16 channel")?;
            }
            DataType::Int32 => {
                cn.data = linear_calculation::<i32>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of i32 channel")?;
            }
            DataType::Int64 => {
                cn.data = linear_calculation::<i64>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of i64 channel")?;
            }
            DataType::Float16 => {
                cn.data = linear_calculation::<f32>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of f16 channel")?;
            }
            DataType::Float32 => {
                cn.data = linear_calculation::<f32>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of f32 channel")?;
            }
            DataType::Float64 => {
                cn.data = linear_calculation::<f64>(cn.data.as_ref(), p1, p2)
                    .context("failed linear conversion of f64 channel")?;
            }
            _ => warn!(
                "not possible to apply linear conversion to the data type of channel {}",
                cn.unique_name,
            ),
        }
    }
    Ok(())
}

/// Generic function calculating rational conversion
#[inline]
fn rational_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: &[f64],
) -> Result<Box<dyn Array>, Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let mut array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    arity_assign::unary(&mut array_f64, |x| {
        (x * x * p1 + x * p2 + p3) / (x * x * p4 + x * p5 + p6)
    });
    Ok(array_f64.to_boxed())
}

/// Apply rational conversion to get physical data
fn rational_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = rational_calculation::<u8>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = rational_calculation::<u16>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = rational_calculation::<u32>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = rational_calculation::<u64>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = rational_calculation::<i8>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = rational_calculation::<i16>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = rational_calculation::<i32>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = rational_calculation::<i64>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = rational_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = rational_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = rational_calculation::<f64>(cn.data.as_ref(), cc_val)
                .context("failed rational conversion of f64 channel")?;
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
fn polynomial_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: &[f64],
) -> Result<Box<dyn Array>, Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let mut array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    arity_assign::unary(&mut array_f64, |x| {
        (p2 - (p4 * (x - p5 - p6))) / (p3 * (x - p5 - p6) - p1)
    });
    Ok(array_f64.to_boxed())
}

/// Apply polynomial conversion to get physical data
fn polynomial_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = polynomial_calculation::<u8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = polynomial_calculation::<u16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = polynomial_calculation::<u32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = polynomial_calculation::<u64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = polynomial_calculation::<i8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = polynomial_calculation::<i16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = polynomial_calculation::<i32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = polynomial_calculation::<i64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = polynomial_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = polynomial_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = polynomial_calculation::<f64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f64 channel")?;
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
fn exponential_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: &[f64],
) -> Result<Option<Box<dyn Array>>, Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let mut new_array = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    if p4 == 0.0 {
        arity_assign::unary(&mut new_array, |x| (((x - p7) * p6 - p3) / p1).ln() / p2);
        Ok(Some(new_array.boxed()))
    } else if p1 == 0.0 {
        arity_assign::unary(&mut new_array, |x| ((p3 / (x - p7) - p6) / p4).ln() / p5);
        Ok(Some(new_array.boxed()))
    } else {
        Ok(None)
    }
}

/// Apply exponential conversion to get physical data
fn exponential_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            if let Some(new_array) = exponential_calculation::<u8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u8 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt16 => {
            if let Some(new_array) = exponential_calculation::<u16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt32 => {
            if let Some(new_array) = exponential_calculation::<u32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt64 => {
            if let Some(new_array) = exponential_calculation::<u64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u64 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int8 => {
            if let Some(new_array) = exponential_calculation::<i8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i8 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int16 => {
            if let Some(new_array) = exponential_calculation::<i16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int32 => {
            if let Some(new_array) = exponential_calculation::<i32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int64 => {
            if let Some(new_array) = exponential_calculation::<i64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i64 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float16 => {
            if let Some(new_array) = exponential_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float32 => {
            if let Some(new_array) = exponential_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float64 => {
            if let Some(new_array) = exponential_calculation::<f64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f64 channel")?
            {
                cn.data = new_array
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
fn logarithmic_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: &[f64],
) -> Result<Option<Box<dyn Array>>, Error> {
    let p1 = cc_val[0];
    let p2 = cc_val[1];
    let p3 = cc_val[2];
    let p4 = cc_val[3];
    let p5 = cc_val[4];
    let p6 = cc_val[5];
    let p7 = cc_val[6];
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let mut new_array = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    if p4 == 0.0 {
        arity_assign::unary(&mut new_array, |x| (((x - p7) * p6 - p3) / p1).exp() / p2);
        Ok(Some(new_array.boxed()))
    } else if p1 == 0.0 {
        arity_assign::unary(&mut new_array, |x| ((p3 / (x - p7) - p6) / p4).exp() / p5);
        Ok(Some(new_array.boxed()))
    } else {
        Ok(None)
    }
}

/// Apply exponential conversion to get physical data
fn logarithmic_conversion(cn: &mut Cn3, cc_val: &[f64]) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            if let Some(new_array) = logarithmic_calculation::<u8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u8 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt16 => {
            if let Some(new_array) = logarithmic_calculation::<u16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt32 => {
            if let Some(new_array) = logarithmic_calculation::<u32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::UInt64 => {
            if let Some(new_array) = logarithmic_calculation::<u64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of u64 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int8 => {
            if let Some(new_array) = logarithmic_calculation::<i8>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i8 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int16 => {
            if let Some(new_array) = logarithmic_calculation::<i16>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int32 => {
            if let Some(new_array) = logarithmic_calculation::<i32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Int64 => {
            if let Some(new_array) = logarithmic_calculation::<i64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of i64 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float16 => {
            if let Some(new_array) = logarithmic_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f16 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float32 => {
            if let Some(new_array) = logarithmic_calculation::<f32>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f32 channel")?
            {
                cn.data = new_array
            };
        }
        DataType::Float64 => {
            if let Some(new_array) = logarithmic_calculation::<f64>(cn.data.as_ref(), cc_val)
                .context("failed polynomial conversion of f64 channel")?
            {
                cn.data = new_array
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
fn alegbraic_conversion_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    compiled: &Instruction,
    slab: &Slab,
    cycle_count: &usize,
    formulae: &str,
    name: &str,
) -> Result<Box<dyn Array>, Error> {
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    let mut new_array = vec![0f64; *cycle_count];
    new_array
        .iter_mut()
        .zip(array_f64)
        .for_each(|(new_array, a)| {
            let mut map = BTreeMap::new();
            map.insert("X".to_string(), a.unwrap_or_default());
            match compiled.eval(slab, &mut map) {
                Ok(res) => *new_array = res,
                Err(error_message) => {
                    *new_array = a.unwrap_or_default();
                    warn!(
                        "{}\n Could not compute formulae {} for channel {} and value {}",
                        error_message,
                        formulae,
                        name,
                        a.unwrap_or_default()
                    );
                }
            }
        });
    Ok(PrimitiveArray::<f64>::from_vec(new_array).boxed())
}

/// Apply algebraic conversion to get physical data
fn algebraic_conversion(cn: &mut Cn3, formulae: &str, cycle_count: &u32) -> Result<(), Error> {
    let parser = fasteval::Parser::new();
    let mut slab = fasteval::Slab::new();
    let compiled_instruction = parser.parse(formulae, &mut slab.ps);
    if let Ok(compiled_instruct) = compiled_instruction {
        let compiled = compiled_instruct
            .from(&slab.ps)
            .compile(&slab.ps, &mut slab.cs);
        match cn.data.data_type() {
            DataType::UInt8 => {
                cn.data = alegbraic_conversion_calculation::<u8>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of u8 channel")?;
            }
            DataType::Int8 => {
                cn.data = alegbraic_conversion_calculation::<i8>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of i8 channel")?;
            }
            DataType::Int16 => {
                cn.data = alegbraic_conversion_calculation::<i16>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of i16 channel")?;
            }
            DataType::UInt16 => {
                cn.data = alegbraic_conversion_calculation::<u16>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of u16 channel")?;
            }
            DataType::Float16 => {
                cn.data = alegbraic_conversion_calculation::<f32>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of f32 channel")?;
            }
            DataType::Int32 => {
                cn.data = alegbraic_conversion_calculation::<i32>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of i32 channel")?;
            }
            DataType::UInt32 => {
                cn.data = alegbraic_conversion_calculation::<u32>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of u32 channel")?;
            }
            DataType::Float32 => {
                cn.data = alegbraic_conversion_calculation::<f32>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of f32 channel")?;
            }
            DataType::Int64 => {
                cn.data = alegbraic_conversion_calculation::<i64>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of i64 channel")?;
            }
            DataType::UInt64 => {
                cn.data = alegbraic_conversion_calculation::<u64>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of u64 channel")?;
            }
            DataType::Float64 => {
                cn.data = alegbraic_conversion_calculation::<f64>(
                    cn.data.as_ref(),
                    &compiled,
                    &slab,
                    &(*cycle_count as usize),
                    formulae,
                    &cn.unique_name,
                )
                .context("failed algebraic conversion of f64 channel")?;
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
fn value_to_value_with_interpolation_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: Vec<f64>,
    cycle_count: &usize,
) -> Result<Box<dyn Array>, Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    let mut new_array = vec![0f64; *cycle_count];
    new_array
        .iter_mut()
        .zip(array_f64)
        .for_each(|(new_array, a)| {
            *new_array = match val.binary_search_by(|&(xi, _)| {
                let a64 = a.unwrap_or_default();
                xi.partial_cmp(&a64).expect("Could not compare values")
            }) {
                Ok(idx) => *val[idx].1,
                Err(0) => *val[0].1,
                Err(idx) if idx >= val.len() => *val[idx - 1].1,
                Err(idx) => {
                    let a64 = a.unwrap_or_default();
                    let (x0, y0) = val[idx - 1];
                    let (x1, y1) = val[idx];
                    (y0 * (x1 - a64) + y1 * (a64 - x0)) / (x1 - x0)
                }
            };
        });
    Ok(PrimitiveArray::<f64>::from_vec(new_array).boxed())
}

/// Apply value to value with interpolation conversion to get physical data
fn value_to_value_with_interpolation(
    cn: &mut Cn3,
    cc_val: Vec<f64>,
    cycle_count: &u32,
) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = value_to_value_with_interpolation_calculation::<u8>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = value_to_value_with_interpolation_calculation::<u16>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = value_to_value_with_interpolation_calculation::<u32>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = value_to_value_with_interpolation_calculation::<u64>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = value_to_value_with_interpolation_calculation::<i8>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = value_to_value_with_interpolation_calculation::<i16>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = value_to_value_with_interpolation_calculation::<i32>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = value_to_value_with_interpolation_calculation::<i64>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = value_to_value_with_interpolation_calculation::<f32>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = value_to_value_with_interpolation_calculation::<f32>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = value_to_value_with_interpolation_calculation::<f64>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value with interpolation conversion of f64 channel")?;
        }
        _=> warn!(
            "not possible to apply value to value with interpolation conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}

/// Generic function calculating algebraic expression
#[inline]
fn value_to_value_without_interpolation_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val: Vec<f64>,
    cycle_count: &usize,
) -> Result<Box<dyn Array>, Error> {
    let val: Vec<(&f64, &f64)> = cc_val.iter().tuples().collect();
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    let mut new_array = vec![0f64; *cycle_count];
    new_array
        .iter_mut()
        .zip(array_f64)
        .for_each(|(new_array, a)| {
            let a64 = a.unwrap_or_default();
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
    Ok(PrimitiveArray::<f64>::from_vec(new_array).boxed())
}

/// Apply value to value without interpolation conversion to get physical data
fn value_to_value_without_interpolation(
    cn: &mut Cn3,
    cc_val: Vec<f64>,
    cycle_count: &u32,
) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = value_to_value_without_interpolation_calculation::<u8>(cn.data.as_ref(), cc_val,&(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = value_to_value_without_interpolation_calculation::<u16>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = value_to_value_without_interpolation_calculation::<u32>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = value_to_value_without_interpolation_calculation::<u64>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = value_to_value_without_interpolation_calculation::<i8>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = value_to_value_without_interpolation_calculation::<i16>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = value_to_value_without_interpolation_calculation::<i32>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = value_to_value_without_interpolation_calculation::<i64>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = value_to_value_without_interpolation_calculation::<f32>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = value_to_value_without_interpolation_calculation::<f32>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = value_to_value_without_interpolation_calculation::<f64>(cn.data.as_ref(), cc_val, &(*cycle_count as usize))
                .context("failed value to value without interpolation conversion of f64 channel")?;
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
fn value_to_text_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val_ref: &[(f64, String)],
    cycle_count: &usize,
) -> Result<Box<dyn Array>, Error> {
    let mut new_array = MutableUtf8ValuesArray::<i64>::with_capacities(*cycle_count, 32);
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    array_f64.values_iter().for_each(|val| {
        let matched_key = cc_val_ref.iter().find(|&x| x.0 == *val);
        if let Some(key) = matched_key {
            new_array.push(key.1.clone());
        }
    });
    Ok(new_array.as_box())
}

/// Apply value to text or scale conversion to get physical data
fn value_to_text(
    cn: &mut Cn3,
    cc_val_ref: &[(f64, String)],
    cycle_count: &u32,
) -> Result<(), Error> {
    // identify max string length in cc_val_ref
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = value_to_text_calculation::<u8>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = value_to_text_calculation::<u16>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = value_to_text_calculation::<u32>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = value_to_text_calculation::<u64>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = value_to_text_calculation::<i8>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = value_to_text_calculation::<i16>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = value_to_text_calculation::<i32>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = value_to_text_calculation::<i64>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = value_to_text_calculation::<f32>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = value_to_text_calculation::<f32>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = value_to_text_calculation::<f64>(
                cn.data.as_ref(),
                cc_val_ref,
                &(*cycle_count as usize),
            )
            .context("value to text conversion of f64 channel")?;
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
fn value_range_to_text_calculation<T: NativeType + AsPrimitive<f64> + Display>(
    array: &dyn Array,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
    cycle_count: usize,
) -> Result<Box<dyn Array>, Error> {
    let mut new_array = MutableUtf8ValuesArray::<i64>::with_capacity(cycle_count);
    let parray = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .context("rational conversion could not downcast to primitive array")?;
    let array_f64 = primitive_as_primitive::<T, f64>(parray, &DataType::Float64);
    array_f64.values_iter().for_each(|a| {
        let matched_key = cc_val_ref
            .0
            .iter()
            .enumerate()
            .find(|&x| (x.1 .0 <= *a) && (*a < x.1 .1));
        if let Some(key) = matched_key {
            new_array.push(key.1 .2.clone());
        } else {
            new_array.push(cc_val_ref.1.clone());
        }
    });
    Ok(new_array.as_box())
}

/// Apply value range to text or scale conversion to get physical data
fn value_range_to_text(
    cn: &mut Cn3,
    cc_val_ref: &(Vec<(f64, f64, String)>, String),
    cycle_count: &u32,
) -> Result<(), Error> {
    match cn.data.data_type() {
        DataType::UInt8 => {
            cn.data = value_range_to_text_calculation::<u8>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of u8 channel")?;
        }
        DataType::UInt16 => {
            cn.data = value_range_to_text_calculation::<u16>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of u16 channel")?;
        }
        DataType::UInt32 => {
            cn.data = value_range_to_text_calculation::<u32>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of u32 channel")?;
        }
        DataType::UInt64 => {
            cn.data = value_range_to_text_calculation::<u64>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of u64 channel")?;
        }
        DataType::Int8 => {
            cn.data = value_range_to_text_calculation::<i8>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of i8 channel")?;
        }
        DataType::Int16 => {
            cn.data = value_range_to_text_calculation::<i16>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of i16 channel")?;
        }
        DataType::Int32 => {
            cn.data = value_range_to_text_calculation::<i32>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of i32 channel")?;
        }
        DataType::Int64 => {
            cn.data = value_range_to_text_calculation::<i64>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of i64 channel")?;
        }
        DataType::Float16 => {
            cn.data = value_range_to_text_calculation::<f32>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of f16 channel")?;
        }
        DataType::Float32 => {
            cn.data = value_range_to_text_calculation::<f32>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of f32 channel")?;
        }
        DataType::Float64 => {
            cn.data = value_range_to_text_calculation::<f64>(
                cn.data.as_ref(),
                cc_val_ref,
                *cycle_count as usize,
            )
            .context("value range to text conversion of f64 channel")?;
        }
        _ => warn!(
            "not possible to apply value to text conversion to the data type of channel {}",
            cn.unique_name,
        ),
    }
    Ok(())
}
