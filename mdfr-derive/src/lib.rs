//! Procedural macro `#[derive(MdfRecord)]` for zero-boilerplate typed MDF 4.3 channel groups.
//!
//! # Example
//! ```ignore
//! use mdfr_core::logger::MdfRecord;
//!
//! #[derive(MdfRecord)]
//! #[mdf(group = "Engine")]
//! pub struct EngineSample {
//!     #[mdf(unit = "rpm", linear(0.0, 0.5))]
//!     pub rpm: u16,
//!     #[mdf(name = "CoolantTemp", unit = "degC")]
//!     pub temp: f32,
//!     pub gear: u8,
//!     pub payload: [u8; 8],
//! }
//! ```
use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Expr, Fields, Lit, Meta, Type, parse_macro_input};

#[proc_macro_derive(MdfRecord, attributes(mdf))]
pub fn derive_mdf_record(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_mdf_record(&input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct FieldSpec {
    ident: syn::Ident,
    ch_name: String,
    unit: Option<String>,
    comment: Option<String>,
    linear: Option<(f64, f64)>,
    rational: Option<[f64; 6]>,
    big_endian: bool,
    kind: FieldKind,
}

enum FieldKind {
    Scalar {
        cn_data_type_le: proc_macro2::TokenStream,
        cn_data_type_be: proc_macro2::TokenStream,
        bit_count: u32,
        byte_len: usize,
        is_bool: bool,
    },
    ByteArray {
        len: usize,
    },
    TypedArray {
        elem_data_type_le: proc_macro2::TokenStream,
        elem_data_type_be: proc_macro2::TokenStream,
        elem_bit_count: u32,
        elem_byte_len: usize,
        len: usize,
    },
}

fn parse_f64_expr(expr: &Expr) -> syn::Result<f64> {
    match expr {
        Expr::Lit(el) => match &el.lit {
            Lit::Float(f) => f.base10_parse::<f64>(),
            Lit::Int(i) => i.base10_parse::<f64>(),
            _ => Err(syn::Error::new_spanned(expr, "expected numeric literal")),
        },
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Neg(_)) => Ok(-parse_f64_expr(&u.expr)?),
        _ => Err(syn::Error::new_spanned(expr, "expected numeric literal")),
    }
}

fn parse_struct_attributes(attrs: &[Attribute], default_name: &str) -> syn::Result<(String, bool)> {
    let mut group_name = default_name.to_string();
    let mut struct_big_endian = false;
    for attr in attrs {
        if !attr.path().is_ident("mdf") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("group") {
                let v = meta.value()?;
                let s: syn::LitStr = v.parse()?;
                group_name = s.value();
                Ok(())
            } else if meta.path.is_ident("big_endian") || meta.path.is_ident("be") {
                struct_big_endian = true;
                Ok(())
            } else {
                Err(meta.error(
                    "unsupported struct-level #[mdf(...)] attribute; expected `group = \"...\"` or `big_endian`",
                ))
            }
        })?;
    }
    Ok((group_name, struct_big_endian))
}

fn classify_type(ty: &Type) -> syn::Result<FieldKind> {
    match ty {
        Type::Path(tp) if tp.qself.is_none() => {
            let Some(ident) = tp.path.get_ident() else {
                return Err(syn::Error::new_spanned(
                    ty,
                    "unsupported field type for #[derive(MdfRecord)]",
                ));
            };
            let s = ident.to_string();
            match s.as_str() {
                "u8" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                    bit_count: 8,
                    byte_len: 1,
                    is_bool: false,
                }),
                "u16" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                    bit_count: 16,
                    byte_len: 2,
                    is_bool: false,
                }),
                "u32" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                    bit_count: 32,
                    byte_len: 4,
                    is_bool: false,
                }),
                "u64" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                    bit_count: 64,
                    byte_len: 8,
                    is_bool: false,
                }),
                "i8" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                    bit_count: 8,
                    byte_len: 1,
                    is_bool: false,
                }),
                "i16" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                    bit_count: 16,
                    byte_len: 2,
                    is_bool: false,
                }),
                "i32" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                    bit_count: 32,
                    byte_len: 4,
                    is_bool: false,
                }),
                "i64" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                    bit_count: 64,
                    byte_len: 8,
                    is_bool: false,
                }),
                "f32" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::FloatBE),
                    bit_count: 32,
                    byte_len: 4,
                    is_bool: false,
                }),
                "f64" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::FloatBE),
                    bit_count: 64,
                    byte_len: 8,
                    is_bool: false,
                }),
                "bool" => Ok(FieldKind::Scalar {
                    cn_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    cn_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 8,
                    byte_len: 1,
                    is_bool: true,
                }),
                _ => Err(syn::Error::new_spanned(
                    ty,
                    "unsupported field type for #[derive(MdfRecord)]; expected u8/u16/u32/u64, i8/i16/i32/i64, f32/f64, bool, or [T; N]",
                )),
            }
        }
        Type::Array(arr) => {
            let len = if let Expr::Lit(el) = &arr.len
                && let Lit::Int(li) = &el.lit
            {
                li.base10_parse::<usize>()?
            } else {
                return Err(syn::Error::new_spanned(
                    &arr.len,
                    "expected integer literal array length",
                ));
            };
            if len == 0 {
                return Err(syn::Error::new_spanned(
                    arr,
                    "zero-length arrays are not supported in #[derive(MdfRecord)]",
                ));
            }

            if let Type::Path(tp) = &*arr.elem
                && let Some(ident) = tp.path.get_ident()
            {
                match ident.to_string().as_str() {
                    "u8" => Ok(FieldKind::ByteArray { len }),
                    "u16" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                        elem_bit_count: 16,
                        elem_byte_len: 2,
                        len,
                    }),
                    "u32" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                        elem_bit_count: 32,
                        elem_byte_len: 4,
                        len,
                    }),
                    "u64" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::UIntBE),
                        elem_bit_count: 64,
                        elem_byte_len: 8,
                        len,
                    }),
                    "i8" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                        elem_bit_count: 8,
                        elem_byte_len: 1,
                        len,
                    }),
                    "i16" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                        elem_bit_count: 16,
                        elem_byte_len: 2,
                        len,
                    }),
                    "i32" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                        elem_bit_count: 32,
                        elem_byte_len: 4,
                        len,
                    }),
                    "i64" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::IntBE),
                        elem_bit_count: 64,
                        elem_byte_len: 8,
                        len,
                    }),
                    "f32" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::FloatBE),
                        elem_bit_count: 32,
                        elem_byte_len: 4,
                        len,
                    }),
                    "f64" => Ok(FieldKind::TypedArray {
                        elem_data_type_le: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                        elem_data_type_be: quote!(::mdfr_core::blocks::cn::CnDataType::FloatBE),
                        elem_bit_count: 64,
                        elem_byte_len: 8,
                        len,
                    }),
                    _ => Err(syn::Error::new_spanned(
                        arr,
                        "unsupported array element type for #[derive(MdfRecord)]; expected numeric primitive (u8..u64, i8..i64, f32/f64)",
                    )),
                }
            } else {
                Err(syn::Error::new_spanned(
                    arr,
                    "unsupported array element type for #[derive(MdfRecord)]",
                ))
            }
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported field type for #[derive(MdfRecord)]",
        )),
    }
}

fn impl_mdf_record(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_ident = &input.ident;
    let (group_name, struct_big_endian) =
        parse_struct_attributes(&input.attrs, &struct_ident.to_string())?;

    let Data::Struct(data_struct) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(MdfRecord)] can only be applied to structs with named fields",
        ));
    };
    let Fields::Named(fields_named) = &data_struct.fields else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(MdfRecord)] requires named struct fields",
        ));
    };

    let mut specs = Vec::new();
    let mut record_bytes: usize = 8; // 8-byte master timestamp at 0..8

    for field in &fields_named.named {
        let ident = field.ident.clone().unwrap();
        let mut ch_name = ident.to_string();
        let mut unit: Option<String> = None;
        let mut comment: Option<String> = None;
        let mut linear: Option<(f64, f64)> = None;
        let mut rational: Option<[f64; 6]> = None;
        let mut big_endian = struct_big_endian;

        for attr in &field.attrs {
            if !attr.path().is_ident("mdf") {
                continue;
            }
            let nested = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            )?;
            for meta in nested {
                match meta {
                    Meta::Path(p) if p.is_ident("big_endian") || p.is_ident("be") => {
                        big_endian = true;
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("name") => {
                        if let Expr::Lit(el) = &nv.value
                            && let Lit::Str(s) = &el.lit
                        {
                            ch_name = s.value();
                        } else {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "expected string literal for `name`",
                            ));
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("unit") => {
                        if let Expr::Lit(el) = &nv.value
                            && let Lit::Str(s) = &el.lit
                        {
                            unit = Some(s.value());
                        } else {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "expected string literal for `unit`",
                            ));
                        }
                    }
                    Meta::NameValue(nv) if nv.path.is_ident("comment") => {
                        if let Expr::Lit(el) = &nv.value
                            && let Lit::Str(s) = &el.lit
                        {
                            comment = Some(s.value());
                        } else {
                            return Err(syn::Error::new_spanned(
                                nv,
                                "expected string literal for `comment`",
                            ));
                        }
                    }
                    Meta::List(ml) if ml.path.is_ident("linear") => {
                        let args = ml.parse_args_with(
                            syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated,
                        )?;
                        if args.len() != 2 {
                            return Err(syn::Error::new_spanned(
                                ml,
                                "linear(...) expects 2 arguments: `linear(p1, p2)` where Y = p1 + p2 * X",
                            ));
                        }
                        linear = Some((parse_f64_expr(&args[0])?, parse_f64_expr(&args[1])?));
                    }
                    Meta::List(ml) if ml.path.is_ident("rational") => {
                        let args = ml.parse_args_with(
                            syn::punctuated::Punctuated::<Expr, syn::Token![,]>::parse_terminated,
                        )?;
                        if args.len() != 6 {
                            return Err(syn::Error::new_spanned(
                                ml,
                                "rational(...) expects 6 arguments: `rational(p1, p2, p3, p4, p5, p6)`",
                            ));
                        }
                        rational = Some([
                            parse_f64_expr(&args[0])?,
                            parse_f64_expr(&args[1])?,
                            parse_f64_expr(&args[2])?,
                            parse_f64_expr(&args[3])?,
                            parse_f64_expr(&args[4])?,
                            parse_f64_expr(&args[5])?,
                        ]);
                    }
                    other => {
                        return Err(syn::Error::new_spanned(
                            other,
                            "unsupported #[mdf(...)] attribute; supported: `name = \"...\"`, `unit = \"...\"`, `comment = \"...\"`, `big_endian`, `linear(p1, p2)`, `rational(p1..p6)`",
                        ));
                    }
                }
            }
        }

        let kind = classify_type(&field.ty)?;
        match &kind {
            FieldKind::Scalar { byte_len, .. } => record_bytes += *byte_len,
            FieldKind::ByteArray { len } => record_bytes += *len,
            FieldKind::TypedArray {
                elem_byte_len, len, ..
            } => record_bytes += *len * *elem_byte_len,
        }
        specs.push(FieldSpec {
            ident,
            ch_name,
            unit,
            comment,
            linear,
            rational,
            big_endian,
            kind,
        });
    }

    let mut register_stmts = Vec::new();
    let mut encode_stmts = Vec::new();
    let mut offset: usize = 8;

    for spec in &specs {
        let ident = &spec.ident;
        let ch_name = &spec.ch_name;
        let is_be = spec.big_endian;
        let add_stmt = match &spec.kind {
            FieldKind::Scalar {
                cn_data_type_le,
                cn_data_type_be,
                bit_count,
                byte_len,
                is_bool,
            } => {
                let start = offset;
                let end = offset + *byte_len;
                offset = end;
                let dt = if is_be {
                    cn_data_type_be
                } else {
                    cn_data_type_le
                };
                let expr = if *is_bool {
                    quote!([self.#ident as u8])
                } else if is_be {
                    quote!(self.#ident.to_be_bytes())
                } else {
                    quote!(self.#ident.to_le_bytes())
                };
                encode_stmts.push(quote! {
                    buf[#start..#end].copy_from_slice(&#expr);
                });
                quote! {
                    let ch = builder.add_channel(group, #ch_name, #dt, #bit_count)?;
                }
            }
            FieldKind::ByteArray { len } => {
                let start = offset;
                let end = offset + *len;
                offset = end;
                let bit_count = (*len as u32) * 8;
                encode_stmts.push(quote! {
                    buf[#start..#end].copy_from_slice(&self.#ident);
                });
                quote! {
                    let ch = builder.add_channel(
                        group,
                        #ch_name,
                        ::mdfr_core::blocks::cn::CnDataType::ByteArray,
                        #bit_count,
                    )?;
                }
            }
            FieldKind::TypedArray {
                elem_data_type_le,
                elem_data_type_be,
                elem_bit_count,
                elem_byte_len,
                len,
            } => {
                let total_bytes = *len * *elem_byte_len;
                let start = offset;
                let end = offset + total_bytes;
                offset = end;
                let dt = if is_be {
                    elem_data_type_be
                } else {
                    elem_data_type_le
                };
                let elem_expr = if is_be {
                    quote!(elem.to_be_bytes())
                } else {
                    quote!(elem.to_le_bytes())
                };
                encode_stmts.push(quote! {
                    for (i, elem) in self.#ident.iter().enumerate() {
                        let s = #start + i * #elem_byte_len;
                        let e = s + #elem_byte_len;
                        buf[s..e].copy_from_slice(&#elem_expr);
                    }
                });
                quote! {
                    let ch = builder.add_array_channel(
                        group,
                        #ch_name,
                        #dt,
                        #elem_bit_count,
                        ::mdfr_core::alloc::vec![#len as u64],
                    )?;
                }
            }
        };
        register_stmts.push(add_stmt);

        if let Some(ref u) = spec.unit {
            register_stmts.push(quote! {
                builder.set_channel_unit(ch, #u)?;
            });
        }
        if let Some(ref c) = spec.comment {
            register_stmts.push(quote! {
                builder.set_channel_comment(ch, #c)?;
            });
        }
        if let Some((p1, p2)) = spec.linear {
            register_stmts.push(quote! {
                builder.set_channel_conversion(
                    ch,
                    ::mdfr_core::logger::ConversionDef::Linear { p1: #p1, p2: #p2 },
                )?;
            });
        }
        if let Some([p1, p2, p3, p4, p5, p6]) = spec.rational {
            register_stmts.push(quote! {
                builder.set_channel_conversion(
                    ch,
                    ::mdfr_core::logger::ConversionDef::Rational([#p1, #p2, #p3, #p4, #p5, #p6]),
                )?;
            });
        }
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics ::mdfr_core::logger::MdfRecord for #struct_ident #ty_generics #where_clause {
            const GROUP_NAME: &'static str = #group_name;
            const RECORD_BYTES: usize = #record_bytes;

            fn register_channels(
                builder: &mut ::mdfr_core::logger::LoggerBuilder,
                group: ::mdfr_core::logger::GroupHandle,
            ) -> ::mdfr_core::error::Result<()> {
                #(#register_stmts)*
                Ok(())
            }

            #[inline]
            fn encode_record(&self, buf: &mut [u8]) {
                #(#encode_stmts)*
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn test_derive_valid_struct_all_types() {
        let input: DeriveInput = parse_quote! {
            #[mdf(group = "VehicleTelemetry")]
            pub struct TelemetrySample {
                #[mdf(name = "EngineSpeed", unit = "rpm", comment = "crankshaft RPM", linear(0.0, 0.5))]
                pub rpm: u16,
                #[mdf(rational(0.0, 1.0, 0.0, 0.0, 0.0, 1.0))]
                pub temp: f32,
                pub flag: bool,
                pub u8_val: u8,
                pub u32_val: u32,
                pub u64_val: u64,
                pub i8_val: i8,
                pub i16_val: i16,
                pub i32_val: i32,
                pub i64_val: i64,
                pub f64_val: f64,
                pub raw_payload: [u8; 8],
            }
        };

        let res = impl_mdf_record(&input);
        assert!(
            res.is_ok(),
            "Failed to implement MdfRecord: {:?}",
            res.err()
        );
        let tokens = res.unwrap().to_string();
        assert!(tokens.contains("impl :: mdfr_core :: logger :: MdfRecord for TelemetrySample"));
        assert!(tokens.contains("const GROUP_NAME : & 'static str = \"VehicleTelemetry\""));
        // 8 master + 2 + 4 + 1 + 1 + 4 + 8 + 1 + 2 + 4 + 8 + 8 + 8 = 59 bytes
        assert!(tokens.contains("const RECORD_BYTES : usize = 59usize"));
    }

    #[test]
    fn test_derive_negative_numbers() {
        let input: DeriveInput = parse_quote! {
            struct Sample {
                #[mdf(linear(-10.5, -2))]
                val1: f32,
                #[mdf(rational(-1, -2.5, -3, -4, -5, -6.2))]
                val2: f64,
            }
        };
        let res = impl_mdf_record(&input);
        assert!(res.is_ok(), "Failed on negative numbers: {:?}", res.err());
    }

    #[test]
    fn test_derive_error_tuple_struct() {
        let input: DeriveInput = parse_quote! {
            struct TupleSample(u8, u16);
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("requires named struct fields"));
    }

    #[test]
    fn test_derive_error_enum() {
        let input: DeriveInput = parse_quote! {
            enum EnumSample {
                A,
                B,
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("can only be applied to structs with named fields"));
    }

    #[test]
    fn test_derive_error_unsupported_field_type() {
        let input: DeriveInput = parse_quote! {
            struct BadField {
                msg: String,
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("unsupported field type for #[derive(MdfRecord)]"));
    }

    #[test]
    fn test_derive_typed_arrays_and_big_endian() {
        let input: DeriveInput = parse_quote! {
            #[mdf(group = "SensorArray", big_endian)]
            pub struct ArraySample {
                pub accel: [f32; 3],
                #[mdf(name = "Voltages", unit = "mV")]
                pub cell_mv: [u16; 12],
                #[mdf(big_endian)]
                pub single_be: u32,
            }
        };

        let res = impl_mdf_record(&input);
        assert!(res.is_ok(), "Failed: {:?}", res.err());
        let tokens = res.unwrap().to_string();
        assert!(tokens.contains("add_array_channel"));
        assert!(tokens.contains("CnDataType :: FloatBE"));
        assert!(tokens.contains("CnDataType :: UIntBE"));
        // 8 master + 3*4 + 12*2 + 4 = 8 + 12 + 24 + 4 = 48 bytes
        assert!(tokens.contains("const RECORD_BYTES : usize = 48usize"));
    }

    #[test]
    fn test_derive_error_unsupported_array() {
        let input: DeriveInput = parse_quote! {
            struct BadArray {
                payload: [String; 4],
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("unsupported array element type for #[derive(MdfRecord)]"));
    }

    #[test]
    fn test_derive_error_invalid_struct_attr() {
        let input: DeriveInput = parse_quote! {
            #[mdf(invalid_attr = "val")]
            struct BadAttr {
                x: u8,
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("unsupported struct-level #[mdf(...)] attribute"));
    }

    #[test]
    fn test_derive_error_linear_arguments() {
        let input: DeriveInput = parse_quote! {
            struct BadLinear {
                #[mdf(linear(1.0))]
                x: u8,
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("linear(...) expects 2 arguments"));
    }

    #[test]
    fn test_derive_error_rational_arguments() {
        let input: DeriveInput = parse_quote! {
            struct BadRational {
                #[mdf(rational(1.0, 2.0))]
                x: u8,
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("rational(...) expects 6 arguments"));
    }
}
