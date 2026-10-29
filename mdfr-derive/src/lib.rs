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
    kind: FieldKind,
}

enum FieldKind {
    Scalar {
        cn_data_type: proc_macro2::TokenStream,
        bit_count: u32,
        byte_len: usize,
        encode_expr: fn(&syn::Ident) -> proc_macro2::TokenStream,
    },
    ByteArray {
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

fn parse_struct_group_name(attrs: &[Attribute], default_name: &str) -> syn::Result<String> {
    let mut group_name = default_name.to_string();
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
            } else {
                Err(meta.error(
                    "unsupported struct-level #[mdf(...)] attribute; expected `group = \"...\"`",
                ))
            }
        })?;
    }
    Ok(group_name)
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
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 8,
                    byte_len: 1,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "u16" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 16,
                    byte_len: 2,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "u32" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 32,
                    byte_len: 4,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "u64" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 64,
                    byte_len: 8,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "i8" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    bit_count: 8,
                    byte_len: 1,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "i16" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    bit_count: 16,
                    byte_len: 2,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "i32" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    bit_count: 32,
                    byte_len: 4,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "i64" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::IntLE),
                    bit_count: 64,
                    byte_len: 8,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "f32" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                    bit_count: 32,
                    byte_len: 4,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "f64" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::FloatLE),
                    bit_count: 64,
                    byte_len: 8,
                    encode_expr: |id| quote!(self.#id.to_le_bytes()),
                }),
                "bool" => Ok(FieldKind::Scalar {
                    cn_data_type: quote!(::mdfr_core::blocks::cn::CnDataType::UIntLE),
                    bit_count: 8,
                    byte_len: 1,
                    encode_expr: |id| quote!([self.#id as u8]),
                }),
                _ => Err(syn::Error::new_spanned(
                    ty,
                    "unsupported field type for #[derive(MdfRecord)]; expected u8/u16/u32/u64, i8/i16/i32/i64, f32/f64, bool, or [u8; N]",
                )),
            }
        }
        Type::Array(arr) => {
            if let Type::Path(tp) = &*arr.elem
                && tp.path.is_ident("u8")
                && let Expr::Lit(el) = &arr.len
                && let Lit::Int(li) = &el.lit
            {
                let n = li.base10_parse::<usize>()?;
                if n > 0 {
                    return Ok(FieldKind::ByteArray { len: n });
                }
            }
            Err(syn::Error::new_spanned(
                ty,
                "unsupported array field for #[derive(MdfRecord)]; only non-empty [u8; N] is supported",
            ))
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported field type for #[derive(MdfRecord)]",
        )),
    }
}

fn impl_mdf_record(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let struct_ident = &input.ident;
    let group_name = parse_struct_group_name(&input.attrs, &struct_ident.to_string())?;

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

        for attr in &field.attrs {
            if !attr.path().is_ident("mdf") {
                continue;
            }
            let nested = attr.parse_args_with(
                syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
            )?;
            for meta in nested {
                match meta {
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
                            "unsupported #[mdf(...)] attribute; supported: `name = \"...\"`, `unit = \"...\"`, `comment = \"...\"`, `linear(p1, p2)`, `rational(p1..p6)`",
                        ));
                    }
                }
            }
        }

        let kind = classify_type(&field.ty)?;
        match &kind {
            FieldKind::Scalar { byte_len, .. } => record_bytes += *byte_len,
            FieldKind::ByteArray { len } => record_bytes += *len,
        }
        specs.push(FieldSpec {
            ident,
            ch_name,
            unit,
            comment,
            linear,
            rational,
            kind,
        });
    }

    let mut register_stmts = Vec::new();
    let mut encode_stmts = Vec::new();
    let mut offset: usize = 8;

    for spec in &specs {
        let ident = &spec.ident;
        let ch_name = &spec.ch_name;
        let add_stmt = match &spec.kind {
            FieldKind::Scalar {
                cn_data_type,
                bit_count,
                byte_len,
                encode_expr,
            } => {
                let start = offset;
                let end = offset + *byte_len;
                offset = end;
                let expr = encode_expr(ident);
                encode_stmts.push(quote! {
                    buf[#start..#end].copy_from_slice(&#expr);
                });
                quote! {
                    let ch = builder.add_channel(group, #ch_name, #cn_data_type, #bit_count)?;
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
        assert!(res.is_ok(), "Failed to implement MdfRecord: {:?}", res.err());
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
    fn test_derive_error_unsupported_array() {
        let input: DeriveInput = parse_quote! {
            struct BadArray {
                payload: [u16; 4],
            }
        };
        let err = impl_mdf_record(&input).unwrap_err().to_string();
        assert!(err.contains("only non-empty [u8; N] is supported"));
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
