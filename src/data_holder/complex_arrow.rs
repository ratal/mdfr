//! complex number stored in primitive builders, fixedsizearraybuilder being too restricted
#[cfg(feature = "ndarray")]
use anyhow::{Context, Error, Result};
use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::{ArrowPrimitiveType, Float32Type, Float64Type},
};
#[cfg(feature = "ndarray")]
use ndarray::{Array, Ix2};

/// Complex struct
#[derive(Debug)]
pub struct ComplexArrow<T: ArrowPrimitiveType> {
    /// The validity mask booolean buffer
    null_buffer_builder: Option<BooleanBuffer>,
    /// the primitive builder
    values_builder: PrimitiveBuilder<T>,
    /// the number of real and imaginary pairs in the array
    len: usize,
}

/// Complex array implementation
impl<T: ArrowPrimitiveType> ComplexArrow<T> {
    /// Create new empty complex array
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
    /// create new complex array with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            len: 0,
        }
    }
    /// create new complex array from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer) -> Self {
        let length = values_buffer.len() / 2;
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            len: length,
        }
    }
    /// Create a new complex array from a primitive builder and optionally its validity buffer
    pub fn new_from_primitive(
        primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
    ) -> Self {
        let length = primitive_builder.len() / 2;
        if let Some(null_buffer_builder) = null_buffer {
            assert_eq!(null_buffer_builder.len() * 2, primitive_builder.len())
        };
        let null_buffer_builder = null_buffer.cloned();
        Self {
            null_buffer_builder,
            values_builder: primitive_builder,
            len: length,
        }
    }
    /// returns the mutable reference of the internal primitive builder array
    pub fn values(&mut self) -> &mut PrimitiveBuilder<T> {
        &mut self.values_builder
    }
    /// retunrs the length of the complex array (number of real and imaginary pairs)
    pub fn len(&self) -> usize {
        self.len
    }
    /// returns True if the complex array is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// returns the internal primitive builder as a slice
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }
    /// returns the validity array
    pub fn nulls(&self) -> Option<&BooleanBuffer> {
        self.null_buffer_builder.as_ref()
    }
    /// returns a finished cloned primitive array of the inner primitive builder
    pub fn finish_cloned(&self) -> PrimitiveArray<T> {
        self.values_builder.finish_cloned()
    }
    /// returns a finished primitive array of the inner primitive builder
    pub fn finish(&mut self) -> PrimitiveArray<T> {
        self.values_builder.finish()
    }
    /// overwrite the validity array
    pub fn set_validity(&mut self, mask: &mut BooleanBufferBuilder) {
        self.null_buffer_builder = Some(mask.finish());
    }
    /// returns the optional validity array as a slice
    pub fn validity_slice(&self) -> Option<&[u8]> {
        self.null_buffer_builder.as_ref().map(|s| s.values())
    }
}

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float32Type> {
    /// to convert ComplexArrow into ndarray
    pub fn to_ndarray(&self) -> Result<Array<f32, Ix2>, Error> {
        let vector: Vec<f32> = self.values_builder.values_slice().to_vec();
        Array::from_shape_vec((self.len(), 2), vector)
            .context("Failed reshaping f32 complex arrow into ndarray")
    }
}

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float64Type> {
    /// to convert ComplexArrow into ndarray
    pub fn to_ndarray(&self) -> Result<Array<f64, Ix2>, Error> {
        let vector: Vec<f64> = self.values_builder.values_slice().to_vec();
        Array::from_shape_vec((vector.len() / 2, 2), vector)
            .context("Failed reshaping f64 complex arrow into ndarray")
    }
}

impl<T: ArrowPrimitiveType> Default for ComplexArrow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for ComplexArrow<Float32Type> {
    fn eq(&self, other: &Self) -> bool {
        if self.values_builder.finish_cloned() == other.values_builder.finish_cloned() {
            match &self.null_buffer_builder {
                Some(buffer) => match &other.null_buffer_builder {
                    Some(other_buffer) => buffer == other_buffer,
                    None => false,
                },
                None => other.null_buffer_builder.is_none(),
            }
        } else {
            false
        }
    }
}

impl PartialEq for ComplexArrow<Float64Type> {
    fn eq(&self, other: &Self) -> bool {
        if self.values_builder.finish_cloned() == other.values_builder.finish_cloned() {
            match &self.null_buffer_builder {
                Some(buffer) => match &other.null_buffer_builder {
                    Some(other_buffer) => buffer == other_buffer,
                    None => false,
                },
                None => other.null_buffer_builder.is_none(),
            }
        } else {
            false
        }
    }
}

impl Clone for ComplexArrow<Float32Type> {
    fn clone(&self) -> Self {
        Self {
            null_buffer_builder: self.null_buffer_builder.clone(),
            values_builder: self
                .values_builder
                .finish_cloned()
                .into_builder()
                .expect("failed getting builder from Primitive array"),
            len: self.len,
        }
    }
}

impl Clone for ComplexArrow<Float64Type> {
    fn clone(&self) -> Self {
        Self {
            null_buffer_builder: self.null_buffer_builder.clone(),
            values_builder: self
                .values_builder
                .finish_cloned()
                .into_builder()
                .expect("failed getting builder from Primitive array"),
            len: self.len,
        }
    }
}
