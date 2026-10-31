//! complex number stored in primitive arrays / builders, fixedsizearraybuilder being too restricted
use crate::data_holder::tensor_arrow::Order;
#[cfg(feature = "ndarray")]
use anyhow::{Context, Error, Result};
#[cfg(feature = "ndarray")]
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::{
    array::{BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer, ScalarBuffer},
    datatypes::ArrowPrimitiveType,
};
#[cfg(feature = "ndarray")]
use ndarray::{Array as NdArray, IxDyn};
use std::mem::size_of;

/// Immutable Complex array wrapping an Arrow [`PrimitiveArray`]
#[derive(Debug)]
pub struct ComplexArrow<T: ArrowPrimitiveType> {
    /// The validity mask boolean buffer
    pub(crate) null_buffer: Option<BooleanBuffer>,
    /// The immutable Arrow primitive array
    pub(crate) values: PrimitiveArray<T>,
    /// The number of real and imaginary pairs in the array
    pub(crate) len: usize,
    /// Shape of tensor (without the implicit trailing 2 for real/imag)
    pub(crate) shape: Vec<usize>,
    /// Order of tensor, row or column major
    pub(crate) order: Order,
}

/// Complex array implementation
impl<T: ArrowPrimitiveType> ComplexArrow<T> {
    /// Create new empty complex array
    pub fn new() -> Self {
        Self {
            null_buffer: None,
            values: PrimitiveArray::<T>::new(ScalarBuffer::default(), None),
            len: 0,
            shape: vec![1],
            order: Order::RowMajor,
        }
    }

    /// Create new complex array from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let pair_size = shape.iter().product::<usize>() * 2;
        let num_elements = values_buffer.len() / size_of::<T::Native>();
        let length = num_elements.checked_div(pair_size).unwrap_or(0);
        let buffer: arrow::buffer::Buffer = values_buffer.into();
        let scalar_buffer = ScalarBuffer::<T::Native>::new(buffer, 0, num_elements);
        let values = PrimitiveArray::<T>::new(scalar_buffer, None);
        Self {
            null_buffer: None,
            values,
            len: length,
            shape,
            order,
        }
    }

    /// Create a new complex array from a primitive array and optionally its validity buffer
    pub fn new_from_array(
        values: PrimitiveArray<T>,
        null_buffer: Option<BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let pair_size = shape.iter().product::<usize>() * 2;
        let length = values.len().checked_div(pair_size).unwrap_or(0);
        Self {
            null_buffer,
            values,
            len: length,
            shape,
            order,
        }
    }

    /// Create a new complex array from a primitive builder (for compatibility)
    #[allow(dead_code)]
    pub fn new_from_primitive(
        mut primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let values = primitive_builder.finish();
        let pair_size = shape.iter().product::<usize>() * 2;
        let length = values.len().checked_div(pair_size).unwrap_or(0);
        Self {
            null_buffer: null_buffer.cloned(),
            values,
            len: length,
            shape,
            order,
        }
    }

    /// Slices the complex array with zero-copy
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let pair_size = self.shape.iter().product::<usize>() * 2;
        let values_slice = self.values.slice(offset * pair_size, length * pair_size);
        let null_buffer = self.null_buffer.as_ref().map(|nb| nb.slice(offset, length));
        Self {
            null_buffer,
            values: values_slice,
            len: length,
            shape: self.shape.clone(),
            order: self.order.clone(),
        }
    }

    /// returns the reference of the internal primitive array
    pub fn values(&self) -> &PrimitiveArray<T> {
        &self.values
    }

    /// returns the length of the complex array (number of real and imaginary pairs)
    pub fn len(&self) -> usize {
        self.len
    }

    /// returns True if the complex array is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// returns the internal primitive array as a slice
    pub fn values_slice(&self) -> &[T::Native] {
        self.values.values()
    }

    /// returns the validity buffer
    pub fn nulls(&self) -> Option<&BooleanBuffer> {
        self.null_buffer.as_ref()
    }

    /// returns a cloned primitive array of the inner primitive array (O(1) Arc clone)
    pub fn finish_cloned(&self) -> PrimitiveArray<T> {
        self.values.clone()
    }

    /// returns a cloned primitive array of the inner primitive array (O(1) Arc clone)
    #[allow(dead_code)]
    pub fn finish(&self) -> PrimitiveArray<T> {
        self.values.clone()
    }

    /// overwrite the validity array
    pub fn set_validity(&mut self, mask: &mut BooleanBufferBuilder) {
        self.null_buffer = Some(mask.finish());
    }

    /// returns the optional validity array as a slice
    pub fn validity_slice(&self) -> Option<&[u8]> {
        self.null_buffer
            .as_ref()
            .map(arrow::buffer::BooleanBuffer::values)
    }

    /// returns the shape of the complex array (excluding the implicit trailing 2 for real/imag)
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// returns the memory layout order of the complex array
    pub fn order(&self) -> &Order {
        &self.order
    }
}

/// Mutable Complex builder for staging complex arrays during reading
#[derive(Debug)]
pub struct ComplexBuilder<T: ArrowPrimitiveType> {
    pub(crate) null_buffer_builder: Option<BooleanBuffer>,
    pub(crate) values_builder: PrimitiveBuilder<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) order: Order,
}

impl<T: ArrowPrimitiveType> ComplexBuilder<T> {
    /// Create new empty complex builder
    pub fn new() -> Self {
        Self::with_capacity(1024, vec![1], Order::RowMajor)
    }

    /// create new complex builder with capacity
    pub fn with_capacity(capacity: usize, shape: Vec<usize>, order: Order) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            shape,
            order,
        }
    }

    /// create new complex builder from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            shape,
            order,
        }
    }

    /// returns the mutable reference of the internal primitive builder array
    #[allow(dead_code)]
    pub fn values(&mut self) -> &mut PrimitiveBuilder<T> {
        &mut self.values_builder
    }

    /// returns the internal primitive builder as a mutable slice
    pub fn values_slice_mut(&mut self) -> &mut [T::Native] {
        self.values_builder.values_slice_mut()
    }

    /// returns the internal primitive builder as a slice
    #[allow(dead_code)]
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }

    /// finishes building and converts to an immutable ComplexArrow (0 copies)
    pub fn finish(mut self) -> ComplexArrow<T> {
        let values = self.values_builder.finish();
        let pair_size = self.shape.iter().product::<usize>() * 2;
        let len = values.len().checked_div(pair_size).unwrap_or(0);
        ComplexArrow {
            null_buffer: self.null_buffer_builder,
            values,
            len,
            shape: self.shape,
            order: self.order,
        }
    }
}

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float32Type> {
    /// to convert ComplexArrow into ndarray
    #[allow(dead_code)]
    pub fn to_ndarray(&self) -> Result<NdArray<f32, IxDyn>, Error> {
        let vector: Vec<f32> = self.values.values().to_vec();
        let mut full_shape = vec![self.len()];
        full_shape.extend(self.shape.iter());
        full_shape.push(2); // complex is always a pair
        NdArray::from_shape_vec(IxDyn(&full_shape), vector)
            .context("Failed reshaping f32 complex arrow into ndarray")
    }
}

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float64Type> {
    /// to convert ComplexArrow into ndarray
    #[allow(dead_code)]
    pub fn to_ndarray(&self) -> Result<NdArray<f64, IxDyn>, Error> {
        let vector: Vec<f64> = self.values.values().to_vec();
        let mut full_shape = vec![self.len()];
        full_shape.extend(self.shape.iter());
        full_shape.push(2); // complex is always a pair
        NdArray::from_shape_vec(IxDyn(&full_shape), vector)
            .context("Failed reshaping f64 complex arrow into ndarray")
    }
}

impl<T: ArrowPrimitiveType> Default for ComplexArrow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType> Default for ComplexBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType> PartialEq for ComplexArrow<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
            && self.null_buffer == other.null_buffer
            && self.shape == other.shape
            && self.order == other.order
    }
}

impl<T: ArrowPrimitiveType> Clone for ComplexArrow<T> {
    fn clone(&self) -> Self {
        Self {
            null_buffer: self.null_buffer.clone(),
            values: self.values.clone(), // O(1) Arc clone
            len: self.len,
            shape: self.shape.clone(),
            order: self.order.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Float32Type, Float64Type};

    #[test]
    fn test_new_and_default() {
        let c: ComplexArrow<Float64Type> = ComplexArrow::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert!(c.nulls().is_none());

        let d: ComplexArrow<Float32Type> = ComplexArrow::default();
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let mut cb: ComplexBuilder<Float64Type> =
            ComplexBuilder::with_capacity(512, vec![1], Order::RowMajor);
        cb.values_slice_mut();
        let c = cb.finish();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_from_buffer_and_methods() {
        // 4 f64 values = 2 complex pairs
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = MutableBuffer::from_iter(data.iter().copied());
        let c = ComplexArrow::<Float64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
        assert!(!c.is_empty());
        assert_eq!(c.values_slice().len(), 4);
        assert_eq!(c.len(), 2);

        let arr = c.finish_cloned();
        assert_eq!(arr.len(), 4);

        assert!(c.nulls().is_none());
        assert!(c.validity_slice().is_none());
    }

    #[test]
    fn test_from_primitive() {
        let mut pb = PrimitiveBuilder::<Float32Type>::with_capacity(6);
        pb.append_value(1.0);
        pb.append_value(2.0);
        pb.append_value(3.0);
        pb.append_value(4.0);
        pb.append_value(5.0);
        pb.append_value(6.0);
        let c = ComplexArrow::new_from_primitive(pb, None, vec![1], Order::RowMajor);
        assert_eq!(c.len(), 3); // 6 values / 2 = 3 pairs
        assert!(!c.is_empty());
        assert!(c.nulls().is_none());
    }

    #[test]
    fn test_partial_eq_f32() {
        let buf1 = MutableBuffer::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().copied());
        let buf2 = MutableBuffer::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().copied());
        let c1 = ComplexArrow::<Float32Type>::new_from_buffer(buf1, vec![1], Order::RowMajor);
        let c2 = ComplexArrow::<Float32Type>::new_from_buffer(buf2, vec![1], Order::RowMajor);
        assert_eq!(c1, c2);

        let buf3 = MutableBuffer::from_iter([5.0f32, 6.0].iter().copied());
        let c3 = ComplexArrow::<Float32Type>::new_from_buffer(buf3, vec![1], Order::RowMajor);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_partial_eq_f64() {
        let buf1 = MutableBuffer::from_iter([1.0f64, 2.0].iter().copied());
        let buf2 = MutableBuffer::from_iter([1.0f64, 2.0].iter().copied());
        let c1 = ComplexArrow::<Float64Type>::new_from_buffer(buf1, vec![1], Order::RowMajor);
        let c2 = ComplexArrow::<Float64Type>::new_from_buffer(buf2, vec![1], Order::RowMajor);
        assert_eq!(c1, c2);

        let buf3 = MutableBuffer::from_iter([3.0f64, 4.0].iter().copied());
        let c3 = ComplexArrow::<Float64Type>::new_from_buffer(buf3, vec![1], Order::RowMajor);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_clone_f32() {
        let buf = MutableBuffer::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().copied());
        let c = ComplexArrow::<Float32Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
        let cloned = c.clone();
        assert_eq!(c, cloned);
    }

    #[test]
    fn test_clone_f64() {
        let buf = MutableBuffer::from_iter([10.0f64, 20.0].iter().copied());
        let c = ComplexArrow::<Float64Type>::new_from_buffer(buf, vec![1], Order::RowMajor);
        let cloned = c.clone();
        assert_eq!(c, cloned);
    }
}
