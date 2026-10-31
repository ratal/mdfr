//! tensor arrow array adapted to mdf4 specificities (samples of tensors)
#[cfg(feature = "ndarray")]
use anyhow::{Context, Error, Result};
#[allow(unused_imports)]
use arrow::datatypes::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow::{
    array::{BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer, ScalarBuffer},
    datatypes::ArrowPrimitiveType,
};
#[cfg(feature = "ndarray")]
use ndarray::{Array as NdArray, IxDyn};
use std::mem::size_of;

/// Immutable Tensor wrapping an Arrow [`PrimitiveArray`]
#[derive(Debug)]
pub struct TensorArrow<T: ArrowPrimitiveType> {
    /// The validity mask boolean buffer
    pub(crate) null_buffer: Option<BooleanBuffer>,
    /// The immutable Arrow primitive array
    pub(crate) values: PrimitiveArray<T>,
    /// Number of samples
    pub(crate) len: usize,
    /// Shape of tensor
    pub(crate) shape: Vec<usize>,
    /// Order of tensor, row or column major
    pub(crate) order: Order,
}

/// Mutable Tensor builder for staging tensor arrays during reading
#[derive(Debug)]
pub struct TensorBuilder<T: ArrowPrimitiveType> {
    pub(crate) null_buffer_builder: Option<BooleanBuffer>,
    pub(crate) values_builder: PrimitiveBuilder<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) order: Order,
}

/// Order of the array, Row or Column Major (first)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Order {
    #[default]
    RowMajor,
    ColumnMajor,
}

impl<T: ArrowPrimitiveType> TensorArrow<T> {
    /// Create new empty tensor
    pub fn new() -> Self {
        Self {
            null_buffer: None,
            values: PrimitiveArray::<T>::new(ScalarBuffer::default(), None),
            len: 0,
            shape: vec![1],
            order: Order::RowMajor,
        }
    }

    /// Create new tensor from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let elem_count = shape.iter().product::<usize>();
        let num_elements = values_buffer.len() / size_of::<T::Native>();
        let length = num_elements.checked_div(elem_count).unwrap_or(0);
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

    /// Create a new tensor from a primitive array and optionally its validity buffer
    pub fn new_from_array(
        values: PrimitiveArray<T>,
        null_buffer: Option<BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let elem_count = shape.iter().product::<usize>();
        let length = values.len().checked_div(elem_count).unwrap_or(0);
        Self {
            null_buffer,
            values,
            len: length,
            shape,
            order,
        }
    }

    /// Create a new tensor from a primitive builder (for compatibility)
    #[allow(dead_code)]
    pub fn new_from_primitive(
        mut primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let values = primitive_builder.finish();
        let elem_count = shape.iter().product::<usize>();
        let length = values.len().checked_div(elem_count).unwrap_or(0);
        Self {
            null_buffer: null_buffer.cloned(),
            values,
            len: length,
            shape,
            order,
        }
    }

    /// Slices the tensor array with zero-copy
    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let elem_count = self.shape.iter().product::<usize>();
        let values_slice = self.values.slice(offset * elem_count, length * elem_count);
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

    /// returns the shape of the tensor
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// number of samples
    pub fn len(&self) -> usize {
        self.len
    }

    /// number of dimensions, length of shape vector
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// returns True if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// returns order of tensor, either row or column major
    pub fn order(&self) -> &Order {
        &self.order
    }

    /// returns inner primitive array as a slice
    pub fn values_slice(&self) -> &[T::Native] {
        self.values.values()
    }

    /// returns the validity array
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
}

impl<T: ArrowPrimitiveType> TensorBuilder<T> {
    /// Create new empty tensor builder
    pub fn new() -> Self {
        Self::with_capacity(1024, vec![1], Order::RowMajor)
    }

    /// Create new tensor builder with capacity
    pub fn with_capacity(capacity: usize, shape: Vec<usize>, order: Order) -> Self {
        let elem_count = shape.iter().product::<usize>().max(1);
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * elem_count),
            shape,
            order,
        }
    }

    /// Create new tensor builder from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            shape,
            order,
        }
    }

    /// returns inner primitive builder as a mutable slice
    pub fn values_slice_mut(&mut self) -> &mut [T::Native] {
        self.values_builder.values_slice_mut()
    }

    /// returns inner primitive builder as a slice
    #[allow(dead_code)]
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }

    /// finishes building and converts to an immutable TensorArrow (0 copies)
    pub fn finish(mut self) -> TensorArrow<T> {
        let values = self.values_builder.finish();
        let prod = self.shape.iter().product::<usize>();
        let len = values.len().checked_div(prod).unwrap_or(0);
        TensorArrow {
            null_buffer: self.null_buffer_builder,
            values,
            len,
            shape: self.shape,
            order: self.order,
        }
    }
}

impl<T: ArrowPrimitiveType> Default for TensorArrow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType> Default for TensorBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType> PartialEq for TensorArrow<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
            && self.null_buffer == other.null_buffer
            && self.shape == other.shape
            && self.order == other.order
    }
}

impl<T: ArrowPrimitiveType> Clone for TensorArrow<T> {
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

#[macro_export]
macro_rules! tensor_arrow_to_ndarray {
    ($arrow_type:ty, $rust_type:ty) => {
        #[cfg(feature = "ndarray")]
        impl TensorArrow<$arrow_type> {
            /// to convert TensorArrow into ndarray
            #[allow(dead_code)]
            pub fn to_ndarray(&self) -> Result<NdArray<$rust_type, IxDyn>, Error> {
                let vector: Vec<$rust_type> = self.values.values().iter().copied().collect();
                let mut shape = self.shape().clone();
                shape.push(self.len());
                NdArray::from_shape_vec(IxDyn(&shape), vector)
                    .context("Failed reshaping tensor arrow into ndarray")
            }
        }
    };
}

tensor_arrow_to_ndarray!(UInt8Type, u8);
tensor_arrow_to_ndarray!(Int8Type, i8);
tensor_arrow_to_ndarray!(Int16Type, i16);
tensor_arrow_to_ndarray!(UInt16Type, u16);
tensor_arrow_to_ndarray!(Int32Type, i32);
tensor_arrow_to_ndarray!(UInt32Type, u32);
tensor_arrow_to_ndarray!(Int64Type, i64);
tensor_arrow_to_ndarray!(UInt64Type, u64);
tensor_arrow_to_ndarray!(Float32Type, f32);
tensor_arrow_to_ndarray!(Float64Type, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_default() {
        let t: TensorArrow<Float64Type> = TensorArrow::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.shape(), &vec![1]);
        assert_eq!(t.order(), &Order::RowMajor);
        assert!(t.nulls().is_none());

        let d: TensorArrow<Int32Type> = TensorArrow::default();
        assert!(d.is_empty());
        assert_eq!(d.shape(), &vec![1]);
    }

    #[test]
    fn test_with_capacity() {
        let tb = TensorBuilder::<Float32Type>::with_capacity(100, vec![2, 3], Order::ColumnMajor);
        let t = tb.finish();
        assert!(t.is_empty());
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.shape(), &vec![2, 3]);
        assert_eq!(t.order(), &Order::ColumnMajor);
    }

    #[test]
    fn test_from_buffer() {
        // 6 f64 values with shape [2,3] = 1 sample of 2x3 tensor
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = MutableBuffer::from_iter(data.iter().copied());
        let t = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![2, 3], Order::RowMajor);
        assert!(!t.is_empty());
        assert_eq!(t.values_slice().len(), 6);
        assert_eq!(t.len(), 1);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.shape(), &vec![2, 3]);
    }

    #[test]
    fn test_from_primitive() {
        let mut pb = PrimitiveBuilder::<Float32Type>::with_capacity(12);
        for i in 0..12 {
            pb.append_value(i as f32);
        }
        let t = TensorArrow::new_from_primitive(pb, None, vec![3, 4], Order::RowMajor);
        assert_eq!(t.len(), 1); // 12 / (3*4) = 1 sample
        assert!(!t.is_empty());
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.shape(), &vec![3, 4]);
        assert_eq!(t.order(), &Order::RowMajor);
        assert!(t.nulls().is_none());
        assert!(t.validity_slice().is_none());
    }

    #[test]
    fn test_finish_cloned() {
        let mut pb = PrimitiveBuilder::<Float64Type>::with_capacity(4);
        pb.append_value(1.0);
        pb.append_value(2.0);
        pb.append_value(3.0);
        pb.append_value(4.0);
        let t = TensorArrow::new_from_primitive(pb, None, vec![2, 2], Order::RowMajor);
        let arr = t.finish_cloned();
        assert_eq!(arr.len(), 4);
    }

    #[test]
    fn test_partial_eq() {
        let buf1 = MutableBuffer::from_iter([1.0f64, 2.0, 3.0].iter().copied());
        let buf2 = MutableBuffer::from_iter([1.0f64, 2.0, 3.0].iter().copied());
        let t1 = TensorArrow::<Float64Type>::new_from_buffer(buf1, vec![3], Order::RowMajor);
        let t2 = TensorArrow::<Float64Type>::new_from_buffer(buf2, vec![3], Order::RowMajor);
        assert_eq!(t1, t2);

        let buf3 = MutableBuffer::from_iter([4.0f64, 5.0, 6.0].iter().copied());
        let t3 = TensorArrow::<Float64Type>::new_from_buffer(buf3, vec![3], Order::RowMajor);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_clone() {
        let buf = MutableBuffer::from_iter([1.0f64, 2.0, 3.0, 4.0].iter().copied());
        let t = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![2, 2], Order::ColumnMajor);
        let cloned = t.clone();
        assert_eq!(t, cloned);
        assert_eq!(cloned.shape(), &vec![2, 2]);
        assert_eq!(cloned.order(), &Order::ColumnMajor);
    }

    #[test]
    fn test_values_slice_mut() {
        let buf = MutableBuffer::from_iter([1.0f64, 2.0, 3.0].iter().copied());
        let mut tb = TensorBuilder::<Float64Type>::new_from_buffer(buf, vec![3], Order::RowMajor);
        let slice = tb.values_slice_mut();
        slice[0] = 99.0;
        let t = tb.finish();
        assert_eq!(t.values_slice()[0], 99.0);
    }
}
