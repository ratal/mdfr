//! complex number stored in primitive builders, fixedsizearraybuilder being too restricted
use crate::data_holder::tensor_arrow::Order;
#[cfg(feature = "ndarray")]
use anyhow::{Context, Error, Result};
#[cfg(feature = "ndarray")]
use arrow::datatypes::{Float32Type, Float64Type};
use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::ArrowPrimitiveType,
};
#[cfg(feature = "ndarray")]
use ndarray::{Array, IxDyn};

/// Complex struct
#[derive(Debug)]
pub struct ComplexArrow<T: ArrowPrimitiveType> {
    /// The validity mask booolean buffer
    null_buffer_builder: Option<BooleanBuffer>,
    /// the primitive builder
    values_builder: PrimitiveBuilder<T>,
    /// the number of real and imaginary pairs in the array
    len: usize,
    /// shape of tensor (without the implicit trailing 2 for real/imag)
    shape: Vec<usize>,
    /// order of tensor, row or column major
    order: Order,
}

/// Complex array implementation
impl<T: ArrowPrimitiveType> ComplexArrow<T> {
    /// Create new empty complex array
    pub fn new() -> Self {
        Self::with_capacity(1024, vec![1], Order::RowMajor)
    }
    /// create new complex array with capacity
    pub fn with_capacity(capacity: usize, shape: Vec<usize>, order: Order) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            len: 0,
            shape,
            order,
        }
    }
    /// create new complex array from a MutableBuffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let length = values_buffer.len() / (shape.iter().product::<usize>() * 2);
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            len: length,
            shape,
            order,
        }
    }
    /// Create a new complex array from a primitive builder and optionally its validity buffer
    pub fn new_from_primitive(
        primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let length = primitive_builder.len() / (shape.iter().product::<usize>() * 2);
        if let Some(null_buffer_builder) = null_buffer {
            assert_eq!(
                null_buffer_builder.len() * shape.iter().product::<usize>() * 2,
                primitive_builder.len()
            );
        }
        let null_buffer_builder = null_buffer.cloned();
        Self {
            null_buffer_builder,
            values_builder: primitive_builder,
            len: length,
            shape,
            order,
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
        self.null_buffer_builder
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

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float32Type> {
    /// to convert ComplexArrow into ndarray
    pub fn to_ndarray(&self) -> Result<Array<f32, IxDyn>, Error> {
        let vector: Vec<f32> = self.values_builder.values_slice().to_vec();
        let mut full_shape = vec![self.len()];
        full_shape.extend(self.shape.iter());
        full_shape.push(2); // complex is always a pair
        Array::from_shape_vec(IxDyn(&full_shape), vector)
            .context("Failed reshaping f32 complex arrow into ndarray")
    }
}

#[cfg(feature = "ndarray")]
impl ComplexArrow<Float64Type> {
    /// to convert ComplexArrow into ndarray
    pub fn to_ndarray(&self) -> Result<Array<f64, IxDyn>, Error> {
        let vector: Vec<f64> = self.values_builder.values_slice().to_vec();
        let mut full_shape = vec![self.len()];
        full_shape.extend(self.shape.iter());
        full_shape.push(2); // complex is always a pair
        Array::from_shape_vec(IxDyn(&full_shape), vector)
            .context("Failed reshaping f64 complex arrow into ndarray")
    }
}

impl<T: ArrowPrimitiveType> Default for ComplexArrow<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType> PartialEq for ComplexArrow<T>
where
    PrimitiveArray<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.values_builder.finish_cloned() == other.values_builder.finish_cloned()
            && self.null_buffer_builder == other.null_buffer_builder
    }
}

impl<T: ArrowPrimitiveType> Clone for ComplexArrow<T> {
    fn clone(&self) -> Self {
        Self {
            null_buffer_builder: self.null_buffer_builder.clone(),
            values_builder: self
                .values_builder
                .finish_cloned()
                .into_builder()
                .expect("failed getting builder from Primitive array"),
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
        let c: ComplexArrow<Float64Type> =
            ComplexArrow::with_capacity(512, vec![1], Order::RowMajor);
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
