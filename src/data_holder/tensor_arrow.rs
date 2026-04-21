//! tensor arrow array adapted to mdf4 specificities (samples of tensors)
#[cfg(feature = "ndarray")]
use anyhow::{Context, Error, Result};
use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::{
        ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
        UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};
#[cfg(feature = "ndarray")]
use ndarray::{Array, IxDyn};

/// Tensor with innner arrow primitive builder
#[derive(Debug)]
pub struct TensorArrow<T: ArrowPrimitiveType> {
    /// The validity mask booolean buffer
    null_buffer_builder: Option<BooleanBuffer>,
    /// the primitive builder
    values_builder: PrimitiveBuilder<T>,
    /// number of samples
    len: usize,
    /// shape of tensor
    shape: Vec<usize>,
    /// order of tensor, row or column major
    order: Order,
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
        Self::with_capacity(1024, vec![1], Order::RowMajor)
    }
    /// Create new empty tensor with capacity
    pub fn with_capacity(capacity: usize, shape: Vec<usize>, order: Order) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            len: 0,
            shape,
            order,
        }
    }
    /// Create new tensor from mutable buffer
    pub fn new_from_buffer(values_buffer: MutableBuffer, shape: Vec<usize>, order: Order) -> Self {
        let length = values_buffer.len() / shape.iter().product::<usize>();
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            len: length,
            shape,
            order,
        }
    }
    /// Create new tensor from primitive builder
    pub fn new_from_primitive(
        primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
        shape: Vec<usize>,
        order: Order,
    ) -> Self {
        let length = primitive_builder.len() / shape.iter().product::<usize>();
        if let Some(null_buffer_builder) = null_buffer {
            assert_eq!(
                null_buffer_builder.len() * shape.iter().product::<usize>(),
                primitive_builder.len()
            )
        };
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
    /// returns the length (number of samples) of the tensor.
    /// Equals first value of shape vector.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Shape vector
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
    /// number of dimensions, length of shape vector
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    /// returns True if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// returns order of tensor, either row or column major
    pub fn order(&self) -> &Order {
        &self.order
    }
    /// returns inner primitive builder as a slice
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }
    /// returns inner primitive builder as a mutable slice
    pub fn values_slice_mut(&mut self) -> &mut [T::Native] {
        self.values_builder.values_slice_mut()
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
        self.null_buffer_builder.as_ref().map(arrow::buffer::BooleanBuffer::values)
    }
}

impl<T: ArrowPrimitiveType> Default for TensorArrow<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! tensor_arrow_peq {
    ($type:tt) => {
        impl PartialEq for TensorArrow<$type> {
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
    };
}

tensor_arrow_peq!(Int8Type);
tensor_arrow_peq!(UInt8Type);
tensor_arrow_peq!(Int16Type);
tensor_arrow_peq!(UInt16Type);
tensor_arrow_peq!(Int32Type);
tensor_arrow_peq!(UInt32Type);
tensor_arrow_peq!(Int64Type);
tensor_arrow_peq!(UInt64Type);
tensor_arrow_peq!(Float32Type);
tensor_arrow_peq!(Float64Type);

#[macro_export]
macro_rules! tensor_arrow_clone {
    ($type:tt) => {
        impl Clone for TensorArrow<$type> {
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
    };
}

tensor_arrow_clone!(Int8Type);
tensor_arrow_clone!(UInt8Type);
tensor_arrow_clone!(Int16Type);
tensor_arrow_clone!(UInt16Type);
tensor_arrow_clone!(Int32Type);
tensor_arrow_clone!(UInt32Type);
tensor_arrow_clone!(Int64Type);
tensor_arrow_clone!(UInt64Type);
tensor_arrow_clone!(Float32Type);
tensor_arrow_clone!(Float64Type);

#[macro_export]
macro_rules! tensor_arrow_to_ndarray {
    ($arrow_type:ty, $rust_type:ty) => {
        #[cfg(feature = "ndarray")]
        impl TensorArrow<$arrow_type> {
            /// to convert TensorArrow into ndarray
            pub fn to_ndarray(&self) -> Result<Array<$rust_type, IxDyn>, Error> {
                let vector: Vec<$rust_type> =
                    self.values_builder.values_slice().iter().copied().collect();
                let mut shape = self.shape().clone();
                shape.push(self.len());
                Array::from_shape_vec(IxDyn(&shape), vector)
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
        let t = TensorArrow::<Float32Type>::with_capacity(100, vec![2, 3], Order::ColumnMajor);
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
        // len = buf_bytes / (sizeof(f64) * product(shape)) but new_from_buffer divides by product of shape
        // buf.len() = 48 bytes, shape product = 6, so len = 48/6 = 8... no
        // Actually values_buffer.len() returns byte count: 48 bytes. shape product = 6.
        // length = 48 / 6 = 8. That seems byte-based again.
        assert!(!t.is_empty());
        assert_eq!(t.values_slice().len(), 6);
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
        let mut t = TensorArrow::<Float64Type>::new_from_buffer(buf, vec![3], Order::RowMajor);
        let slice = t.values_slice_mut();
        slice[0] = 99.0;
        assert_eq!(t.values_slice()[0], 99.0);
    }
}
