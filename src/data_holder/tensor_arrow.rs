use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::{
        ArrowPrimitiveType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
        UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
};

/// Complex

#[derive(Debug)]
pub struct TensorArrow<T: ArrowPrimitiveType> {
    null_buffer_builder: Option<BooleanBuffer>,
    values_builder: PrimitiveBuilder<T>,
    len: usize,
    shape: Vec<usize>,
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
    pub fn new() -> Self {
        Self::with_capacity(1024, vec![1], Order::RowMajor)
    }
    pub fn with_capacity(capacity: usize, shape: Vec<usize>, order: Order) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            len: 0,
            shape,
            order,
        }
    }
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
    pub fn values(&mut self) -> &mut PrimitiveBuilder<T> {
        &mut self.values_builder
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn order(&self) -> &Order {
        &self.order
    }
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }
    pub fn values_slice_mut(&mut self) -> &mut [T::Native] {
        self.values_builder.values_slice_mut()
    }
    pub fn nulls(&self) -> Option<&BooleanBuffer> {
        self.null_buffer_builder.as_ref()
    }
    pub fn finish_cloned(&self) -> PrimitiveArray<T> {
        self.values_builder.finish_cloned()
    }
    pub fn finish(&mut self) -> PrimitiveArray<T> {
        self.values_builder.finish()
    }
    pub fn set_validity(&mut self, mask: &mut BooleanBufferBuilder) {
        self.null_buffer_builder = Some(mask.finish());
    }
    pub fn validity_slice(&self) -> Option<&[u8]> {
        self.null_buffer_builder.as_ref().map(|s| s.values())
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