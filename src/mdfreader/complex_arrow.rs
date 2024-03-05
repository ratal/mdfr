use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::{ArrowPrimitiveType, DataType, Float32Type, Float64Type},
};

/// Complex

#[derive(Debug)]
pub struct ComplexArrow<T: ArrowPrimitiveType> {
    null_buffer_builder: Option<BooleanBuffer>,
    values_builder: PrimitiveBuilder<T>,
    data_type: DataType,
    len: usize,
}

impl<T: ArrowPrimitiveType> ComplexArrow<T> {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            data_type: T::DATA_TYPE,
            len: 0,
        }
    }
    pub fn new_from_buffer(values_buffer: MutableBuffer) -> Self {
        let length = values_buffer.len() / 2;
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            data_type: T::DATA_TYPE,
            len: length,
        }
    }
    pub fn new_from_primitive(
        primitive_builder: PrimitiveBuilder<T>,
        null_buffer: Option<&BooleanBuffer>,
    ) -> Self {
        let length = primitive_builder.len() / 2;
        match null_buffer {
            Some(null_buffer_builder) => {
                assert_eq!(null_buffer_builder.len() * 2, primitive_builder.len())
            }
            None => {}
        };
        let null_buffer_builder = null_buffer.map(|buffer| buffer.clone());
        Self {
            null_buffer_builder,
            values_builder: primitive_builder,
            data_type: T::DATA_TYPE,
            len: length,
        }
    }
    pub fn values(&mut self) -> &mut PrimitiveBuilder<T> {
        &mut self.values_builder
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn data_type(&self) -> DataType {
        self.data_type.clone()
    }
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
    }
    /// Returns the current values buffer as a mutable slice
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
                None => {
                    if other.null_buffer_builder.is_none() {
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        }
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
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
                None => {
                    if other.null_buffer_builder.is_none() {
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        }
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
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
            data_type: self.data_type.clone(),
            len: self.len.clone(),
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
            data_type: self.data_type.clone(),
            len: self.len.clone(),
        }
    }
}
