use arrow::{
    array::{ArrayBuilder, BooleanBufferBuilder, PrimitiveArray, PrimitiveBuilder},
    buffer::{BooleanBuffer, MutableBuffer},
    datatypes::{ArrowPrimitiveType, Float32Type, Float64Type},
};

/// Complex struct
#[derive(Debug)]
pub struct ComplexArrow<T: ArrowPrimitiveType> {
    null_buffer_builder: Option<BooleanBuffer>,
    values_builder: PrimitiveBuilder<T>,
    len: usize,
}

// Complex implementation
impl<T: ArrowPrimitiveType> ComplexArrow<T> {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            null_buffer_builder: None,
            values_builder: PrimitiveBuilder::with_capacity(capacity * 2),
            len: 0,
        }
    }
    pub fn new_from_buffer(values_buffer: MutableBuffer) -> Self {
        let length = values_buffer.len() / 2;
        let values_builder = PrimitiveBuilder::new_from_buffer(values_buffer, None);
        Self {
            null_buffer_builder: None,
            values_builder,
            len: length,
        }
    }
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
    pub fn values(&mut self) -> &mut PrimitiveBuilder<T> {
        &mut self.values_builder
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn values_slice(&self) -> &[T::Native] {
        self.values_builder.values_slice()
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
