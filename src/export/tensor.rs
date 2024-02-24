//! Arrow tensor, not official implementation
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
use std::marker::PhantomData;
use std::mem;

use arrow::array::Array;
use arrow::buffer::Buffer;
use arrow::datatypes::*;

use arrow::error::Result;

/// Computes the strides required assuming a row major memory layout
fn compute_row_major_strides(shape: &[usize]) -> Result<Vec<usize>> {
    let mut total_locations = shape.iter().product();

    Ok(shape
        .iter()
        .map(|val| {
            total_locations /= *val;
            total_locations
        })
        .collect())
}

/// Computes the strides required assuming a column major memory layout
fn compute_column_major_strides<T>(shape: &[usize]) -> Result<Vec<usize>> {
    let mut remaining_bytes = mem::size_of::<T>();
    let mut strides = Vec::<usize>::new();

    for i in shape {
        strides.push(remaining_bytes);

        if let Some(val) = remaining_bytes.checked_mul(*i) {
            remaining_bytes = val;
        } else {
            return Err(Error::Overflow);
        }
    }

    Ok(strides)
}

/// Tensor of primitive types
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: NativeType> {
    data_type: DataType,
    values: Buffer<T>,
    shape: Vec<usize>,
    order: Order,
    strides: Option<Vec<usize>>,
    names: Option<Vec<String>>,
    _marker: PhantomData<T>,
}

/// Order of the array, Row or Column Major (first)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Order {
    #[default]
    RowMajor,
    ColumnMajor,
}

#[allow(dead_code)]
impl<T: NativeType> Tensor<T> {
    /// Creates a new `Tensor`
    pub fn try_new(
        data_type: DataType,
        values: Buffer<T>,
        shape: Option<Vec<usize>>,
        order: Option<Order>,
        strides: Option<Vec<usize>>,
        names: Option<Vec<String>>,
    ) -> Result<Self> {
        match shape {
            None => {
                if strides.is_some() {
                    return Err(Error::InvalidArgumentError(
                        "expected None strides for tensor with no shape".to_string(),
                    ));
                }

                if names.is_some() {
                    return Err(Error::InvalidArgumentError(
                        "expected None names for tensor with no shape".to_string(),
                    ));
                }
            }

            Some(ref s) => {
                if let Some(ref st) = strides {
                    if st.len() != s.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "shape {} and stride {} dimensions differ",
                            s.len(),
                            st.len()
                        )));
                    }
                }

                if let Some(ref n) = names {
                    if n.len() != s.len() {
                        return Err(Error::InvalidArgumentError(format!(
                            "number of dimensions {} and number of dimension names differ {}",
                            s.len(),
                            n.len()
                        )));
                    }
                }

                let total_elements: usize = s.iter().product();
                if total_elements != values.len() {
                    return Err(Error::InvalidArgumentError(format!(
                        "number of elements {} in buffer does not match dimensions {}",
                        total_elements,
                        values.len()
                    )));
                }
            }
        };

        // Checking that the tensor strides used for construction are correct
        // otherwise a row major stride is calculated and used as value for the tensor
        let tensor_strides = {
            if let Some(st) = strides {
                if let Some(ref s) = shape {
                    if compute_row_major_strides(s)? == st
                        || compute_column_major_strides::<T>(s)? == st
                    {
                        Some(st)
                    } else {
                        return Err(Error::InvalidArgumentError(
                            "the input stride does not match the selected shape".to_string(),
                        ));
                    }
                } else {
                    Some(st)
                }
            } else if let Some(ref s) = shape {
                match order {
                    Some(Order::RowMajor) => Some(compute_row_major_strides(s)?),
                    Some(Order::ColumnMajor) => Some(compute_column_major_strides::<T>(s)?),
                    None => Some(compute_row_major_strides(s)?),
                }
            } else {
                None
            }
        };

        let order = order.unwrap_or_default();

        let shape = shape.unwrap_or_default();

        Ok(Self {
            data_type,
            values,
            shape,
            order,
            strides: tensor_strides,
            names,
            _marker: PhantomData,
        })
    }

    /// Creates a new Tensor using row major memory layout
    pub fn new_row_major(
        data_type: DataType,
        values: Buffer<T>,
        shape: Option<Vec<usize>>,
        names: Option<Vec<String>>,
    ) -> Result<Self> {
        if let Some(ref s) = shape {
            let strides = Some(compute_row_major_strides(s)?);

            Self::try_new(
                data_type,
                values,
                shape,
                Some(Order::RowMajor),
                strides,
                names,
            )
        } else {
            Err(Error::InvalidArgumentError(
                "shape required to create row major tensor".to_string(),
            ))
        }
    }

    /// Creates a new Tensor using column major memory layout
    pub fn new_column_major(
        data_type: DataType,
        values: Buffer<T>,
        shape: Option<Vec<usize>>,
        names: Option<Vec<String>>,
    ) -> Result<Self> {
        if let Some(ref s) = shape {
            let strides = Some(compute_column_major_strides::<T>(s)?);

            Self::try_new(
                data_type,
                values,
                shape,
                Some(Order::ColumnMajor),
                strides,
                names,
            )
        } else {
            Err(Error::InvalidArgumentError(
                "shape required to create column major tensor".to_string(),
            ))
        }
    }

    /// Returns a clone of this PrimitiveArray sliced by an offset and length.
    /// # Implementation
    /// This operation is `O(1)` as it amounts to increase two ref counts.
    /// # Panic
    /// This function panics iff `offset + length > self.len()`.
    #[inline]
    pub fn slice(&mut self, offset: usize, length: usize) {
        assert!(
            offset + length <= self.len(),
            "offset + length may not exceed length of array"
        );
        unsafe { self.values.slice_unchecked(offset, length) }
    }

    /// Returns a clone of this PrimitiveArray sliced by an offset and length.
    /// # Implementation
    /// This operation is `O(1)` as it amounts to increase two ref counts.
    /// # Safety
    /// The caller must ensure that `offset + length <= self.len()`.
    #[inline]
    pub unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.values.slice_unchecked(offset, length);
    }

    #[must_use]
    pub fn with_validity(&self, _validity: Option<Bitmap>) -> Self {
        self.clone()
    }

    /// The sizes of the dimensions
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn order(&self) -> &Order {
        &self.order
    }

    /// Returns the arrays' [`DataType`].
    #[inline]
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Returns a reference to the underlying `Buffer`
    #[inline]
    pub fn values_buffer(&self) -> &Buffer<T> {
        &self.values
    }

    /// Returns a reference to the underlying `Buffer`
    #[inline]
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }

    /// Update the values of this [`Tensor`].
    /// # Panics
    /// This function panics iff `values.len() != self.len()`.
    pub fn set_values(&mut self, values: Buffer<T>) {
        assert_eq!(
            values.len(),
            self.len(),
            "values' length must be equal to this arrays' length"
        );
        self.values = values;
    }

    /// Returns an option of a mutable reference to the values of this [`PrimitiveArray`].
    pub fn get_mut_values(&mut self) -> Option<&mut [T]> {
        self.values.get_mut_slice()
    }

    /// Returns a reference to a value in the buffer
    /// The value is accessed via an index of the same size
    /// as the strides vector
    ///
    /// # Examples
    /// Creating a 2x2 vector and accessing the (1,0) element in the tensor
    /// ```
    /// use arrow2::buffer::Buffer;
    /// let buffer: Buffer<i32> = vec![0i32, 1, 2, 3].into();
    /// let shape = Some(vec![2, 2]);
    /// let tensor = Tensor::try_new(buffer, shape, None, None).unwrap();
    /// assert_eq!(tensor.value(&[1, 0]), Some(&2));
    /// ```
    pub fn value(&self, index: &[usize]) -> Option<&T> {
        // Check if this tensor has strides. The 1x1 doesn't
        // have strides that define the tensor
        match self.strides.as_ref() {
            None => Some(&self.values()[0]),
            Some(strides) => {
                // If the index doesn't have the same len as
                // the strides vector then a None is returned
                if index.len() != strides.len() {
                    return None;
                }

                // The index in the buffer is calculated using the
                // row strides.
                // index = sum(index[i] * stride[i])
                let buf_index = strides
                    .iter()
                    .zip(index)
                    .fold(0usize, |acc, (s, i)| acc + (s * i));

                Some(&self.values()[buf_index])
            }
        }
    }

    /// The number of bytes between elements in each dimension
    pub fn strides(&self) -> Option<&Vec<usize>> {
        self.strides.as_ref()
    }

    /// The names of the dimensions
    pub fn names(&self) -> Option<&Vec<String>> {
        self.names.as_ref()
    }

    /// The number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The name of dimension i
    pub fn dim_name(&self, i: usize) -> Option<String> {
        self.names.as_ref().map(|names| names[i].clone())
    }

    /// The total number of elements in the `Tensor`
    pub fn size(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Boxes self into a [`Box<dyn Array>`].
    pub fn boxed(self) -> Box<dyn Array> {
        Box::new(self)
    }

    /// Boxes self into a [`std::sync::Arc<dyn Array>`].
    pub fn arced(self) -> std::sync::Arc<dyn Array> {
        std::sync::Arc::new(self)
    }

    /// Indicates if the data is laid out contiguously in memory
    pub fn is_contiguous(&self) -> bool {
        self.is_row_major() || self.is_column_major()
    }

    /// Indicates if the memory layout row major
    pub fn is_row_major(&self) -> bool {
        match self.order {
            Order::RowMajor => true,
            Order::ColumnMajor => false,
        }
    }

    /// Indicates if the memory layout column major
    pub fn is_column_major(&self) -> bool {
        match self.order {
            Order::RowMajor => false,
            Order::ColumnMajor => true,
        }
    }

    /// Creates a (non-null) [`Tensor`] from a vector of values.
    /// This function is `O(1)`.
    pub fn from_vec(
        values: Vec<T>,
        shape: Option<Vec<usize>>,
        order: Option<Order>,
        strides: Option<Vec<usize>>,
        names: Option<Vec<String>>,
    ) -> Self {
        Self::new(
            T::PRIMITIVE.into(),
            values.into(),
            shape,
            order,
            strides,
            names,
        )
    }

    /// Alias for `Self::try_new(..).unwrap()`.
    /// # Panics
    /// This function errors iff:
    /// * The validity is not `None` and its length is different from `values`'s length
    /// * The `data_type`'s [`PhysicalType`] is not equal to [`PhysicalType::Primitive`].
    pub fn new(
        data_type: DataType,
        values: Buffer<T>,
        shape: Option<Vec<usize>>,
        order: Option<Order>,
        strides: Option<Vec<usize>>,
        names: Option<Vec<String>>,
    ) -> Self {
        Self::try_new(data_type, values, shape, order, strides, names).unwrap()
    }
}

impl<T: NativeType> Array for Tensor<T> {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn data_type(&self) -> &DataType {
        self.data_type()
    }

    fn validity(&self) -> Option<&Bitmap> {
        None
    }

    fn slice(&mut self, offset: usize, length: usize) {
        self.values.slice(offset, length)
    }
    unsafe fn slice_unchecked(&mut self, offset: usize, length: usize) {
        self.values.slice_unchecked(offset, length)
    }
    fn with_validity(&self, validity: Option<Bitmap>) -> Box<dyn Array> {
        Box::new(self.with_validity(validity))
    }
    fn to_boxed(&self) -> Box<dyn Array> {
        Box::new(self.clone())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn null_count(&self) -> usize {
        if self.data_type() == &DataType::Null {
            return self.len();
        };
        self.validity()
            .as_ref()
            .map(|x| x.unset_bits())
            .unwrap_or(0)
    }

    fn is_null(&self, i: usize) -> bool {
        self.validity()
            .as_ref()
            .map(|x| !x.get_bit(i))
            .unwrap_or(false)
    }

    fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }
}

/// Cast [`Tensor`] as a [`Tensor`]
/// Same as `number as to_number_type` in rust
pub fn tensor_as_tensor<I, O>(from: &Tensor<I>, to_type: &DataType) -> Tensor<O>
where
    I: NativeType + num_traits::AsPrimitive<O>,
    O: NativeType,
{
    unary(from, num_traits::AsPrimitive::<O>::as_, to_type.clone())
}

/// Applies an unary and infallible function to a [`Tensor`]. This is the
/// fastest way to perform an operation on a [`Tensor`] when the benefits
/// of a vectorized operation outweighs the cost of branching nulls and
/// non-nulls.
///
/// # Implementation
/// This will apply the function for all values, including those on null slots.
/// This implies that the operation must be infallible for any value of the
/// corresponding type or this function may panic.
#[inline]
pub fn unary<I, F, O>(array: &Tensor<I>, op: F, data_type: DataType) -> Tensor<O>
where
    I: NativeType,
    O: NativeType,
    F: Fn(I) -> O,
{
    let values = array.values().iter().map(|v| op(*v)).collect::<Vec<_>>();

    Tensor::<O>::new(
        data_type,
        values.into(),
        Some(array.shape().clone()),
        Some(array.order().clone()),
        array.strides().cloned(),
        array.names().cloned(),
    )
}

/// Applies an unary function to a [`Tensor`], optionally in-place.
///
/// # Implementation
/// This function tries to apply the function directly to the values of the array.
/// If that region is shared, this function creates a new region and writes to it.
///
/// # Panics
/// This function panics iff
/// * the arrays have a different length.
/// * the function itself panics.
#[inline]
pub fn unary_assign<I, F>(array: &mut Tensor<I>, op: F)
where
    I: NativeType,
    F: Fn(I) -> I,
{
    if let Some(values) = array.get_mut_values() {
        // mutate in place
        values.iter_mut().for_each(|l| *l = op(*l));
    } else {
        // alloc and write to new region
        let values = array.values().iter().map(|l| op(*l)).collect::<Vec<_>>();
        array.set_values(values.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow2::buffer::Buffer;

    #[test]
    fn test_compute_row_major_strides() {
        assert_eq!(
            vec![6_usize, 1],
            compute_row_major_strides(&[4_usize, 6]).unwrap()
        );
        assert_eq!(
            vec![6_usize, 1],
            compute_row_major_strides(&[4_usize, 6]).unwrap()
        );
        assert_eq!(
            vec![6_usize, 1],
            compute_row_major_strides(&[4_usize, 6]).unwrap()
        );
    }

    #[test]
    fn test_compute_column_major_strides() {
        assert_eq!(
            vec![8_usize, 32],
            compute_column_major_strides::<i64>(&[4_usize, 6]).unwrap()
        );
        assert_eq!(
            vec![4_usize, 16],
            compute_column_major_strides::<i32>(&[4_usize, 6]).unwrap()
        );
        assert_eq!(
            vec![1_usize, 4],
            compute_column_major_strides::<i8>(&[4_usize, 6]).unwrap()
        );
    }

    #[test]
    fn test_zero_dim() {
        let buf = Buffer::<u8>::from(vec![1]);
        let tensor = Tensor::<u8>::try_new(DataType::UInt8, buf, None, None, None, None).unwrap();
        assert_eq!(0, tensor.size());
        assert_eq!(&Vec::<usize>::new(), tensor.shape());
        assert_eq!(None, tensor.names());
        assert_eq!(0, tensor.ndim());
        assert!(tensor.is_row_major());
        assert!(!tensor.is_column_major());
        assert!(tensor.is_contiguous());

        let buf = Buffer::<i32>::from(vec![1, 2, 2, 2]);
        let tensor = Tensor::<i32>::try_new(DataType::Int32, buf, None, None, None, None).unwrap();
        assert_eq!(0, tensor.size());
        assert_eq!(&Vec::<usize>::new(), tensor.shape());
        assert_eq!(None, tensor.names());
        assert_eq!(0, tensor.ndim());
        assert!(tensor.is_row_major());
        assert!(!tensor.is_column_major());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let tensor =
            Tensor::<i32>::try_new(DataType::Int32, buf, Some(vec![2, 8]), None, None, None)
                .unwrap();
        assert_eq!(16, tensor.size());
        assert_eq!(&vec![2_usize, 8], tensor.shape());
        assert_eq!(Some(vec![8_usize, 1]).as_ref(), tensor.strides());
        assert_eq!(2, tensor.ndim());
        assert_eq!(None, tensor.names());
    }

    #[test]
    fn test_new_row_major() {
        let buf: Buffer<i32> =
            Vec::<i32>::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).into();
        let tensor =
            Tensor::<i32>::new_row_major(DataType::Int32, buf, Some(vec![2, 8]), None).unwrap();
        assert_eq!(16, tensor.size());
        assert_eq!(&vec![2_usize, 8], tensor.shape());
        assert_eq!(Some(vec![8_usize, 1]).as_ref(), tensor.strides());
        assert_eq!(None, tensor.names());
        assert_eq!(2, tensor.ndim());
        assert!(tensor.is_row_major());
        assert!(!tensor.is_column_major());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_new_column_major() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let tensor =
            Tensor::<i32>::new_column_major(DataType::Int32, buf, Some(vec![2, 8]), None).unwrap();
        assert_eq!(16, tensor.size());
        assert_eq!(&vec![2_usize, 8], tensor.shape());
        assert_eq!(Some(vec![4_usize, 8]).as_ref(), tensor.strides());
        assert_eq!(None, tensor.names());
        assert_eq!(2, tensor.ndim());
        assert!(!tensor.is_row_major());
        assert!(tensor.is_column_major());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_with_names() {
        let buf = Buffer::<i64>::from(vec![0, 1, 2, 3, 4, 5, 6, 7]);
        let names = vec!["Dim 1".to_owned(), "Dim 2".to_owned()];
        let tensor =
            Tensor::<i64>::new_column_major(DataType::Int64, buf, Some(vec![2, 4]), Some(names))
                .unwrap();
        assert_eq!(8, tensor.size());
        assert_eq!(&vec![2_usize, 4], tensor.shape());
        assert_eq!(Some(vec![8_usize, 16]).as_ref(), tensor.strides());
        assert_eq!("Dim 1", tensor.dim_name(0).unwrap());
        assert_eq!("Dim 2", tensor.dim_name(1).unwrap());
        assert_eq!(2, tensor.ndim());
        assert!(!tensor.is_row_major());
        assert!(tensor.is_column_major());
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_inconsistent_strides() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = Tensor::<i32>::try_new(
            DataType::Int32,
            buf,
            Some(vec![2, 8]),
            None,
            Some(vec![2, 8, 1]),
            None,
        );

        if result.is_ok() {
            panic!("shape and stride dimensions are different")
        }
    }

    #[test]
    fn test_inconsistent_names() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = Tensor::<i32>::try_new(
            DataType::Int32,
            buf,
            Some(vec![2, 8]),
            None,
            Some(vec![4, 8]),
            Some(vec!["1".to_owned(), "2".to_owned(), "3".to_owned()]),
        );

        if result.is_ok() {
            panic!("dimensions and names have different shape")
        }
    }

    #[test]
    fn test_incorrect_shape() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result =
            Tensor::<i32>::try_new(DataType::Int32, buf, Some(vec![2, 6]), None, None, None);

        if result.is_ok() {
            panic!("number of elements does not match for the shape")
        }
    }

    #[test]
    fn test_incorrect_stride() {
        let buf = Buffer::<i32>::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = Tensor::<i32>::try_new(
            DataType::Int32,
            buf,
            Some(vec![2, 8]),
            None,
            Some(vec![30, 4]),
            None,
        );

        if result.is_ok() {
            panic!("the input stride does not match the selected shape")
        }
    }
    #[test]
    fn test_data() {
        let buffer: Buffer<i32> = vec![0i32, 1, 2, 3].into();
        let shape = Some(vec![2, 2]);

        let tensor = Tensor::try_new(DataType::Int32, buffer, shape, None, None, None).unwrap();
        let mut data = tensor.values().iter();

        assert_eq!(data.next(), Some(&0));
        assert_eq!(data.next(), Some(&1));
        assert_eq!(data.next(), Some(&2));
        assert_eq!(data.next(), Some(&3));
    }

    #[test]
    fn test_access_data() {
        let buffer: Buffer<i32> = vec![0i32, 1, 2, 3].into();
        let shape = Some(vec![2, 2]);

        let tensor = Tensor::try_new(DataType::Int32, buffer, shape, None, None, None).unwrap();
        assert_eq!(tensor.value(&[0, 0]), Some(&0));
        assert_eq!(tensor.value(&[0, 1]), Some(&1));
        assert_eq!(tensor.value(&[1, 0]), Some(&2));
        assert_eq!(tensor.value(&[1, 1]), Some(&3));
    }
}
