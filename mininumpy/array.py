# mininumpy/array.py

import math
import operator
import builtins
from functools import reduce

"""!
@brief The list of public symbols exported by the mininumpy module.
These symbols are available when using `from mininumpy import *`.
"""
__all__ = [
    # Classes
    'Array',      #<! The core N-dimensional array class.

    # Array Creation
    'array',      #<! Create an array from a list or nested list.
    'zeros',      #<! Create a new array of given shape, filled with zeros.
    'ones',       #<! Create a new array of given shape, filled with ones.
    'eye',        #<! Return a 2-D array with ones on the diagonal and zeros elsewhere.
    'arange',     #<! Return evenly spaced values within a given interval.
    'linspace',   #<! Return evenly spaced numbers over a specified interval.

    # Math Functions
    'exp',        #<! Calculate the exponential of all elements in the input array.
    'log',        #<! Natural logarithm, element-wise.
    'sqrt',       #<! Return the non-negative square-root of an array, element-wise.
    'abs',        #<! Calculate the absolute value element-wise.

    # Reductions
    'sum',        #<! Sum of all array elements.
    'mean',       #<! Compute the arithmetic mean of all elements.
    'min',        #<! Return the minimum of all elements.
    'max',        #<! Return the maximum of all elements.
    'argmin',     #<! Return the flat index of the first minimum.
    'argmax'      #<! Return the flat index of the first maximum.
]

class Array:
    """!
    @brief A simplified multi-dimensional array class, similar to NumPy's ndarray.

    This class wraps a Python list (or nested lists) and provides methods for
    basic array manipulation, such as shape, transpose, and elementwise operations.
    """

    def __init__(self, data):
        """!
        @brief Constructor for the Array class.

        Initializes an array from a list or nested list. It validates the data,
        determines the shape, and calculates the total size.

        @param data A Python list or nested list.
        @exception TypeError if data is not a list.
        @exception ValueError if rows have inconsistent lengths.
        """
        if not isinstance(data, list):
            raise TypeError("Input data must be a list or nested list.")

        self.data = data
        self.shape = self._derive_shape(data)
        self.ndim = len(self.shape)
        self.size = reduce(operator.mul, self.shape, 1)

        self._validate_shape(self.data, self.shape, 0)

    def _derive_shape(self, data):
        """!
        @brief Recursively determines the shape of the nested list data.
        @param data The list or nested list.
        @return A tuple representing the shape.
        """
        if not isinstance(data, list):
            return ()
        if not data:
            return (0,)
        return (len(data),) + self._derive_shape(data[0])

    def _validate_shape(self, data, shape, level):
        """!
        @brief Recursively validates that all sub-arrays at a certain depth have the same length.
        @param data The data to validate.
        @param shape The target shape.
        @param level The current recursion depth.
        @exception ValueError if shapes are inconsistent.
        """
        if level >= len(shape) - 1:
            return
        if len(data) != shape[level]:
            raise ValueError("Inconsistent shape in input data.")
        expected_len = shape[level + 1]
        for item in data:
            if not isinstance(item, list) or len(item) != expected_len:
                raise ValueError("All rows must have the same length.")
            if level + 1 < len(shape) - 1:
                self._validate_shape(item, shape, level + 1)

    def __str__(self):
        """!
        @brief Provides a user-friendly string representation of the array.
        @return A string that pretty-prints the array.
        """
        return f"Array({self.data},\n      shape={self.shape})"

    def __repr__(self):
        """!
        @brief Provides an unambiguous string representation of the array object.
        @return A string representation of the object.
        """
        return str(self)

    def reshape(self, new_shape):
        """!
        @brief Returns a new Array with a different shape, without changing its data.

        The new shape must be compatible with the original size.

        @param new_shape A tuple of ints giving the new shape.
        @return A new Array object with the specified shape.
        @exception ValueError if the new shape is not compatible with the array's size.
        """
        new_size = reduce(operator.mul, new_shape, 1)
        if new_size != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}")

        flat_list = self.flatten()

        def _build_reshaped(lst, shape):
            if not shape:
                return lst[0]
            n = shape[0]
            sub_len = len(lst) // n
            return [_build_reshaped(lst[i * sub_len:(i + 1) * sub_len], shape[1:]) for i in range(n)]

        return Array(_build_reshaped(flat_list, new_shape))

    def flatten(self):
        """!
        @brief Flattens the multi-dimensional data into a 1D list.
        @return A 1D list containing all elements of the array.
        """
        flat_list = []

        def _recursive_flatten(data):
            for item in data:
                if isinstance(item, list):
                    _recursive_flatten(item)
                else:
                    flat_list.append(item)

        _recursive_flatten(self.data)
        return flat_list

    def transpose(self):
        """!
        @brief Transposes the array. Currently only supports 2D arrays.
        @return A new Array object that is the transpose of the original.
        @exception ValueError if the array is not 2D.
        """
        if self.ndim != 2:
            raise ValueError("Transpose is only supported for 2D arrays.")

        rows, cols = self.shape
        transposed_data = [[self.data[j][i] for j in range(rows)] for i in range(cols)]
        return Array(transposed_data)

    def _elementwise_op(self, other, op):
        """!
        @brief A helper function to perform elementwise operations.

        Handles operations with a scalar or another array of the same shape.

        @param other The other operand (scalar or Array).
        @param op A function (e.g., from the operator module) to apply.
        @return A new Array with the result of the operation.
        @exception ValueError if arrays have mismatched shapes.
        """
        if isinstance(other, (int, float)):
            def _apply_op(data):
                if isinstance(data, list):
                    return [_apply_op(sub_data) for sub_data in data]
                return op(data, other)

            return Array(_apply_op(self.data))

        if self.shape != other.shape:
            raise ValueError(f"Operands could not be broadcast together with shapes {self.shape} and {other.shape}")

        def _apply_op_arrays(data1, data2):
            if isinstance(data1, list):
                return [_apply_op_arrays(sub1, sub2) for sub1, sub2 in zip(data1, data2)]
            return op(data1, data2)

        return Array(_apply_op_arrays(self.data, other.data))

    def __add__(self, other):
        """!
        @brief Implements element-wise addition (operator +).
        @param other Scalar or Array to add.
        @return New Array with the result.
        """
        return self._elementwise_op(other, operator.add)

    def __sub__(self, other):
        """!
        @brief Implements element-wise subtraction (operator -).
        @param other Scalar or Array to subtract.
        @return New Array with the result.
        """
        return self._elementwise_op(other, operator.sub)

    def __mul__(self, other):
        """!
        @brief Implements element-wise multiplication (operator *).
        @param other Scalar or Array to multiply.
        @return New Array with the result.
        """
        return self._elementwise_op(other, operator.mul)

    def __truediv__(self, other):
        """!
        @brief Implements element-wise division (operator /).
        @param other Scalar or Array to divide by.
        @return New Array with the result.
        """
        return self._elementwise_op(other, operator.truediv)

    def __pow__(self, other):
        """!
        @brief Implements element-wise power (operator **).
        @param other Exponent (Scalar or Array).
        @return New Array with the result.
        """
        return self._elementwise_op(other, operator.pow)

    def __matmul__(self, other):
        """!
        @brief Implements the matrix multiplication operator (@).
        @param other The other Array to multiply with.
        @return The result of the matrix multiplication.
        """
        # This delegates the logic to the linalg module to avoid circular imports
        # and keep array logic separate from linear algebra algorithms.
        from . import linalg
        return linalg.matmul(self, other)

    def __rsub__(self, other):
        """!
        @brief Implements the right-side subtraction (e.g., scalar - array).
        @param other The other operand (scalar).
        @return The result of the elementwise subtraction.
        """
        # This is equivalent to: -1 * (self - other)
        return self._elementwise_op(other, lambda a, b: b - a)


# --- Array Creation Functions ---

def array(data):
    """!
    @brief Creates an Array object from a list or nested list.
    @param data The input list.
    @return An Array object.
    """
    return Array(data)


def _create_nested_list(shape, value):
    """!
    @brief Helper to create a nested list of a given shape filled with a value.
    @param shape The desired shape tuple.
    @param value The value to fill the list with.
    @return A nested list.
    """
    if not shape:
        return value
    return [_create_nested_list(shape[1:], value) for _ in range(shape[0])]


def zeros(shape):
    """!
    @brief Creates an array of a given shape, filled with zeros.
    @param shape A tuple of ints specifying the shape.
    @return An Array object filled with zeros.
    """
    return Array(_create_nested_list(shape, 0.0))


def ones(shape):
    """!
    @brief Creates an array of a given shape, filled with ones.
    @param shape A tuple of ints specifying the shape.
    @return An Array object filled with ones.
    """
    return Array(_create_nested_list(shape, 1.0))


def eye(n):
    """!
    @brief Creates an N x N identity matrix.
    @param n The number of rows and columns.
    @return An N x N identity Array.
    """
    data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    return Array(data)


def arange(start, stop=None, step=1):
    """!
    @brief Creates an array with a range of values.
    @param start The start of the interval.
    @param stop The end of the interval.
    @param step The step size.
    @return A 1D Array with the specified range of values.
    """
    if stop is None:
        stop = start
        start = 0
    return Array(list(range(start, stop, step)))


def linspace(start, stop, num):
    """!
    @brief Creates an array with a specified number of evenly spaced values.
    @param start The starting value of the sequence.
    @param stop The ending value of the sequence.
    @param num The number of samples to generate.
    @return A 1D Array of evenly spaced samples.
    """
    if num == 0:
        return Array([])
    if num == 1:
        return Array([float(start)])
    step = (stop - start) / (num - 1)
    return Array([start + i * step for i in range(num)])


# --- Elementwise Functions ---

def _apply_func_to_array(a, func):
    """!
    @brief Helper to apply a function elementwise to an array.
    @param a The input Array.
    @param func The function to apply.
    @return A new Array with the function applied to each element.
    """

    def _recursive_apply(data):
        if isinstance(data, list):
            return [_recursive_apply(sub_data) for sub_data in data]
        return func(data)

    return Array(_recursive_apply(a.data))


def exp(a):
    """! @brief Computes the exponential of all elements in the input array. """
    return _apply_func_to_array(a, math.exp)


def log(a):
    """! @brief Computes the natural logarithm of all elements in the input array. """
    return _apply_func_to_array(a, math.log)


def sqrt(a):
    """! @brief Computes the non-negative square-root of all elements. """
    return _apply_func_to_array(a, math.sqrt)


def abs(a):
    """! @brief Computes the absolute value of all elements. """
    return _apply_func_to_array(a, builtins.abs)


# --- Reductions ---

def sum(a):
    """! @brief Computes the sum of all array elements. """
    # Using dictionary access is more robust against namespace issues.
    return builtins.sum(a.flatten())


def mean(a):
    """! @brief Computes the arithmetic mean of all array elements. """
    return sum(a) / a.size


def min(a):
    """! @brief Finds the minimum value among all array elements. """
    return builtins.min(a.flatten())


def max(a):
    """! @brief Finds the maximum value among all array elements. """
    return builtins.max(a.flatten())


def argmin(a):
    """! @brief Finds the index of the minimum value in the flattened array. """
    flat_list = a.flatten()
    return flat_list.index(min(a))


def argmax(a):
    """! @brief Finds the index of the maximum value in the flattened array. """
    flat_list = a.flatten()
    return flat_list.index(max(a))