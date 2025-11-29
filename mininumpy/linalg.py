# mininumpy/linalg.py

"""!
@brief MiniNumPy Linear Algebra Module.
Contains functions for linear algebra operations like dot product, matrix multiplication,
determinant, and inverse.
"""
import math
from .array import Array, zeros


def dot(a, b):
    """!
    @brief Computes the dot product of two arrays.

    - If both are 1D, it is the inner product.
    - If both are 2D, it is matrix multiplication.

    @param a First input array.
    @param b Second input array.
    @return The dot product of a and b.
    @raises ValueError if dimensions are incompatible.
    """
    if a.ndim == 1 and b.ndim == 1:
        if a.shape[0] != b.shape[0]:
            raise ValueError("1D vectors must have the same length for dot product.")
        return sum(x * y for x, y in zip(a.data, b.data))

    elif a.ndim == 2 and b.ndim == 2:
        return matmul(a, b)

    else:
        # This implementation can be extended for more cases like NumPy.
        raise ValueError("Dot product is only implemented for 1D-1D or 2D-2D arrays.")


def matmul(a, b):
    """!
    @brief Performs matrix multiplication on two 2D arrays.

    @param a A 2D Array with shape (m, n).
    @param b A 2D Array with shape (n, p).
    @return A new 2D Array with shape (m, p).
    @raises ValueError if arrays are not 2D or shapes are incompatible.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Matrix multiplication is only supported for 2D arrays.")

    m, n = a.shape
    n_b, p = b.shape

    if n != n_b:
        raise ValueError(f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}")

    # Transpose b for easier row-wise access
    b_t = b.transpose()

    result_array = zeros((m, p))
    new_data = result_array.data
    for i in range(m):
        for j in range(p):
            # Dot product of row i of a and column j of b
            row_a = a.data[i]
            col_b = b_t.data[j]  # Now a row in the transposed matrix
            sum_val = sum(x * y for x, y in zip(row_a, col_b))
            new_data[i][j] = sum_val

    return result_array


def norm(a):
    """!
    @brief Calculates the vector (L2) or matrix (Frobenius) norm.

    @param a The input array.
    @return The norm of the array (float).
    """
    # from .array import sqrt, sum
    # This is equivalent to sqrt(sum(a**2))
    return math.sqrt(sum(x*x for x in a._flatten()))


def det(a):
    """!
    @brief Calculates the determinant of a square 2D array.

    This implementation uses recursive Laplace expansion. It is not efficient
    for large matrices but is simple to implement.

    @param a A square 2D Array.
    @return The determinant of the array.
    @raises ValueError if the array is not square or not 2D.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Determinant can only be calculated for square 2D arrays.")

    n = a.shape[0]

    # Base cases for recursion
    if n == 1:
        return a.data[0][0]
    if n == 2:
        return a.data[0][0] * a.data[1][1] - a.data[0][1] * a.data[1][0]

    determinant = 0
    for j in range(n):
        # Create the submatrix (minor)
        sub_matrix_data = [row[:j] + row[j + 1:] for row in a.data[1:]]
        sub_matrix = Array(sub_matrix_data)

        # Add to determinant using cofactor expansion along the first row
        sign = (-1) ** j
        determinant += sign * a.data[0][j] * det(sub_matrix)

    return determinant


def inv(a):
    """!
    @brief Calculates the inverse of a square 2D array.

    The inverse is calculated using the formula: inv(A) = (1/det(A)) * adj(A),
    where adj(A) is the adjugate matrix (transpose of the cofactor matrix).

    @param a A square, invertible 2D Array.
    @return The inverse of the array.
    @raises ValueError if the array is not square or is singular (det=0).
    """
    d = det(a)
    if d == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    n = a.shape[0]
    if n == 1:
        return Array([[1.0 / a.data[0][0]]])

    # --- Calculate Cofactor Matrix ---
    cofactor_matrix = zeros((n, n))
    cofactor_data = cofactor_matrix.data
    for i in range(n):
        for j in range(n):
            # Create the submatrix
            sub_matrix_data = [row[:j] + row[j + 1:] for row in (a.data[:i] + a.data[i + 1:])]
            sub_matrix = Array(sub_matrix_data)

            sign = (-1) ** (i + j)
            cofactor_data[i][j] = sign * det(sub_matrix)

    # --- Adjugate is the transpose of the cofactor matrix ---
    adjugate_matrix = cofactor_matrix.transpose()

    # --- Inverse is adjugate / determinant ---
    return adjugate_matrix * (1.0 / d)


def eig(a):
    """!
    @brief (Bonus) Placeholder for eigenvalue and eigenvector calculation.

    Calculating eigenvalues is a complex numerical problem (e.g., using QR algorithm)
    and is beyond the scope of a simple implementation from scratch.

    @param a A square 2D Array.
    @raises NotImplementedError
    """
    raise NotImplementedError("Eigenvalue calculation is a bonus and not implemented.")