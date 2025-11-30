# mininumpy/linalg.py

"""!
@brief MiniNumPy Linear Algebra Module.
Contains functions for linear algebra operations like dot product, matrix multiplication,
determinant, and inverse.
"""
import math
from .array import Array, zeros, eye


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
    return math.sqrt(sum(x*x for x in a.flatten()))


def lu_decompose(a):
    """!
    @brief Performs LUP decomposition of a square matrix.
    Decomposes matrix A such that P @ A = L @ U.

    @param a The input square Array.
    @return A tuple (P, L, U, swaps) where:
            - P is the permutation Array.
            - L is the lower triangular Array.
            - U is the upper triangular Array.
            - swaps is the number of row swaps performed (for determinant sign).
    @raises ValueError if matrix is not square.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("LU decomposition is only for square matrices.")

    n = a.shape[0]

    # Create distinct copies for U (initially A) and P (initially Identity)
    # We use explicit loops to copy data to avoid referencing the original list
    U_data = [row[:] for row in a.data]
    L_matrix = eye(n)
    L_data = L_matrix.data
    P_matrix = eye(n)
    P_data = P_matrix.data

    swaps = 0

    for k in range(n):
        # --- Partial Pivoting ---
        # Find the row with the largest absolute value in the current column k
        pivot_row = k
        max_val = abs(U_data[k][k])
        for i in range(k + 1, n):
            if abs(U_data[i][k]) > max_val:
                max_val = abs(U_data[i][k])
                pivot_row = i

        # Swap rows in U, P, and L (only the computed part of L) if needed
        if pivot_row != k:
            U_data[k], U_data[pivot_row] = U_data[pivot_row], U_data[k]
            P_data[k], P_data[pivot_row] = P_data[pivot_row], P_data[k]
            # For L, we only swap the elements to the left of the diagonal (0 to k-1)
            # to maintain the lower triangular structure logic after permutation
            for col_idx in range(k):
                L_data[k][col_idx], L_data[pivot_row][col_idx] = \
                    L_data[pivot_row][col_idx], L_data[k][col_idx]
            swaps += 1

        # Check for singular matrix (pivot is zero)
        if abs(U_data[k][k]) < 1e-12:
            # Depending on strictness, we might raise error or continue
            # For determinant calculation, a zero pivot implies det=0.
            continue

            # --- Elimination ---
        for i in range(k + 1, n):
            factor = U_data[i][k] / U_data[k][k]
            L_data[i][k] = factor
            # Update row i of U: Row_i = Row_i - factor * Row_k
            for j in range(k, n):
                U_data[i][j] -= factor * U_data[k][j]

    return P_matrix, L_matrix, Array(U_data), swaps


def det(a):
    """!
    @brief Calculates the determinant using LUP decomposition.
    Complexity: O(n^3).

    @param a The input square Array.
    @return The determinant value.
    """
    try:
        _, _, U, swaps = lu_decompose(a)
    except ValueError:
        return 0.0  # Handle singular matrices nicely

    n = a.shape[0]
    determinant = 1.0

    # Det(A) = (-1)^swaps * Product(diag(U))
    for i in range(n):
        determinant *= U.data[i][i]

    return determinant * ((-1) ** swaps)


def _solve_lu(L, U, P, b):
    """!
    @brief Helper to solve Ax = b given the LUP decomposition.
    Solves LUx = Pb.
    """
    n = L.shape[0]

    # 1. Apply Permutation to b: b_p = P @ b
    # b can be a vector (list) or an Array object. We treat it as list data here.
    b_data = b if isinstance(b, list) else b.flatten()
    pb = [0.0] * n
    for i in range(n):
        # P is a permutation matrix, simple dot product picks the right element
        pb[i] = sum(P.data[i][j] * b_data[j] for j in range(n))

    # 2. Forward Substitution: Solve Ly = pb
    y = [0.0] * n
    for i in range(n):
        s = sum(L.data[i][j] * y[j] for j in range(i))
        y[i] = pb[i] - s  # L[i][i] is always 1

    # 3. Backward Substitution: Solve Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U.data[i][j] * x[j] for j in range(i + 1, n))
        if abs(U.data[i][i]) < 1e-12:
            raise ValueError("Matrix is singular.")
        x[i] = (y[i] - s) / U.data[i][i]

    return x


def inv(a):
    """!
    @brief Calculates the inverse of a matrix using LUP decomposition.
    Solves Ax = e_i for each column of the identity matrix.
    Complexity: O(n^3).

    @param a The input square Array.
    @return The inverse Array.
    @raises ValueError if matrix is singular.
    """
    n = a.shape[0]
    P, L, U, _ = lu_decompose(a)

    # Check singularity based on U's diagonal
    for i in range(n):
        if abs(U.data[i][i]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted.")

    inv_matrix = zeros((n, n))
    inv_data = inv_matrix.data  # Get reference to modify data

    # Solve for each column of Identity
    for j in range(n):
        # Create standard basis vector e_j (e.g., [0, 1, 0...])
        e_j = [1.0 if i == j else 0.0 for i in range(n)]

        # Solve Ax = e_j
        x_j = _solve_lu(L, U, P, e_j)

        # Store result in the j-th column of inverse matrix
        for i in range(n):
            inv_data[i][j] = x_j[i]

    return inv_matrix


def eig(a):
    """!
    @brief (Bonus) Placeholder for eigenvalue and eigenvector calculation.

    Calculating eigenvalues is a complex numerical problem (e.g., using QR algorithm)
    and is beyond the scope of a simple implementation from scratch.

    @param a A square 2D Array.
    @raises NotImplementedError
    """
    raise NotImplementedError("Eigenvalue calculation is a bonus and not implemented.")