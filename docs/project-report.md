# MiniNumPy Project Report

## Objectives

MiniNumPy explores how a numerical library represents arrays, applies operations across dimensions, and solves matrix problems using only Python's standard library. The notebook compares representative results with NumPy and applies the library to geometry and grayscale images.

## Array representation

`Array` stores values in nested Python lists. The constructor derives the shape from the nesting and checks row lengths. The `ndim` and `size` attributes describe the rank and element count. Python lists make the structure easy to inspect, while the separate lists and Python objects add overhead compared with contiguous numeric storage.

`flatten()` collects elements into a one-dimensional list in traversal order. `reshape()` builds a new nested structure with the same element count, and `transpose()` swaps the axes of a matrix.

## Elementwise operations and reductions

Operator methods delegate to a recursive helper. It applies a scalar at each leaf or visits matching positions in two arrays of the same shape. Reflected subtraction supports expressions such as `255 - image`, preserving the order of the operands at each pixel.

Reductions work on flattened data. The implementation calls `builtins.sum`, `builtins.min`, and `builtins.max` explicitly because the package exposes functions with those same names.

## Determinants and inverses

The first determinant implementation used recursive cofactor expansion. Its factorial growth made it impractical as matrix size increased. The current implementation uses LU decomposition with partial pivoting, which takes cubic time for an n-by-n matrix.

At each step, the algorithm chooses the remaining row with the largest absolute value in the pivot column. It records row swaps in a permutation matrix `P` and computes triangular factors `L` and `U` satisfying `P @ A = L @ U` for a nonsingular matrix with pivots above the implementation's tolerance.

- The determinant is the product of the diagonal entries of `U`, with the sign flipped for an odd number of row swaps.
- The inverse is built one column at a time. For each identity column `e`, forward substitution solves `L y = P e`, then backward substitution solves `U x = y`. The solution `x` becomes the corresponding column of the inverse.

Choosing a large pivot helps limit rounding error during elimination. The implementation uses an absolute pivot tolerance of `1e-12`, so the scale of a matrix can affect whether inversion treats it as singular.

## Demonstrations and trade-offs

The notebook compares determinants and inverses with NumPy, including for a five-by-five matrix. It also uses homogeneous coordinates to scale, rotate, and shear a shape, and applies scalar arithmetic to brighten and invert a grayscale image.

The main performance gain comes from the change in algorithm. Nested lists and interpreted loops still carry overhead, while NumPy uses optimized numerical routines. MiniNumPy keeps the operations visible for study.
