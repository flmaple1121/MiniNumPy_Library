"""!
@mainpage MiniNumPy Project Report

@tableofcontents

@section sec_intro 1. Introduction & Objectives

The goal of the **MiniNumPy** project is to design and implement a simplified, pure-Python version of the industry-standard **NumPy** library. This project serves as a deep dive into the internal architecture of numerical computing systems, requiring a ground-up implementation of N-dimensional arrays, element-wise arithmetic, broadcasting rules, and advanced linear algebra algorithms.

By avoiding external dependencies (like real NumPy or SciPy) for the core logic, this project explores:
- **Memory Management**: How multi-dimensional data is structured in memory.
- **Algorithmic Complexity**: The difference between naive implementations (e.g., recursive determinant) and optimized approaches (e.g., LU decomposition).
- **Numerical Stability**: Handling floating-point errors and singular matrices.

@section sec_design 2. Architecture & Design Choices

@subsection sub_storage Core Data Structure
The heart of the library is the `Array` class defined in `array.py`.
- **Storage Strategy**: We utilized nested Python lists (`List[List[float]]`) to store matrix data.
    - *Pros*: Simple to implement; native Python support; dynamic sizing.
    - *Cons*: Non-contiguous memory allocation leads to cache misses; iterating over nested lists is slower than C-contiguous arrays.
- **Shape Management**: The `shape` attribute is derived dynamically. We implemented a recursive validation step `_validate_shape` to ensure that the input data forms a regular grid (i.e., no jagged arrays).

@subsection sub_broadcast Operator Overloading & Broadcasting
To mimic NumPy's intuitive syntax (e.g., `A + 10`), we overloaded Python's magic methods:
- `__add__`, `__sub__`, `__mul__`, `__truediv__`.
- **Reflected Arithmetic Operations (`__rsub__`)**: While addition is commutative (\f$ A+B = B+A \f$), subtraction is not. To support intuitive operations where a scalar is the left operand (e.g., calculating a photographic negative via \f$ 255 - \text{image} \f$), we implemented the `__rsub__` method.
- **Mechanism**: When Python encounters `scalar - Array`, it invokes `Array.__rsub__`. Our implementation ensures the operation order is correctly handled as `scalar - element` rather than `element - scalar`.
- **Broadcasting Implementation**: The `_elementwise_op` helper function detects if the operand is a scalar or an array. If it is a scalar, the operation is applied recursively to every element of the nested list structure.

@section sec_algos 3. Algorithms & Linear Algebra

The most significant engineering effort was placed in the `linalg.py` module. We transitioned from naive mathematical definitions to computationally efficient algorithms.

@subsection sub_det Determinant Computation
**Initial Approach: Laplace Expansion**
Initially, we implemented the determinant using recursive Laplace expansion (cofactor expansion).
\f[
\det(A) = \sum_{j=1}^{n} (-1)^{i+j} A_{ij} \det(M_{ij})
\f]
- **Complexity**: \f$ O(n!) \f$.
- **Issue**: This approach became computationally infeasible for matrices larger than \f$ 10 \times 10 \f$.

**Optimized Approach: LU Decomposition**
We replaced the recursive method with **LUP Decomposition**:
\f[
P A = L U
\f]
where \f$ P \f$ is a permutation matrix, \f$ L \f$ is lower triangular, and \f$ U \f$ is upper triangular.
- **Complexity**: \f$ O(n^3) \f$.
- **Determinant Calculation**: The determinant is simply the product of the diagonal elements of \f$ U \f$, adjusted by the sign of the permutations (\f$ s \f$):
\f[
\det(A) = (-1)^s \prod_{i=1}^{n} U_{ii}
\f]

@subsection sub_inv Matrix Inversion
Instead of using the Adjugate Matrix method (which requires \f$ n^2 \f$ determinant calculations), we used the LUP decomposition to solve for the inverse column by column.
To find \f$ A^{-1} \f$, we solve the linear system \f$ A x_j = e_j \f$ for each standard basis vector \f$ e_j \f$ (columns of the identity matrix).
The solution involves two rapid steps:
1.  **Forward Substitution**: Solve \f$ L y = P e_j \f$.
2.  **Backward Substitution**: Solve \f$ U x_j = y \f$.

@section sec_challenges 4. Challenges & Solutions
@subsection sub_namespace Namespace Conflicts & Built-ins
**Problem**: To replicate the NumPy API, we named our reduction functions `sum`, `min`, and `max`. However, these names shadow Python's built-in functions. Using them naively inside the library (e.g., using `sum()` to calculate list totals within `Array` methods) caused recursive errors or IDE warnings.<br>
**Solution**:
- We explicitly imported Python's `builtins` module.
- Inside our library functions, we access standard Python operations via `builtins.sum()`, `builtins.min()`, etc.
- This approach allows us to expose a NumPy-compatible API (e.g., `mnp.sum(arr)`) to the user while maintaining robust access to native Python functionality internally, resolving static analysis warnings and potential runtime bugs.

@subsection sub_stability Numerical Stability & Partial Pivoting
**Problem**: During Gaussian elimination (LU decomposition), if a pivot element on the diagonal is zero (or very close to zero), division by zero occurs, causing the algorithm to crash or produce inaccurate results.<br>
**Solution**: We implemented **Partial Pivoting**. Before eliminating a column, the algorithm searches for the largest absolute value in that column and swaps the current row with the row containing the maximum value. This ensures numerical stability and prevents division by zero for non-singular matrices.

@subsection sub_recursion Python Recursion Limit
**Problem**: The initial recursive implementation of `reshape` and `determinant` hit Python's maximum recursion depth for larger arrays.<br>
**Solution**:
- For `determinant`, we switched to the iterative LU decomposition.
- For `flatten`, we ensured it returns a flat list efficiently to support reshaping without deep recursion stacks.

@subsection sub_viz Image Processing "Inversion"
**Problem**: During the demo application (grayscale image processing), the inverted image appeared visually incorrect (white background for 0 values) due to IDE theming.<br>
**Solution**: We verified the mathematical correctness by inspecting the raw pixel values (0 for black, 255 for white) and saving the plot to a raw PNG file, proving the logic was correct despite the IDE's dark mode rendering artifacts.

@section sec_perf 5. Performance Analysis

While `MiniNumPy` is functionally correct, it is an interpreted Python implementation.
- **MiniNumPy**: \f$ O(n^3) \f$ complexity, but with high constant factors due to Python's dynamic typing and object overhead.
- **Real NumPy**: Uses BLAS/LAPACK (highly optimized C/Fortran libraries).

For a \f$ 100 \times 100 \f$ matrix multiplication:
- **MiniNumPy** takes approximately 0.5 - 1.0 seconds.
- **NumPy** takes microseconds.

However, the implementation of LU decomposition allowed `MiniNumPy` to handle matrices of size \f$ 20 \times 20 \f$ and larger instantaneously, whereas the naive recursive determinant would have taken millions of years to compute.

@section sec_concl 6. Conclusion
This project successfully replicates the core API of NumPy. By implementing **LUP Decomposition from scratch**, we achieved a robust Linear Algebra module capable of handling general square matrices efficiently. The project highlights the importance of algorithmic selection (LU vs. Recursion) over micro-optimizations in high-level languages.
"""

# --- Import core functionality to expose them at package level ---
from .array import *
from . import linalg

__version__ = "0.1.3"