# MiniNumPy

MiniNumPy is a pure-Python library inspired by NumPy. It implements multidimensional arrays and linear algebra from scratch so you can follow how each operation works with nested Python lists.

## Quick start

Clone the repository and run Python from the project root. The library uses only the Python standard library. On Windows, use `py -3` in place of `python` if you use the Python launcher.

```bash
git clone https://github.com/flmaple1121/MiniNumPy_Library.git
cd MiniNumPy_Library
python
```

```python
import mininumpy as mnp

a = mnp.array([[0, 2], [1, 3]])

print(a.shape)             # (2, 2)
print((a + 1).data)        # [[1, 3], [2, 4]]
print((a @ a).data)        # [[2, 6], [3, 11]]
print(mnp.linalg.det(a))   # -2.0
```

## Features

| Area | Supported operations |
| --- | --- |
| Array creation | `array`, `zeros`, `ones`, `eye`, `arange`, `linspace` |
| Shape and layout | `shape`, `ndim`, `size`, `reshape`, `flatten`, and 2D `transpose` |
| Elementwise math | Arithmetic with scalars or same-shape arrays, plus `exp`, `log`, `sqrt`, and `abs` |
| Reductions | `sum`, `mean`, `min`, and `max` over all elements, with `argmin` and `argmax` returning flat indices |
| Linear algebra | Vector dot products (`dot`), 2D matrix multiplication (`matmul` or `@`), `norm`, `lu_decompose`, `det`, and `inv` |

Determinants and inverses use LU decomposition with partial pivoting on square matrices.

## Notebook demo

The notebook compares MiniNumPy with NumPy and shows how to transform a 2D shape and edit a grayscale image. [Open the demo notebook](demo.ipynb)

To run it locally with JupyterLab, use these commands from the project root.

```bash
python -m pip install -r requirements.txt jupyterlab
python -m jupyterlab demo.ipynb
```

## Source and documentation

- Array creation, reshaping, arithmetic, and reductions. [Array implementation](mininumpy/array.py)
- Matrix operations and pivoted LU decomposition. [Linear algebra implementation](mininumpy/linalg.py)
- Data model, algorithms, and numerical trade-offs. [Project report](docs/project-report.md)

To generate the API reference with Doxygen, run `doxygen Doxyfile` from the project root and open `html/index.html`.
