"""
Build the _hermite_cext C++ extension in-place:

    python setup_ext.py build_ext --inplace

Requires: Cython, numpy, and a C++17-capable compiler
(Xcode Command Line Tools on macOS, g++ on Linux).
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="_hermite_cext",
    sources=["_hermite_cext.pyx", "hermite_ext.c"],
    include_dirs=[np.get_include()],
    language="c",
    extra_compile_args=[
        "-std=c99",
        "-O3",
        "-mcpu=native",
        "-fno-math-errno",   # allows sqrt/exp2/ldexp to be inlined as intrinsics
        # Note: -ffast-math is intentionally omitted — it can reorder operations
        # across the frexp boundary and potentially break the log-scale invariant.
    ],
)

setup(
    name="hermite_cext",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    ),
)
