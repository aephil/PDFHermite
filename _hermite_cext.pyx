# cython: language_level=3, boundscheck=False, wraparound=False
"""
Thin Cython wrapper around the C++ hermite_basis_forward_cpp function.
Allocates a Fortran-contiguous output array and dispatches with nogil.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "hermite_ext.h":
    void hermite_basis_forward_c(
        const double* x,
        int N,
        int nh,
        double* out
    ) nogil


def hermite_basis_forward(
    cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] x,
    int nh
):
    """C implementation of _hermite_basis_forward.

    Parameters
    ----------
    x  : 1-D float64 C-contiguous array of N evaluation points
    nh : number of odd-order Hermite functions to compute

    Returns
    -------
    out : ndarray, shape (N, nh), Fortran-contiguous
        Column ih contains F_{2ih+1}(x).
    """
    cdef int N = x.shape[0]
    cdef cnp.ndarray out = np.empty((N, nh), dtype=np.float64, order='F')

    hermite_basis_forward_c(
        <const double*> x.data,
        N,
        nh,
        <double*> out.data,
    )
    return out
