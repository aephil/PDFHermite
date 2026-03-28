"""
Mathematical functions for the PDFHermite package.

Provides Hermite and Chebyshev basis construction, Q-space convolution,
instrument resolution broadening, background pre-fitting, and D(r) computation.
"""

import numpy as np
from scipy.signal import convolve
from scipy.special import eval_chebyt
from scipy.optimize import minimize

try:
    from _hermite_cext import hermite_basis_forward as _hermite_basis_forward_c
    _USE_C_HERMITE = True
except ImportError:
    _USE_C_HERMITE = False


# ---------------------------------------------------------------------------
# Hermite functions
# ---------------------------------------------------------------------------

def fockstate(n, x):
    """Evaluate the normalized Hermite (Fock state) function of order *n*.

    Uses Clenshaw's recurrence relation for the polynomial F_n = H_n/sqrt(2^n n!),
    with log2-scale tracking to prevent floating-point overflow for large *n*.
    This is a direct port of the MATLAB ``fockstate`` function.

    Parameters
    ----------
    n : int
        Order of the Hermite function (must be >= 0).
    x : array_like
        Evaluation points (one-dimensional).

    Returns
    -------
    ndarray
        Values of pi^(-1/4) * exp(-x^2/2) * F_n(x) at each point in *x*.
    """
    x = np.asarray(x, dtype=float)
    orig_shape = x.shape
    xs = x.ravel()
    m = len(xs)

    # Coefficient vector: 1 at position n, 0 elsewhere
    cs = np.zeros(n + 1)
    cs[n] = 1.0
    N = n  # highest degree

    B = np.zeros((m, 2))     # [b_current, b_previous]
    logB = np.zeros(m)       # accumulated binary exponent for scaling

    for k in range(N, -1, -1):
        a_k = -xs * np.sqrt(2.0 / (k + 1))
        b_k = np.sqrt((k + 1.0) / (k + 2))
        new_b0 = cs[k] - a_k * B[:, 0] - b_k * B[:, 1]
        B[:, 1] = B[:, 0]
        B[:, 0] = new_b0
        # Extract binary mantissa/exponent to keep values in [0.5, 1) and
        # accumulate the exponent in logB; equivalent to MATLAB [F,E]=log2(B).
        F, E = np.frexp(B[:, 0])
        logB += E
        B[:, 0] = F
        B[:, 1] /= np.ldexp(1.0, E)   # divide by 2^E element-wise

    y = np.pi ** (-0.25) * np.exp(-xs ** 2 / 2.0 + np.log(2) * logB) * B[:, 0]
    return y.reshape(orig_shape)


def _hermite_basis_forward(x, nh):
    """Compute all nh odd-order normalized Hermite functions at points *x*.

    Uses a log-scaled forward 3-term recurrence relation.  Define the unscaled
    quantity G_n(x) = F_n(x) * exp(x²/2) / pi^(-1/4), which satisfies the same
    recurrence but with non-decaying initial conditions:
      G_0 = 1,  G_1 = sqrt(2) * x
      G_{n+1} = sqrt(2/(n+1)) * x * G_n - sqrt(n/(n+1)) * G_{n-1}

    Each G_n is tracked as (mantissa, binary_exponent) using frexp so that
    overflow is avoided for large x.  The final result is recovered as
      F_n(x) = G_n_mantissa * 2^(G_n_exp - x²/(2*log2)) * pi^(-1/4)

    This single forward sweep avoids the O(nh²) cost of calling the backward
    Clenshaw recurrence separately for each order.

    Parameters
    ----------
    x  : ndarray, shape (N,)
    nh : int

    Returns
    -------
    out : ndarray, shape (N, nh)
        Columns are F_1, F_3, …, F_{2nh-1} evaluated at *x*.
    """
    N = len(x)
    out = np.zeros((N, nh))

    # G_0 = 1 for all x.  Rescale immediately.
    G_prev_m, G_prev_e = np.frexp(np.ones(N))           # (0.5, ones*1)
    G_prev_e = G_prev_e.astype(float)

    # G_1 = sqrt(2) * x
    G1_raw = np.sqrt(2.0) * x
    G_curr_m, E = np.frexp(G1_raw)
    G_curr_e = E.astype(float)

    # Common additive term: log2(pi^(-1/4) * exp(-x^2/2))
    # = -0.25*log2(pi) - x^2 / (2*log(2))
    log2_common = -0.25 * np.log2(np.pi) - x ** 2 / (2.0 * np.log(2.0))

    def _store(m, e, col):
        total = m * np.exp2(e + log2_common)
        out[:, col] = total

    _store(G_curr_m, G_curr_e, 0)   # F_1

    ih = 1
    for n in range(1, 2 * nh - 1):
        An = np.sqrt(2.0 / (n + 1))   # scalar
        Bn = np.sqrt(n / (n + 1.0))   # scalar

        # G_{n+1} = An*x * G_curr - Bn * G_prev
        # Use a common binary scale to subtract safely
        # term1 = An*x * G_curr_m * 2^G_curr_e
        # term2 = Bn * G_prev_m * 2^G_prev_e
        t1_m = An * x * G_curr_m       # shape (N,)
        t1_e = G_curr_e                # shape (N,)
        t2_m = Bn * G_prev_m           # shape (N,)
        t2_e = G_prev_e                # shape (N,)

        # Bring to a common exponent (use the larger one)
        max_e = np.maximum(t1_e, t2_e)
        diff = (t1_m * np.exp2(t1_e - max_e)
                - t2_m * np.exp2(t2_e - max_e))

        G_next_m, E2 = np.frexp(diff)
        G_next_e = max_e + E2

        G_prev_m = G_curr_m
        G_prev_e = G_curr_e
        G_curr_m = G_next_m
        G_curr_e = G_next_e

        # n+1 is odd when n is even → store as the next odd-order function
        if n % 2 == 0 and ih < nh:
            _store(G_curr_m, G_curr_e, ih)
            ih += 1

    return out


def build_hermite_basis(q, npts, nh, qp):
    """Build the Hermite function basis matrix for all banks.

    Evaluates odd-order Hermite functions F_1, F_3, F_5, … at Q/qp for each
    data point in each bank, using a single-pass forward recurrence.

    Parameters
    ----------
    q : ndarray, shape (len_max, nbanks)
    npts : array of int, shape (nbanks,)
    nh : int
        Number of Hermite functions.
    qp : float
        Characteristic Q scale (Q').

    Returns
    -------
    xh : ndarray, shape (len_max, nh, nbanks)
    """
    len_max, nbanks = q.shape
    xh = np.zeros((len_max, nh, nbanks))
    for ib in range(nbanks):
        n = npts[ib]
        x = q[:n, ib] / qp
        xh[:n, :, ib] = (_hermite_basis_forward_c if _USE_C_HERMITE
                         else _hermite_basis_forward)(x, nh)
    return xh


# ---------------------------------------------------------------------------
# Chebyshev basis
# ---------------------------------------------------------------------------

def chebyshev_basis(q, qfitmax, nbanks, npts, npoly):
    """Build an odd-order Chebyshev polynomial basis for background pre-fitting.

    Uses T(2n-1, Q/qfitmax) for n = 1, 2, …, npuse, where npuse is scaled
    by max(Q_bank)/qfitmax so that banks with smaller Q ranges get fewer terms.

    Parameters
    ----------
    q : ndarray, shape (len_max, nbanks)
    qfitmax : float
    nbanks : int
    npts : array of int
    npoly : int
        Base number of Chebyshev terms (scaled per bank).

    Returns
    -------
    Cheb : ndarray, shape (len_max, nc_max, nbanks)
    nc : ndarray of int, shape (nbanks,)
        Actual number of Chebyshev terms used per bank.
    """
    nc_max = int(round(npoly * np.max(q) / qfitmax))
    len_max = q.shape[0]
    Cheb = np.zeros((len_max, max(nc_max, 1), nbanks))
    nc = np.zeros(nbanks, dtype=int)

    for ib in range(nbanks):
        n = npts[ib]
        Q = q[:n, ib]
        npuse = int(round(npoly * np.max(Q) / qfitmax))
        nc[ib] = npuse
        if npuse == 0:
            continue
        cols = []
        for m in range(1, npuse + 1):
            cols.append(eval_chebyt(2 * m - 1, Q / qfitmax))
        Cheb[:n, :npuse, ib] = np.column_stack(cols)

    return Cheb, nc


# ---------------------------------------------------------------------------
# Resolution-kernel helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma, dx):
    """Normalised Gaussian kernel with standard deviation *sigma*, step *dx*."""
    qlim = np.floor(10.0 * sigma / dx) * dx
    x = np.arange(-qlim, qlim + dx / 2, dx)
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    return y


def _pseudo_voigt_kernel(sigma, gamma, ratio, dx):
    """Pseudo-Voigt kernel: ratio * Gaussian + (1-ratio) * Lorentzian."""
    qlim = np.floor(10.0 * gamma / dx) * dx
    x = np.arange(-qlim, qlim + dx / 2, dx)
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    lorentz = (1.0 / np.pi) * gamma / (x ** 2 + gamma ** 2)
    return ratio * gauss + (1.0 - ratio) * lorentz


def _back_to_back_kernel(sigma, gamma, ratio, lam_up, lam_down, dx):
    """Back-to-back exponential convolved with a pseudo-Voigt profile."""
    qlim = np.floor(10.0 / min(lam_down, lam_up) / dx) * dx
    x = np.arange(-qlim, qlim + dx / 2, dx)
    pv = _pseudo_voigt_kernel(sigma, gamma, ratio, dx)
    # Build back-to-back exponential using same x grid
    qlim2 = np.floor(10.0 / min(lam_down, lam_up) / dx) * dx
    x2 = np.arange(-qlim2, qlim2 + dx / 2, dx)
    btb = np.where(
        x2 <= 0,
        lam_up * lam_down / (lam_up + lam_down) * np.exp(lam_up * x2),
        lam_up * lam_down / (lam_up + lam_down) * np.exp(-lam_down * x2),
    )
    return convolve(pv, btb, mode='same')


def _read_resolution_param(res_file, key, ibank, nbanks):
    """Read a per-bank resolution parameter from a resolution info file."""
    from io_utils import read_config
    cfg = read_config(res_file)
    if nbanks == 1:
        val = cfg.get(key, '')
    else:
        val = cfg.get(f'{key}{ibank + 1}', '')  # 1-indexed in file
    return float(val) if val else None


def _load_resolution_config(res_file):
    """Return the full parsed resolution config dict."""
    from io_utils import read_config
    return read_config(res_file)


# ---------------------------------------------------------------------------
# Resolution broadening applied to Qi(Q) data
# ---------------------------------------------------------------------------

def apply_broadening(qiq, nbanks, npts, len_max, dq, res_file):
    """Convolve each bank's Qi(Q) data with the instrument resolution function.

    Supports four resolution types: ``gaussian``, ``gaussian_tof``,
    ``pseudo_voigt``, and ``back_to_back``.

    Parameters
    ----------
    qiq : ndarray, shape (len_max, nbanks)  — modified in place
    nbanks, npts, len_max : int / array
    dq : float — Q-space step size
    res_file : str — path to resolution parameter file

    Returns
    -------
    ndarray, shape (len_max, nbanks)
    """
    cfg = _load_resolution_config(res_file)
    res_type = cfg.get('resolution_type', '').lower()
    dx = dq
    qiq = qiq.copy()

    for ib in range(nbanks):
        N = npts[ib]
        data = qiq[:N, ib]

        if res_type == 'gaussian':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            kernel = _gaussian_kernel(sigma, dx)
            qiq[:N, ib] = convolve(data, kernel, mode='same') * dx
            qiq[N:, ib] = 0.0

        elif res_type == 'gaussian_tof':
            a = _get_param(cfg, 'a', ib, nbanks)
            qiq[:N, ib] = _apply_gauss_tof_1d(data, a, dx)

        elif res_type == 'pseudo_voigt':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            gamma = _get_param(cfg, 'gamma', ib, nbanks)
            ratio = _get_param(cfg, 'ratio', ib, nbanks)
            kernel = _pseudo_voigt_kernel(sigma, gamma, ratio, dx)
            qiq[:N, ib] = convolve(data, kernel, mode='same') * dx
            qiq[N:, ib] = 0.0

        elif res_type == 'back_to_back':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            gamma = _get_param(cfg, 'gamma', ib, nbanks)
            ratio = _get_param(cfg, 'ratio', ib, nbanks)
            lam_up = _get_param(cfg, 'lambda_up', ib, nbanks)
            lam_down = _get_param(cfg, 'lambda_down', ib, nbanks)
            kernel = _back_to_back_kernel(sigma, gamma, ratio, lam_up, lam_down, dx)
            qiq[:N, ib] = convolve(data, kernel, mode='same') * dx
            qiq[N:, ib] = 0.0

    return qiq


def _apply_gauss_tof_1d(data, a, dq):
    """Apply Q-dependent Gaussian (gaussian_tof) broadening to a 1-D array.

    At position j the Gaussian width is sigma_j = (j+1)*a*dq.  Below the
    threshold sigma < 0.01 the data are left unchanged (identity mapping).

    The result at each point j is computed as a dot product of a Gaussian
    weight vector with the data slice around j, avoiding N full convolution
    calls and instead using numpy matrix operations for efficiency.
    """
    N = len(data)
    result = data.copy()
    for j in range(N):
        sigma = (j + 1) * a * dq
        if sigma < 0.01:
            continue   # identity
        half_w = int(np.floor(10.0 * sigma / dq))
        k0 = max(0, j - half_w)
        k1 = min(N, j + half_w + 1)
        offsets = np.arange(k0, k1) - j
        g = (np.exp(-offsets ** 2 * dq ** 2 / (2 * sigma ** 2))
             / np.sqrt(2 * np.pi * sigma ** 2) * dq)
        result[j] = np.dot(data[k0:k1], g)
    return result


# ---------------------------------------------------------------------------
# Resolution broadening applied to the Hermite basis
# ---------------------------------------------------------------------------

def apply_resolution_to_basis(xh, q, nh, nbanks, npts, qp, dq, res_file):
    """Convolve each Hermite basis function with the instrument resolution.

    If the Q grid is non-uniform (std of dQ >= 0.01) the basis is first
    interpolated to a uniform grid, convolved, then interpolated back.

    Parameters
    ----------
    xh : ndarray, shape (len_max, nh, nbanks)
    q  : ndarray, shape (len_max, nbanks)
    nh, nbanks : int
    npts : array of int
    qp  : float — characteristic Q scale
    dq  : float — Q spacing of the lowest-Q bank
    res_file : str

    Returns
    -------
    ndarray, same shape as *xh*
    """
    cfg = _load_resolution_config(res_file)
    res_type = cfg.get('resolution_type', '').lower()
    y = xh.copy()

    for ib in range(nbanks):
        N = npts[ib]
        diffs = np.diff(q[:N, ib])
        uniform = diffs.std() < 1e-2
        dx = dq if uniform else 0.01

        if not uniform:
            # Interpolate each Hermite function to a uniform x = Q/qp grid
            x_orig = q[:N, ib] / qp
            x_uni = np.arange(x_orig[0], x_orig[-1] + dx / 2, dx)
            N_uni = len(x_uni)
            xh_uni = np.zeros((N_uni, nh))
            for ih in range(nh):
                xh_uni[:, ih] = np.interp(x_uni, x_orig, xh[:N, ih, ib])
        else:
            xh_uni = xh[:N, :, ib]   # (N, nh) view
            N_uni = N

        if res_type == 'gaussian':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            kernel = _gaussian_kernel(sigma, dx)
            for ih in range(nh):
                xh_uni[:, ih] = convolve(xh_uni[:, ih], kernel, mode='same') * dx

        elif res_type == 'gaussian_tof':
            a = _get_param(cfg, 'a', ib, nbanks)
            c_key = f'c{ib + 1}' if nbanks > 1 else 'c'
            c_val = cfg.get(c_key, cfg.get('c', ''))
            c_offset = float(c_val) if c_val else 0.0
            xh_uni = _apply_gauss_tof_basis(xh_uni, N_uni, nh, a, dx, c_offset, qp,
                                             q[0, ib])

        elif res_type == 'pseudo_voigt':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            gamma = _get_param(cfg, 'gamma', ib, nbanks)
            ratio = _get_param(cfg, 'ratio', ib, nbanks)
            kernel = _pseudo_voigt_kernel(sigma, gamma, ratio, dx)
            for ih in range(nh):
                xh_uni[:, ih] = convolve(xh_uni[:, ih], kernel, mode='same') * dx

        elif res_type == 'back_to_back':
            sigma = _get_param(cfg, 'sigma', ib, nbanks)
            gamma = _get_param(cfg, 'gamma', ib, nbanks)
            ratio = _get_param(cfg, 'ratio', ib, nbanks)
            lam_up = _get_param(cfg, 'lambda_up', ib, nbanks)
            lam_down = _get_param(cfg, 'lambda_down', ib, nbanks)
            kernel = _back_to_back_kernel(sigma, gamma, ratio, lam_up, lam_down, dx)
            for ih in range(nh):
                xh_uni[:, ih] = convolve(xh_uni[:, ih], kernel, mode='same') * dx

        if not uniform:
            # Interpolate back to the original (possibly non-uniform) Q grid
            x_orig = q[:N, ib] / qp
            for ih in range(nh):
                y[:N, ih, ib] = np.interp(x_orig, x_uni, xh_uni[:, ih])
        else:
            y[:N, :, ib] = xh_uni

    return y


def _apply_gauss_tof_basis(xh_bank, N, nh, a, dq, c_offset, qp, q_min):
    """Apply gaussian_tof broadening to the (N, nh) Hermite basis for one bank.

    For each output position j the Gaussian width is
    sigma_j = (j+1)*a*dq + c_offset (uniform-spacing case).

    The result at position j is computed as a weighted sum over nearby rows of
    xh_bank, using numpy dot products so all nh functions are handled together.

    Returns
    -------
    ndarray, shape (N, nh)
    """
    result = xh_bank.copy()
    for j in range(N):
        sigma = (j + 1) * a * dq + c_offset
        if sigma < 0.01:
            continue   # identity
        half_w = int(np.floor(10.0 * sigma / dq))
        k0 = max(0, j - half_w)
        k1 = min(N, j + half_w + 1)
        offsets = np.arange(k0, k1) - j
        g = (np.exp(-offsets ** 2 * dq ** 2 / (2 * sigma ** 2))
             / np.sqrt(2 * np.pi * sigma ** 2) * dq)
        # (nh,) = (k1-k0, nh).T @ (k1-k0,)
        result[j, :] = xh_bank[k0:k1, :].T @ g
    return result


def _get_param(cfg, key, ibank, nbanks):
    """Read a scalar parameter that may be bank-indexed (key1, key2, …)."""
    if nbanks == 1:
        val = cfg.get(key, '')
    else:
        val = cfg.get(f'{key}{ibank + 1}', '')
    if not val:
        raise KeyError(f"Resolution parameter '{key}' not found for bank {ibank + 1}.")
    return float(val)


# ---------------------------------------------------------------------------
# Q-space convolution  (the r-space cutoff integral)
# ---------------------------------------------------------------------------

def conv1(rmax, q, qiq, npts):
    """Apply the Q-space convolution corresponding to an r-space cutoff at rmax.

    For each output Q_m:
      Qi_new(Q_m) = (1/pi) * sum_n [ Qi(Q_n)
                     * (sin(|Q_n-Q_m|*rmax)/|Q_n-Q_m| - sin((Q_n+Q_m)*rmax)/(Q_n+Q_m))
                     * dQ_n ]
    with the special case |Q_n - Q_m| = 0:
      kernel = rmax - sin((Q_n+Q_m)*rmax)/(Q_n+Q_m)

    Parameters
    ----------
    rmax : float
    q    : ndarray, shape (len_max, nbanks)
    qiq  : ndarray, shape (len_max, nbanks)
    npts : array of int

    Returns
    -------
    ndarray, same shape as *qiq*
    """
    nbanks = q.shape[1]
    out = qiq.copy()

    for ib in range(nbanks):
        N = npts[ib]
        Q = q[:N, ib]
        IQ = qiq[:N, ib]

        # dQ per point: spacing to next point; last point reuses previous spacing
        dq_arr = np.empty(N)
        dq_arr[:-1] = np.abs(np.diff(Q))
        dq_arr[-1] = dq_arr[-2]

        # N x N arrays (m = row, n = column)
        B = np.abs(Q[:, None] - Q[None, :])   # |Q_n - Q_m|
        T = Q[:, None] + Q[None, :]            # Q_n + Q_m

        with np.errstate(divide='ignore', invalid='ignore'):
            kernel = np.where(
                B == 0,
                rmax - np.sin(T * rmax) / T,
                np.sin(B * rmax) / B - np.sin(T * rmax) / T,
            )

        # Sum over n (columns): (N,) = (N,N) @ (N,) element-wise then sum
        out[:N, ib] = (kernel * IQ[None, :] * dq_arr[None, :]).sum(axis=1) / np.pi

    return out


# ---------------------------------------------------------------------------
# Chebyshev pre-fitting (background removal)
# ---------------------------------------------------------------------------

def prefit_chebyshev(q, qiq, nbanks, npts, qfitmin, Cheb, nc):
    """Fit and subtract a Chebyshev polynomial background from Qi(Q).

    Uses least-squares (equivalent to fminsearch on a linear model) restricted
    to the range Q > qfitmin.

    Returns
    -------
    yfit : ndarray, shape (len_max, nbanks) — fitted background
    qiq_corrected : ndarray — Qi(Q) with background subtracted
    """
    yfit = np.zeros_like(qiq)
    qiq_corrected = qiq.copy()

    for ib in range(nbanks):
        N = npts[ib]
        Q = q[:N, ib]
        Y = qiq[:N, ib]
        npuse = nc[ib]
        if npuse == 0:
            continue

        C = Cheb[:N, :npuse, ib]
        mask = Q > qfitmin
        p, _, _, _ = np.linalg.lstsq(C[mask], Y[mask], rcond=None)
        yfit[:N, ib] = C @ p
        qiq_corrected[:N, ib] = Y - yfit[:N, ib]

    return yfit, qiq_corrected


# ---------------------------------------------------------------------------
# Coefficient fitting
# ---------------------------------------------------------------------------

def fit_coefficients(xh, qiq, nbanks, npts, nh):
    """Fit Hermite expansion coefficients by least squares across all banks.

    Assembles the full design matrix from all banks and solves the system
    using numpy's least-squares solver (equivalent to MATLAB's backslash).

    Parameters
    ----------
    xh   : ndarray, shape (len_max, nh, nbanks)  [or (len_max, nh+nc, nbanks)]
    qiq  : ndarray, shape (len_max, nbanks)
    nbanks, npts, nh : int / array

    Returns
    -------
    coefficients : ndarray, shape (ncols,)
    """
    ncols = xh.shape[1]
    total_pts = int(npts.sum())
    hval = np.zeros((total_pts, ncols))
    qiqval = np.zeros(total_pts)

    row = 0
    for ib in range(nbanks):
        N = npts[ib]
        hval[row:row + N, :] = xh[:N, :, ib]
        qiqval[row:row + N] = qiq[:N, ib]
        row += N

    coefficients, _, _, _ = np.linalg.lstsq(hval, qiqval, rcond=None)
    return coefficients


# ---------------------------------------------------------------------------
# D(r) computation
# ---------------------------------------------------------------------------

def compute_dr(xxx, nh, qp, coefficients):
    """Compute D(r) from fitted Hermite coefficients.

    Evaluates the analytic inverse Fourier transform of the Hermite expansion:
      D(r) = qp * sqrt(2/pi) * sum_{ih=0}^{nh-1} (-1)^ih * c_ih * H_{2ih+1}(x)
    where x = Q/qp and Q ranges over the fine grid *xxx* (already in Q/qp units).

    Parameters
    ----------
    xxx          : ndarray, shape (M,) — dimensionless grid, Q/qp
    nh           : int
    qp           : float
    coefficients : ndarray, shape (>= nh,)

    Returns
    -------
    Dr : ndarray, shape (M,)
    r  : ndarray, shape (M,) — physical r grid in Å (= xxx / qp)
    xxh1 : ndarray, shape (M, nh) — Hermite functions on the fine grid
    """
    M = len(xxx)
    xxh1 = (_hermite_basis_forward_c if _USE_C_HERMITE
            else _hermite_basis_forward)(xxx, nh)

    signs = np.array([(-1) ** ih for ih in range(nh)], dtype=float)
    Dr = xxh1 @ (signs * coefficients[:nh]) * qp * np.sqrt(2.0 / np.pi)
    r = xxx / qp
    return Dr, r, xxh1
