"""
PDFHermite: fit Hermite functions to total scattering data Qi(Q) and compute
the pair distribution function D(r) as their analytic Fourier transform.

Inspired by:
  Krylov, A.S. & Vvedenskii, A.V. (1995). Software package for radial
  distribution function calculation. J. Non-Cryst. Solids 192-193, 683-687.
  doi:10.1016/0022-3093(95)00424-6

Project devised by Martin Dove, implemented by Min Gao, Yinze Qin and
Martin Dove, tested by Martin Dove (MATLAB version, finalised May 2023).
Python port by Claude Code, March 2026.

Usage
-----
    python hermite_fit.py <rootname>

where <rootname>.dat is the configuration file.  Outputs are written to the
same directory as the configuration file.
"""

import sys
import os
import numpy as np

import io_utils
import functions
import plotting


def run(rootname):
    """Execute the full PDFHermite fitting pipeline for *rootname*.

    Parameters
    ----------
    rootname : str
        Base name (with path) of the ``.dat`` configuration file, without the
        ``.dat`` extension.
    """
    # -----------------------------------------------------------------------
    # File paths
    # -----------------------------------------------------------------------
    dat_file = rootname + '.dat'
    config_dir = os.path.dirname(os.path.abspath(dat_file))
    out_qiqfile = rootname + '.qiqout'
    out_drfile = rootname + '.drout'

    # -----------------------------------------------------------------------
    # Read configuration
    # -----------------------------------------------------------------------
    config = io_utils.read_config(dat_file)

    qfitmax = float(config.get('qfitmax', 50.0))
    qfitmin = float(config.get('qfitmin', 0.0))
    rmax = float(config.get('rmax', 30.0))
    rspacing = float(config.get('rspacing', 0.02))
    data_type = config.get('data_type', 'i(q)').lower()

    convolution = config.get('convolution', 'false').lower() == 'true'
    lorch = config.get('lorch', 'false').lower() == 'true'
    pre_fitting = config.get('pre_fitting', 'false').lower() == 'true'
    chebyshev_fitting = config.get('chebyshev_fitting', 'false').lower() == 'true'
    broaden_data = config.get('broaden_data', 'false').lower() == 'true'
    fit_with_resolution = (
        config.get('fit_with_instrument_resolution', 'false').lower() == 'true')

    chebyshev_str = config.get('chebyshev', '0').strip()
    chebyshev_order = int(float(chebyshev_str)) if chebyshev_str else 0
    if chebyshev_order == 0:
        pre_fitting = False
        chebyshev_fitting = False

    nhermites_str = config.get('nhermites', '').strip().lower()
    try:
        nhermites = int(float(nhermites_str))
    except (ValueError, TypeError):
        nhermites = None   # 'default' or empty → compute from rmax and qmax

    pdf_file_str = config.get('pdf_file', '').strip()
    pdf_file = os.path.join(config_dir, pdf_file_str) if pdf_file_str else None

    res_file_str = config.get('resolution_information', '').strip()
    res_file = os.path.join(config_dir, res_file_str) if res_file_str else None

    broaden_file_str = config.get('broaden_information', '').strip()
    broaden_file = os.path.join(config_dir, broaden_file_str) if broaden_file_str else None

    # -----------------------------------------------------------------------
    # Load scattering data
    # -----------------------------------------------------------------------
    q, qiq, qrange, npts = io_utils.read_data(config, config_dir)
    nbanks = q.shape[1]
    len_max = q.shape[0]

    # -----------------------------------------------------------------------
    # Determine number of Hermite functions (nh) and characteristic scale (qp)
    # -----------------------------------------------------------------------
    qmax = np.max(q)
    if nhermites is not None:
        rmax = round(4 * nhermites / qmax)
        nh = nhermites
    else:
        nh = int(round(rmax * qmax / 4))

    print(f'The nhermites is: {nh}')
    print(f'The rmax is: {rmax}')

    # qp = sqrt(qmax_data / rmax) is the characteristic Q scale Q'
    qp = np.sqrt(np.max(qrange[1, :]) / rmax)

    # -----------------------------------------------------------------------
    # Identify the lowest-Q bank and extrapolate down towards Q = 0
    # -----------------------------------------------------------------------
    # Q spacing for each bank (difference between first two data points)
    dq = q[1, :] - q[0, :]

    qmin = np.min(q[q > 0])
    minbank = int(np.where(q[0, :] == qmin)[0][0])
    qiqinit = qiq[0, minbank]    # qiqoffset is 0 throughout

    print(f'The initial value of Qi(Q) is: {qiqinit}')

    dq_min = dq[minbank]
    nadd = int(round(qmin / dq_min)) - 1 if (qmin / dq_min) > 0 else 0

    if nadd > 0:
        # Resize arrays if needed to accommodate the extra points
        needed = npts[minbank] + nadd
        if needed > len_max:
            extra = needed - len_max
            q = np.vstack([q, np.zeros((extra, nbanks))])
            qiq = np.vstack([qiq, np.zeros((extra, nbanks))])
            len_max = q.shape[0]

        # Shift existing data forward
        N = npts[minbank]
        q[nadd:nadd + N, minbank] = q[:N, minbank]
        qiq[nadd:nadd + N, minbank] = qiq[:N, minbank]

        # Fill extrapolated points with linear ramp from 0 to qiqinit
        for k in range(1, nadd + 1):
            q[k - 1, minbank] = dq_min * k
            qiq[k - 1, minbank] = qiqinit * (dq_min * k) / qmin

        npts[minbank] += nadd

    # -----------------------------------------------------------------------
    # Build Hermite function basis
    # -----------------------------------------------------------------------
    print('Building Hermite basis...')
    xh = functions.build_hermite_basis(q, npts, nh, qp)

    # Save unbroadened copy for resolution comparison
    if fit_with_resolution:
        xh1 = xh.copy()

    # -----------------------------------------------------------------------
    # Optionally apply instrument resolution to the Hermite basis
    # -----------------------------------------------------------------------
    if fit_with_resolution and res_file:
        print('Applying instrument resolution to Hermite basis...')
        xh = functions.apply_resolution_to_basis(
            xh, q, nh, nbanks, npts, qp, dq_min, res_file)

    # -----------------------------------------------------------------------
    # Optional Chebyshev background pre-fitting
    # -----------------------------------------------------------------------
    yfit_cheb = None   # Chebyshev background curves (for plotting)

    if pre_fitting or chebyshev_fitting:
        print('Building Chebyshev basis...')
        Cheb, nc = functions.chebyshev_basis(
            q, qfitmax, nbanks, npts, chebyshev_order)

        # Apply Lorch damping to Chebyshev basis if requested
        if lorch:
            for ib in range(nbanks):
                N = npts[ib]
                L = np.sinc(q[:N, ib] / qfitmax)
                Cheb[:N, :nc[ib], ib] *= L[:, None]

        if pre_fitting:
            # Save original Qi(Q) before background removal (for plots)
            qiq_before_prefit = qiq.copy()
            yfit_cheb, qiq = functions.prefit_chebyshev(
                q, qiq, nbanks, npts, qfitmin, Cheb, nc)

            # Diagnostic pre-fitting plots
            for ib in range(nbanks):
                N = npts[ib]
                iq_before = qiq_before_prefit[:N, ib] / q[:N, ib]
                iq_after = qiq[:N, ib] / q[:N, ib]
                plotting.plot_prefit_iq(q, iq_before[None, :].T,
                                        iq_after[None, :].T,
                                        npts, ib, config_dir)
                plotting.plot_prefit_qiq(q, qiq_before_prefit, qiq,
                                         yfit_cheb, npts, ib, config_dir)

        if chebyshev_fitting:
            # Combine Hermite and Chebyshev columns into a single design matrix
            nc_max = int(np.max(nc))
            xhch = np.zeros((len_max, nh + nc_max, nbanks))
            for ib in range(nbanks):
                xhch[:, :nh, ib] = xh[:, :, ib]
                xhch[:, nh:nh + nc[ib], ib] = Cheb[:, :nc[ib], ib]
            fit_basis = xhch
        else:
            fit_basis = xh
    else:
        fit_basis = xh

    # -----------------------------------------------------------------------
    # Optional copy of qiq for broadening comparison
    # -----------------------------------------------------------------------
    if broaden_data and fit_with_resolution:
        qiq1 = qiq.copy()

    # -----------------------------------------------------------------------
    # Optional Q-space convolution (r-space cutoff at rmax)
    # -----------------------------------------------------------------------
    if convolution:
        print('Applying Q-space convolution...')
        qiq = functions.conv1(rmax, q, qiq, npts)
        if broaden_data and fit_with_resolution:
            qiq1 = functions.conv1(rmax, q, qiq1, npts)

    # -----------------------------------------------------------------------
    # Optional Lorch window correction
    # -----------------------------------------------------------------------
    if lorch:
        for ib in range(nbanks):
            N = npts[ib]
            L = np.sinc(q[:N, ib] / qfitmax)
            qiq[:N, ib] *= L
        if broaden_data and fit_with_resolution:
            for ib in range(nbanks):
                N = npts[ib]
                L = np.sinc(q[:N, ib] / qfitmax)
                qiq1[:N, ib] *= L

    # -----------------------------------------------------------------------
    # Optional data broadening (instrument resolution on Qi(Q) itself)
    # -----------------------------------------------------------------------
    if broaden_data and broaden_file:
        print('Applying broadening to Qi(Q)...')
        qiq = functions.apply_broadening(
            qiq, nbanks, npts, len_max, dq_min, broaden_file)

    # -----------------------------------------------------------------------
    # Least-squares fit: solve for Hermite (and optional Chebyshev) coefficients
    # -----------------------------------------------------------------------
    print('Fitting Hermite coefficients...')
    coefficients = functions.fit_coefficients(fit_basis, qiq, nbanks, npts, nh)

    if fit_with_resolution:
        # Also fit with the unbroadened basis (naive, without deconvolution)
        coefficients1 = functions.fit_coefficients(xh1, qiq, nbanks, npts, nh)

    # -----------------------------------------------------------------------
    # Reconstruct Hermite-fitted Qi(Q) and compute residuals per bank
    # -----------------------------------------------------------------------
    qiqhermite = np.zeros_like(qiq)
    if fit_with_resolution:
        deconvolved_qiq = np.zeros_like(qiq)

    for ib in range(nbanks):
        N = npts[ib]
        if chebyshev_fitting:
            # Use only the Hermite part of the combined basis for the fit curve
            qiqhermite[:N, ib] = fit_basis[:N, :nh, ib] @ coefficients[:nh]
        else:
            qiqhermite[:N, ib] = fit_basis[:N, :, ib] @ coefficients

        if fit_with_resolution:
            # Deconvolved Qi(Q): unbroadened basis × resolution-corrected coefficients
            deconvolved_qiq[:N, ib] = xh1[:N, :, ib] @ coefficients[:nh]

    # -----------------------------------------------------------------------
    # Plot Qi(Q) fit for each bank and save PDFs
    # -----------------------------------------------------------------------
    print('Saving Qi(Q) fit plots...')
    plotting.plot_qiq_fit(
        q, qiq, qiqhermite, npts, nbanks, config_dir,
        yfit=yfit_cheb if pre_fitting else None)

    # -----------------------------------------------------------------------
    # Compute D(r) on a fine r-space grid
    # -----------------------------------------------------------------------
    print('Computing D(r)...')
    qmax_all = np.max(qrange[1, :])
    xxx = np.arange(rspacing, qmax_all + rspacing / 2, rspacing) / qp

    Dr, r, xxh1 = functions.compute_dr(xxx, nh, qp, coefficients)

    if fit_with_resolution:
        Dr1, _, _ = functions.compute_dr(xxx, nh, qp, coefficients1)

    # -----------------------------------------------------------------------
    # Plot D(r) and save PDFs
    # -----------------------------------------------------------------------
    print('Saving D(r) plots...')
    dr_pdf = os.path.join(config_dir, 'Dr.pdf')
    plotting.plot_dr(r, Dr, dr_pdf)

    dr_res_pdf = os.path.join(config_dir, 'Dr_resolution.pdf')
    if pdf_file and os.path.isfile(pdf_file):
        # Overlay with a reference D(r) from file
        drdata = np.loadtxt(pdf_file)
        if drdata[0, 0] == 0.0:
            drdata = drdata[1:, :]
        plotting.plot_dr_with_reference(r, Dr, drdata, dr_res_pdf)

    elif fit_with_resolution:
        # Compare resolution-corrected D(r) (Drcase1) with naive D(r) (Drcase2)
        # Legend follows MATLAB: Dr1 = "without resolution" (naive),
        # Dr = "with resolution" (deconvolved)
        plotting.plot_dr_resolution_comparison(r, Dr, Dr1, dr_res_pdf)
    else:
        # Nothing to compare against; save the plain D(r) again
        plotting.plot_dr(r, Dr, dr_res_pdf)

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    print(f'Writing {out_drfile} ...')
    io_utils.write_dr(out_drfile, r, Dr)

    print(f'Writing {out_qiqfile} ...')
    io_utils.write_qiq(out_qiqfile, q, qiqhermite, npts)

    print('Done.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python hermite_fit.py <rootname>')
        print('  <rootname>.dat must exist in the current directory or be a path.')
        sys.exit(1)

    rootname = sys.argv[1]
    # Strip trailing .dat if accidentally supplied
    if rootname.endswith('.dat'):
        rootname = rootname[:-4]

    run(rootname)
