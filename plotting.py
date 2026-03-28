"""
Plotting functions for the PDFHermite package.

Generates publication-quality figures of the Hermite fit to Qi(Q) and the
resulting pair distribution function D(r), saved as vector PDF files.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; change to 'TkAgg' for windows
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_FONTSIZE = 18
_FONT = 'DejaVu Sans'   # closest freely available equivalent to Helvetica


def _style_axes(ax):
    """Apply publication-quality styling to an Axes object."""
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
        label.set_fontname(_FONT)
        label.set_fontsize(_FONTSIZE)
        label.set_fontweight('normal')
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontname(_FONT)
            text.set_fontsize(_FONTSIZE)
        leg.get_frame().set_visible(False)
    ax.tick_params(length=6, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.tick_params(labelsize=_FONTSIZE)
    ax.figure.patch.set_facecolor('white')


# ---------------------------------------------------------------------------
# Qi(Q) fit plots
# ---------------------------------------------------------------------------

def plot_qiq_fit(q, qiq, qiqhermite, npts, nbanks, out_dir, yfit=None):
    """Save one Qi(Q) fit plot per bank.

    Each plot shows the data (black), Hermite fit (red), and residual (blue)
    offset below for clarity.  If *yfit* is supplied (Chebyshev background)
    it is drawn in cyan.

    Parameters
    ----------
    q, qiq, qiqhermite : ndarray, shape (len_max, nbanks)
    npts   : array of int
    nbanks : int
    out_dir : str — directory for output PDFs
    yfit   : ndarray or None — Chebyshev background array
    """
    import os
    colors_pre = {'data': 'k', 'fit': 'r', 'diff': 'b', 'cheb': 'c'}
    for ib in range(nbanks):
        N = npts[ib]
        Q = q[:N, ib]
        Y = qiq[:N, ib]
        Yfit = qiqhermite[:N, ib]
        offset = min(np.min(Yfit), np.min(Y)) * 1.1
        diff_curve = Y - Yfit + offset

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(Q, Y, '-', color=colors_pre['data'], label='Qi(Q)')
        ax.plot(Q, Yfit, '-', color=colors_pre['fit'], label='Fitted')
        ax.plot(Q, diff_curve, '-', color=colors_pre['diff'], label='Difference')
        if yfit is not None:
            ax.plot(Q, yfit[:N, ib], '-', color=colors_pre['cheb'], label='Chebyshev bg')

        ymin = min(np.min(Yfit), np.min(Y)) * 1.3
        ymax = max(np.max(Yfit), np.max(Y)) * 1.1
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'Q ($\AA^{-1}$)')
        ax.set_ylabel('Qi(Q)')
        ax.legend(frameon=False)
        ax.box = True
        _style_axes(ax)
        plt.tight_layout()

        if nbanks > 1:
            fname = os.path.join(out_dir, f'Qi(Q)_bank{ib + 2}.pdf')
        else:
            fname = os.path.join(out_dir, 'Qi(Q).pdf')
        fig.savefig(fname, format='pdf', bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# D(r) plots
# ---------------------------------------------------------------------------

def plot_dr(r, Dr, out_path, xlim=(0, 25)):
    """Save a D(r) plot.

    Parameters
    ----------
    r       : ndarray — r values in Å
    Dr      : ndarray — D(r) values
    out_path: str
    xlim    : tuple
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, Dr, 'k-')
    ymin = np.min(Dr) * 1.2
    ymax = np.max(Dr) * 1.05
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r'r ($\AA$)')
    ax.set_ylabel('D(r)')
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_dr_with_reference(r, Dr, drdata, out_path, xlim=(0, 25)):
    """Save a D(r) plot overlaid with a reference PDF.

    Parameters
    ----------
    r, Dr   : computed D(r)
    drdata  : ndarray, shape (M, 2) — reference (r, D(r))
    out_path: str
    """
    r_ref, Dr_ref = drdata[:, 0], drdata[:, 1]
    A = len(Dr)
    offset = min(np.min(Dr), np.min(Dr_ref)) * 1.1
    diff_curve = Dr_ref[:A] - Dr + offset

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_ref, Dr_ref, 'k-', label='Original PDF')
    ax.plot(r, Dr, 'r-', label='Reconstructed PDF')
    ax.plot(r, diff_curve, 'b-', label='Difference')

    ymin = min(np.min(Dr), np.min(Dr_ref)) * 1.2
    ymax = max(np.max(Dr), np.max(Dr_ref)) * 1.1
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r'r ($\AA$)')
    ax.set_ylabel('D(r)')
    ax.legend(frameon=False)
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_dr_resolution_comparison(r, Dr_no_res, Dr_with_res, out_path, xlim=(0, 25)):
    """Save a plot comparing D(r) with and without instrument resolution.

    Parameters
    ----------
    Dr_no_res   : D(r) computed without resolution correction
    Dr_with_res : D(r) computed with resolution correction
    """
    offset = min(np.min(Dr_no_res), np.min(Dr_with_res)) * 1.1
    diff_curve = Dr_with_res - Dr_no_res + offset

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, Dr_with_res, 'k-', label='without resolution function')
    ax.plot(r, Dr_no_res, 'r-', label='with resolution function')
    ax.plot(r, diff_curve, 'b-', label='difference')

    ymin = min(np.min(Dr_no_res), np.min(Dr_with_res)) * 1.2
    ymax = max(np.max(Dr_no_res), np.max(Dr_with_res)) * 1.05
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r'r ($\AA$)')
    ax.set_ylabel('D(r)')
    ax.legend(frameon=False)
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pre-fitting diagnostic plots
# ---------------------------------------------------------------------------

def plot_prefit_iq(q, iq_before, iq_after, npts, ibank, out_dir):
    """Save i(Q) before/after Chebyshev pre-fitting for one bank."""
    import os
    N = npts[ibank]
    Q = q[:N, ibank]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='0.75', linewidth=0.8, zorder=0)
    ax.plot(Q, iq_before[:N], '-', color='lightcoral', label='before pre-fitting')
    ax.plot(Q, iq_after[:N], '-', color='k', label='after pre-fitting')
    ax.set_title(f'data of bank{ibank + 1}')
    ax.set_xlabel(r'Q ($\AA^{-1}$)')
    ax.set_ylabel('i(Q)')
    ax.legend(frameon=False)
    _style_axes(ax)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'prefit_iq_bank{ibank + 1}.pdf')
    fig.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_prefit_qiq(q, qiq_before, qiq_after, yfit, npts, ibank, out_dir):
    """Save Qi(Q) before/after Chebyshev pre-fitting for one bank."""
    import os
    N = npts[ibank]
    Q = q[:N, ibank]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color='0.75', linewidth=0.8, zorder=0)
    ax.plot(Q, qiq_before[:N, ibank], '-', color='lightcoral', label='before pre-fitting')
    ax.plot(Q, qiq_after[:N, ibank], '-', color='k', label='after pre-fitting')
    ax.plot(Q, yfit[:N, ibank], '-', color='b', label='pre-fitting chebyshev')
    ax.set_title(f'data of bank{ibank + 1}')
    ax.set_xlabel(r'Q ($\AA^{-1}$)')
    ax.set_ylabel('Qi(Q)')
    ax.legend(frameon=False)
    _style_axes(ax)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'prefit_qiq_bank{ibank + 1}.pdf')
    fig.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close(fig)
