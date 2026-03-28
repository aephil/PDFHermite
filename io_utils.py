"""
Input/output utilities for the PDFHermite package.

Handles config file parsing, scattering data loading (individual files and
Gudrun multi-bank format), and output file writing.
"""

import os
import numpy as np


def read_config(filename):
    """Parse a PDFHermite configuration file with KEY :: VALUE syntax.

    Lines are case-insensitive. Whitespace around keys and values is stripped.
    The keywords ``data_files`` and ``gudrun_files`` are special: their integer
    value gives the number of detector banks, and the filenames follow on
    subsequent lines (one file for Gudrun format, N files for data_files format).

    Parameters
    ----------
    filename : str
        Path to the ``.dat`` configuration file.

    Returns
    -------
    dict
        Dictionary of configuration key-value pairs.  All keys are lower-case.
        Numeric values are stored as strings; the caller converts as needed.
        The key ``'data_filenames'`` contains a list of data file names when
        ``data_files`` or ``gudrun_files`` is present.
    """
    config = {}
    with open(filename, 'r') as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if '::' in line:
            key_part, _, val_part = line.partition('::')
            key = key_part.strip().lower()
            val = val_part.strip()

            if key in ('data_files', 'gudrun_files'):
                nbanks = int(val)
                config[key] = nbanks
                # For gudrun_files there is exactly one data file; for
                # data_files there are nbanks files (one per bank).
                max_files = 1 if key == 'gudrun_files' else nbanks
                filenames = []
                j = i + 1
                while j < len(lines) and len(filenames) < max_files:
                    next_line = lines[j].strip()
                    j += 1
                    if not next_line:
                        continue
                    if '::' in next_line:
                        j -= 1  # back up so outer loop sees this keyword
                        break
                    if key == 'data_files':
                        # Format: "N filename.ext" — strip the leading digit
                        parts = next_line.split(None, 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            filenames.append(parts[1])
                        else:
                            filenames.append(next_line)
                    else:
                        filenames.append(next_line)
                config['data_filenames'] = filenames
                i = j
                continue
            else:
                config[key] = val
        i += 1

    return config


def read_data(config, config_dir):
    """Load scattering data described by *config*.

    Supports two formats:
    - ``data_files``: individual two-column (Q, intensity) ASCII files, one per
      detector bank.
    - ``gudrun_files``: a single Gudrun ``.dcs01`` file with 14 comment-line
      header rows; columns are Q, I1, err1, I2, err2, … (11 columns for 5 banks).

    The self-scattering term is subtracted if ``self_term`` is given.  Data are
    then converted to Qi(Q) according to ``data_type``:
    - ``i(q)``:  Qi(Q) = I(Q) * Q
    - ``s(q)``:  Qi(Q) = (S(Q) - 1) * Q

    Only rows with ``qfitmin < Q <= qfitmax`` are kept.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary from :func:`read_config`.
    config_dir : str
        Directory containing the configuration file; data file paths are
        resolved relative to this directory.

    Returns
    -------
    q : ndarray, shape (len_max, nbanks)
        Q values, zero-padded to the maximum bank length.
    qiq : ndarray, shape (len_max, nbanks)
        Qi(Q) values, zero-padded.
    qrange : ndarray, shape (2, nbanks)
        Effective Q range [qmin, qmax] per bank after clipping to fit bounds.
    npts : ndarray of int, shape (nbanks,)
        Number of valid data points per bank.
    """
    qfitmin = float(config.get('qfitmin', 0.0))
    qfitmax = float(config.get('qfitmax', 50.0))
    data_type = config.get('data_type', '').strip().lower()
    self_term_str = config.get('self_term', '').strip()
    self_term = float(self_term_str) if self_term_str else None

    filenames = config.get('data_filenames', [])

    if 'data_files' in config:
        return _read_data_files(
            filenames, config_dir, qfitmin, qfitmax, data_type, self_term)
    elif 'gudrun_files' in config:
        nbanks = config['gudrun_files']
        return _read_gudrun_file(
            filenames[0], config_dir, nbanks, qfitmin, qfitmax,
            data_type, self_term)
    else:
        raise ValueError("Config must contain either 'data_files' or 'gudrun_files'.")


def _apply_data_type(Q, I, data_type, self_term):
    """Subtract self term and convert to Qi(Q).

    If *data_type* is empty the data are already Qi(Q) and are used as-is
    (after self_term subtraction).  Valid non-empty values are ``'i(q)'``
    (multiply by Q) and ``'s(q)'`` (subtract 1, then multiply by Q).
    """
    if self_term is not None:
        I = I - self_term
    if data_type == 'i(q)':
        return I * Q
    elif data_type == 's(q)':
        return (I - 1.0) * Q
    elif data_type == '':
        return I   # already Qi(Q)
    else:
        raise ValueError(f"Unknown data_type '{data_type}'. Use 'i(q)' or 's(q)'.")


def _filter_q_range(Q, QiQ, qfitmin, qfitmax):
    """Return arrays filtered to qfitmin < Q <= qfitmax."""
    mask = (Q > qfitmin) & (Q <= qfitmax)
    return Q[mask], QiQ[mask]


def _pack_banks(bank_q, bank_qiq, nbanks):
    """Stack per-bank arrays into zero-padded 2-D arrays."""
    npts = np.array([len(bq) for bq in bank_q], dtype=int)
    len_max = int(npts.max())
    q = np.zeros((len_max, nbanks))
    qiq = np.zeros((len_max, nbanks))
    qrange = np.zeros((2, nbanks))
    for ib, (bq, bqiq) in enumerate(zip(bank_q, bank_qiq)):
        n = npts[ib]
        q[:n, ib] = bq
        qiq[:n, ib] = bqiq
        qrange[0, ib] = bq[0]
        qrange[1, ib] = bq[-1]
    return q, qiq, qrange, npts


def _read_data_files(filenames, config_dir, qfitmin, qfitmax, data_type, self_term):
    """Read individual data files with at least two columns (Q, I[, err, …]).

    Automatically skips header lines that cannot be parsed as numbers.
    Only the first two columns are used.
    """
    nbanks = len(filenames)
    bank_q, bank_qiq = [], []
    for fname in filenames:
        path = os.path.join(config_dir, fname)
        # Count header lines: skip any leading lines that cannot be parsed
        # as at least two whitespace-separated floats.
        skiprows = 0
        with open(path) as _fh:
            for _line in _fh:
                parts = _line.split()
                try:
                    if len(parts) >= 2:
                        float(parts[0]); float(parts[1])
                        break   # valid data line found
                except ValueError:
                    pass
                skiprows += 1
        data = np.loadtxt(path, usecols=(0, 1), skiprows=skiprows)
        Q, I = data[:, 0], data[:, 1]
        QiQ = _apply_data_type(Q, I, data_type, self_term)
        Qf, QiQf = _filter_q_range(Q, QiQ, qfitmin, qfitmax)
        bank_q.append(Qf)
        bank_qiq.append(QiQf)
    return _pack_banks(bank_q, bank_qiq, nbanks)


def _read_gudrun_file(filename, config_dir, nbanks, qfitmin, qfitmax,
                      data_type, self_term):
    """Read a Gudrun multi-bank data file.

    The file has 14 header lines (all starting with ``#``) followed by rows of
    Q, I1, err1, I2, err2, …  Only rows where the bank intensity is positive
    are retained.
    """
    path = os.path.join(config_dir, filename)
    raw = np.loadtxt(path, skiprows=14)
    Q_all = raw[:, 0]

    bank_q, bank_qiq = [], []
    for ib in range(nbanks):
        col = 2 * ib + 1   # column index for bank ib intensity (0-based)
        I_all = raw[:, col]
        # Keep only rows where the bank has positive (measured) intensity
        pos_mask = I_all > 0
        Q = Q_all[pos_mask]
        I = I_all[pos_mask]
        QiQ = _apply_data_type(Q, I, data_type, self_term)
        Qf, QiQf = _filter_q_range(Q, QiQ, qfitmin, qfitmax)
        bank_q.append(Qf)
        bank_qiq.append(QiQf)

    return _pack_banks(bank_q, bank_qiq, nbanks)


def write_dr(filename, r, Dr):
    """Write the pair distribution function D(r) to a two-column ASCII file.

    Parameters
    ----------
    filename : str
        Output file path.
    r : array_like
        r values in Å.
    Dr : array_like
        D(r) values.
    """
    data = np.column_stack([np.asarray(r), np.asarray(Dr)])
    with open(filename, 'w') as fh:
        for row in data:
            fh.write(f'{row[0]:6.2f} {row[1]:12.6f} \n')


def write_qiq(filename, q_banks, qiq_banks, npts):
    """Write Hermite-fitted Qi(Q) for all banks to an ASCII file.

    Parameters
    ----------
    filename : str
        Output file path.
    q_banks : ndarray, shape (len_max, nbanks)
    qiq_banks : ndarray, shape (len_max, nbanks)
    npts : array of int
    """
    nbanks = q_banks.shape[1]
    with open(filename, 'w') as fh:
        for ib in range(nbanks):
            n = npts[ib]
            for k in range(n):
                fh.write(f'{q_banks[k, ib]:10.4f} {qiq_banks[k, ib]:14.8f}\n')
            fh.write('\n')
