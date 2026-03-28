"""
Helper utilities for the PDFHermite Streamlit web interface.

Handles temp-directory management, config round-tripping, output parsing,
and Plotly figure construction.  No modifications to the core pipeline are
needed; this module calls into the existing io_utils / hermite_fit code.
"""

import io as _io
import os
import shutil
import tempfile
import zipfile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# io_utils lives in the same package directory
import io_utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_DATASETS = [
    'ScF3_Gudrun', 'silica', 'Acetylene', 'Calcite',
    'Cristobalite', 'Quartz', 'andalusite', 'Cu2P2O7', 'Fe2O3', 'SmB6',
]

DEFAULT_PARAMS = {
    'qfitmin': 0.5,
    'qfitmax': 50.0,
    'rmax': 30.0,
    'rspacing': 0.02,
    'lorch': True,
    'convolution': True,
    'data_type': 'i(q)',
    'nhermites': 'default',
    'self_term': '',
    'chebyshev': 0,
    'pre_fitting': False,
    'chebyshev_fitting': False,
    'fit_with_instrument_resolution': False,
    'broaden_data': False,
    'resolution_information': '',
    'broaden_information': '',
    'pdf_file': '',
    '_data_filenames': [],
    '_use_gudrun': False,
    '_nbanks': 1,
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _find_dat_file(directory):
    """Return the path to the PDFHermite config .dat file in *directory*.

    Identifies the config file by the presence of '::' (KEY :: VALUE syntax),
    which distinguishes it from plain-text data files that also use .dat extension.
    """
    candidates = sorted(
        name for name in os.listdir(directory)
        if name.endswith('.dat') and not name.startswith('._')
    )
    for name in candidates:
        path = os.path.join(directory, name)
        try:
            with open(path) as fh:
                if any('::' in line for line in fh):
                    return path
        except (OSError, UnicodeDecodeError):
            continue
    raise FileNotFoundError(f'No PDFHermite .dat config file found in {directory}')


def _config_to_defaults(config):
    """Convert a parsed config dict (from io_utils.read_config) to a UI defaults dict."""
    d = DEFAULT_PARAMS.copy()

    for key in ('qfitmin', 'qfitmax', 'rmax', 'rspacing'):
        raw = config.get(key, '').strip()
        if raw:
            try:
                d[key] = float(raw)
            except ValueError:
                pass

    for key in ('data_type', 'nhermites', 'self_term',
                'resolution_information', 'broaden_information', 'pdf_file'):
        if key in config:
            d[key] = config[key]

    raw_cheb = config.get('chebyshev', '').strip()
    if raw_cheb:
        try:
            d['chebyshev'] = int(float(raw_cheb))
        except ValueError:
            pass

    for key in ('lorch', 'convolution', 'pre_fitting', 'chebyshev_fitting',
                'fit_with_instrument_resolution', 'broaden_data'):
        if key in config:
            d[key] = config[key].strip().lower() == 'true'

    d['_data_filenames'] = config.get('data_filenames', [])
    d['_use_gudrun'] = 'gudrun_files' in config
    d['_nbanks'] = config.get('gudrun_files', config.get('data_files', 1))
    if isinstance(d['_nbanks'], str):
        try:
            d['_nbanks'] = int(d['_nbanks'])
        except ValueError:
            d['_nbanks'] = 1

    return d


def load_sample_defaults(sample_name, package_dir):
    """Parse a sample dataset's .dat file and return a defaults dict for the UI."""
    dat_path = _find_dat_file(os.path.join(package_dir, sample_name))
    return _config_to_defaults(io_utils.read_config(dat_path))


def load_sample_files(sample_name, package_dir):
    """Return an ordered dict of {filename: text} for the .dat and referenced .txt files.

    Loads the config file first, then any resolution/broadening .txt files named
    in the config.  Missing or unreadable files are silently skipped.
    """
    sample_dir = os.path.join(package_dir, sample_name)
    dat_path = _find_dat_file(sample_dir)
    config = io_utils.read_config(dat_path)

    files = {}

    # Always include the .dat
    with open(dat_path) as fh:
        files[os.path.basename(dat_path)] = fh.read()

    # Include any referenced .txt control files
    for key in ('resolution_information', 'broaden_information'):
        fname = config.get(key, '').strip()
        if not fname:
            continue
        full_path = os.path.join(sample_dir, fname)
        if os.path.isfile(full_path):
            with open(full_path) as fh:
                files[fname] = fh.read()

    return files


def defaults_from_bytes(dat_bytes, dat_filename):
    """Parse a .dat file supplied as raw bytes and return a defaults dict."""
    tmp = tempfile.NamedTemporaryFile(suffix='.dat', delete=False)
    try:
        tmp.write(dat_bytes)
        tmp.close()
        config = io_utils.read_config(tmp.name)
    finally:
        os.unlink(tmp.name)
    return _config_to_defaults(config)


def write_dat_from_params(dat_path, params):
    """Write a complete .dat config file to *dat_path* from a params dict."""
    lines = []

    scalar_keys = [
        ('qfitmax',  'Qfitmax'),
        ('qfitmin',  'Qfitmin'),
        ('rmax',     'rmax'),
        ('rspacing', 'Rspacing'),
        ('data_type','data_type'),
        ('nhermites','nhermites'),
        ('self_term','self_term'),
        ('chebyshev','chebyshev'),
    ]
    for param_key, file_key in scalar_keys:
        lines.append(f'{file_key} :: {params.get(param_key, "")}')

    bool_keys = [
        ('convolution',                  'convolution'),
        ('lorch',                        'Lorch'),
        ('pre_fitting',                  'pre_fitting'),
        ('chebyshev_fitting',            'chebyshev_fitting'),
        ('fit_with_instrument_resolution','fit_with_instrument_resolution'),
        ('broaden_data',                 'broaden_data'),
    ]
    for param_key, file_key in bool_keys:
        val = 'true' if params.get(param_key, False) else 'false'
        lines.append(f'{file_key} :: {val}')

    for param_key, file_key in [
        ('resolution_information', 'resolution_information'),
        ('broaden_information',    'broaden_information'),
        ('pdf_file',               'PDF_file'),
    ]:
        lines.append(f'{file_key} :: {params.get(param_key, "")}')

    # Data file block
    filenames = params.get('_data_filenames', [])
    nbanks = params.get('_nbanks', len(filenames))
    if params.get('_use_gudrun', False):
        lines.append(f'Gudrun_files :: {nbanks}')
        if filenames:
            lines.append(filenames[0])
    else:
        lines.append(f'Data_files :: {nbanks}')
        for i, fname in enumerate(filenames):
            lines.append(f'{i + 1} {fname}')

    with open(dat_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Temp-directory management
# ---------------------------------------------------------------------------

def prepare_sample_run(sample_name, package_dir, params):
    """Copy a sample dataset directory to a fresh temp dir, write .dat from params.

    Returns the rootname (temp_dir/stem) suitable for hermite_fit.run().
    """
    sample_dir = os.path.join(package_dir, sample_name)
    original_dat = _find_dat_file(sample_dir)
    stem = os.path.splitext(os.path.basename(original_dat))[0]

    tmp = tempfile.mkdtemp(prefix='pdfhermite_')
    for name in os.listdir(sample_dir):
        src = os.path.join(sample_dir, name)
        if os.path.isfile(src) and not name.startswith('._'):
            shutil.copy2(src, tmp)

    write_dat_from_params(os.path.join(tmp, stem + '.dat'), params)
    return os.path.join(tmp, stem)


def prepare_upload_run(dat_name, upload_map, params):
    """Write uploaded files to a fresh temp dir, write .dat from params.

    Parameters
    ----------
    dat_name    : str   basename of the config file (used to derive the rootname)
    upload_map  : dict  {filename: bytes} for all non-dat data/resolution files
    params      : dict  current form parameters

    Returns the rootname for hermite_fit.run().
    """
    tmp = tempfile.mkdtemp(prefix='pdfhermite_')
    for fname, content in upload_map.items():
        with open(os.path.join(tmp, fname), 'wb') as fh:
            fh.write(content)

    stem = os.path.splitext(os.path.basename(dat_name))[0]
    write_dat_from_params(os.path.join(tmp, stem + '.dat'), params)
    return os.path.join(tmp, stem)


def cleanup_old_run(tmp_dir):
    """Remove a temp directory created by a previous run."""
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def collect_results(rootname):
    """Parse all outputs written by hermite_fit.run() into a results dict."""
    tmp_dir = os.path.dirname(rootname)
    drout_path  = rootname + '.drout'
    qiqout_path = rootname + '.qiqout'

    r, Dr       = parse_drout(drout_path)
    qiq_banks   = parse_qiqout(qiqout_path)
    pdf_paths   = sorted(
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir)
        if f.endswith('.pdf')
    )
    has_prefit = any('prefit' in os.path.basename(p) for p in pdf_paths)

    return {
        'tmp_dir':     tmp_dir,
        'rootname':    rootname,
        'drout_path':  drout_path,
        'qiqout_path': qiqout_path,
        'pdf_paths':   pdf_paths,
        'r':           r,
        'Dr':          Dr,
        'qiq_banks':   qiq_banks,
        'nbanks':      len(qiq_banks),
        'has_prefit':  has_prefit,
    }


def parse_drout(filepath):
    """Return (r, Dr) ndarrays from a .drout file."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]


def parse_qiqout(filepath):
    """Return a list of (q, qiq) ndarray pairs, one per bank, from a .qiqout file."""
    banks, q_cur, qiq_cur = [], [], []
    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                if q_cur:
                    banks.append((np.array(q_cur), np.array(qiq_cur)))
                    q_cur, qiq_cur = [], []
            else:
                parts = stripped.split()
                q_cur.append(float(parts[0]))
                qiq_cur.append(float(parts[1]))
    if q_cur:
        banks.append((np.array(q_cur), np.array(qiq_cur)))
    return banks


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def make_dr_figure(r, Dr):
    """Return a Plotly Figure for D(r)."""
    fig = go.Figure(go.Scatter(
        x=r, y=Dr,
        mode='lines',
        line=dict(color='black', width=1.5),
        name='D(r)',
    ))
    fig.update_layout(
        xaxis_title='r (Å)',
        yaxis_title='D(r)',
        height=430,
        margin=dict(l=60, r=20, t=30, b=60),
        hovermode='x unified',
    )
    return fig


def make_qiq_figure(banks):
    """Return a Plotly Figure with one subplot per bank showing fitted Qi(Q)."""
    n = len(banks)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f'Bank {i + 1}' for i in range(n)],
        shared_xaxes=True,
        vertical_spacing=max(0.04, 0.5 / n) if n > 1 else 0,
    )
    for i, (q, qiq) in enumerate(banks):
        fig.add_trace(
            go.Scatter(
                x=q, y=qiq,
                mode='lines',
                name=f'Bank {i + 1}',
                line=dict(color='crimson', width=1.5),
            ),
            row=i + 1, col=1,
        )
        fig.update_yaxes(title_text='Qi(Q)', row=i + 1, col=1)
    fig.update_xaxes(title_text='Q (Å⁻¹)', row=n, col=1)
    fig.update_layout(
        height=max(380, 240 * n),
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Zip helper
# ---------------------------------------------------------------------------

def zip_files(file_paths):
    """Return a bytes ZIP archive containing the given files."""
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            zf.write(fp, arcname=os.path.basename(fp))
    return buf.getvalue()
