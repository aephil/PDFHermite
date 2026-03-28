"""
PDFHermite web interface — run with:

    streamlit run app.py
"""

import contextlib
import glob as _glob
import io as _io
import os
import subprocess as _subprocess
import sys

# Ensure the package directory is importable
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

# ── Compile C extension if not already built ─────────────────────────────────
# Runs on every Streamlit script execution; the glob check is a fast no-op
# once the .so is present.  On a fresh deployment the subprocess call runs
# once (~5–10 s) before the page first renders.
if not _glob.glob(os.path.join(PACKAGE_DIR, '_hermite_cext*.so')):
    _result = _subprocess.run(
        [sys.executable, 'setup_ext.py', 'build_ext', '--inplace'],
        cwd=PACKAGE_DIR,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if _result.returncode != 0:
        print('Hermite C extension build failed:', _result.stderr or _result.stdout,
              file=sys.stderr)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

import hermite_fit
import web_utils

_DT_OPTIONS = ['i(Q)', 'S(Q)', 'Qi(Q)']   # 'Qi(Q)' means already in Qi(Q) form (empty in config)


def _apply_res_info_to_session(prefix, info):
    """Pre-fill per-bank resolution widget keys from an info dict."""
    res_type = info.get('type') or ''
    if res_type:   # only set if valid; empty leaves widget at its own default
        st.session_state[f'{prefix}_type'] = res_type
    for ib, bparams in enumerate(info.get('params', [])):
        for pk, val in bparams.items():
            st.session_state[f'{prefix}_{pk}_{ib + 1}'] = float(val)


def _apply_defaults_to_session(d):
    """Write a defaults dict directly into session state widget keys.

    Streamlit ignores value= on a widget whose key already exists in session
    state, so the only reliable way to update widgets is to write the desired
    values here before the widgets are created.
    """
    _dt_raw = str(d.get('data_type', '')).strip().lower()
    # Map config values (case-insensitive) to display labels
    _dt_sel = {'': 'Qi(Q)', 'i(q)': 'i(Q)', 's(q)': 'S(Q)'}.get(_dt_raw, 'i(Q)')
    st.session_state['qfitmin']     = float(d.get('qfitmin', 0.5))
    st.session_state['qfitmax']     = float(d.get('qfitmax', 50.0))
    st.session_state['rmax']        = float(d.get('rmax', 30.0))
    st.session_state['rspacing']    = float(d.get('rspacing', 0.02))
    st.session_state['lorch']       = bool(d.get('lorch', True))
    st.session_state['convolution'] = bool(d.get('convolution', True))
    st.session_state['data_type']   = _dt_sel
    st.session_state['nhermites']   = str(d.get('nhermites', 'default'))
    st.session_state['self_term']   = str(d.get('self_term', ''))
    st.session_state['chebyshev']   = int(d.get('chebyshev', 0))
    st.session_state['pre_fitting'] = bool(d.get('pre_fitting', False))
    st.session_state['fit_with_instrument_resolution'] = bool(d.get('fit_with_instrument_resolution', False))
    st.session_state['broaden_data'] = bool(d.get('broaden_data', False))
    st.session_state['_nbanks']     = int(d.get('_nbanks', 1))
    _apply_res_info_to_session('res', d.get('_resolution_info', {'type': None, 'params': []}))
    _apply_res_info_to_session('brd', d.get('_broadening_info', {'type': None, 'params': []}))


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='PDFHermite', layout='wide')
st.title('PDFHermite')
st.caption('Pair Distribution Function fitting via Hermite Functions')
st.caption(
    'Wang, S., Gao, M., Qin, Y., Zhang, S., Tan, L. & Dove, M. T. (2025). '
    '[Accounting for instrument resolution in the pair distribution functions '
    'obtained from total scattering data using Hermite functions.]'
    '(https://doi.org/10.1107/S1600576725004340) '
    '*J. Appl. Cryst.* **58**, 1269–1287.'
)

# ── Input mode ───────────────────────────────────────────────────────────────
input_mode = st.radio('Input', ['Sample dataset', 'Upload files'], horizontal=True)

# Clear results when the user switches input mode
if st.session_state.get('_last_input_mode') != input_mode:
    st.session_state.pop('run_results', None)
    st.session_state['_last_input_mode'] = input_mode

# ── Sample dataset selection ─────────────────────────────────────────────────
if input_mode == 'Sample dataset':
    selected_sample = st.selectbox(
        'Dataset', web_utils.SAMPLE_DATASETS,
        index=web_utils.SAMPLE_DATASETS.index('silica'),
        format_func=lambda k: web_utils.SAMPLE_DISPLAY_NAMES[k],
        key='sample_select',
    )

    if st.session_state.get('_last_sample') != selected_sample:
        st.session_state['_last_sample'] = selected_sample
        d = web_utils.load_sample_defaults(selected_sample, PACKAGE_DIR)
        st.session_state['_defaults'] = d
        st.session_state['_sample_files'] = web_utils.load_sample_files(
            selected_sample, PACKAGE_DIR
        )
        st.session_state.pop('run_results', None)
        _apply_defaults_to_session(d)

    defaults = st.session_state.setdefault(
        '_defaults',
        web_utils.load_sample_defaults(selected_sample, PACKAGE_DIR),
    )

    sample_files = st.session_state.setdefault(
        '_sample_files',
        web_utils.load_sample_files(selected_sample, PACKAGE_DIR),
    )
    with st.expander(f'Configuration files ({len(sample_files)} loaded)'):
        for fname, content in sample_files.items():
            st.caption(fname)
            st.code(content, language=None)

# ── File upload ───────────────────────────────────────────────────────────────
else:
    defaults = web_utils.DEFAULT_PARAMS.copy()

    data_uploads = st.file_uploader(
        'Data file(s) — Gudrun .dcs01 or plain two-column Q/I files',
        accept_multiple_files=True,
        key='data_uploads',
    )

    config_uploads = st.file_uploader(
        'Config files (optional — .dat pre-fills parameters; '
        'include resolution/broadening .txt files if referenced)',
        type=['dat', 'txt'],
        accept_multiple_files=True,
        key='config_uploads',
    )

    # When data files change: auto-detect format and bank count
    if data_uploads:
        data_hash = hash(b''.join(f.getvalue()[:200] for f in data_uploads))
        if st.session_state.get('_last_data_hash') != data_hash:
            st.session_state['_last_data_hash'] = data_hash
            detected = web_utils.detect_data_format(data_uploads[0].getvalue())
            st.session_state['_use_gudrun']  = detected['format'] == 'gudrun'
            st.session_state['_nbanks']      = detected['nbanks']
            st.session_state['_data_filenames'] = [f.name for f in data_uploads]
            st.session_state.pop('run_results', None)
            # Reset resolution/broadening widgets when file set changes
            for key in ('res_type', 'brd_type'):
                st.session_state.pop(key, None)

    # When config files change: parse .dat for parameters, .txt for resolution/broadening
    if config_uploads:
        cfg_hash = hash(b''.join(f.getvalue() for f in config_uploads))
        if st.session_state.get('_last_cfg_hash') != cfg_hash:
            st.session_state['_last_cfg_hash'] = cfg_hash
            # Build a name→bytes map of all uploaded config files
            cfg_map = {f.name: f.getvalue() for f in config_uploads}
            # Identify the .dat (must contain '::')
            dat_file = next(
                (f for f in config_uploads
                 if f.name.endswith('.dat')
                 and b'::' in f.getvalue()),
                None,
            )
            try:
                if dat_file is not None:
                    parsed = web_utils.defaults_from_bytes(dat_file.getvalue(),
                                                            dat_file.name)
                    # Resolve resolution/broadening files from the same upload set
                    for info_key, file_key in [
                        ('_resolution_info', 'resolution_information'),
                        ('_broadening_info', 'broaden_information'),
                    ]:
                        fname = parsed.get(file_key, '').strip()
                        if fname and fname in cfg_map:
                            parsed[info_key] = web_utils.parse_resolution_bytes(
                                cfg_map[fname])
                    # Preserve data-file metadata derived from data uploads
                    for preserve_key in ('_use_gudrun', '_nbanks', '_data_filenames'):
                        parsed[preserve_key] = st.session_state.get(
                            preserve_key, parsed.get(preserve_key))
                    st.session_state['_defaults'] = parsed
                    _apply_defaults_to_session(parsed)
            except Exception as exc:
                st.warning(f'Could not parse config file: {exc}')

    defaults = st.session_state.get('_defaults', defaults)

# ── Parameter form ────────────────────────────────────────────────────────────
st.subheader('Parameters')
col1, col2 = st.columns(2)

with col1:
    qfitmin = st.number_input('Minimum Q to fit (Å⁻¹)', step=0.1,  format='%.2f', key='qfitmin')
    rmax    = st.number_input('Maximum r (Å)',            step=1.0,  format='%.1f', key='rmax')
    lorch   = st.checkbox('Lorch window',                                           key='lorch')

with col2:
    qfitmax  = st.number_input('Maximum Q to fit (Å⁻¹)', step=1.0,  format='%.1f',  key='qfitmax')
    rspacing = st.number_input('r spacing (Å)',           step=0.005, format='%.4f', key='rspacing')
    convolution = st.checkbox('Q-space convolution',                                 key='convolution')

data_type = st.selectbox('Input data type', _DT_OPTIONS, key='data_type')
nhermites = st.text_input('Number of Hermite functions (or "default")',              key='nhermites')

with st.expander('Advanced options'):
    self_term = st.text_input('Self-scattering term (leave blank if none)',          key='self_term')
    st.caption(r'Expected value of $S(Q \rightarrow 0) = \sum_j c_j \bar{b_j^2}$, '
               r'required for the conversion from S(Q) to i(Q).')
    chebyshev = st.number_input('Number of Chebyshev polynomials (0 = off)',
                                min_value=0, step=1,                                 key='chebyshev')
    pre_fitting = st.checkbox('Chebyshev background pre-fitting',                    key='pre_fitting')
    fit_res     = st.checkbox('Fit with instrument resolution',
                              key='fit_with_instrument_resolution')
    broaden     = st.checkbox('Broaden data',                                        key='broaden_data')

    nbanks_ui = st.session_state.get('_nbanks', 1)

    def _res_param_inputs(prefix, res_type, nbanks):
        """Render per-bank parameter inputs for a resolution/broadening type."""
        param_keys = web_utils._RES_PARAMS.get(res_type, [])
        for ib in range(nbanks):
            label = f'Bank {ib + 1}' if nbanks > 1 else ''
            if label:
                st.caption(label)
            cols = st.columns(len(param_keys))
            for ic, pk in enumerate(param_keys):
                hint = ' (optional)' if pk == 'c' else ''
                cols[ic].number_input(
                    f'{pk}{hint}',
                    key=f'{prefix}_{pk}_{ib + 1}',
                    value=0.0,
                    format='%g',
                    step=0.001,
                )

    _RES_TYPE_OPTIONS = list(web_utils._RES_PARAMS.keys())

    if fit_res:
        st.markdown('**Instrument resolution parameters**')
        res_type_sel = st.selectbox('Resolution type', _RES_TYPE_OPTIONS,
                                    key='res_type')
        _res_param_inputs('res', res_type_sel, nbanks_ui)

    if broaden:
        st.markdown('**Data broadening parameters**')
        brd_type_sel = st.selectbox('Broadening type', _RES_TYPE_OPTIONS,
                                    key='brd_type')
        _res_param_inputs('brd', brd_type_sel, nbanks_ui)

def _collect_res_info(prefix, type_key, nbanks):
    """Build a resolution info dict from session-state widget values."""
    res_type = st.session_state.get(type_key, '')
    if not res_type:
        return {'type': None, 'params': []}
    param_keys = web_utils._RES_PARAMS.get(res_type, [])
    banks = []
    for ib in range(nbanks):
        d = {}
        for pk in param_keys:
            val = st.session_state.get(f'{prefix}_{pk}_{ib + 1}', 0.0)
            d[pk] = val
        banks.append(d)
    return {'type': res_type, 'params': banks}


# Bundle current form values + internal data-file metadata into one dict
_nbanks_now = st.session_state.get('_nbanks', 1)
params = {
    'qfitmin':   qfitmin,
    'qfitmax':   qfitmax,
    'rmax':      rmax,
    'rspacing':  rspacing,
    'lorch':     lorch,
    'convolution': convolution,
    'data_type': '' if data_type == 'Qi(Q)' else data_type,
    'nhermites': nhermites,
    'self_term': self_term,
    'chebyshev': chebyshev,
    'pre_fitting':                    pre_fitting,
    'chebyshev_fitting':              False,
    'fit_with_instrument_resolution': fit_res,
    'broaden_data':                   broaden,
    # Resolution/broadening generated from form widgets
    '_resolution_info': (_collect_res_info('res', 'res_type', _nbanks_now)
                         if fit_res else {'type': None, 'params': []}),
    '_broadening_info': (_collect_res_info('brd', 'brd_type', _nbanks_now)
                         if broaden else {'type': None, 'params': []}),
    # Not exposed in the form — carried through from defaults
    'pdf_file':          defaults.get('pdf_file', ''),
    '_data_filenames':   (st.session_state.get('_data_filenames', [])
                          if input_mode == 'Upload files'
                          else defaults.get('_data_filenames', [])),
    '_use_gudrun':       (st.session_state.get('_use_gudrun', False)
                          if input_mode == 'Upload files'
                          else defaults.get('_use_gudrun', False)),
    '_nbanks':           _nbanks_now,
}

# ── Run button ────────────────────────────────────────────────────────────────
run_ready = True
if input_mode == 'Upload files':
    if not data_uploads:
        st.warning('Please upload at least one data file.')
        run_ready = False

if run_ready and st.button('Run PDFHermite', type='primary'):
    # Clean up previous temp directory
    if 'run_results' in st.session_state:
        web_utils.cleanup_old_run(st.session_state['run_results']['tmp_dir'])
        del st.session_state['run_results']

    try:
        if input_mode == 'Sample dataset':
            rootname = web_utils.prepare_sample_run(selected_sample, PACKAGE_DIR, params)
        else:
            upload_map = {f.name: f.getvalue() for f in data_uploads}
            rootname = web_utils.prepare_upload_run(upload_map, params)

        log_buf = _io.StringIO()
        with st.spinner('Running PDFHermite…'):
            with contextlib.redirect_stdout(log_buf):
                hermite_fit.run(rootname)

        st.session_state['run_results'] = web_utils.collect_results(rootname)
        st.session_state['run_log'] = log_buf.getvalue()

    except Exception as exc:
        st.error(f'Pipeline error: {exc}')
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
if 'run_results' in st.session_state:
    res = st.session_state['run_results']

    st.subheader('Results')
    tab_dr, tab_qiq, tab_prefit = st.tabs(['D(r)', 'Qi(Q) Fit', 'Pre-fit Diagnostics'])

    with tab_dr:
        st.plotly_chart(
            web_utils.make_dr_figure(res['r'], res['Dr']),
            width='stretch',
        )

    with tab_qiq:
        st.plotly_chart(
            web_utils.make_qiq_figure(res['qiq_banks']),
            width='stretch',
        )
        st.caption(
            'Shows the Hermite-fitted Qi(Q) per bank.  '
            'Download the PDF plots below for the full data / fit / residual comparison.'
        )

    with tab_prefit:
        if res['has_prefit']:
            st.info(
                'Pre-fit diagnostic plots (i(Q) and Qi(Q) before/after Chebyshev removal) '
                'are included in the PDF ZIP download below.'
            )
        else:
            st.info('Pre-fitting was not enabled for this run.')

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader('Downloads')
    dcol1, dcol2, dcol3 = st.columns(3)

    with open(res['drout_path'], 'rb') as fh:
        dcol1.download_button(
            'D(r) data (.drout)', fh.read(),
            file_name=os.path.basename(res['drout_path']),
            mime='text/plain',
        )

    with open(res['qiqout_path'], 'rb') as fh:
        dcol2.download_button(
            'Qi(Q) data (.qiqout)', fh.read(),
            file_name=os.path.basename(res['qiqout_path']),
            mime='text/plain',
        )

    if res['pdf_paths']:
        dcol3.download_button(
            f'All plots — {len(res["pdf_paths"])} PDF(s) (ZIP)',
            web_utils.zip_files(res['pdf_paths']),
            file_name='pdfhermite_plots.zip',
            mime='application/zip',
        )

    with st.expander('Pipeline log'):
        st.text(st.session_state.get('run_log', '(no output captured)'))
