"""
PDFHermite web interface — run with:

    streamlit run app.py
"""

import contextlib
import io as _io
import os
import sys

import streamlit as st

# Ensure the package directory is importable
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

import hermite_fit
import web_utils

_DT_OPTIONS = ['i(Q)', 'S(Q)', 'Qi(Q)']   # 'Qi(Q)' means already in Qi(Q) form (empty in config)


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


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title='PDFHermite', layout='wide')
st.title('PDFHermite')
st.caption('Pair Distribution Function fitting via Hermite Functions')

# ── Input mode ───────────────────────────────────────────────────────────────
input_mode = st.radio('Input', ['Sample dataset', 'Upload files'], horizontal=True)

# Clear results when the user switches input mode
if st.session_state.get('_last_input_mode') != input_mode:
    st.session_state.pop('run_results', None)
    st.session_state['_last_input_mode'] = input_mode

# ── Sample dataset selection ─────────────────────────────────────────────────
if input_mode == 'Sample dataset':
    selected_sample = st.selectbox('Dataset', web_utils.SAMPLE_DATASETS,
                                   key='sample_select')

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

    dat_upload = st.file_uploader('Config file (.dat)', type=['dat', 'txt'],
                                  key='dat_upload')
    data_uploads = st.file_uploader(
        'Data files (and any resolution / broadening files referenced in the config)',
        accept_multiple_files=True,
        key='data_uploads',
    )

    if dat_upload is not None:
        dat_hash = hash(dat_upload.getvalue())
        if st.session_state.get('_last_dat_hash') != dat_hash:
            st.session_state['_last_dat_hash'] = dat_hash
            try:
                defaults = web_utils.defaults_from_bytes(dat_upload.getvalue(),
                                                          dat_upload.name)
                st.session_state['_defaults'] = defaults
                _apply_defaults_to_session(defaults)
            except Exception as exc:
                st.warning(f'Could not parse config file: {exc}')

        defaults = st.session_state.get('_defaults', defaults)

        # Tell the user which extra files are expected
        expected = list(defaults.get('_data_filenames', []))
        for key in ('resolution_information', 'broaden_information', 'pdf_file'):
            v = defaults.get(key, '').strip()
            if v:
                expected.append(v)
        if expected:
            uploaded_names = {f.name for f in (data_uploads or [])}
            missing = [f for f in expected if f not in uploaded_names]
            if missing:
                st.info(f"Also upload: {', '.join(missing)}")

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
    chebyshev = st.number_input('Number of Chebyshev polynomials (0 = off)',
                                min_value=0, step=1,                                 key='chebyshev')
    pre_fitting = st.checkbox('Chebyshev background pre-fitting',                    key='pre_fitting')
    fit_res     = st.checkbox('Fit with instrument resolution',
                              key='fit_with_instrument_resolution')
    broaden     = st.checkbox('Broaden data',                                        key='broaden_data')

# Bundle current form values + internal data-file metadata into one dict
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
    'chebyshev_fitting':              chebyshev > 0,
    'fit_with_instrument_resolution': fit_res,
    'broaden_data':                   broaden,
    # Not exposed in the form — carried through from defaults
    'resolution_information': defaults.get('resolution_information', ''),
    'broaden_information':    defaults.get('broaden_information', ''),
    'pdf_file':               defaults.get('pdf_file', ''),
    '_data_filenames':        defaults.get('_data_filenames', []),
    '_use_gudrun':            defaults.get('_use_gudrun', False),
    '_nbanks':                defaults.get('_nbanks', 1),
}

# ── Run button ────────────────────────────────────────────────────────────────
run_ready = True
if input_mode == 'Upload files':
    if dat_upload is None:
        st.warning('Please upload a .dat config file.')
        run_ready = False
    elif not data_uploads:
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
            rootname = web_utils.prepare_upload_run(dat_upload.name, upload_map, params)

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
