"""
Contamination guard, loaded automatically (PYTHONPATH) into every analysis process started by run_analysis.py.

Logs every file the process reads -- Python open() (audit hook), and the GDAL-based readers pyogrio / geopandas /
fiona / rasterio, which bypass Python open() -- to $GUARD_LOG, and raises if a path matches a forbidden pattern:
historical tree datasets (unsplit or mixed-vintage GeoJSONs, the repository's committed tree points), cached
census-unit tables with tree columns, and old analysis outputs. GUARD_ALLOW (os.pathsep-separated substrings) exempts
paths a step is entitled to read (e.g. the committed census tables for their NON-tree columns in build_unit_table.py).
"""
import os, sys

FORBIDDEN = [
    '/ripley_data_0429206/all_trees/', '/ripley_data_0429206/all_trees_corrected/',
    '/ripley_data_0429206/all_trees_split/', '/ripley_data_0429206/all_trees_split_full/',
    '/manuscript_repo_original/', '/data/tree_points/', '/sherlock_venv/analysis_output/',
    '/sherlock_venv/boundary_denominator_sensitivity/all_cities', '/Revisions_code/',
    '/covariates_nontree/', '/seam_analysis/consistent_pipeline_sensitivity/final/',
]
EXTRA = [p for p in os.environ.get('GUARD_FORBID', '').split(os.pathsep) if p]
ALLOW = [p for p in os.environ.get('GUARD_ALLOW', '').split(os.pathsep) if p]
LOG = os.environ.get('GUARD_LOG')
_seen = set()


def _check(path, how):
    try:
        p = os.path.realpath(os.fspath(path)) if not str(path).startswith(('zip:', '/vsi')) else str(path)
    except TypeError:
        return
    for raw in {str(path), p}:
        if any(a in raw for a in ALLOW):
            break
        bad = [f for f in FORBIDDEN + EXTRA if f in raw]
        if bad:
            raise PermissionError(f'[guard] forbidden input {raw} (pattern {bad[0]}) via {how}')
    if LOG and (p, how) not in _seen and not p.startswith(('/Library/Frameworks', '/System', '/usr', '/dev', sys.prefix)) \
            and '/site-packages/' not in p and not p.endswith(('.pyc', '.py')):
        _seen.add((p, how))
        with open(LOG, 'a') as f:  # audit hook ignores LOG itself (below)
            f.write(f'{how}\t{p}\n')


def _hook(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]) if isinstance(args[0], bytes) else os.fspath(args[0])
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else 'r'
        if LOG and os.path.realpath(path) == os.path.realpath(LOG):
            return
        if 'r' in mode or '+' in mode or mode == '':
            _check(path, 'open')


if os.environ.get('GUARD_ACTIVE') == '1':
    sys.addaudithook(_hook)

    def _wrap(mod, name, how):
        f = getattr(mod, name)

        def g(path, *a, **k):
            if isinstance(path, (str, os.PathLike)):
                _check(path, how)
            return f(path, *a, **k)
        g.__wrapped__ = f
        setattr(mod, name, g)

    try:
        import pyogrio
        for n in ('read_dataframe', 'read_info', 'read_arrow', 'read_bounds'):
            if hasattr(pyogrio, n):
                _wrap(pyogrio, n, f'pyogrio.{n}')
    except ImportError:
        pass
    try:
        import rasterio
        _wrap(rasterio, 'open', 'rasterio.open')
    except ImportError:
        pass
    try:
        import fiona
        _wrap(fiona, 'open', 'fiona.open')
    except ImportError:
        pass

# optional figure audit (FIG_AUDIT_LOG); no effect otherwise
if os.environ.get("FIG_AUDIT_LOG"):
    import figaudit  # noqa: E402,F401
