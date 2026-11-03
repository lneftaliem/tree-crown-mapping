"""
paths.py -- single place for the pipeline's input/output locations.

Defaults reproduce the current (archive-coverage) run. A complete re-separated tile set is
used by setting environment variables, without editing any script and without touching the
current outputs:
  SPLIT_DIR    folder of center_based_relabeled_*.tif           (default: Urban_data/split)
  PRED_DIR     folder of original pred_*.tif; tiles not found there are read from
               used_predictions.zip                              (default: zip only)
  DIAG_DIR     diagnostics folder (crown tables, georef table)   (default: split_audit_20261031/diagnostics)
  METADATA_CSV tile metadata                                     (default: sherlock_venv/tile_metadata.csv)
"""
import os

ROOT = '/path/to/Urban_data'
AUDIT = os.environ.get('AUDIT_DIR', f'{ROOT}/split_audit_20261031')  # v1 build: set to the v1 work folder
ZIP_FILE = f'{ROOT}/prediction_tifs/used_predictions.zip'
ZIP = f'zip://{ZIP_FILE}!'
SPLIT_DIR = os.environ.get('SPLIT_DIR', f'{ROOT}/split')
PRED_DIR = os.environ.get('PRED_DIR')
DIAG = os.environ.get('DIAG_DIR', f'{AUDIT}/diagnostics')
METADATA_CSV = os.environ.get('METADATA_CSV', f'{ROOT}/sherlock_venv/tile_metadata.csv')
CENSUS = f'{ROOT}/ripley_data_0429206/census_boundaries'


def pred_path(name):
    """Original prediction raster: PRED_DIR if it holds the file, otherwise the archive zip."""
    if PRED_DIR and os.path.exists(os.path.join(PRED_DIR, name)):
        return os.path.join(PRED_DIR, name)
    return ZIP + name


def split_path(name):
    return f'{SPLIT_DIR}/center_based_relabeled_{name}'
