#!/bin/bash
# build_v1.sh -- builds the v1 consistently processed (NDVI + PAN, tau 0.7, Sherlock job 50693038), instance-separated
# four-city dataset (restartable; writes into work_v1/ and dataset/ only; every earlier prediction, split raster and
# dataset is read-only and none of them is an input).
#   00 census attributes (verified authoritative sources)              -> work_v1/diag/bg_lookup_*.csv
#   01v1 one v1 prediction per in-city window (347), checks, aliases     -> work_v1/tiles/{tile_inventory,window_aliases}.csv
#   02v1 empty scan base + empty artefact list (no historical tile reused)
#   16 scan all 347 tiles (01_tile_scan.scan, unchanged)                -> work_v1/diag/tile_scan.csv, crowns/
#   08 georeferencing table; 04 city crown tables + dedup audit
#   07 artefact screen of every component >= 20,000 px -> large_components.csv + thumbnails; STOPS until the flagged
#      components have been reviewed visually and work_v1/diag/artifact_components.csv written (artifact_review.done)
#   06 final GeoJSONs (>= 3 m2 projected, 1.0 m cross-tile dedup, centroid in city, artefacts excluded, > 250 m2 flagged)
#   11 validation; 12 coverage
# v1b (input channels in training order, Sherlock job 50750471; see code/sherlock/regen_v1b.py) is built by the same script with
#   V1=.../v1b_pan_ndvi_20261101 WORK=$P/work_v1b FINAL=$P/dataset/tree_crowns_v1b EXPECT_INPUTS=ch0_band1_pan+ch1_band2_ndvi
# The design for the PAN + PAN selection that this replaces is kept in provenance/build_v1_panpan_design.sh.txt.
set -euo pipefail
P=/path/to/Urban_data/tree_crowns_v1_20261101
V1=${V1:-/path/to/Urban_data/v1_ndvi_pan_20261101}
W=${WORK:-$P/work_v1}
PY=$(grep ^PY= $P/../split_audit_20261031/scripts/env_full.sh | cut -d= -f2 | cut -d' ' -f1)
export DIAG_DIR=$W/diag AUDIT_DIR=$W SPLIT_DIR=$W/tiles/split_links PRED_DIR=$W/tiles/pred_links
export COVERAGE_OUT=$W/coverage FINAL_DIR=${FINAL:-$P/dataset/tree_crowns_v1}
cd $P/code/pipeline
mkdir -p $W/logs $W/diag $FINAL_DIR
step() { local n=$1; shift; if [ -e "$W/logs/$n.done" ]; then echo "[skip] $n"; else echo "[run ] $n"; "$@" > $W/logs/$n.log 2>&1 && touch $W/logs/$n.done || { tail -30 $W/logs/$n.log; exit 1; }; fi; }
step 00_census   $PY 00_census_attributes.py
step 01v1_inv    $PY 01v1_tile_inventory.py $V1 $W/tiles
step 02v1_diag   $PY - <<'PY'
import os, pandas as pd
F = '/path/to/Urban_data/split_audit_20261031/diagnostics_full'
D = os.environ['DIAG_DIR']
pd.read_csv(f'{F}/tile_scan.csv', nrows=0).to_csv(f'{D}/tile_scan_base.csv', index=False)
print('empty scan base: all v1 tiles are scanned')
PY
step 16_scan     $PY 16_scan_new_tiles.py $DIAG_DIR/tile_scan_base.csv 6
step 08_georef   $PY 08_georef_table.py
step 04_audit_a  $PY 04_city_pipeline_audit.py bloomington cupertino
step 04_audit_b  $PY 04_city_pipeline_audit.py surrey austin
step 07_screen   $PY 07_artifact_screen.py 6
if [ ! -e $W/logs/artifact_review.done ]; then
  echo "STOP: review $DIAG_DIR/large_components.csv and thumbnails, write $DIAG_DIR/artifact_components.csv"
  echo "      (columns city,tile,parent_cc8,npx,max_edge,share_in_solid_101px_windows), then touch $W/logs/artifact_review.done"
  exit 2
fi
step 06_build    $PY 06_build_city_outputs.py --out $FINAL_DIR --exclude-artifacts --unresolved-m2 250 austin bloomington cupertino surrey
step 11_validate $PY 11_validate_outputs.py $FINAL_DIR
step 12_coverage $PY 12_block_group_coverage.py
echo "build complete: $FINAL_DIR"
