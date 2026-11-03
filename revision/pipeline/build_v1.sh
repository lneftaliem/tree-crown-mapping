#!/bin/bash
# build_v1.sh -- builds the final instance-separated, NDVI-masked four-city dataset from the v1b predictions and split
# rasters (Sherlock: revision/sherlock/12_v1b_inference.sbatch, U-Net tau 0.7 + separate_instances -cn 8 -m 15).
# Restartable: each finished step leaves logs/<step>.done in the work folder.
#   00 census attributes (verified authoritative sources)              -> $WORK/diag/bg_lookup_*.csv
#   01v1 one prediction per in-city window (347), checks, aliases       -> $WORK/tiles/{tile_inventory,window_aliases}.csv
#   02v1 empty scan base (every tile is scanned)
#   16 scan all 347 tiles (01_tile_scan.scan)                           -> $WORK/diag/tile_scan.csv, crowns/
#   08 georeferencing table; 04 city crown tables + dedup audit
#   07 artefact screen of every component >= 20,000 px -> large_components.csv + thumbnails. The visual review of the
#      final run flagged no component; its result is inputs/artifact_components.csv (header only), which is copied in.
#   06 final GeoJSONs (>= 3 m2 projected, 1.0 m cross-tile dedup, centroid in city, artefacts excluded, > 250 m2 flagged)
#   11 validation; 12 coverage
#   17 / 17b mean NDVI under every crown; 20 vegetation mask (drop crowns with mean NDVI < 0.2) -> $OUT (final dataset)
# Inputs: V1 = the downloaded Sherlock output folder (predictions/, split/, stats/); URBAN_DATA = data folder holding the
# NASA PGC mosaics and income tables (see REPRODUCING.md). Example:
#   URBAN_DATA=/data/urban V1=/data/urban/v1b_pan_ndvi WORK=/data/urban/work_v1b \
#   FINAL=/data/urban/tree_crowns_v1b OUT=/data/urban/tree_crowns_v1b_ndvi02 bash build_v1.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${V1:?set V1 to the Sherlock output folder}" "${WORK:?set WORK}" "${FINAL:?set FINAL}" "${OUT:?set OUT}"
W=$WORK
PY=${PY:-python}
NDVI_THRESHOLD=${NDVI_THRESHOLD:-0.2}
export EXPECT_INPUTS=${EXPECT_INPUTS:-ch0_band1_pan+ch1_band2_ndvi}  # v1b: input channels in training order
export DIAG_DIR=$W/diag AUDIT_DIR=$W SPLIT_DIR=$W/tiles/split_links PRED_DIR=$W/tiles/pred_links
export COVERAGE_OUT=$W/coverage FINAL_DIR=$FINAL
cd $HERE
mkdir -p $W/logs $W/diag $FINAL_DIR
step() { local n=$1; shift; if [ -e "$W/logs/$n.done" ]; then echo "[skip] $n"; else echo "[run ] $n"; "$@" > $W/logs/$n.log 2>&1 && touch $W/logs/$n.done || { tail -30 $W/logs/$n.log; exit 1; }; fi; }
step 00_census   $PY 00_census_attributes.py
step 01v1_inv    $PY 01v1_tile_inventory.py $V1 $W/tiles
step 02v1_diag   $PY - <<'PY'
import os, shutil
shutil.copy('inputs/tile_scan_base.csv', os.environ['DIAG_DIR'])
print('empty scan base: all v1 tiles are scanned')
PY
step 16_scan     $PY 16_scan_new_tiles.py $DIAG_DIR/tile_scan_base.csv 6
step 08_georef   $PY 08_georef_table.py
step 04_audit_a  $PY 04_city_pipeline_audit.py bloomington cupertino
step 04_audit_b  $PY 04_city_pipeline_audit.py surrey austin
step 07_screen   $PY 07_artifact_screen.py 6
if [ ! -e $W/logs/artifact_review.done ] && [ "${USE_RECORDED_REVIEW:-1}" = 1 ]; then
  cp inputs/artifact_components.csv $DIAG_DIR/ && touch $W/logs/artifact_review.done
fi
if [ ! -e $W/logs/artifact_review.done ]; then
  echo "STOP: review $DIAG_DIR/large_components.csv and thumbnails, write $DIAG_DIR/artifact_components.csv"
  echo "      (columns city,tile,parent_cc8,npx,max_edge,share_in_solid_101px_windows), then touch $W/logs/artifact_review.done"
  exit 2
fi
step 06_build    $PY 06_build_city_outputs.py --out $FINAL_DIR --exclude-artifacts --unresolved-m2 250 austin bloomington cupertino surrey
step 11_validate $PY 11_validate_outputs.py $FINAL_DIR
step 12_coverage $PY 12_block_group_coverage.py
step 17_ndvi     $PY 17_crown_ndvi.py $W/tiles/tile_inventory.csv 6
step 17b_join    $PY 17b_crown_ndvi_join.py $W $FINAL_DIR
step 20_mask     $PY 20_ndvi_mask.py $FINAL_DIR $W $OUT $NDVI_THRESHOLD
echo "build complete: $OUT"
