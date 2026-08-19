#!/usr/bin/env bash
#
# reproduce.sh — reproduce the figures that can be built from the committed
# public data in this repository (no licensed imagery or model weights needed).
#
# What it does:
#   1. stages the committed data/ archives into the layout the scripts read
#      (analysis_output/ripley_data/ and census_boundaries/exact_boundaries/),
#      reassembling the split Austin/Surrey files and decompressing every city;
#   2. runs the figure scripts that depend only on that public data.
#
# Outputs are written to  $TREE_MAPPING_BASE_DIR/analysis_output/  (default: ./analysis_output).
#
# Usage:
#   ./reproduce.sh                 # all four cities
#   TREE_MAPPING_CITIES=cupertino,bloomington ./reproduce.sh   # quick subset
#
# Requirements: Python 3.10+ with the packages in requirements.txt
# (geopandas, matplotlib, numpy, pandas, shapely). Reassembling + reading the
# full Austin (1.1 GB) and Surrey (717 MB) GeoJSONs needs several GB of free RAM
# and disk; use the TREE_MAPPING_CITIES subset above if you just want to verify
# the pipeline quickly.

set -euo pipefail
cd "$(dirname "$0")"

export TREE_MAPPING_BASE_DIR="${TREE_MAPPING_BASE_DIR:-$(pwd)}"
PY="${PYTHON:-python3}"

echo "=================================================================="
echo " Reproducing figures from committed public data"
echo "   repo            : $(pwd)"
echo "   base dir        : $TREE_MAPPING_BASE_DIR"
echo "   cities          : ${TREE_MAPPING_CITIES:-austin,bloomington,cupertino,surrey}"
echo "=================================================================="

echo
echo "[1/3] Staging committed data (data/ -> analysis_output/ripley_data + census_boundaries/exact_boundaries) ..."
$PY repro_data.py ${TREE_MAPPING_CITIES:+--cities "$TREE_MAPPING_CITIES"}

echo
echo "[2/3] Crown-area histogram + income-vs-crown-area scatter (Extended Data Fig. 2) ..."
$PY plot_crown_area_histogram.py

echo
echo "[3/3] Tree-density choropleth maps (Fig. 1) ..."
$PY generate_density_map_from_geojson.py

echo
echo "Done. Figures written to: $TREE_MAPPING_BASE_DIR/analysis_output/"
ls -1 "$TREE_MAPPING_BASE_DIR/analysis_output/" 2>/dev/null | grep -Ei '\.(png|pdf)$' || true
