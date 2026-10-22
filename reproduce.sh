#!/usr/bin/env bash
#
# reproduce.sh - reproduce the figures that can be built from the committed
# public data in this repository (no licensed imagery or model weights needed).
#
# What it does:
#   1. stages the committed data/ archives into the layout the scripts read
#      (analysis_output/ripley_data/, census_boundaries/exact_boundaries/,
#      analysis_output/equity_results/ and redlining_data/), reassembling the
#      split Austin/Surrey files and decompressing every city;
#   2. runs the figure scripts that depend only on that public data;
#   3. reruns the revision analyses (latitude-corrected crown areas, Surrey
#      crown-merging sensitivity, Austin spatial models, Austin HOLC redlining).
#
# Reference copies of every revision output are committed under results/ so
# you can diff your run against them.
#
# Outputs are written to  $TREE_MAPPING_BASE_DIR/analysis_output/  (default: ./analysis_output).
#
# Usage:
#   bash reproduce.sh                 # all four cities
#   TREE_MAPPING_CITIES=cupertino,bloomington bash reproduce.sh   # quick subset
#
# Requirements: Python 3.10+ with the packages in requirements.txt.
# Reassembling + reading the full Austin (1.1 GB) and Surrey (717 MB) GeoJSONs
# needs several GB of free RAM and disk; use the TREE_MAPPING_CITIES subset above if you just want to verify
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

CITIES="${TREE_MAPPING_CITIES:-austin,bloomington,cupertino,surrey}"
has_city() { [[ ",$CITIES," == *",$1,"* ]]; }

echo
echo "[1/8] Staging committed data (data/ -> analysis_output/, census_boundaries/, redlining_data/) ..."
$PY repro_data.py ${TREE_MAPPING_CITIES:+--cities "$TREE_MAPPING_CITIES"}

echo
echo "[2/8] Crown-area histogram + income-vs-crown-area scatter (Extended Data Fig. 2) ..."
$PY plot_crown_area_histogram.py

echo
echo "[3/8] Tree-density choropleth maps (Fig. 1) ..."
$PY generate_density_map_from_geojson.py

echo
echo "[4/8] Latitude-corrected crown areas and census-unit re-aggregation ..."
$PY run_crown_area_correction_pipeline.py --cities "$CITIES"

echo
echo "[5/8] Corrected cross-city income scatter plots ..."
$PY generate_plots_corrected.py

echo
echo "[6/8] Austin exploratory spatial analysis (Spearman, OLS, spatial error models) ..."
$PY run_austin_exploratory_spatial_analysis.py

if has_city surrey; then
  echo
  echo "[7/8] Surrey crown-merging sensitivity analysis (Reviewer 2) ..."
  $PY run_surrey_reviewer2_sensitivity.py
  $PY plot_surrey_reviewer2_figure.py
else
  echo
  echo "[7/8] Skipping Surrey sensitivity analysis (surrey not in TREE_MAPPING_CITIES)"
fi

if has_city austin; then
  echo
  echo "[8/8] Austin HOLC redlining figures ..."
  $PY generate_redlining_figures.py
else
  echo
  echo "[8/8] Skipping Austin redlining figures (austin not in TREE_MAPPING_CITIES)"
fi

echo
echo "Done. Figures written to: $TREE_MAPPING_BASE_DIR/analysis_output/"
ls -1 "$TREE_MAPPING_BASE_DIR/analysis_output/" 2>/dev/null | grep -Ei '\.(png|pdf)$' || true
