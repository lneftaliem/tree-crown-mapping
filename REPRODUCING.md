# Reproducing the revised manuscript

This guide lists, in order, every step from imagery to the numbers, figures and tables in the revised manuscript, and the code and inputs for each. Steps 1–3 need the licensed imagery and a GPU/CPU cluster (they were run on Stanford's Sherlock cluster with Slurm); steps 4–5 run on a workstation; step 5 can be run from the data committed in this repository.

Clone with the instance-separation code included:

```bash
git clone --recurse-submodules https://github.com/lneftaliem/tree-crown-mapping
```

Paths to local data are set with environment variables rather than edited in the code: `URBAN_DATA` (data folder; defaults to `/path/to/Urban_data`), plus the per-step variables listed below. Sherlock job scripts use Sherlock's `$GROUP_SCRATCH` and `$HOME` layout and expect `revision/sherlock/` to be at `$HOME/unet_workflow` and `revision/separation/` at `$HOME/split_workflow`; adjust those paths for another cluster.

## Overview

| Step | What it does | Code | Needs |
|---|---|---|---|
| 1 | Training data and U-Net fine-tuning | `revision/model/` | Imagery, annotations, Brandt et al. (2020) weights |
| 2 | City-wide inference (τ = 0.7) | `revision/sherlock/12_v1b_inference.sbatch`, `regen_v1b.py`, `unet_infer.py` | Imagery mosaics, fine-tuned weights |
| 3 | Instance separation | `revision/separation/`, `external/separate_instances` (pinned) | Step 2 predictions |
| 4 | Crown polygons and points, deduplication, census join, NDVI mask | `revision/pipeline/build_v1.sh` | Steps 2–3 outputs, imagery mosaics (NDVI), census tables |
| 5 | Statistics, figures and tables | `revision/analysis/run_analysis.py` | `data/` in this repository (+ inventories for Fig. S5) |
| 6 | Model validation (Supplementary Note 2, Figs. S1–S3) | `revision/sherlock/08`, `09`, `13_*.sbatch` | Annotations, imagery, weights |

## 1. Model fine-tuning (`revision/model/`)

- `1-Preprocessing.ipynb` (settings in `config/Preprocessing.py`) cuts the 113 annotated 256 × 256 training areas from the imagery and writes, for each, the NDVI and panchromatic bands, the crown annotation and the boundary weight map. Note that it writes mosaic band 1 (panchromatic) to the files named `ndvi_*` and band 2 (NDVI) to `pan_*`, so the network's channel 0 is panchromatic and channel 1 is NDVI; inference (step 2) uses the same order.
- `2-UNetTraining.ipynb` / `2-UNetTraining.py` (settings in `config/UNetTraining.py`) fine-tunes the U-Net of `core/UNet.py`, initialised from the Sahel/Sudan weights of Brandt et al. (2020) (`PRETRAINED_WEIGHTS`, obtained from the authors), with the Adadelta optimizer and the weighted Tversky loss in `core/losses.py`, and imgaug augmentation (`core/dataset_generator.py`). It saves `trees_<date>_AdaDelta_weightmap_tversky_012_256.keras`.
- The fine-tuned model used in the paper, `trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras`, is attached to the [v1.0 release](https://github.com/lneftaliem/tree-crown-mapping/releases/tag/v1.0).
- `core/` and `config/` follow the framework of [Brandt et al. (2020)](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) (MIT), with the changes made for this study.
- The training annotations are derived from the licensed imagery and are not redistributed; `data/validation/image_manifest_113.csv` lists the 113 training areas.

## 2. City-wide inference (`revision/sherlock/`)

`12_v1b_inference.sbatch` runs one Slurm task per mosaic scene (23 scenes, `v1_chunks/v1_NNN.txt`; the 347 in-city 5,000 × 5,000-pixel windows are listed in `v1_windows.csv`):

```bash
sbatch --export=ALL,THRESH=0.7 revision/sherlock/12_v1b_inference.sbatch
```

- `regen_v1b.py` reprojects the two-band (panchromatic, NDVI) mosaic to EPSG:4326 (`reproject_verbatim.py`) and calls `unet_infer.detect_tree` on every window: 256 × 256 patches with a 224-pixel stride, per-patch per-channel normalization, maximum merging of overlapping patches, and a binary mask at probability ≥ 0.7. It writes `prob/`, `predictions/` and `stats/` (one JSON line per window).
- The same job then separates the predictions into crowns (step 3).
- Inputs: the NASA PGC mosaics (`RAW_DIR`; image IDs of every source scene are in `data/imagery_cutlines/`) and the fine-tuned model (`MODEL`).

## 3. Instance separation (`revision/separation/`, `external/separate_instances`)

Touching crowns are split with the center-based instance separation written by Christian Igel and Ankit Kariryaa ([`separate_instances`](https://github.com/ankitkariryaa/separate_instances), GPL-3.0; the "detect centre and relabel" method of Mugabowindekwe et al., 2023, *Nature Climate Change* 13, 91–97), included as a git submodule pinned to commit `d81cac7`. Parameters: 15-pixel maximum filter, 8-connectivity (`config.sh`).

- `00_setup_env.sbatch` builds the Python environment; `01_test_reproduce.sbatch` checks the installation on test tiles.
- `run_tile.sh <tile> regen` separates one prediction tile and checks that the output has the input's grid and footprint. It is called by step 2's job for every window.
- Output: `split/center_based_relabeled_<tile>.tif`, one integer label per crown.

## 4. Dataset build (`revision/pipeline/`)

Copy the Sherlock output folder (`predictions/`, `split/`, `stats/`) to a local folder `V1`, then:

```bash
cd revision/pipeline
URBAN_DATA=/data/urban V1=/data/urban/v1b_pan_ndvi WORK=/data/urban/work_v1b \
FINAL=/data/urban/tree_crowns_v1b OUT=/data/urban/tree_crowns_v1b_ndvi02 bash build_v1.sh
```

| Step | Script | Result |
|---|---|---|
| 00 | `00_census_attributes.py` | Census income and demographics per unit (original-release attributes in `inputs/original_bg_lookup/`; new units from ACS 2019–2023 B19013/B03002 and StatCan 98-401-X2021006) |
| 01v1 | `01v1_tile_inventory.py` | One prediction per in-city window, with checks (threshold 0.7, input channel order) |
| 16, 08, 04 | `16_scan_new_tiles.py`, `08_georef_table.py`, `04_city_pipeline_audit.py` | Crown tables per tile, georeferencing, city crown tables |
| 07 | `07_artifact_screen.py` | Screen of all components ≥ 20,000 px; the visual review flagged none (`inputs/artifact_components.csv`) |
| 06 | `06_build_city_outputs.py` | Crowns ≥ 3 m², 1.0 m cross-tile deduplication, centroid inside the municipality, > 250 m² flagged |
| 11, 12 | `11_validate_outputs.py`, `12_block_group_coverage.py` | Checks and coverage |
| 17, 17b, 20 | `17_crown_ndvi.py`, `17b_crown_ndvi_join.py`, `20_ndvi_mask.py` | Mean NDVI under each crown; crowns with mean NDVI < 0.2 removed |

The result is the final dataset: the points are in `data/tree_points/` (with `build_log.json` and `ndvi_mask_log.json`) and the crown outlines in `data/crown_polygons/`. Census units come from `data/boundaries/`. Steps 00 and 17 need the ACS and StatCan tables and the NASA PGC mosaics under `URBAN_DATA` (see `paths.py` and the script headers).

## 5. Statistics, figures and tables (`revision/analysis/`)

Reassemble the city files in `data/tree_points/` into one folder (see `data/tree_points/README.md`), add the crown polygons with `python revision/analysis/rebuild_polygons.py <tree points folder>` (from `data/crown_polygons/`), then:

```bash
python revision/analysis/run_analysis.py --dataset <tree points folder> --label final --out <run folder> \
    --build-logs 'data/tree_points/build_log.json'
```

| Manuscript item | Step in `run_analysis.py` | Script |
|---|---|---|
| Census-unit table (density, cover, mean crown area, covariates) | `units` | `build_unit_table.py` |
| Fig. 1 | `fig1` | `repo_adapted/generate_density_map_from_geojson.py` |
| Figs. 2–4 | `fig2_4` | `repo_adapted/generate_plots_corrected.py` |
| Unadjusted Spearman correlations (main text) | `stats` | `manuscript_stats.py` |
| Extended Data Figs. 3–4 (OLS with HC3, spatial error models, BH correction) | `ed3_4_models`, `ed3_4_plots` | `adjusted_models.py`, `plot_adjusted.py` |
| Extended Data Table 1 | `ed_table1` | `ed_table1.py` (`ed_table1_png.py` renders it) |
| Extended Data Fig. 1 | `ed_fig1` | `ed_fig1.py` |
| Extended Data Fig. 2 | `ed_fig2` | `repo_adapted/plot_crown_area_histogram.py` |
| Fig. S5, Supplementary Note 4 (municipal inventories) | `figS5` | `repo_adapted/inventory_correlation_analysis.py` (set `INVENTORY_CSV`) |
| Fig. S6, Supplementary Note 5 (Ripley's L) | `figS6` | `ripley_igel/` (Christian Igel's code, UTM, convex-hull area; see its README) |
| Figs. S9–S11 (HOLC redlining, Austin) | `figS9_11` | `repo_adapted/generate_redlining_figures.py` (`data/redlining/`) |
| Fig. S12 (Surrey crown-merging sensitivity) | `figS12_stats`, `figS12_plot` | `repo_adapted/run_surrey_reviewer2_sensitivity.py`, `repo_adapted/plot_surrey_s12_three_panel.py` |
| Sensitivity analyses | `sensitivity` | `sensitivity.py` |

Other items:

- **Fig. 5 and Fig. S8** (imagery acquisition maps and MODIS NDVI seasonality): `imagery_metadata/generate_image_year_map.py`, with the MODIS curves in `imagery_metadata/city_seasonal_ndvi_curves.csv`; it reads the mosaic cutline metadata.
- **Climate and built-up context** (Köppen–Geiger × GHS-BUILT-S shares in the Introduction): `climate_context/audit_and_recalculate_independent_city_analysis.py` (inputs: Beck et al. 2018 Köppen–Geiger 1 km, GHS-BUILT-S R2023A 2020 30″, Natural Earth admin-1); results in `climate_context/*.csv`.
- **Fig. 6** (example detections): `fig6_review/` (candidate windows, scoring, contact sheets, final figure); needs the NDVI mosaics.
- **Fig. S4** (density of municipal inventory trees) and **Fig. S7** (resolution comparison) are unchanged from the original submission: Fig. S4 uses only the municipal inventories, not the model output, and Fig. S7 is an illustration made from the imagery.
- The Ripley's L analysis (Fig. S6) is Christian Igel's code (`revision/analysis/ripley_igel/`, from github.com/christian-igel/city-trees), run in the UTM zone of each city with the convex hull of each polygon's trees as the area (pointpats uses the bounding box; see its README); its results are in `results/final_v1b/ripley_igel/`, with the previous bounding-box results in `utm_bbox_previous/`. The earlier convex-hull re-implementation (`ripley_l.py`, `results/final_v1b/ripley/`) is kept as the reference step `ripley_hull_reference`.
- The final statistics produced by this step are in `results/final_v1b/`, for comparison.

## 6. Model validation (`revision/sherlock/`)

`08_validation.sbatch` and `09_threshold_validation.sbatch` predict the 113 annotated patches, binarize them (τ = 0.3 / 0.5 / 0.7), separate instances as in step 3, and compute pixel- and object-level metrics with `repo_adapted/evaluate_object_accuracy.py` (patch georeferencing in `validation/tile_metadata.csv`). `13_validation_figfix.sbatch` makes Figs. S1–S3 at τ = 0.7 (`repo_adapted/evaluate_object_accuracy_figfix.py`). Results: `results/validation/`.
