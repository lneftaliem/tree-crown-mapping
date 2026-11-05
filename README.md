# Mapping tree crowns across four North American cities

Code accompanying the manuscript **"Mapping tree crowns across four North American cities"** (Neftaliem et al.). The paper has been submitted for publication.

We fine-tune a U-Net convolutional neural network, originally developed for mapping dryland woody vegetation in the African Sahel ([Brandt et al., 2020](https://doi.org/10.1038/s41586-020-2824-5); [Tucker et al., 2023](https://doi.org/10.1126/science.abg1740)), to delineate individual tree crowns in Austin (TX), Bloomington (IN), Cupertino (CA), and Surrey (BC) from 50 cm multispectral satellite imagery. The model is fine-tuned on 113 annotated images (4,559 crowns) and applied without per-city retraining. Binary predictions (τ = 0.7) are split into individual crowns by center-based instance separation, giving about 3.6 million detected tree crowns across the four cities. We then relate detected-tree density, detected-crown cover, and mean detected crown area to census-level median household income.

**Final analysis.** The results in the revised manuscript come from the code in `revision/` and the final datasets in `data/tree_points/` and `data/covariates/`; the final statistics tables are in `results/final_v1b/`. **[`REPRODUCING.md`](REPRODUCING.md) walks through every step, from model fine-tuning to each figure and table.** The root-level scripts and `reproduce.sh` reproduce the original submission from `data/original_submission/`.

Clone with `git clone --recurse-submodules` to include the instance-separation code (`external/separate_instances`).

## Final analysis

| Folder | Contents |
|---|---|
| `revision/model/` | Training-data preparation and U-Net fine-tuning (`1-Preprocessing.ipynb`, `2-UNetTraining.ipynb` / `.py`, `core/`, `config/`). The fine-tuned weights are on the [v1.0 release](https://github.com/lneftaliem/tree-crown-mapping/releases/tag/v1.0). |
| `revision/separation/`, `external/separate_instances` | Center-based instance separation written by Christian Igel and Ankit Kariryaa ([`separate_instances`](https://github.com/ankitkariryaa/separate_instances); method described in Mugabowindekwe et al., 2023), included as a git submodule pinned to `d81cac7`, and the Sherlock wrapper that runs it on each prediction tile |
| `revision/sherlock/` | Slurm jobs for city-wide inference (`12_v1b_inference.sbatch`, `regen_v1b.py`, `unet_infer.py`, and the scene and window lists) and model validation on the 113 annotated patches |
| `revision/pipeline/` | Crown polygons and points from the separated tiles, 1.0 m cross-tile deduplication, census joins and the NDVI mask (`build_v1.sh` runs every step; `inputs/` holds the recorded manual decisions and census attributes it uses) |
| `revision/analysis/` | `run_analysis.py`, a single entry point that regenerates every tree-derived statistic, figure and table from one named dataset, plus the adjusted models, Ripley's L, Extended Data tables and figures, sensitivity analyses and land-cover diagnostics. `repo_adapted/` holds the versions of the root-level scripts used for the final run; `imagery_metadata/` makes Fig. 5 and Fig. S8; `climate_context/` computes the climate and built-up shares; `fig6_review/` makes Fig. 6. |
| `results/final_v1b/` | Final statistics tables (unadjusted correlations, adjusted and spatial models, Ripley's L, inventory comparison, sensitivity analyses, Extended Data Table 1) and comparisons with the original submission (`comparison_to_original/`) |

The crown outlines are in `data/crown_polygons/`. To rerun the analysis, decompress the four city files in `data/tree_points/` into one folder, rebuild the full polygon files there with `python revision/analysis/rebuild_polygons.py <folder>`, and run:

```bash
python revision/analysis/run_analysis.py --dataset <folder> --label final --out <run folder>
```

Some steps need inputs not stored here: the licensed imagery, the TIGER/Line and StatCan census tables, and the municipal tree inventories. Paths to local data are set with the `URBAN_DATA` environment variable (default `/path/to/Urban_data`); see [`REPRODUCING.md`](REPRODUCING.md).

## Repository contents (original submission)

| Script | Stage | Purpose |
|---|---|---|
| `3-RasterAnalysis.py` | Inference | Runs the trained U-Net over satellite image tiles (NDVI + panchromatic) to produce per-pixel canopy probability rasters. |
| `prepare_ripley_data.py` | Post-processing | Converts predicted rasters into per-tree point datasets (centroid, crown area) in projected UTM coordinates, with cross-tile spatial deduplication. |
| `csv_to_geojson.py` | Post-processing | Converts per-city tree point CSVs into GeoJSON. |
| `assign_income_to_trees.py` | Analysis prep | Joins detected trees to census block groups / dissemination areas and attaches median household income. |
| `city_tree_analysis.py` | Analysis | Main driver: aggregates tree counts, canopy cover, and crown area to census units and produces the city-level figures. |
| `csv_tree_density_map.py` / `generate_density_map_from_geojson.py` | Figures | Publication-style tree density choropleth maps (Fig. 1). |
| `inventory_correlation_analysis.py` | Analysis | Compares model-detected tree counts against municipal tree inventories. |
| `tree_size_vs_income.py` | Analysis | Income vs. mean/median crown area per census unit (Fig. 4). |
| `plot_crown_area_histogram.py` | Figures | Crown area distribution histograms across cities (Extended Data Fig. 2). |
| `prepare_ripley_data.py` / `run_sensitivity.py` | Analysis | Ripley's K spatial clustering analysis and deduplication-threshold sensitivity test (Extended Data Fig. 1, Fig. S6). |
| `revision/analysis/ripley_igel/` | Analysis | Ripley's L analysis for Fig. S6 and Supplementary Note 5, written by Christian Igel ([christian-igel/city-trees](https://github.com/christian-igel/city-trees)); revised version in UTM coordinates with the convex hull of the trees as the area. |
| `repro_data.py` | Reproducibility | Stages the committed `data/` archives (reassembles the split files, decompresses) into the layout the figure scripts read from. Imported automatically by the reproducible scripts. |
| `reproduce.sh` | Reproducibility | One command that stages the committed data and regenerates every figure and table that can be built from the public data (no imagery or model weights required). |

### Revision analyses

| Script | Purpose | Reference outputs |
|---|---|---|
| `run_crown_area_correction_pipeline.py` | Converts each tree's pixel count to m² using the true ground area of a pixel at that tree's latitude, then re-aggregates tree count, crown area, and canopy cover to census units. | `data/covariates/all_cities_urban_covariates_corrected.csv`, `results/crown_area_correction/` |
| `generate_plots_corrected.py` | Cross-city income vs. tree density, canopy cover, and mean crown area scatter plots using the corrected crown areas. | `results/equity_corrected/` |
| `run_park_covariate_pipeline.py` | Park and natural-area coverage per census unit from each city's official GIS park layers (downloaded live). | `data/covariates/all_cities_park_covariates.csv` |
| `run_surrey_reviewer2_sensitivity.py` / `plot_surrey_reviewer2_figure.py` | Tests whether merging of touching crowns in dense canopy drives the negative income vs. crown area association in Surrey (size filters, quantiles, dense-canopy exclusion, bootstrap CIs). | `results/surrey_reviewer2/`, `docs/SURREY_SENSITIVITY_METHODOLOGICAL_REPORT.md` |
| `run_austin_exploratory_spatial_analysis.py` | Austin Spearman correlations with FDR correction, sequential OLS, spatial error models, distance-to-downtown and income-quartile breakdowns. | `results/austin_spatial/` |
| `generate_redlining_figures.py` | Exploratory comparison of 1937 HOLC grades with present-day detected trees in Austin, with leave-one-polygon-out sensitivity. | `results/austin_redlining/`, `docs/METHODS_REDLINING_ANALYSIS.md`, `docs/REDLINING_FIGURE_INTERPRETATION.md` |
| `evaluate_object_accuracy.py` + `slurm/*.sbatch` | Pixel- and object-level validation on the 113 annotated patches, broken down by training split and canopy stratum (runs on Sherlock; needs the annotations, imagery, and weights). | `results/validation/`, `data/validation/image_manifest_113.csv`, `docs/VALIDATION_METRICS_AUDIT_REPORT.md` |

A model card for the fine-tuned network is provided in [`MODEL_CARD.md`](MODEL_CARD.md).

## Data

Final, published datasets live under `data/`:

```
data/
├── boundaries/
│   ├── <city>_exact_boundary.geojson        # city/study-area outline
│   └── <city>_exact_block_groups.geojson    # census block groups / dissemination areas used in the analysis
├── tree_points/                                   # FINAL per-tree datasets (see data/tree_points/README.md)
│   ├── cupertino_ripley_points_with_income.geojson.gz
│   ├── bloomington_ripley_points_with_income.geojson.gz
│   ├── austin_ripley_points_with_income.geojson.gz.part00 ... part20
│   └── surrey_ripley_points_with_income.geojson.gz.part00 ... part06
├── crown_polygons/                                # FINAL crown outlines, joined to the points by tree_id (see data/crown_polygons/README.md)
│   └── <city>_crown_polygons.geojson.gz.partNN
├── parks/surrey/                                  # City of Surrey park and natural-area layers used for the park covariate
├── covariates/                                    # FINAL census-unit tables (1,344 analysis units)
│   ├── all_cities_urban_covariates.csv            # income, housing, distance to downtown, tree metrics
│   ├── all_cities_urban_covariates_corrected.csv  # same, with latitude-corrected crown area and canopy cover
│   └── all_cities_park_covariates.csv             # park and natural-area coverage per census unit
├── original_submission/                           # tree points and covariates of the original submission (1,283 units)
├── redlining/
│   └── austin_redlining_data.json                 # 1937 HOLC polygons (Mapping Inequality, CC BY-NC-SA 4.0; see data/redlining/README.md)
├── imagery_cutlines/
│   └── <city>_mosaic_cutlines.geojson             # source-image footprints, image IDs and acquisition metadata for every mosaic tile (see data/imagery_cutlines/README.md)
└── validation/
    └── image_manifest_113.csv                     # the 113 annotated validation patches
```

`results/` holds the reference outputs (tables and figures) of the revision analyses, and `docs/` the accompanying methods notes and data citations (`docs/DATA_CITATIONS.md`, `docs/data_citations.bib`).

Each `<city>_ripley_points_with_income` file is a deduplicated per-tree point dataset (crown centroid, crown area, census join, and income). The final files are in `data/tree_points/`; field definitions are in `data/tree_points/README.md`.

Cupertino's and Bloomington's are small enough to commit as a single gzip file (~9 MB and ~10 MB compressed); decompress with:

```bash
gunzip -k data/tree_points/cupertino_ripley_points_with_income.geojson.gz
```

Austin's and Surrey's datasets are 2.3 GB and 670 MB, too large for a single file on GitHub (100 MB hard limit) even after gzip (~190 MB and ~57 MB compressed), and too large for the web upload UI (25 MB per file) either way. They're split into ~9 MB chunks with `split`. Reassemble and decompress with:

```bash
cat data/tree_points/austin_ripley_points_with_income.geojson.gz.part* > austin_ripley_points_with_income.geojson.gz
gunzip austin_ripley_points_with_income.geojson.gz

cat data/tree_points/surrey_ripley_points_with_income.geojson.gz.part* > surrey_ripley_points_with_income.geojson.gz
gunzip surrey_ripley_points_with_income.geojson.gz
```

### Latitude-corrected crown areas

The revision analyses use crown areas in m² that account for latitude: image pixels have a fixed size in degrees, so their ground area is `DEG_PER_PX² × 111320² × cos(latitude)` (about 0.32 m² per pixel in Surrey vs. 0.25 m² nominal). The corrected per-tree files (`<city>_ripley_points_with_income_corrected.geojson`) are an exact, deterministic function of the committed point data, so they are not stored separately. `repro_data.load_corrected_points()` rebuilds them in memory when a script needs them, and `python run_crown_area_correction_pipeline.py --write-points` writes them to `analysis_output/ripley_data_corrected/` (~2.2 GB for all four cities).

## Reproducing the original submission (from committed data only)

The scripts in this section read the original-submission data in `data/original_submission/`.

A subset of the paper's figures can be regenerated end-to-end from the data committed in this repository, with **no licensed imagery and no model weights**, only the Python packages in `requirements.txt`. After `pip install -r requirements.txt`:

```bash
bash reproduce.sh
```

This stages the committed `data/` archives into the locations the scripts read from (reassembling the split Austin/Surrey files and decompressing every city), then regenerates the figures and tables below into `analysis_output/`:

| Output file | Paper figure | Inputs (all committed) |
|---|---|---|
| `crown_area_histogram.png` | Crown-area distributions (Extended Data Fig. 2) | `data/original_submission/tree_points/` |
| `combined_income_crown_area_scatter.{png,pdf}` | Income vs. crown area | `data/original_submission/tree_points/` |
| `identical_tree_density_map.{png,pdf}` | Tree-density choropleth (Fig. 1) | `data/original_submission/tree_points/` + `data/boundaries/` |
| `equity_results/all_cities_urban_covariates_corrected.csv`, `crown_area_correction_verification.csv` | Latitude-corrected census-unit metrics | `data/original_submission/tree_points/` + `data/original_submission/covariates/` |
| `equity_results/combined_income_zscore_*_scatter*.{png,pdf,svg}` | Income vs. density, canopy cover, crown area (corrected) | `data/original_submission/covariates/` |
| `austin_spatial_analysis/` | Austin correlations, regressions, spatial error models, figures A to C | `data/original_submission/covariates/` + `data/boundaries/` |
| `surrey_reviewer2_analysis/` | Surrey crown-merging sensitivity tables, report, and figure | `data/original_submission/tree_points/` + `data/original_submission/covariates/` |
| `redlining_figures/` | Austin HOLC figures 1 to 3 and polygon/grade/sensitivity tables | `data/original_submission/tree_points/` + `data/redlining/` + `data/boundaries/` |

Each table should match its copy in `results/` (checked on a clean copy of the repository; figures can differ slightly in font rendering if Arial is not installed). The park covariates are the one exception: `run_park_covariate_pipeline.py` downloads each city's current park layer, so a fresh run can drift from the committed `all_cities_park_covariates.csv` if a city edits its data.

Reassembling and reading the full Austin (1.1 GB) and Surrey (717 MB) GeoJSONs needs several GB of free RAM and disk. To verify the pipeline quickly on just the two small, single-file cities (this skips the Surrey and Austin-only analyses):

```bash
TREE_MAPPING_CITIES=cupertino,bloomington bash reproduce.sh
```

You can also run any of the scripts directly (for example `python plot_crown_area_histogram.py` or `python run_surrey_reviewer2_sensitivity.py`), as each one auto-stages the committed data via `repro_data.py` on import. Staging is idempotent and never overwrites existing outputs.

The remaining figures (canopy cover and inventory-comparison analyses, Fig. 4's temperature-mediation model, and the raster-based detection maps) require inputs that cannot be redistributed here (licensed imagery, the model weights, municipal inventories, and raw census/temperature layers); see **What you'll need to add** below. The committed per-tree datasets already carry each tree's crown area, census unit, median household income, and demographic shares, so most census-level analyses can be recomputed directly from `data/original_submission/tree_points/` without rerunning detection.

## What you'll need to add

This repo contains the analysis and post-processing code. To run the full pipeline end-to-end you'll also need the following, which are **not included here**:

1. **The base model architecture and training code (`core/` and `config/`).** The versions used in this study are in `revision/model/core/` and `revision/model/config/`; copy them next to the script you run. They follow Ankit Kariryaa's tree-detection framework,
   [ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) (MIT licensed).

2. **Fine-tuned model weights.** `trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras` (~376 MB) is too large for a plain GitHub push (GitHub blocks files over 100 MB), so it is distributed as a **GitHub Release asset**: download it from the repository's [**Releases** page](https://github.com/lneftaliem/tree-crown-mapping/releases/latest) (provided as a ~327 MB zip; unzip to get the ~376 MB `.keras` file).

   Once you have the file, point the scripts at it via the `TREE_MAPPING_BASE_DIR` environment variable (see below) or place it at `saved_models/UNet/` under your base directory.

3. **Satellite imagery.** Vantor imagery (QuickBird-2, GeoEye-1, WorldView-2, WorldView-3) was acquired by NASA under the Electro-Optical Commercial Layer License. The copyright remains with Vantor, and redistribution is not possible. The imagery is available to U.S. government-funded researchers through [EarthExplorer](https://earthexplorer.usgs.gov/) or from Vantor. The image IDs and acquisition metadata for every scene in our mosaics are listed in `data/imagery_cutlines/`.

4. **Auxiliary data**, obtained from public sources and not redistributed in this repo:
   - Census boundaries: US Census Bureau TIGER/Line block groups (`tl_2023_<state>_bg`) and place boundaries; Statistics Canada dissemination area boundaries (`lda_000b21a_e`) for Surrey.
   - Income: ACS 5-Year Estimates 2019–2023, Table B19013 (US); 2021 Census Profile median household income (Canada).
   - Municipal tree inventories (used only for the inventory-comparison analysis), obtained from each city's open data portal.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`geopandas`/`rasterio`/`fiona` depend on GDAL; if `pip install` fails on your system, install via conda instead:

```bash
conda create -n tree-crown python=3.10
conda activate tree-crown
conda install -c conda-forge geopandas rasterio fiona rtree gdal
pip install -r requirements.txt
```

## Configuring paths

The scripts no longer hardcode any machine-specific paths. Set an environment variable pointing to your working data directory before running anything:

```bash
export TREE_MAPPING_BASE_DIR=/path/to/your/data
```

Scripts default to the current working directory if this isn't set. Expected subdirectories under `TREE_MAPPING_BASE_DIR` (created as needed by each stage):

```
<TREE_MAPPING_BASE_DIR>/
├── cutouts/                     # extracted image tiles for inference
├── saved_models/UNet/            # trained .keras model weights
├── notebooks/census_boundaries/  # TIGER/Line + StatCan boundary shapefiles
├── notebooks/income_data/        # ACS / StatCan income tables
├── notebooks/urbantreedata_2023/ # municipal inventory CSVs
├── all_trees/                    # per-city Ripley point datasets (CSV/GeoJSON)
└── analysis_output/              # all figures and derived tables get written here
```

## Pipeline order (original submission)

For the revised manuscript, follow [`REPRODUCING.md`](REPRODUCING.md) instead.

1. Fine-tune / obtain the U-Net model (see `core`/`config` note above).
2. `3-RasterAnalysis.py`: run inference over image tiles to get canopy probability rasters.
3. `prepare_ripley_data.py`: extract per-tree points and crown areas, deduplicate across tile overlaps.
4. `csv_to_geojson.py`, `assign_income_to_trees.py`: attach census geography and income to detected trees.
5. `city_tree_analysis.py`, `csv_tree_density_map.py` / `generate_density_map_from_geojson.py`, `inventory_correlation_analysis.py`, `tree_size_vs_income.py`, `plot_crown_area_histogram.py`, `run_sensitivity.py`: analysis and figure generation (can be run independently once step 4's outputs exist).
6. `run_crown_area_correction_pipeline.py`: latitude-corrected crown areas and census-unit metrics, followed by the revision analyses (`generate_plots_corrected.py`, `run_park_covariate_pipeline.py`, `run_surrey_reviewer2_sensitivity.py` + `plot_surrey_reviewer2_figure.py`, `run_austin_exploratory_spatial_analysis.py`, `generate_redlining_figures.py`).
7. `evaluate_object_accuracy.py` (via `slurm/run_object_validation.sbatch` on Sherlock): validation against the 113 annotated patches. `python evaluate_object_accuracy.py --run_unit_tests` checks the matching logic on synthetic cases without any cluster data.

## Citation

If you use this code, please cite the manuscript and the original detection framework:

> Neftaliem, L., Anderson, C.J., Igel, C., Field, C.B., Jackson, R.B., Kariryaa, A., Small, J., Tucker, C.J. Mapping tree crowns across four North American cities. Manuscript submitted for publication.

> Brandt, M. et al. An unexpectedly large count of trees in the West African Sahara and Sahel. *Nature* 587, 78–82 (2020).
