# Individual Tree Crown Mapping Across Four North American Cities

Code accompanying the manuscript **"Individual tree crown mapping across four North American cities from high-resolution satellite imagery"** (Neftaliem et al., in preparation for *Nature Cities*).

We fine-tune a U-Net convolutional neural network — originally developed for mapping dryland woody vegetation in the African Sahel ([Brandt et al., 2020](https://doi.org/10.1038/s41586-020-2824-5); [Tucker et al., 2023](https://doi.org/10.1126/science.abg1740)) — to delineate individual tree crowns in Austin (TX), Bloomington (IN), Cupertino (CA), and Surrey (BC) from 50 cm multispectral satellite imagery. The model is fine-tuned on 113 annotated images (4,589 crowns) and applied to map ~3.3 million trees across the four cities without per-city retraining. We then relate crown-level tree density, canopy cover, and crown size to census-level median household income.

## Repository contents

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

## What you'll need to add

This repo contains the analysis and post-processing code. To run the full pipeline end-to-end you'll also need the following, which are **not included here**:

1. **The base model architecture and training code (`core/` and `config/`).** The U-Net implementation, loss functions (`tversky`, `dice_coef`, ...), optimizers, and data generators that `3-RasterAnalysis.py` imports (`from core.UNet import UNet`, `from config import RasterAnalysis`, etc.) come from Ankit Kariryaa's tree-detection framework:
   [ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) (MIT licensed).
   Clone that repo and copy its `notebooks/core/` and a `notebooks/config/RasterAnalysis.py` (see that repo's `configTemplate/`) into your working directory, or add it as a git submodule. Please retain their license/attribution.

2. **Fine-tuned model weights.** `trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras` (~376 MB) is too large for a plain GitHub push (GitHub blocks files over 100 MB) and isn't included in this repo. To make it available to collaborators, either:
   - Track it with [Git LFS](https://git-lfs.com/) (`git lfs track "*.keras"`) if you want it versioned in this repo, or
   - Host it externally (e.g., [Zenodo](https://zenodo.org/), Hugging Face, or a lab Google Drive/Box) and link the download URL here once the paper is public.

   Once you have the file locally, point the scripts at it via the `TREE_MAPPING_BASE_DIR` environment variable (see below) or place it at `saved_models/UNet/` under your base directory.

3. **Satellite imagery.** Imagery was licensed from Vantor (QuickBird-2, GeoEye-1, WorldView-2, WorldView-3) under the Next View License and cannot be redistributed here. You'll need your own licensed source of comparable sub-meter multispectral imagery.

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

## Pipeline order

1. Fine-tune / obtain the U-Net model (see `core`/`config` note above).
2. `3-RasterAnalysis.py` — run inference over image tiles to get canopy probability rasters.
3. `prepare_ripley_data.py` — extract per-tree points and crown areas, deduplicate across tile overlaps.
4. `csv_to_geojson.py`, `assign_income_to_trees.py` — attach census geography and income to detected trees.
5. `city_tree_analysis.py`, `csv_tree_density_map.py` / `generate_density_map_from_geojson.py`, `inventory_correlation_analysis.py`, `tree_size_vs_income.py`, `plot_crown_area_histogram.py`, `run_sensitivity.py` — analysis and figure generation (can be run independently once step 4's outputs exist).

## Citation

If you use this code, please cite the manuscript (citation to be updated once published) and the original detection framework:

> Neftaliem, L., Anderson, C., Igel, C., Field, C.B., Jackson, R.B., Small, J., Tucker, C.J. Individual tree crown mapping across four North American cities from high-resolution satellite imagery. *Nature Cities* (in preparation).

> Brandt, M. et al. An unexpectedly large count of trees in the western Sahara and Sahel. *Nature* 587, 78–82 (2020).
