# Model Card: Individual Tree Crown Segmentation U-Net (4-City North American Fine-Tune)

A fully convolutional **U-Net** that segments individual tree crowns from 50 cm
multispectral satellite imagery. It is a fine-tune of the dryland-vegetation
U-Net of Brandt et al. (2020) / Tucker et al. (2023), adapted to delineate urban
and peri-urban tree crowns in four North American cities.

> This card follows the Hugging Face
> [annotated model card template](https://huggingface.co/docs/hub/model-card-annotated).
> Fields that depend on the manuscript's final reported numbers are marked
> **[More Information Needed]** and should be filled from the paper before release.

## Table of Contents

- [Model Details](#model-details)
- [Uses](#uses)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation](#citation)
- [Glossary](#glossary)
- [More Information](#more-information)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)

## Model Details

### Model Description

A U-Net convolutional neural network that takes a 2-channel raster patch
(NDVI + panchromatic) and outputs a per-pixel probability of belonging to a tree
crown. Connected components of the thresholded probability map are treated as
individual crowns, from which centroids and crown areas are derived. The model
was fine-tuned on 113 manually annotated images (4,589 delineated crowns) and
applied to map ~3.3 million trees across Austin (TX), Bloomington (IN),
Cupertino (CA), and Surrey (BC) **without per-city retraining**.

- **Developed by:** L. Neftaliem, C. Anderson, C. Igel, C. B. Field, R. B. Jackson, J. Small, C. J. Tucker
- **Funded by:** [More Information Needed]
- **Shared by:** [lneftaliem](https://github.com/lneftaliem)
- **Model type:** Semantic segmentation (fully convolutional U-Net) for individual tree crown delineation
- **Language(s):** Not applicable (geospatial imagery model)
- **License:** [More Information Needed] — the upstream architecture/training framework (Kariryaa et al.) is MIT-licensed; the fine-tuned weights are **available on request** from the corresponding author. Specify the weights' redistribution terms here.
- **Finetuned from model:** Sahel dryland-vegetation U-Net of [Brandt et al., 2020](https://doi.org/10.1038/s41586-020-2824-5) / [Tucker et al., 2023](https://doi.org/10.1126/science.abg1740), built on the [ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) framework (MIT).

### Model Sources

- **Repository:** https://github.com/lneftaliem/tree-crown-mapping
- **Paper:** Neftaliem et al., "Individual tree crown mapping across four North American cities from high-resolution satellite imagery." Manuscript submitted for publication. [DOI to be added]
- **Demo:** N/A
- **Weights file:** `trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras` (~376 MB; not hosted in the repo — available on request)

## Uses

### Direct Use

Delineating individual tree crowns and estimating per-tree crown area from
sub-meter (≈50 cm) multispectral imagery over the four study cities and
comparable temperate North American urban/peri-urban settings. Downstream, the
per-tree outputs support estimates of tree density, canopy cover, and crown-size
distributions at the census-unit level, including comparisons to socioeconomic
variables such as median household income.

### Downstream Use

- Urban forestry inventories and canopy-equity analyses.
- Baselines / transfer-learning starting points for tree-crown segmentation in
  new cities with similar imagery.
- Change detection when applied to multi-date imagery of the same area
  (with appropriate caution; see limitations).

### Out-of-Scope Use

- Imagery whose resolution, spectral bands, sensor, or landscape differ
  substantially from the ≈50 cm NDVI+panchromatic training imagery
  (e.g., 10 m Sentinel-2, RGB drone imagery, dense closed-canopy tropical
  forest) — expect degraded and unvalidated performance.
- **Species identification, tree health, height, biomass, or carbon estimation** —
  the model outputs crown extent only, not any of these attributes.
- Legal, regulatory, property-boundary, or tax determinations about individual
  parcels or trees.
- Any use presenting per-tree detections as a ground-truth census; outputs are
  model estimates with non-zero omission and commission error.

## Bias, Risks, and Limitations

- **Geographic/sensor domain shift:** trained and validated on four specific
  cities and a specific set of commercial sensors (QuickBird-2, GeoEye-1,
  WorldView-2, WorldView-3). Accuracy elsewhere is unknown.
- **Crown crowding / merging:** in closed-canopy or tightly spaced settings,
  adjacent crowns may merge into one detection or single crowns may split,
  biasing counts and crown-area statistics. A deduplication + Ripley's K
  sensitivity analysis is included in the repository to characterise this.
- **Imagery conditions:** phenology, sun angle, shadow, off-nadir view, and
  seasonal leaf-off conditions affect NDVI and detection.
- **Small/understory trees** beneath larger canopies are systematically
  under-detected at this resolution.
- **Equity analyses inherit model error:** because detections carry omission and
  commission error that may correlate with neighbourhood structure (lot size,
  planting patterns), care is needed when relating detections to demographic or
  income variables so model error is not read as a social signal.

### Recommendations

Report and propagate detection uncertainty into any downstream density/canopy/
equity estimates. Validate against an independent local reference (e.g., a
municipal inventory or manual annotation) before using the model in a new city.
Treat crown-area statistics as relative rather than absolute, and prefer
aggregate (census-unit) conclusions over per-tree claims.

## How to Get Started with the Model

The model is applied through `3-RasterAnalysis.py` in the repository, which
tiles the input rasters into 256×256 patches, normalises each channel, runs
inference with overlap, and merges patch predictions into per-tile canopy
probability rasters. In outline:

```bash
# 1. Obtain the base framework (core/ + config/) and the fine-tuned weights
#    (see the repository README, "What you'll need to add").
export TREE_MAPPING_BASE_DIR=/path/to/your/data   # weights at saved_models/UNet/

# 2. Provide co-registered NDVI + panchromatic tiles under cutouts/, then:
python 3-RasterAnalysis.py

# 3. Post-process rasters into per-tree points (centroid + crown area):
python prepare_ripley_data.py
```

To reproduce the **published figures** from the already-processed per-tree
datasets committed in the repo — no imagery or weights required — run
`./reproduce.sh` (see the README).

## Training Details

### Training Data

113 manually annotated images containing 4,589 delineated tree crowns across the
four study cities, drawn from the same ≈50 cm multispectral imagery used at
inference. Annotations delineate individual crown polygons used to derive the
segmentation target (with a weight map emphasising crown boundaries).
[More Information Needed: per-city annotation counts and train/val/test split.]

### Training Procedure

#### Preprocessing

- Two input channels per patch: **NDVI** and **panchromatic**.
- Imagery tiled into **256 × 256** patches; each channel normalised
  independently over the patch (per-channel standardisation).
- Boundary-weighted target masks (`weightmap`) to sharpen crown separation.

#### Training Hyperparameters

- **Architecture:** U-Net (encoder–decoder with skip connections).
- **Loss:** Tversky loss (`tversky`); monitored with Dice coefficient,
  accuracy, specificity, and sensitivity.
- **Optimizer:** AdaDelta.
- **Patch size:** 256 × 256; **inference stride:** 128 (50% overlap), predictions
  merged with a MAX operator.
- **Epochs / batch size / Tversky α–β / learning-rate schedule:** [More Information Needed]

#### Speeds, Sizes, Times

- **Trained weights file size:** ~376 MB (`.keras`).
- **Training wall-clock time / throughput:** [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Held-out annotated crowns and, for external validation, municipal tree
inventories from each city's open-data portal (used in
`inventory_correlation_analysis.py`). [More Information Needed: exact held-out set definition.]

#### Factors

Performance should be disaggregated by **city**, and ideally by land-cover /
neighbourhood context and canopy density, since these drive crown-merging error.

#### Metrics

Segmentation quality via Dice coefficient / IoU, sensitivity (recall) and
specificity; count-level agreement via correlation between detected counts and
municipal inventory counts per unit. These metrics are appropriate because the
task is both pixel-level segmentation and object-level counting.

### Results

[More Information Needed — insert the manuscript's reported detection/segmentation
metrics and the model-vs-inventory correlation coefficients here. Do not populate
with estimated values.]

#### Summary

The fine-tuned model maps ~3.3 million trees across the four cities from a single
set of weights (no per-city retraining), and detected counts are compared against
independent municipal inventories in the accompanying manuscript.

## Environmental Impact

Carbon emissions can be estimated with the
[ML CO2 Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** [More Information Needed] (fine-tuning performed on the Stanford Sherlock HPC cluster per the code paths)
- **Hours used:** [More Information Needed]
- **Cloud Provider:** N/A (on-premise HPC)
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications

### Model Architecture and Objective

U-Net fully convolutional encoder–decoder with skip connections. Input: a
2-channel (NDVI + panchromatic) 256×256 patch. Output: a single-channel
per-pixel tree-crown probability map, optimised with Tversky loss. Individual
crowns are obtained by thresholding and connected-component labelling; crown area
is the labelled region area converted to projected (UTM) units.

### Compute Infrastructure

#### Hardware

[More Information Needed] — GPU-class hardware for training; the repository's
inference and post-processing scripts run on CPU/GPU workstations. Reassembling
and reading the full per-city GeoJSONs needs several GB of RAM/disk.

#### Software

- Python 3.10
- TensorFlow/Keras (`>=2.10,<2.16`) for model inference
- Geospatial stack: geopandas, shapely, rasterio, fiona, rtree, affine (GDAL)
- Scientific stack: numpy, pandas, scipy, scikit-image, statsmodels
- See `requirements.txt` for the full list.

## Citation

**BibTeX:**

```bibtex
@article{neftaliem_tree_crown_mapping,
  title   = {Individual tree crown mapping across four North American cities from high-resolution satellite imagery},
  author  = {Neftaliem, L. and Anderson, C. and Igel, C. and Field, C. B. and Jackson, R. B. and Small, J. and Tucker, C. J.},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}

@article{brandt2020unexpectedly,
  title   = {An unexpectedly large count of trees in the western Sahara and Sahel},
  author  = {Brandt, Martin and others},
  journal = {Nature},
  volume  = {587},
  pages   = {78--82},
  year    = {2020},
  doi     = {10.1038/s41586-020-2824-5}
}
```

**APA:**

Neftaliem, L., Anderson, C., Igel, C., Field, C. B., Jackson, R. B., Small, J.,
& Tucker, C. J. (2026). *Individual tree crown mapping across four North
American cities from high-resolution satellite imagery* [Manuscript submitted
for publication].

## Glossary

- **NDVI** — Normalised Difference Vegetation Index, a red/near-infrared ratio
  that highlights live vegetation.
- **Panchromatic** — a single high-resolution broadband intensity channel.
- **Crown** — the above-ground canopy extent of an individual tree.
- **Tversky loss** — a generalisation of the Dice loss with tunable
  false-positive/false-negative weighting, useful for class-imbalanced
  segmentation.
- **Ripley's K** — a spatial statistic used here to characterise clustering of
  detected trees and the sensitivity of results to the deduplication threshold.

## More Information

See the repository [README](README.md) for the full pipeline, the list of inputs
that cannot be redistributed (licensed imagery, weights, census/inventory data),
and the `reproduce.sh` workflow for regenerating figures from the committed data.

## Model Card Authors

Drafted from the repository and manuscript metadata; to be reviewed and completed
by L. Neftaliem and co-authors.

## Model Card Contact

Corresponding author of the manuscript (see the repository README / paper).
