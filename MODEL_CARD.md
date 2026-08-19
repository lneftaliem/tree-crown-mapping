# Model Card: Individual Tree Crown Segmentation U-Net (4-City North American Fine-Tune)

A fully convolutional **U-Net** that segments individual tree crowns from 50 cm
multispectral satellite imagery. It is a fine-tune of the dryland-vegetation
U-Net of Brandt et al. (2020) / Tucker et al. (2023), adapted to delineate urban
and peri-urban tree crowns in four North American cities.

> This card follows the Hugging Face
> [annotated model card template](https://huggingface.co/docs/hub/model-card-annotated).
> Values are taken from the accompanying manuscript (Neftaliem et al., submitted
> to *Nature Communications Sustainability*). A few fields the manuscript does not
> report (e.g. training wall-clock time and carbon) remain marked
> **[Not reported]**.

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
crown. A probability threshold of 0.3 produces a binary canopy mask, and
connected-component labelling treats each spatially contiguous canopy object as
an individual crown, from which centroids and crown areas are derived. The model
was fine-tuned on 113 manually annotated 256×256 images (4,589 delineated crowns,
≈0.74 km², ~0.03% of the mapped area) and applied to map ≈3.29 million trees
across Austin (TX), Bloomington (IN), Cupertino (CA), and Surrey (BC)
**without per-city retraining**.

- **Developed by:** Leona Neftaliem, Connor J. Anderson, Christian Igel, Christopher B. Field, Robert B. Jackson, Jennifer Small, Compton J. Tucker (Stanford University; NASA Goddard Space Flight Center / SSAI; University of Copenhagen; University of Maryland Baltimore County)
- **Funded by:** U.S. National Science Foundation Graduate Research Fellowship Program, Grant No. DGE-2146755 (to L. Neftaliem). Computation on the Stanford Sherlock cluster (Stanford Research Computing).
- **Shared by:** [lneftaliem](https://github.com/lneftaliem)
- **Model type:** Semantic segmentation (fully convolutional U-Net, Ronneberger et al. 2015) for individual tree-crown delineation from remote-sensing imagery
- **Language(s):** Not applicable (geospatial imagery model)
- **License:** MIT (free to use, modify, and redistribute with attribution; see [`LICENSE`](LICENSE)). This covers the fine-tuning / analysis code and the derived per-tree datasets in this repository, and matches the MIT-licensed upstream framework (Kariryaa et al.; Brandt et al., 2020). The fine-tuned weights are **available on request** from the corresponding author and are intended for free use under the same terms; the pretrained base weights originate from Brandt et al. (2020) (via ORNL DAAC), so retain their attribution when redistributing weight derivatives.
- **Finetuned from model:** Sahel dryland-vegetation U-Net of [Brandt et al., 2020](https://doi.org/10.1038/s41586-020-2824-5) / [Tucker et al., 2023](https://doi.org/10.1126/science.abg1740), pretrained on 89,899 manually annotated trees across an environmental gradient in the African Sahel, built on the [ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel) framework (MIT). Pretrained weights obtained from the Brandt et al. (2020) authors (available via ORNL DAAC).

### Model Sources

- **Repository:** https://github.com/lneftaliem/tree-crown-mapping
- **Paper:** Neftaliem, L., Anderson, C. J., Igel, C., Field, C. B., Jackson, R. B., Small, J., & Tucker, C. J. "Mapping tree crowns across four North American cities." Manuscript submitted to *Nature Communications Sustainability*. [DOI to be added]
- **Demo:** N/A
- **Weights file:** `trees_20260120-1928_AdaDelta_weightmap_tversky_012_256_final.keras` (~376 MB; not hosted in the repo — available on request). Note: the manuscript reports the fine-tune used the Adam optimizer; the "AdaDelta" token in the filename is legacy naming inherited from the base framework.

## Uses

### Direct Use

Delineating individual tree crowns and estimating per-tree crown area from
sub-meter (≈50 cm) NDVI + panchromatic imagery over the four study cities and
comparable temperate/subtropical North American urban and peri-urban settings.
The four cities span humid subtropical (Austin), humid continental (Bloomington),
Mediterranean (Cupertino), and temperate oceanic (Surrey) climates.

### Downstream Use

- Per-tree outputs support estimates of tree density, canopy cover, and
  crown-size distributions at the census-unit level, including comparisons to
  socioeconomic variables such as median household income.
- Urban-forestry inventories and canopy-equity analyses, especially where
  private-land and unmanaged-greenspace trees are invisible to municipal
  street/park inventories.
- A transfer-learning starting point for tree-crown segmentation in new cities
  with comparable sub-meter multispectral imagery.

### Out-of-Scope Use

- Imagery whose resolution, spectral bands, sensor, or landscape differ
  substantially from the ≈50 cm NDVI+panchromatic training imagery. At 4 m
  (e.g. PlanetScope) small crowns span fewer than two pixels and detection is
  unreliable (Supplementary Note 6).
- **Species identification, tree health, height, biomass, or carbon estimation** —
  the model outputs crown extent only.
- Legal, regulatory, property-boundary, or tax determinations about individual
  parcels or trees.
- Presenting per-tree detections as a ground-truth census; outputs carry non-zero
  omission and commission error (see below).

## Bias, Risks, and Limitations

- **Crown crowding / merging:** connected-component labelling can merge
  physically touching crowns into one detection, causing modest undercounting in
  dense-canopy areas (Methods; Fig. 6).
- **Boundary precision:** instance-segmentation overlap is moderate (Dice = 0.55,
  IoU = 0.379); crown geometry is approximate, and precision (0.495) reflects the
  difficulty of exact delineation in heterogeneous urban scenes.
- **Recall-weighted operating point:** the Tversky loss (α = 0.11, β = 0.89) and
  0.3 threshold favour recall (0.618), so the model tends to over-predict at crown
  peripheries and where vegetation is spectrally similar to background.
- **Geographic/sensor domain:** trained and evaluated on four cities and four
  commercial sensors (QuickBird-2, GeoEye-1, WorldView-2/3); accuracy elsewhere is
  unverified, though cross-city generalisation without retraining is demonstrated.
- **Imagery conditions:** phenology, sun angle, shadow, and off-nadir geometry
  affect NDVI and detection; imagery here was screened to May–November growing
  season.
- **Equity analyses inherit model error:** omission/commission error may correlate
  with neighbourhood structure (lot size, planting patterns), so care is needed
  not to read model error as a social signal.

### Recommendations

Report and propagate detection uncertainty into downstream density/canopy/equity
estimates; validate against an independent local reference (e.g. a municipal
inventory or manual annotation) before use in a new city; treat crown-area
statistics as relative rather than absolute; and prefer aggregate (census-unit)
conclusions over per-tree claims.

## How to Get Started with the Model

The model is applied through `3-RasterAnalysis.py` in the repository, which tiles
the input rasters into 256×256 patches, normalises each channel, runs inference
with 50% overlap, and merges patch predictions into per-tile canopy probability
rasters. In outline:

```bash
# 1. Obtain the base framework (core/ + config/) and the fine-tuned weights
#    (see the repository README, "What you'll need to add").
export TREE_MAPPING_BASE_DIR=/path/to/your/data   # weights at saved_models/UNet/

# 2. Provide co-registered NDVI + panchromatic tiles under cutouts/, then:
python 3-RasterAnalysis.py            # -> per-pixel canopy probability rasters
python prepare_ripley_data.py         # -> per-tree points (centroid + crown area)
```

To reproduce the **published figures** from the already-processed per-tree
datasets committed in the repo — no imagery or weights required — run
`./reproduce.sh` (see the README).

## Training Details

### Training Data

113 manually annotated 256×256-pixel images containing 4,589 delineated tree
crowns across the four study cities (≈0.74 km² total; ~0.03% of the mapped area),
drawn from the same ≈50 cm pan-sharpened multispectral imagery used at inference.
Training tiles were chosen with an **active-learning** approach that iteratively
targeted regions where model predictions were weakest. The 113 images were
partitioned approximately **80/10/10** into training / validation / test sets.

### Training Procedure

#### Preprocessing

- Two input channels per patch: **NDVI** (from pan-sharpened red + NIR) and
  **panchromatic**; each channel normalised independently per patch.
- Imagery orthorectified to a common reference and pan-sharpened to 0.5 m; tiled
  into **256 × 256** patches.
- Data augmentation followed Brandt et al. (2020) (via `imgaug`), with added
  perspective transforms (10%) and linear contrast adjustment over a 0.3–1.2
  factor range.

#### Training Hyperparameters

- **Architecture:** U-Net encoder–decoder with skip connections (Ronneberger et al., 2015), weights initialised from Brandt et al. (2020).
- **Loss:** Tversky loss with **α = 0.11, β = 0.89** (penalising false negatives more heavily, given ~6% tree-pixel class imbalance); ε = 1×10⁻⁷; pixel-wise (weightmap) weighting.
- **Optimizer:** **Adam**, learning rate **5.0 × 10⁻⁵**.
- **Epochs:** **100**.
- **Patch size:** 256 × 256; 2-channel input.
- **Inference:** probability threshold **0.3** (chosen on the validation set to maximise F1 while favouring recall); patches predicted with 50% overlap and merged; individual crowns via connected-component analysis (no minimum-size threshold).
- **Framework:** TensorFlow / Keras.

#### Speeds, Sizes, Times

- **Trained weights file size:** ~376 MB (`.keras`).
- **Compute:** Stanford Sherlock HPC cluster.
- **Training wall-clock time / throughput:** [Not reported]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Held-out test patches drawn from all four cities (the ~10% test split), evaluated
over 7,471,104 validation pixels (Supplementary Note 2; Fig. S1). Independent
external comparison uses municipal street/park tree inventories from each city's
open-data portal.

#### Factors

Performance is reported city-agnostically at the pixel level; qualitative results
(Figs. S2–S3) show it varies with canopy density and crown clarity — high on
isolated, well-defined crowns (Dice 0.80–0.82) and lower in dense or spectrally
ambiguous areas (Dice as low as 0.32).

#### Metrics

Pixel-level accuracy, precision, recall, specificity, F1, Dice coefficient, and
IoU (Jaccard) — complementary overlap and classification measures appropriate
under strong class imbalance. Object-level agreement is assessed via Spearman
correlation between model-detected and municipally inventoried counts per census
unit.

### Results

Pixel-level segmentation performance on held-out data (Fig. S1):

| Metric | Value |
|---|---|
| Accuracy | 93.9% |
| Precision | 0.495 |
| Recall (sensitivity) | 0.618 |
| Specificity | 0.960 |
| F1-score | ≈0.55 (equals Dice for binary segmentation) |
| Dice coefficient | 0.550 |
| IoU (Jaccard) | 0.379 |

Confusion (normalised): 95.2% true-negative rate (non-tree), 65.7% true-positive
rate (tree). Object-level correspondence with municipal inventories was
non-significant in Austin (ρ = 0.04) and significant in Bloomington (ρ = 0.38),
Cupertino (ρ = 0.67), and Surrey (ρ = 0.39); model counts exceeded inventory
counts in every city (2.5×–36×), reflecting detection of trees on private land
and in unmanaged green space.

#### Summary

From a single set of fine-tuned weights (no per-city retraining), the model maps
≈3.29 million trees across the four cities — Austin ≈1,900,000; Surrey ≈1,220,000;
Bloomington ≈117,000; Cupertino ≈52,700 — at 93.9% pixel accuracy, with moderate
instance-segmentation overlap (Dice 0.55, IoU 0.379) reflecting the difficulty of
exact crown-boundary delineation.

## Environmental Impact

Carbon emissions can be estimated with the
[ML CO2 Impact calculator](https://mlco2.github.io/impact#compute).

- **Hardware Type:** Stanford Sherlock HPC cluster (GPU nodes) — specific GPU model [Not reported]
- **Hours used:** [Not reported]
- **Cloud Provider:** N/A (on-premise university HPC)
- **Compute Region:** Stanford, California, USA
- **Carbon Emitted:** [Not reported]

## Technical Specifications

### Model Architecture and Objective

U-Net fully convolutional encoder–decoder with skip connections. Input: a
2-channel (NDVI + panchromatic) 256×256 patch. Output: a single-channel per-pixel
tree-crown probability map, optimised with a Tversky loss (α = 0.11, β = 0.89).
Individual crowns are obtained by thresholding at 0.3 and connected-component
labelling; crown area is the labelled region area in projected (UTM) units. A
KD-Tree deduplication step removes cross-tile double counts within 1.0 m
(sensitivity-tested at 0.8/1.0/1.5 m).

### Compute Infrastructure

#### Hardware

Stanford Sherlock HPC cluster (GPU) for fine-tuning; the repository's inference
and post-processing scripts run on CPU/GPU workstations. Reassembling and reading
the full per-city GeoJSONs needs several GB of RAM/disk.

#### Software

- Python 3.10
- TensorFlow / Keras (`>=2.10,<2.16`) for model training and inference
- `imgaug` for data augmentation
- Geospatial stack: geopandas, shapely, rasterio, fiona, rtree, affine (GDAL)
- Scientific stack: numpy, pandas, scipy, scikit-image, statsmodels
- See `requirements.txt` for the full list.

## Citation

**BibTeX:**

```bibtex
@article{neftaliem_tree_crown_mapping,
  title   = {Mapping tree crowns across four North American cities},
  author  = {Neftaliem, Leona and Anderson, Connor J. and Igel, Christian and Field, Christopher B. and Jackson, Robert B. and Small, Jennifer and Tucker, Compton J.},
  year    = {2026},
  note    = {Manuscript submitted to Nature Communications Sustainability}
}

@article{brandt2020unexpectedly,
  title   = {An unexpectedly large count of trees in the West African Sahara and Sahel},
  author  = {Brandt, Martin and Tucker, Compton J. and Kariryaa, Ankit and Rasmussen, Kjeld and Abel, Christin and Small, Jennifer and others},
  journal = {Nature},
  volume  = {587},
  number  = {7832},
  pages   = {78--82},
  year    = {2020},
  doi     = {10.1038/s41586-020-2824-5}
}

@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  pages     = {234--241},
  year      = {2015}
}
```

**APA:**

Neftaliem, L., Anderson, C. J., Igel, C., Field, C. B., Jackson, R. B., Small, J.,
& Tucker, C. J. (2026). *Mapping tree crowns across four North American cities*
[Manuscript submitted for publication, Nature Communications Sustainability].

## Glossary

- **NDVI** — Normalised Difference Vegetation Index, a red/near-infrared ratio
  that highlights live vegetation.
- **Panchromatic** — a single high-resolution broadband intensity channel; here
  it also carries the cast-shadow cue used to define a tree.
- **Crown** — the above-ground canopy extent of an individual tree.
- **Tversky loss** — a generalisation of the Dice loss with tunable
  false-positive/false-negative weighting (here α = 0.11, β = 0.89), useful for
  class-imbalanced segmentation.
- **Dice / IoU** — overlap metrics between predicted and reference masks; IoU is
  the stricter (Jaccard) measure.
- **Ripley's L** — a density-independent second-order spatial statistic used to
  characterise clustering of detected trees.

## More Information

See the repository [README](README.md) for the full pipeline, the inputs that
cannot be redistributed (licensed Vantor imagery, weights, census/inventory data),
and the `reproduce.sh` workflow for regenerating figures from the committed data.

## Model Card Authors

Drafted from the manuscript and repository metadata; reviewed by L. Neftaliem and
co-authors.

## Model Card Contact

Leona Neftaliem — leonan@stanford.edu (corresponding author).
