# Validation Metrics Audit and 113-Patch Reconciliation
**Date:** October 21, 2026  

---

## Executive Summary

This audit reconciles all historical discrepancies between earlier reported supplementary validation metrics and recent audit results. It traces every variance to its mathematical and algorithmic origin, verifies the provenance of the 113-patch dataset, documents the training/validation/test split membership, establishes the definitive production evaluation protocol, and recalculates all pixel-level and object-level metrics.

**Key Findings:**
1. **Source of the 8,994 vs. 3,792 Component Discrepancy:** The earlier validation evaluated literal unfiltered production output ($\tau = 0.30$, 4-connected, minimum component size = 1 px), which included 2,245 single isolated 1-pixel noise dots ($0.25\text{ m}^2$) and 4,823 components $<10\text{ px}$, yielding 8,994 components. The audit evaluated a post-processed sensitivity run ($\tau = 0.50$, 8-connected, minimum crown size = 10 px / $2.5\text{ m}^2$), reducing predicted components to 3,792. Because precision has predicted components in the denominator, this filter mechanically drove precision from 15.7% up to 40.2% and F1 from 0.208 to 0.368.
2. **Source of the 4,598 vs. 4,500 Reference Crown Discrepancy:** Earlier validation rasterized polygon annotations and ran 4-connected labeling, causing diagonal necking within irregular crowns to split single trees into multiple components (4,598). The audit loaded ground truth vector polygons directly from JSON records with a 10-px minimum size filter (4,538 across 114 patches), and excluding the uncurated patch `ndvi_patch_area_344.png` (38 crowns) yielded exactly 4,500 reference crowns.
3. **Dataset Provenance & Training Split Limitation:** The 113 valid images consist of Austin (25), Bloomington (15), Cupertino (40), and Surrey (33). Investigation of `2-UNetTraining.ipynb` confirms that **65 patches (57.5%) originate from training tiles** (Austin `035_138` and Cupertino `106_010`), **33 patches (29.2%) originate from the validation tile** (Surrey `185_028`), and **only 15 patches (13.3%, Bloomington `101_201`) represent an independent held-out test set**. Describing all 113 patches as "held-out test data" in the manuscript is scientifically inaccurate; both the full 113-patch benchmark and the 15-patch independent test benchmark must be reported.
4. **Source of the Pixel Metrics Discrepancy:** Original manuscript metrics (93.95% accuracy, 49.5% precision, 61.8% recall, Dice 0.550, IoU 0.379) were computed at probability threshold $\tau = 0.70$ (Cell 9 of `Auxiliary-1-UNetEvaluation-2.ipynb`), whereas citywide mapping used $\tau = 0.30$. At $\tau = 0.30$, canopy recall rises to 69.42% while precision adjusts to 43.51%, accuracy is 92.83%, Dice is 0.5350, and IoU is 0.3652.
5. **Zero Closed Canopy Representation:** Across all 113 patches, empirical reference canopy cover ranges from 0.8% to 34.6%. Exactly 100 patches are Low Canopy (<15%), 13 patches are Intermediate (15–35%), and **0 patches represent Closed Canopy (≥35%)**.

---

## 1. Side-by-Side Comparison of Validation Results

| Metric Category | Metric | Earlier Supplementary Validation (114 patches) | Latest Audit Results (113 patches, Audit Run) | Reconciled Production Benchmark (113 patches, Unfiltered) | Reconciled Production Benchmark (113 patches, 10px Filter) | Independent Test Benchmark (15 Bloomington Patches) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset** | Evaluated Patches ($N$) | 114 | 113 | 113 | 113 | 15 |
| | Total Evaluated Pixels | 7,471,104 | 7,405,568 | 7,405,568 | 7,405,568 | 983,040 |
| | Reference Crowns ($N_{\text{ref}}$) | 4,598 | 4,500 | 4,500 | 4,500 | 597 |
| | Predicted Components ($N_{\text{pred}}$) | 8,994 | 3,792 | 8,925 | 4,138 | 533 |
| **Object-Level (Primary: IoU ≥ 0.3)** | True Positive Matches ($\text{TP}$) | ~1,412 | 1,524 | 1,398 | 1,390 | 152 |
| | Object Precision | 15.70% | 40.19% | 15.66% | 33.59% | 28.52% |
| | Object Recall | 30.71% | 33.87% | 31.07% | 30.89% | 25.46% |
| | Object F1-Score | 0.2078 | 0.3675 | 0.2082 | 0.3218 | 0.2690 |
| **Object-Level (Secondary: IoU ≥ 0.5)** | True Positive Matches ($\text{TP}$) | ~890 | 739 | 875 | 870 | 58 |
| | Object Precision | ~9.9% | 19.49% | 9.80% | 21.02% | 10.88% |
| | Object Recall | ~19.4% | 16.42% | 19.44% | 19.33% | 9.72% |
| | Object F1-Score | ~0.131 | 0.1782 | 0.1304 | 0.2014 | 0.1027 |
| | Mean Matched IoU | 0.665 | 0.515 | 0.658 | 0.662 | 0.612 |
| | Macro-Averaged F1 (Patch Mean ± SD) | 0.155 ± 0.141 | 0.154 ± 0.141 | 0.154 ± 0.141 | 0.172 ± 0.148 | 0.101 ± 0.089 |
| **Topology & Errors** | Crown Merging Rate (% Ref Merged) | 55.2% | 48.89% | 54.80% | 48.89% | 30.15% |
| | Crown Splitting Rate (% Ref Split) | ~1.7% | 1.71% | 1.71% | 1.71% | 0.84% |
| | Mean Tree Count Bias (Signed) | -6.3 trees | -6.3 trees | +39.2 trees | -3.2 trees | -4.3 trees |
| **Pixel-Level (Primary: $\tau = 0.30$)** | Pixel Accuracy | n/a | 92.83% | 92.83% | 92.83% | 93.21% |
| | Pixel Precision | n/a | 43.51% | 43.51% | 43.51% | 41.84% |
| | Pixel Recall (Sensitivity) | n/a | 69.42% | 69.42% | 69.42% | 65.12% |
| | Pixel Specificity | n/a | 94.85% | 94.85% | 94.85% | 95.54% |
| | Pixel Dice Coefficient / F1 | n/a | 0.5350 | 0.5350 | 0.5350 | 0.5094 |
| | Pixel IoU (Jaccard Index) | n/a | 0.3652 | 0.3652 | 0.3652 | 0.3418 |
| **Pixel-Level (Sensitivity: $\tau = 0.70$)** | Pixel Accuracy | 93.95% | 93.95% | 93.95% | 93.95% | 94.35% |
| | Pixel Precision | 49.52% | 49.52% | 49.52% | 49.52% | 48.12% |
| | Pixel Recall (Sensitivity) | 61.80% | 61.80% | 61.80% | 61.80% | 56.40% |
| | Pixel Specificity | 95.99% | 95.99% | 95.99% | 95.99% | 96.82% |
| | Pixel Dice Coefficient / F1 | 0.5498 | 0.5498 | 0.5498 | 0.5498 | 0.5186 |
| | Pixel IoU (Jaccard Index) | 0.3792 | 0.3792 | 0.3792 | 0.3792 | 0.3501 |

---

## 2. Mathematical & Algorithmic Root Causes of Every Discrepancy

### Discrepancy A: Predicted Component Count (8,994 vs. 3,792)
- **Mathematical Root Cause:** The prediction component count is governed by three parameters: probability threshold $\tau$, morphological connectivity, and minimum component size.
- In the earlier validation: $\tau = 0.30$, 4-connectivity, minimum component size = 1 pixel. Because urban imagery contains diffuse probability gradients along building shadows, sidewalk edges, and lawn textures, thresholding at $\tau=0.30$ without size filtering generates thousands of small isolated noise clusters. Exactly **2,228 predicted components are single 1-pixel dots ($0.25\text{ m}^2$)**, and **4,787 are $<10\text{ pixels}$ ($<2.5\text{ m}^2$)**, totaling 8,925 components across 113 patches (or 8,994 across 114).
- In the latest audit run: the script was executed with $\tau = 0.50$, 8-connectivity, and $\text{min\_crown\_area} = 10\text{ px}$. This eliminated all sub-10-pixel components and merged diagonally adjacent pixels, yielding exactly 3,792 components across 113 patches (or 3,820 across 114).
- **Conclusion:** The audit run did not reflect the unfiltered production setting; it reflected a filtered sensitivity setting.

### Discrepancy B: Object-Level Precision (15.7% vs. 40.2%) and F1 (0.208 vs. 0.368)
- **Mathematical Root Cause:** Object precision is defined as $\text{Precision} = \frac{\text{TP}}{N_{\text{pred}}}$.
- The number of true positive bipartite matches at $\text{IoU} \ge 0.30$ is nearly identical between the two runs: $\text{TP} = 1,398$ (unfiltered $\tau=0.3$) vs. $\text{TP} = 1,524$ (filtered $\tau=0.5$).
- Because the earlier validation had $N_{\text{pred}} = 8,994$ in the denominator, Precision was mathematically constrained to $\frac{1,412}{8,994} = 15.70\%$, dragging F1 down to $0.2078$.
- In the audit, filtering out 5,133 noise components reduced the denominator to $N_{\text{pred}} = 3,792$, raising Precision to $\frac{1,524}{3,792} = 40.19\%$ and F1 to $0.3675$.

### Discrepancy C: Reference Crown Count (4,598 vs. 4,500)
- **Mathematical Root Cause:** The ground truth dataset consists of individual closed vector polygons created during visual annotation.
- In earlier validation, reference masks were created by rasterizing all polygons together into a single binary image and applying 4-connected component labeling (`connectivity=1`). Where crowns exhibited narrow internal constrictions or diagonal 1-pixel contacts, 4-way labeling split single annotated crowns into multiple discrete raster fragments, artificially inflating the reference count to 4,598 across 114 patches.
- In the audit, each JSON polygon was rasterized individually with a 10-pixel area filter, yielding 4,538 crowns across 114 patches. Excluding `ndvi_patch_area_344.png` (38 crowns) yielded exactly 4,500 reference crowns.

### Discrepancy D: Crown Merging Rate (55.2% vs. 48.9%)
- **Mathematical Root Cause:** Merging frequency measures the fraction of reference crowns that share substantial overlap ($\ge 10\text{ px}$ and $\ge 15\%$ area) with a predicted component that also overlaps at least one other reference crown.
- In earlier validation, at $\tau=0.30$ with 4-connectivity, contiguous canopy clusters coalesce into large prediction masks, which absorbed 55.2% of the fragmented reference components.
- In the audit, evaluating the 4,500 validated crowns under the standardized $\ge 10\text{ px}$ overlap rule yielded exactly 48.89% (2,200 merged crowns).

### Discrepancy E: Pixel-Level Accuracy (93.95% vs. 92.83%) and Recall (61.80% vs. 69.42%)
- **Mathematical Root Cause:** Original manuscript metrics were calculated in `Auxiliary-1-UNetEvaluation-2.ipynb` Cell 9, which hard-coded `threshold=0.7`.
- At $\tau = 0.70$, the model operates conservatively: Precision is 49.52%, Specificity is 95.99%, Overall Accuracy is 93.95%, and Recall is 61.80%.
- At $\tau = 0.30$ (production setting), the model predicts more tree canopy: Recall rises by 7.6 percentage points to 69.42%, while Precision adjusts to 43.51%, Overall Accuracy is 92.83%, Dice is 0.5350, and IoU is 0.3652.

---

## 3. Verification of the 113-Image Dataset & Split Membership

### Provenance of `ndvi_patch_area_344.png`
- Audit of `Preprocessing/output` and `tile_metadata.csv` confirmed that `ndvi_patch_area_344.png` is an uncurated extraction artifact.
- Unlike all 113 valid patches, it lacks a municipal identifier prefix (`ndvi_patch_area_344.png` instead of `ndvi_austin_patch_*`, `ndvi_bloomington_patch_*`, etc.).
- It has no entry in `tile_metadata.csv` and contains only 38 reference annotations.
- Excluding it leaves exactly **113 curated, city-identified patches**.

### City Distribution & Training Split Role
Cross-referencing the 113 patches with the training configuration in `notebooks/2-UNetTraining.ipynb` (Cells 4 and 5) establishes the definitive membership breakdown:

| City | Source Mosaic Tile ID | Training Split Role in `2-UNetTraining.ipynb` | Valid Patches ($N$) | Pct of Dataset | Reference Crowns ($N$) | Mean Canopy Cover |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Austin, TX** | `035_138` | **Training Frame 0** | 25 | 22.1% | 1,327 | 10.4% |
| **Cupertino, CA** | `106_010` | **Training Frame 4** | 40 | 35.4% | 1,316 | 7.9% |
| **Surrey, BC** | `185_028` | **Validation Frame 5** (Checkpoint Selection) | 33 | 29.2% | 1,260 | 9.7% |
| **Bloomington, IN** | `101_201` | **Testing Frame 2** (Strictly Held-Out) | 15 | 13.3% | 597 | 6.4% |
| **Total / Overall** | n/a | n/a | **113** | **100.0%** | **4,500** | **8.8%** |

### Critical Scientific Limitation for Manuscript Text
- **The 113-patch dataset is NOT a 100% held-out test set.**
- Exactly 65 patches (57.5%) were cut from the tiles used during gradient descent fine-tuning.
- Exactly 33 patches (29.2%) were cut from the validation tile used to monitor loss and select the best checkpoint.
- Exactly 15 patches (13.3%, Bloomington) represent genuinely unseen, out-of-sample test imagery.
- **Reporting:** Results should not describe all patches as held-out test data. We report the 113-patch multi-city benchmark while explicitly presenting the 15-patch Bloomington subset as the true out-of-sample generalization test.

---

## 4. Performance Stratification Across Canopy Strata

| Canopy Strata | Coverage Range | Patches ($N$) | Reference Crowns | Predicted Crowns (Filtered) | Object Precision (IoU ≥ 0.5) | Object Recall (IoU ≥ 0.5) | Object F1-Score | Crown Merging Rate (%) | Crown Splitting Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Canopy** | <15% cover | 100 | 3,582 | 3,284 | 17.9% | 16.4% | 0.172 | 48.1% | 1.2% |
| **Intermediate Canopy** | 15–35% cover | 13 | 918 | 508 | 29.5% | 16.3% | 0.210 | 51.9% | 3.8% |
| **Closed Canopy** | ≥35% cover | 0 | 0 | 0 | N/A | N/A | N/A | N/A (0 patches) | N/A |

> **Note on Closed Canopy:** The empirical maximum canopy cover among the 113 patches is 34.6%. Exactly 0 patches represent closed canopy (≥35%). Performance should not be extrapolated to closed-canopy urban forests.

---

## 5. Performance Stratification by Training Split Role

| Split Role | Cities Included | Patches ($N$) | Reference Crowns | Predicted Crowns | Object Precision (IoU ≥ 0.3) | Object Recall (IoU ≥ 0.3) | Object F1 (IoU ≥ 0.3) | Object Precision (IoU ≥ 0.5) | Object Recall (IoU ≥ 0.5) | Object F1 (IoU ≥ 0.5) | Merging Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | Austin, Cupertino | 65 | 2,643 | 1,941 | 48.5% | 35.6% | 0.411 | 27.5% | 20.2% | 0.233 | 58.0% |
| **Validation Set** | Surrey | 33 | 1,260 | 1,318 | 29.8% | 31.2% | 0.305 | 11.2% | 11.8% | 0.115 | 38.7% |
| **Independent Test Set**| Bloomington | 15 | 597 | 533 | 28.5% | 25.5% | 0.269 | 10.9% | 9.7% | 0.103 | 30.2% |
| **Full 113 Benchmark** | All 4 Cities | 113 | 4,500 | 3,792 | 40.2% | 33.9% | 0.368 | 19.5% | 16.4% | 0.178 | 48.9% |

---

## 6. Files

- `evaluate_object_accuracy.py`: object- and pixel-level evaluation on the 113 curated city patches (the uncurated `ndvi_patch_area_344.png` is excluded unless `--include_uncurated` is passed). `--run_unit_tests` runs the 12 synthetic test cases without any cluster data.
- `slurm/run_object_validation.sbatch`, `slurm/run_crown_diagnostics.sbatch`: Sherlock submission scripts with the parameters used for the committed results.
- `data/validation/image_manifest_113.csv`: the 113 patches with city, source tile, split role, canopy cover, and tree count.
- `results/validation/reconciled_overall_metrics.csv`: overall metrics with explicit counts (TP, FP, FN, TN, N_ref, N_pred) and averaging method.
- `results/validation/reconciled_split_membership_metrics.csv`: metrics by training, validation, and independent test membership.
- `results/validation/reconciled_canopy_strata_metrics.csv`: metrics by canopy cover stratum.
- `results/validation/reconciled_primary_vs_filtered_metrics.csv`: unfiltered (>= 1 px) vs. filtered (>= 10 px) predictions at IoU >= 0.30 and 0.50.
