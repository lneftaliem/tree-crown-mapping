# Ripley's L analysis (Fig. S6, Supplementary Note 5)

**The Ripley's L analysis was designed and written by Christian Igel** (University of Copenhagen), who ran it for the original submission. `process_city_trees.py` is his code, included **unchanged** from his repository [christian-igel/city-trees](https://github.com/christian-igel/city-trees) (commit `3c57d4c`) with his permission. `analyze_notebook.py` reproduces cells 2 and 4 of his notebook `City Trees Analysis.ipynb` from the same repository with the same logic. Please credit him when reusing this analysis.

## Method (Christian Igel's code)

- **Trees:** trees are assigned to each polygon of a census area (block group or Surrey dissemination area) by spatial join (`within`).
- **Statistic:** for each polygon with at least 50 trees, L̂(r) − r at r = 2, 4, …, 50 m:
  - K̂(r) = A · #{(i, j): i ≠ j, d_ij < r} / n²
  - A is the area of the convex hull of the polygon's trees.
  - Polygons with more than 100,000 trees are subsampled to 100,000; none in the final data are.
- **Multi-polygon areas:** L̂ is averaged over the polygons, weighted by the convex-hull area of their trees (`l += area * L`, `l /= area_auc`).
- **AUC:** area under L̂(r) − r over 2–50 m (`sklearn.metrics.auc`).
- **Census areas analysed:** income available and at least 50 trees in the analysed polygons.
- **Test:** Spearman correlation of AUC with median household income.
- **Quartile curves:** within-city income quartiles; L̂ averaged per quartile, weighted by the analysed convex-hull area.
- **No edge correction.**
- **Projection:** UTM zone of each city (EPSG 32614 Austin, 32616 Bloomington, 32610 Cupertino and Surrey). `process_city_trees_utm.py` is his script with only its two `to_crs(epsg=3857)` calls replaced; his original uses Web Mercator.
- **Area:**
  - His code computes the convex hull of each polygon's trees and passes it to `pointpats.l_test(..., hull=...)`.
  - pointpats ignores `hull=` for the observed statistic and uses the bounding box of the points instead.
  - As agreed with him, `run_ripley.py` replaces `pointpats.l_test` with `l_test_hull`, the same statistic with A = the hull his code passes in.
  - `RIPLEY_AREA=bbox` gives the pointpats (bounding-box) statistic.

## Files

| File | Author | Purpose |
|---|---|---|
| `process_city_trees.py` | C. Igel, unchanged | Per-census-area L̂(r) and AUC (EPSG:3857) |
| `process_city_trees_utm.py` | C. Igel; projection lines changed | Same in the UTM zone of each city (revised Fig. S6) |
| `analyze_notebook.py` | logic from C. Igel's notebook | Filter, Spearman test, quartile curves, 2 × 2 figure |
| `run_ripley.py` | this revision | Runs his script with `l_test_hull` (convex-hull area; KD-tree pair counts, no distance matrix) |
| `tests_ripley_hull.py` | this revision | Validation tests (82 checks) |
| `check_exact.py` | this revision | Checks the bounding-box variant against pointpats on real census areas |
| `hull_geometry_check.py` | this revision | Checks that hull areas larger than a census polygon come from concave polygons, not geometry errors |
| `figS6_step.py` | this revision | The `figS6` step of `run_analysis.py` |

### What `tests_ripley_hull.py` checks

- **Hull area:** shapely vs scipy, and analytic squares and triangles.
- **Pair counting:** vs brute force, including ties at d = r and duplicate points.
- **Normalisation and L transformation.**
- **Bounding-box variant:** equal to `pointpats.l_test` (difference ≤ 1.4 × 10⁻¹⁴ on synthetic patterns and real census polygons).
- **Convex-hull variant:** equal to the pointpats statistic rescaled to the hull area.
- **Complete spatial randomness:** matches the analytic expectation without edge correction.
- **His `process_city()`, unchanged, on synthetic census areas:**
  - the MultiPolygon weighting;
  - polygons with fewer than 50 or exactly 50 trees;
  - degenerate hulls, which are skipped;
  - census areas with no analysed polygon, which give NaN and are filtered out;
  - trees on a polygon boundary, which are dropped;
  - subsampling.

## Run

From the repository root, after reassembling the tree points (see `REPRODUCING.md`):

```bash
python revision/analysis/run_analysis.py --dataset <tree points folder> --label final --out <run folder> --only figS6
```

Or directly, on a folder holding `census_boundaries/<city>_exact_block_groups.geojson` and `<city>_ripley_points_with_income.geojson`:

```bash
python revision/analysis/ripley_igel/run_ripley.py <folder> austin bloomington cupertino surrey
FIG_DPI=300 python revision/analysis/ripley_igel/analyze_notebook.py <folder> final
python revision/analysis/ripley_igel/tests_ripley_hull.py <tree points folder>
```

## Results

Results are in `results/final_v1b/ripley_igel/`. The previous revision's UTM bounding-box results are in `utm_bbox_previous/`.

**Reproduction:** on the original (published) tree data, `RIPLEY_PROJ=3857 RIPLEY_AREA=bbox` reproduces the published Fig. S6 and the original notebook output exactly:
- census areas: 601 / 56 / 28 / 563;
- ρ = 0.037, 0.024, 0.071, 0.028.

**Final data, UTM, convex hull (revised Fig. S6):**

| City | Census areas | ρ (AUC vs income) | p |
|---|---|---|---|
| Austin | 606 | 0.19 | 2.3 × 10⁻⁶ |
| Bloomington | 58 | 0.19 | 0.15 |
| Cupertino | 28 | 0.015 | 0.94 |
| Surrey | 632 | 0.085 | 0.033 |

- **Sign of L̂(r) − r:** negative at 2–4 m (up to 6 m in Cupertino); positive from about 6 m (8 m in Cupertino). See `comparison_sign_change.csv`.
- **Other variants:** `variants_and_reproduction.csv` and `comparison_published_bbox_hull.csv` list the published, Web Mercator and bounding-box variants.
- **Geometry check:** `hull_geometry_check.csv` covers the 1,466 analysed polygons:
  - every tree lies within its polygon;
  - every tree hull lies within the polygon's convex hull;
  - UTM areas agree with geodesic areas to within 0.08 %;
  - hulls exceed the polygon area only for concave polygons.
