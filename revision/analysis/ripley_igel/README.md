# Ripley's L analysis (Fig. S6, Supplementary Note 5)

**The Ripley's L analysis was designed and written by Christian Igel** (University of Copenhagen), who ran it for the original submission. `process_city_trees.py` is his code, included **unchanged** from his repository [christian-igel/city-trees](https://github.com/christian-igel/city-trees) (commit `3c57d4c`) with his permission. `analyze_notebook.py` reproduces cells 2 and 4 of his notebook `City Trees Analysis.ipynb` from the same repository with the same logic. Please credit him when reusing this analysis.

## Method (Christian Igel's code)

- **Trees:** trees are assigned to each polygon part of a census area (block group or Surrey dissemination area) by spatial join.
- **Statistic:** for each part with at least 50 trees, `pointpats.l_test(coords, r = 2, 4, …, 50 m, n_simulations=0, linearized=True, hull=<convex hull>)`. pointpats computes the observed K̂ with the bounding box of the points as the area; the `hull` argument only affects simulations. Parts with more than 100,000 trees are subsampled to 100,000.
- **Multi-part areas:** averaged, weighted by the convex-hull area of their trees.
- **AUC:** area under L̂(r) − r over 2–50 m (`sklearn.metrics.auc`).
- **Census areas analysed:** income available and at least 50 trees in the analysed polygons.
- **Test:** Spearman correlation of AUC with median household income.
- **Quartile curves:** within-city income quartiles; L̂ averaged per quartile, weighted by the analysed convex-hull area.
- **No edge correction.**
- **Projection:** his original script projects to Web Mercator (EPSG:3857). At his suggestion the revision uses the UTM zone of each city: `process_city_trees_utm.py` is his script with only its two `to_crs(epsg=3857)` calls replaced (EPSG 32614 Austin, 32616 Bloomington, 32610 Cupertino and Surrey).

## Files

| File | Author | Purpose |
|---|---|---|
| `process_city_trees.py` | C. Igel, unchanged | Per-census-area L̂(r) and AUC (EPSG:3857) |
| `process_city_trees_utm.py` | C. Igel; projection lines changed | Same in the UTM zone of each city (revised Fig. S6) |
| `analyze_notebook.py` | logic from C. Igel's notebook | Filter, Spearman test, quartile curves, 2 × 2 figure |
| `run_ripley.py` | this revision | Runs his script with an exact, memory-light `pointpats.l_test` (see below) |
| `check_exact.py` | this revision | Checks that the replacement equals pointpats |
| `figS6_step.py` | this revision | The `figS6` step of `run_analysis.py` |

pointpats builds the full pairwise-distance matrix, which needs more than 100 GB for the largest census areas. `run_ripley.py` therefore swaps in `l_test_exact`, which computes the same statistic with a KD-tree: bounding-box area, pairs with d < r, linearised. On real census areas the difference from pointpats is exactly 0 (`check_exact.py`). The subsample of areas with more than 100,000 trees is seeded (0).

## Run

From the repository root, after reassembling the tree points (see `REPRODUCING.md`):

```bash
python revision/analysis/run_analysis.py --dataset <tree points folder> --label final --out <run folder> --only figS6
```

Or directly, on a folder holding `census_boundaries/<city>_exact_block_groups.geojson` and `<city>_ripley_points_with_income.geojson`:

```bash
RIPLEY_PROJ=utm python revision/analysis/ripley_igel/run_ripley.py <folder> austin bloomington cupertino surrey
FIG_DPI=300 python revision/analysis/ripley_igel/analyze_notebook.py <folder> final
```

## Results

Results are in `results/final_v1b/ripley_igel/`.

**Reproduction.** On the original (published) tree data with `RIPLEY_PROJ=3857`, the code reproduces the published Fig. S6 and the original notebook output exactly: 601 / 56 / 28 / 563 census areas; ρ = 0.037, 0.024, 0.071, 0.028.

**Final data, UTM:**

| City | Census areas | ρ (AUC vs income) | p |
|---|---|---|---|
| Austin | 606 | 0.10 | 0.010 |
| Bloomington | 58 | 0.23 | 0.079 |
| Cupertino | 28 | 0.17 | 0.38 |
| Surrey | 632 | 0.036 | 0.36 |

`variants_and_reproduction.csv` also lists Web Mercator and a convex-hull sensitivity variant. The convex-hull variant is the same as `../ripley_l.py`, which was used before Christian Igel's code was available and is kept as the reference step `ripley_hull_reference`.
