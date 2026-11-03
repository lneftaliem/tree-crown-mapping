# Final per-tree datasets

One file per city (`austin`, `bloomington`, `cupertino`, `surrey`): `<city>_ripley_points_with_income.geojson`, with one point per detected tree crown. These are the crowns used in all analyses in the revised manuscript. Detected crowns per city: Austin 2,586,912; Bloomington 143,323; Cupertino 118,413; Surrey 765,995.

## How the crowns were produced

1. **Inference.** The fine-tuned U-Net was applied to every 16 × 16 km mosaic tile. Model input is channel 0 = panchromatic and channel 1 = NDVI, with per-patch normalization, a 224-pixel stride and maximum merging of overlapping patches.
2. **Binary mask.** Predictions were thresholded at τ = 0.7.
3. **Instance separation.** Contiguous canopy was split into individual crowns by center-based instance separation (15-pixel maximum filter, eight-connectivity).
4. **Size filter and deduplication.**
   - Crowns smaller than 3 m² were removed.
   - Crowns detected twice in overlapping tiles were deduplicated within 1.0 m.
   - Crowns larger than 250 m² are kept but flagged `unresolved_canopy`.
5. **NDVI mask.** Crowns with mean NDVI below 0.2 were removed (0.6–3.5 % of crowns per city; `ndvi_mask_log.json`).

`build_log.json` records the counts at each build stage, including the deduplication-radius sensitivity at 0.8, 1.0 and 1.5 m. The code is in `revision/pipeline/` and `revision/sherlock/`.

## Fields

| Field | Description |
|---|---|
| `tree_id`, `source_tile`, `instance_label` | Crown identifiers |
| `longitude`, `latitude` | Crown centroid (WGS 84) |
| `x_meters`, `y_meters` | Crown centroid in the city's UTM zone |
| `crown_area_px` | Crown area in pixels |
| `crown_area_m2` | Projected crown area (latitude-corrected) |
| `n_parts`, `parent_component_px`, `instances_in_parent` | Instance-separation bookkeeping |
| `unresolved_canopy` | `true` for crowns > 250 m² (likely merged canopy) |
| `block_group_id` | Census block group (U.S.) or dissemination area (Surrey) |
| `median_household_income`, `has_income` | Unit income (USD; Surrey converted at 1.37 CAD per USD) |
| `total_pop`, `pct_*` | Unit population and demographic shares |

## Files

- **Small cities:** Bloomington and Cupertino are single gzip files.
- **Large cities:** Austin and Surrey are split into 9 MB parts. Reassemble with:

```bash
cat data/tree_points/austin_ripley_points_with_income.geojson.gz.part* > austin_ripley_points_with_income.geojson.gz
gunzip austin_ripley_points_with_income.geojson.gz
```

The crown outlines of the same trees are in `data/crown_polygons/` (joined to these points by `tree_id`).

The datasets from the original submission are kept in `data/original_submission/` for comparison.
