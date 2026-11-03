# Crown polygons

One file per city (`austin`, `bloomington`, `cupertino`, `surrey`): `<city>_crown_polygons.geojson.gz`, with the outline of every detected tree crown in the final dataset (WGS 84). These are the same crowns as the points in `data/tree_points/`: Austin 2,586,912; Bloomington 143,323; Cupertino 118,413; Surrey 765,995.

Each crown is the polygonized outline of its pixels in the instance-separated prediction (0.5 m pixels). To keep the files small:

- **Fields.** Each feature carries only `tree_id`. Every other attribute (census unit, income, crown area, flags) is on the point with the same `tree_id` in `data/tree_points/`.
- **Coordinates.** Rounded to 7 decimal degrees (about 1 cm). This changes the summed crown area by less than 0.001 %; the `crown_area_m2` values on the points are unaffected.

## Files

Each city is split into 9 MB parts. Reassemble and decompress with:

```bash
cat data/crown_polygons/austin_crown_polygons.geojson.gz.part* | gunzip > austin_crown_polygons.geojson
```

To recreate the full polygon files with every attribute (`<city>_ripley_polygons_with_income.geojson`, as read by `revision/analysis/run_analysis.py`), decompress the points into a folder and run:

```bash
python revision/analysis/rebuild_polygons.py <folder>
```

The polygons are produced by `revision/pipeline/06_build_city_outputs.py` and filtered by the NDVI mask (`revision/pipeline/20_ndvi_mask.py`); see `data/tree_points/README.md` for the processing steps.
