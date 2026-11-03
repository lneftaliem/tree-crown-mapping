# Mosaic cutlines

One GeoJSON file per city (`austin`, `bloomington`, `cupertino`, `surrey`) giving the footprint of every source image used to build the 16 × 16 km image mosaics (84 tiles: 41 Austin, 12 Bloomington, 12 Cupertino, 19 Surrey). Coordinates are WGS 84 (EPSG:4326).

Each feature is the part of one source image that was used in one mosaic tile. Fields:

| Field | Description |
|---|---|
| `mosaic_tile` | Mosaic tile ID (row_column of the 16 km grid) |
| `in_final_dataset` | `true` if the tile contributed to the final tree-crown dataset (tiles intersecting the municipal boundaries) |
| `IMAGENAME` | Source image file name (scene and product identifiers) |
| `CAT_ID` | Vantor catalog ID of the acquisition |
| `SENSOR` | GE01, QB02, WV02 or WV03 |
| `ACQDATE` | Acquisition date and time (UTC) |
| `RESOLUTION`, `OFF_NADIR`, `SUN_ELEV`, `SUN_AZ`, `SAT_ELEV`, `SAT_AZ`, `CLOUDCOVER`, `TDI`, `SCAN_DIR` | Acquisition geometry and image properties |
| `DATE_DIFF`, `SCORE`, `CLOUD_FRAC` | Mosaic selection inputs |
| `RF_MULT`, `RF_ADD`, `AOD`, `STATS_*` | Radiometric scaling and image statistics |

The imagery itself was acquired by NASA under the Electro-Optical Commercial Layer License. The copyright remains with Vantor, and redistribution is not possible. U.S. government-funded researchers can obtain the scenes through EarthExplorer or from Vantor using the catalog IDs above.
