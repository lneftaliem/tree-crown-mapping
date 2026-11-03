"""
06_build_city_outputs.py -- FINAL GENERATION of the split-crown point and polygon GeoJSONs.
*** Not to be run until the audit has been approved. ***

Pipeline (per city), all inputs read-only:
  1. For every city tile (all 2,006 tiles are accounted for in tile_scan.csv; none is skipped
     silently -- a tile without usable georeferencing raises), read the split raster and the
     instance labels. One crown = one label value (labels are unique within a tile; an instance
     whose pixels form several 8-connected pieces becomes a MultiPolygon, not several trees).
  2. Candidate crowns: npx * (projected pixel area at centroid) >= 2.9 m2 (safety margin), then
     polygonize (rasterio.features.shapes, connectivity=8 -- the connectivity the split was run
     with), dissolve pieces by label, project to the city UTM zone, and compute the exact
     crown_area_m2 from the projected polygon. crown_area_px = true raster pixel count.
  3. Filter crown_area_m2 >= 3.0  (Brandt et al.; area, not diameter).
  4. Optional QA exclusions (configured below after review): artefact prediction blocks.
  5. Cross-tile dedup (manuscript): centroids of the projected crown polygons, KD-tree, r = 1.0 m,
     only pairs from different tiles, deterministic edge-distance priority (dedup.py).
  6. Keep crowns whose CENTROID lies within the exact municipal boundary (not bbox, not polygon
     intersection; polygons are not clipped, so crown area is never truncated by the boundary).
  7. Block-group assignment: centroid 'intersects' block group, first match (identical to the
     original assign_income_ripley.py); socioeconomic fields copied verbatim from the original
     per-city GeoJSONs via the block_group_id -> attributes lookup (bg_lookup_<city>.csv).
  8. Write <city>_ripley_points_with_income.geojson and <city>_ripley_polygons_with_income.geojson
     (EPSG:4326 / CRS84, like the originals) with identical tree_id sets.

Usage: python 06_build_city_outputs.py --out <dir> [--exclude-artifacts] [--unresolved-m2 N] city [...]
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import dedup_cross_tile  # noqa: E402

ROOT = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
import paths  # noqa: E402
AUD = paths.DIAG
CENSUS = paths.CENSUS
SPLIT = paths.SPLIT_DIR
EPSG = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
MIN_AREA = 3.0
ORIG_FIELDS = ['city', 'block_group_id', 'median_household_income', 'has_income', 'longitude', 'latitude',
               'x_meters', 'y_meters', 'crown_area_px', 'total_pop', 'pct_white', 'pct_black', 'pct_asian',
               'pct_hispanic_latino', 'pct_indigenous', 'pct_other_multiracial']
NEW_FIELDS = ['tree_id', 'source_tile', 'instance_label', 'crown_area_m2', 'n_parts', 'parent_component_px',
              'instances_in_parent', 'unresolved_canopy']
ATTR = ['median_household_income', 'has_income', 'total_pop', 'pct_white', 'pct_black', 'pct_asian',
        'pct_hispanic_latino', 'pct_indigenous', 'pct_other_multiracial']


def tile_transform(row):
    """Resolved transform from georef_table.csv (08_georef_table.py: embedded/metadata/base_match/scene)."""
    if not np.isfinite(row.dx):
        raise RuntimeError(f'no georeferencing for {row.filename}')
    return rasterio.Affine(row.dx, 0, row.L, 0, -row.dx, row['T'])


def polygonize_tile(row, cand):
    """cand: crown rows (this tile) to polygonize. Returns GeoDataFrame in UTM."""
    tf = tile_transform(row)
    with rasterio.open(f'{SPLIT}/center_based_relabeled_{row.filename}') as src:
        s = src.read(1)
    want = np.zeros(int(s.max()) + 1, bool)
    want[cand.label.values] = True
    m = want[s]
    geoms = {}
    for g, v in shapes(s.astype(np.int32), mask=m, connectivity=8, transform=tf):
        geoms.setdefault(int(v), []).append(shape(g))
    labs = np.array(sorted(geoms))
    polys = [p[0] if len(p) == 1 else unary_union(p) for p in (geoms[k] for k in labs)]
    # 8-connected pixel outlines can touch themselves at a single vertex (diagonal pixels), which
    # GEOS reports as 'Ring Self-intersection'; the structure-preserving repair keeps every pixel
    # and the exact area (verified: max relative area change < 1e-15) and yields a MultiPolygon.
    polys = list(shapely.make_valid(np.array(polys, dtype=object), method='structure', keep_collapsed=False))
    g = gpd.GeoDataFrame({'label': labs, 'n_parts': [len(geoms[k]) for k in labs]}, geometry=polys, crs=4326)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--exclude-artifacts', action='store_true',
                    help='drop crowns inside components listed in diagnostics/artifact_components.csv')
    ap.add_argument('--unresolved-m2', type=float, default=250.0,
                    help='flag (not drop) crowns larger than this as unresolved_canopy')
    ap.add_argument('cities', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scan = pd.read_csv(f'{AUD}/georef_table.csv')
    art = pd.read_csv(f'{AUD}/artifact_components.csv') if a.exclude_artifacts else None
    if art is not None and 'city' not in art:
        art['city'] = art.tile.str.extract(r'_(austin|bloomington|cupertino|surrey)_')[0]
    log = {}
    for city in a.cities:
        tr = Transformer.from_crs(4326, EPSG[city], always_xy=True)
        crowns = pd.read_parquet(f'{AUD}/crowns_city/{city}_split.parquet')
        st = {'all_instances_in_city_tiles': len(crowns)}
        cand = crowns[crowns.area_m2 >= MIN_AREA * 0.97]
        # Only crowns whose centroid is within 25 m of the municipality can end up in (or interact,
        # via the 1 m dedup, with) the final set; restricting here changes no result, only run time.
        bnd0 = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
        near = gpd.GeoSeries([bnd0], crs=4326).to_crs(EPSG[city]).buffer(25).to_crs(4326).iloc[0]
        shapely.prepare(near)
        cand = cand[shapely.contains_xy(near, cand.lon.values, cand.lat.values)]
        st['candidates_ge2.9m2_within_25m_of_city'] = len(cand)
        if art is not None:
            bad = art[art.city == city][['tile', 'parent_cc8']].assign(_bad=True)
            cand = cand.merge(bad, on=['tile', 'parent_cc8'], how='left')
            bad_mask = cand._bad.eq(True).to_numpy()
            st['excluded_artifact_instances_ge2.9m2_all_tiles'] = int(bad_mask.sum())
            st['excluded_artifact_area_m2_all_tiles'] = float(cand.area_m2[bad_mask].sum())
            cand[bad_mask][['tile', 'label', 'parent_cc8', 'npx', 'area_m2', 'lon', 'lat', 'inside']].to_csv(
                f'{a.out}/excluded_artifact_instances_{city}.csv', index=False)
            cand = cand[~bad_mask].drop(columns='_bad')
        parts = []
        for _, row in scan[scan.city == city].iterrows():
            c = cand[cand.tile == row.filename]
            if len(c) == 0:
                continue
            g = polygonize_tile(row, c)
            g = g.merge(c[['label', 'tile', 'npx', 'edge_m', 'parent_cc8_npx', 'labels_in_parent']], on='label')
            parts.append(g)
        g = pd.concat(parts, ignore_index=True)
        g = gpd.GeoDataFrame(g, geometry='geometry', crs=4326)
        gu = g.to_crs(EPSG[city])
        g['crown_area_m2'] = gu.area.values
        cu = gu.centroid
        g['x_meters'], g['y_meters'] = cu.x.values, cu.y.values
        lon, lat = Transformer.from_crs(EPSG[city], 4326, always_xy=True).transform(cu.x.values, cu.y.values)
        g['longitude'], g['latitude'] = lon, lat
        st['polygonized_candidates'] = len(g)
        g = g[g.crown_area_m2 >= MIN_AREA].reset_index(drop=True)
        st['ge_3m2'] = len(g)
        tb = (g.tile + '|' + g.label.astype(str).str.zfill(9)).values
        # Extended Data Fig. 1: final trees under alternative dedup radii, everything else held constant
        bnd_s = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
        ins_s = shapely.contains_xy(bnd_s, g.longitude.values, g.latitude.values)
        for r in (0.8, 1.0, 1.5):
            k = dedup_cross_tile(g[['x_meters', 'y_meters']].values, g.tile.values, g.edge_m.values, tb, r)
            st[f'dedup_sensitivity_final_r{r}'] = int((k & ins_s).sum())
            st[f'dedup_sensitivity_removed_r{r}'] = int((~k).sum())
        keep = dedup_cross_tile(g[['x_meters', 'y_meters']].values, g.tile.values, g.edge_m.values, tb, 1.0)
        st['cross_tile_duplicates_removed'] = int((~keep).sum())
        g = g[keep].reset_index(drop=True)
        bnd = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
        inside = shapely.contains_xy(bnd, g.longitude.values, g.latitude.values)
        st['centroid_outside_boundary_removed'] = int((~inside).sum())
        g = g[inside].reset_index(drop=True)
        # block groups
        bg = gpd.read_file(f'{CENSUS}/{city}_exact_block_groups.geojson').to_crs(4326)
        idc = 'DAUID' if city == 'surrey' else 'GEOID'
        bg['bgid'] = pd.to_numeric(bg[idc], errors='coerce').astype('Int64')
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(g.longitude, g.latitude), crs=4326)
        j = gpd.sjoin(pts, bg[['bgid', 'geometry']], how='left', predicate='intersects')
        st['points_on_bg_edges_multi_match'] = int(j.index.duplicated().sum())
        j = j[~j.index.duplicated(keep='first')].sort_index()
        g['block_group_id'] = j.bgid.astype('Int64').values
        lk = pd.read_csv(f'{AUD}/bg_lookup_{city}.csv').set_index('block_group_id')
        for c in ATTR:
            g[c] = [lk[c].get(int(v), np.nan) if pd.notna(v) else np.nan for v in g.block_group_id]
        g['total_pop'] = g.total_pop.astype('Int64')
        g['has_income'] = np.where(g.median_household_income.notna(), 'yes', 'no')
        st['no_block_group'] = int(g.block_group_id.isna().sum())
        st['bg_not_in_original_lookup'] = int((g.block_group_id.notna() & ~g.block_group_id.isin(lk.index)).sum())
        g['city'] = city
        g['crown_area_px'] = g.npx.astype(float)
        g['tree_id'] = city + '_' + g.tile.str[:-4] + '_' + g.label.astype(str)
        g['source_tile'] = g.tile
        g['instance_label'] = g.label.astype('int64')
        g['parent_component_px'] = g.parent_cc8_npx.astype('int64')
        g['instances_in_parent'] = g.labels_in_parent.astype('int64')
        g['unresolved_canopy'] = (g.crown_area_m2 > a.unresolved_m2) if a.unresolved_m2 else False
        assert g.tree_id.is_unique
        g = g.sort_values('tree_id').reset_index(drop=True)
        cols = ORIG_FIELDS + [c for c in NEW_FIELDS if c not in ORIG_FIELDS]
        poly = gpd.GeoDataFrame(g[cols], geometry=g.geometry.values, crs=4326)
        pts = gpd.GeoDataFrame(g[cols], geometry=gpd.points_from_xy(g.longitude, g.latitude), crs=4326)
        pts.to_file(f'{a.out}/{city}_ripley_points_with_income.geojson', driver='GeoJSON')
        poly.to_file(f'{a.out}/{city}_ripley_polygons_with_income.geojson', driver='GeoJSON')
        st['final'] = len(g)
        log[city] = st
        print(city, st, flush=True)
    json.dump(log, open(f'{a.out}/build_log_{"_".join(a.cities)}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
