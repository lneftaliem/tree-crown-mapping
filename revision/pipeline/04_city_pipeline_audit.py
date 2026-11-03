"""
04_city_pipeline_audit.py -- city-level stage counts for ORIGINAL (4-connected components) and
SPLIT (instance labels) crowns, from the per-crown tables written by 01_tile_scan.py.

Stages / variants evaluated:
  R  replica of prepare_ripley_data.py: dedup in DEGREES (eps 1e-5) with the unsorted-set loop,
     then centroid 'within' the exact municipal boundary                -> should reproduce 3.29 M
  M  manuscript method: 1.0 m UTM centroid KD-tree, cross-tile only, deterministic
     edge-distance priority (dedup.py), then centroid within boundary
     M_prevloop: same pairs but the previous attempt's sorted loop (index-order priority)
     M_r0.8 / M_r1.5: threshold sensitivity
  A  >= 3 m2 filter, both orders: dedup->filter and filter->dedup
Crown area = true pixel count x projected (UTM) area of one pixel at the crown centroid; this
equals the projected area of the pixel-union polygon to <0.01 % (checked in 05_*).

Writes diagnostics/city_stages.csv, diagnostics/large_crowns.csv, diagnostics/income_strata.csv,
       diagnostics/area_distribution.csv, diagnostics/crowns_city/<city>_<kind>.parquet
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.validation import make_valid
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dedup import dedup_cross_tile, previous_loop  # noqa: E402

ROOT = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
AUD = paths.DIAG
CENSUS = paths.CENSUS
EPSG = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
ORIG_REPORTED = {'austin': 1899163, 'bloomington': 116682, 'cupertino': 52720, 'surrey': 1222292}
MIN_AREA = 3.0


def load_city(city, kind, scan):
    tiles = scan[scan.city == city]
    parts = []
    for _, t in tiles.iterrows():
        p = f'{AUD}/crowns_geo/{t.filename[:-4]}.{kind}.parquet'
        d = pd.read_parquet(p)
        d['dx'] = t.dx; d['dy'] = t.dy
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def pixel_area(lon, lat, dx, dy, tr):
    """Projected area (m2) of one EPSG:4326 pixel of size dx x dy (deg) centred at lon/lat."""
    hx, hy = dx / 2, dy / 2
    xs, ys = [], []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        x, y = tr.transform(lon + sx * hx, lat + sy * hy)
        xs.append(x); ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    return 0.5 * np.abs(np.sum(xs * np.roll(ys, -1, 0) - np.roll(xs, -1, 0) * ys, axis=0))


def degree_replica(lon, lat, tile):
    """prepare_ripley_data.py dedup: KDTree on lon/lat, eps=1e-5 deg, unsorted set iteration."""
    pairs = cKDTree(np.column_stack([lon, lat])).query_pairs(0.00001)
    to_remove = set()
    for i, j in pairs:
        if tile[i] != tile[j]:
            if i not in to_remove:
                to_remove.add(j)
    keep = np.ones(len(lon), bool); keep[list(to_remove)] = False
    return keep


def main(cities):
    scan = pd.read_csv(f'{AUD}/georef_table.csv')  # resolved transforms (08_georef_table.py)
    scan['dy'] = scan.dx
    os.makedirs(f'{AUD}/crowns_city', exist_ok=True)
    stage_rows, big_rows, inc_rows, dist_rows = [], [], [], []
    for city in cities:
        bnd = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
        shapely.prepare(bnd)
        bg = gpd.read_file(f'{CENSUS}/{city}_exact_block_groups.geojson').to_crs(4326)
        idc = 'DAUID' if city == 'surrey' else 'GEOID'
        bg['block_group_id'] = pd.to_numeric(bg[idc], errors='coerce').astype('Int64')
        lk = pd.read_csv(f'{AUD}/bg_lookup_{city}.csv')
        tr = Transformer.from_crs(4326, EPSG[city], always_xy=True)
        for kind in ('orig', 'split'):
            d = load_city(city, kind, scan)
            d['x'], d['y'] = tr.transform(d.lon.values, d.lat.values)
            d['pix_m2'] = pixel_area(d.lon.values, d.lat.values, d.dx.values, d.dy.values, tr)
            d['area_m2'] = d.npx * d.pix_m2
            d['inside'] = shapely.contains_xy(bnd, d.lon.values, d.lat.values)
            big = d.area_m2 >= MIN_AREA
            tile = d.tile.values
            xy = d[['x', 'y']].values
            tb = (d.tile + '|' + d.label.astype(str).str.zfill(9)).values
            st = {'city': city, 'kind': kind, 'n_tiles': d.tile.nunique(), 'S0_all_crowns_in_city_tiles': len(d),
                  'S0_inside_boundary_no_dedup': int(d.inside.sum())}
            # R: replica
            kR = degree_replica(d.lon.values, d.lat.values, tile)
            st['R_deg_dedup_removed'] = int((~kR).sum())
            st['R_final_inside'] = int((kR & d.inside).sum())
            st['R_reported'] = ORIG_REPORTED[city] if kind == 'orig' else np.nan
            # M: manuscript, deterministic
            for r in (0.8, 1.0, 1.5):
                k = dedup_cross_tile(xy, tile, d.edge_m.values, tb, r)
                st[f'M_r{r}_removed'] = int((~k).sum())
                st[f'M_r{r}_final_inside'] = int((k & d.inside).sum())
                if r == 1.0:
                    kM = k
            kP = previous_loop(xy, tile, 1.0, sort=True)
            st['M_prevloop_removed'] = int((~kP).sum())
            st['M_prevloop_final_inside'] = int((kP & d.inside).sum())
            st['M_vs_prevloop_disagree'] = int((kM != kP).sum())
            # A: area filter, both orders
            st['A_dedup_then_filter_inside'] = int((kM & big & d.inside).sum())
            kF = np.zeros(len(d), bool)
            idx = np.nonzero(big.values)[0]
            kF[idx] = dedup_cross_tile(xy[idx], tile[idx], d.edge_m.values[idx], tb[idx], 1.0)
            st['A_filter_then_dedup_inside'] = int((kF & d.inside).sum())
            st['A_order_disagree_inside'] = int(((kF != (kM & big.values)) & d.inside).sum())
            st['A_filter_only_inside_no_dedup'] = int((big & d.inside).sum())
            # crowns lost by dedup->filter that filter->dedup keeps
            stage_rows.append(st)
            print(st, flush=True)

            d['keep_dedup'] = kM
            d['keep_final'] = kF & d.inside.values          # recommended: filter -> dedup -> centroid-in
            d['keep_unfiltered'] = kM & d.inside.values
            # block-group join (centroid within) for income strata
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(d.lon, d.lat), crs=4326)
            sel = d.keep_unfiltered.values
            j = gpd.sjoin(pts[sel], bg[['block_group_id', 'geometry']], how='left', predicate='intersects')
            j = j[~j.index.duplicated(keep='first')]
            d['block_group_id'] = pd.Series(pd.NA, index=d.index, dtype='Int64')
            d.loc[j.index, 'block_group_id'] = j.block_group_id.values
            d.to_parquet(f'{AUD}/crowns_city/{city}_{kind}.parquet', index=False)

            # area distributions
            for lab, m in [('unfiltered', d.keep_unfiltered), ('ge3m2', d.keep_final)]:
                a = d.area_m2[m]
                q = a.quantile([.05, .25, .5, .75, .95, .99, .999]).to_dict()
                dist_rows.append({'city': city, 'kind': kind, 'set': lab, 'n': int(m.sum()),
                                  'mean': a.mean(), 'max': a.max(), **{f'q{k}': v for k, v in q.items()},
                                  'share_1px': float((d.npx[m] == 1).mean()),
                                  'n_gt100m2': int((a > 100).sum()), 'n_gt250m2': int((a > 250).sum()),
                                  'n_gt1000m2': int((a > 1000).sum()),
                                  'area_total_m2': a.sum(),
                                  'area_share_gt250m2': a[a > 250].sum() / a.sum(),
                                  'area_share_gt1000m2': a[a > 1000].sum() / a.sum()})
            top = d[d.keep_final].nlargest(10, 'area_m2')
            for _, t in top.iterrows():
                big_rows.append({'city': city, 'kind': kind, **t[['tile', 'label', 'npx', 'area_m2', 'lon', 'lat',
                                                                  'edge_m']].to_dict(),
                                 **({k: t[k] for k in ('labels_in_parent', 'parent_cc8_npx', 'n_pieces')}
                                    if kind == 'split' else {'n_split_labels': t['n_split_labels']})})
            # income strata (block-group income quintiles, city-specific, BG-level)
            q = lk.dropna(subset=['median_household_income']).copy()
            q['inc_q'] = pd.qcut(q.median_household_income, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            mp = dict(zip(q.block_group_id.astype('int64'), q.inc_q.astype(str)))
            d['inc_q'] = d.block_group_id.map(lambda v: mp.get(int(v), 'NA') if pd.notna(v) else 'no_bg')
            for qq, g in d[d.keep_unfiltered].groupby('inc_q'):
                inc_rows.append({'city': city, 'kind': kind, 'inc_q': qq, 'n_unfiltered': len(g),
                                 'n_ge3m2': int((g.area_m2 >= MIN_AREA).sum()),
                                 'retained_share': float((g.area_m2 >= MIN_AREA).mean()),
                                 'median_area_unfiltered': g.area_m2.median(),
                                 'median_area_ge3': g.area_m2[g.area_m2 >= MIN_AREA].median(),
                                 'share_1px': float((g.npx == 1).mean())})
    sfx = '' if len(cities) == 4 else '_' + '_'.join(cities)
    pd.DataFrame(stage_rows).to_csv(f'{AUD}/city_stages{sfx}.csv', index=False)
    pd.DataFrame(big_rows).to_csv(f'{AUD}/large_crowns{sfx}.csv', index=False)
    pd.DataFrame(inc_rows).to_csv(f'{AUD}/income_strata{sfx}.csv', index=False)
    pd.DataFrame(dist_rows).to_csv(f'{AUD}/area_distribution{sfx}.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:] or ['bloomington', 'cupertino', 'surrey', 'austin'])
