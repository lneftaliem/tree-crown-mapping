"""
12_block_group_coverage.py -- block-group coverage audit (diagnostic only; reads the final GeoJSONs,
never writes them).

Coverage footprint = union of the INDIVIDUAL rectangles of the 2,006 rasters in used_predictions.zip,
each from its resolved transform (08_georef_table.py) and its actual pixel size (not 5000 px assumed).
Areas are computed in the city UTM zone. For each census block group / DA in
<city>_exact_block_groups.geojson that overlaps the municipality:
  area within municipality, area covered by tiles, % covered, area inside excluded artefact blocks,
  original detections (all / inside tile coverage), re-derived original >= 3 m2 on the same tiles,
  split crowns before the 3 m2 filter, final split crowns >= 3 m2, artefact crowns excluded,
  coverage class and diagnosis.
Two coverage measures:
  nominal   = union of tile rectangles (what the archive's extents cover);
  effective = union of CELL x CELL-pixel cells (~250 m) of each tile that contain >= 1 predicted pixel,
              after a 1-cell morphological closing (so isolated genuinely treeless cells inside valid imagery
              are not treated as gaps), minus the excluded artefact blocks. Whole tiles / scene parts with
              zero predictions (e.g. all of Surrey scene 186_028) are no-data or failed inference, not
              treeless land, and fall outside the effective footprint.
Coverage classes use the EFFECTIVE measure: none (< 1 %), partial (1-80 %), substantial (>= 80 %);
'full' = >= 99 % (subset of substantial).
Writes split_audit_20261031/block_group_coverage/{block_group_coverage.csv, city_summary.csv,
income_quintile_coverage.csv, bloomington_missing_explained.csv, maps/*.png}.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
import shapely
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import box, shape
from shapely.validation import make_valid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.environ.get('URBAN_DATA', '/path/to/Urban_data')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
AUD = paths.AUDIT
DIAG = paths.DIAG
OUT = os.environ.get('COVERAGE_OUT', f'{AUD}/block_group_coverage')
FINAL_DIR = os.environ.get('FINAL_DIR', f'{ROOT}/ripley_data_0429206/all_trees_split')  # final GeoJSONs to assess
CENSUS = paths.CENSUS

EPSG = {'austin': 32614, 'bloomington': 32616, 'cupertino': 32610, 'surrey': 32610}
CITIES = ['bloomington', 'austin', 'cupertino', 'surrey']
S8 = ndi.generate_binary_structure(2, 2)
CELL = 500  # pixels (~230-350 m depending on latitude)
# sequential blue ramp (dataviz reference palette), light -> dark
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
NO_COV = '#e6e5e0'
CAT = {'has crowns': '#2a78d6', 'substantial coverage, no crowns': '#eb6834',
       'partial coverage, no crowns': '#eda100', 'no coverage': '#d9d8d2'}
INK, INK2 = '#0b0b0b', '#52514e'


def fix(g):
    """make valid, then snap to a 1 cm grid (avoids GEOS TopologyException on large unions)."""
    return shapely.set_precision(make_valid(g), 0.01)


def tile_boxes(geo, city):
    g = geo[geo.city == city]
    return [box(r.L, r['T'] - r.dx * r.o_h, r.L + r.dx * r.o_w, r['T']) for _, r in g.iterrows()]


def effective_footprint(geo, city, near):
    """Cells (CELL px) with >= 1 predicted pixel, closed by one cell, unioned over tiles near the city."""
    polys = []
    for _, r in geo[geo.city == city].iterrows():
        tb = box(r.L, r['T'] - r.dx * r.o_h, r.L + r.dx * r.o_w, r['T'])
        if not tb.intersects(near) or r.o_px == 0:
            continue
        c = pd.read_parquet(f'{DIAG}/crowns_geo/{r.filename[:-4]}.orig.parquet', columns=['row', 'col'])
        nr, nc = int(np.ceil(r.o_h / CELL)), int(np.ceil(r.o_w / CELL))
        occ = np.zeros((nr, nc), bool)
        occ[(c.row.values // CELL).astype(int), (c.col.values // CELL).astype(int)] = True
        occ = ndi.binary_closing(np.pad(occ, 1), np.ones((3, 3), bool))[1:-1, 1:-1] | occ
        for i, j in zip(*np.nonzero(occ)):
            x0 = r.L + j * CELL * r.dx; y0 = r['T'] - i * CELL * r.dx
            x1 = r.L + min((j + 1) * CELL, r.o_w) * r.dx; y1 = r['T'] - min((i + 1) * CELL, r.o_h) * r.dx
            polys.append(box(x0, y1, x1, y0))
    return shapely.union_all(polys) if polys else shapely.Polygon()


def artifact_polygons(art, geo, city, stride=4):
    """Footprint of the excluded artefact components (block-max downsampled by `stride`, then polygonized)."""
    polys = []
    for tile, grp in art[art.city == city].groupby('tile'):
        r = geo.loc[tile]
        with rasterio.open(paths.pred_path(tile)) as src:
            o = src.read(1) > 0
        lab, _ = ndi.label(ndi.binary_fill_holes(o), S8)
        m = np.isin(lab, grp.parent_cc8.values)
        H, W = m.shape
        m = m[:H - H % stride, :W - W % stride].reshape(H // stride, stride, W // stride, stride).any(axis=(1, 3))
        tf = rasterio.Affine(r.dx * stride, 0, r.L, 0, -r.dx * stride, r['T'])
        polys += [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=tf)]
    return shapely.union_all(polys) if polys else shapely.Polygon()


def count_in(bg, lon, lat, col):
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    j = gpd.sjoin(pts, bg[['bg_key', 'geometry']], how='inner', predicate='intersects')
    j = j[~j.index.duplicated(keep='first')]
    return j.groupby('bg_key').size().rename(col)


def main():
    os.makedirs(f'{OUT}/maps', exist_ok=True)
    geo = pd.read_csv(f'{DIAG}/georef_table.csv').set_index('filename', drop=False)
    art = pd.read_csv(f'{DIAG}/artifact_components.csv')
    rows, city_geo = [], {}
    for city in CITIES:
        utm = EPSG[city]
        bnd = make_valid(gpd.read_file(f'{CENSUS}/{city}_exact_boundary.geojson').to_crs(4326).union_all())
        bnd_u = gpd.GeoSeries([bnd], crs=4326).to_crs(utm).iloc[0]
        cov = shapely.union_all(tile_boxes(geo, city))
        cov_u = gpd.GeoSeries([cov], crs=4326).to_crs(utm).iloc[0]
        artp = artifact_polygons(art, geo, city)
        art_u = gpd.GeoSeries([artp], crs=4326).to_crs(utm).iloc[0]
        near = gpd.GeoSeries([bnd], crs=4326).to_crs(utm).buffer(500).to_crs(4326).iloc[0]
        eff = effective_footprint(geo, city, near)
        # snap to a 1 cm grid: avoids GEOS TopologyException on the large unions
        cov_u, art_u = fix(cov_u), fix(art_u)
        eff_u = fix(fix(gpd.GeoSeries([eff], crs=4326).to_crs(utm).iloc[0]).difference(art_u))
        eff = gpd.GeoSeries([eff_u], crs=utm).to_crs(4326).iloc[0]
        bg = gpd.read_file(f'{CENSUS}/{city}_exact_block_groups.geojson').to_crs(4326)
        idc = 'DAUID' if city == 'surrey' else 'GEOID'
        bg['bg_key'] = pd.to_numeric(bg[idc], errors='coerce').astype('int64')
        bg['geometry'] = bg.geometry.map(lambda g: make_valid(g) if not g.is_valid else g)
        bu = bg.to_crs(utm)
        inc = gpd.read_file(f'{ROOT}/ripley_data_0429206/all_trees/archive/{city}_blockgroups_with_income.geojson')
        inc['bg_key'] = pd.to_numeric(inc[idc], errors='coerce').astype('int64')
        inc = inc.drop_duplicates('bg_key').set_index('bg_key').median_household_income
        lk = pd.read_csv(f'{DIAG}/bg_lookup_{city}.csv').set_index('block_group_id')
        t = pd.DataFrame({'city': city, 'block_group_id': bg.bg_key.values})
        city_part = gpd.GeoSeries([fix(g) for g in bu.geometry.intersection(fix(bnd_u))], crs=utm)
        t['area_in_municipality_m2'] = city_part.area.values
        t['area_covered_nominal_tiles_m2'] = city_part.intersection(cov_u).area.values
        t['pct_covered_nominal_tiles'] = np.where(t.area_in_municipality_m2 > 0,
                                                  100 * t.area_covered_nominal_tiles_m2 / t.area_in_municipality_m2, np.nan)
        t['area_covered_m2'] = city_part.intersection(eff_u).area.values
        t['pct_covered'] = np.where(t.area_in_municipality_m2 > 0, 100 * t.area_covered_m2 / t.area_in_municipality_m2, np.nan)
        nodata_u = fix(cov_u.difference(fix(eff_u.union(art_u))))
        t['area_nominal_but_no_predictions_m2'] = city_part.intersection(nodata_u).area.values
        t['area_in_excluded_artifacts_m2'] = city_part.intersection(art_u).area.values
        # income: original per-tree lookup first, block-group income file for BGs without original trees
        t['median_household_income'] = [lk.median_household_income.get(k, inc.get(k, np.nan)) for k in t.block_group_id]
        t['income_source'] = ['original_trees' if k in lk.index else ('bg_income_file' if pd.notna(inc.get(k, np.nan)) else 'none')
                              for k in t.block_group_id]
        # counts
        orig = pyogrio.read_dataframe(f'{ROOT}/ripley_data_0429206/all_trees/{city}_ripley_points_with_income.geojson',
                                      read_geometry=False, columns=['block_group_id', 'longitude', 'latitude'])
        shapely.prepare(eff)
        orig['in_cov'] = shapely.contains_xy(eff, orig.longitude.values, orig.latitude.values)
        shapely.prepare(cov)
        orig['in_nominal'] = shapely.contains_xy(cov, orig.longitude.values, orig.latitude.values)
        t = t.merge(orig.groupby('block_group_id').size().rename('n_original_detections'), left_on='block_group_id',
                    right_index=True, how='left')
        t = t.merge(orig[orig.in_cov].groupby('block_group_id').size().rename('n_original_in_tile_coverage'),
                    left_on='block_group_id', right_index=True, how='left')
        t = t.merge(orig[orig.in_nominal].groupby('block_group_id').size().rename('n_original_in_nominal_tiles'),
                    left_on='block_group_id', right_index=True, how='left')
        fin = pyogrio.read_dataframe(f'{FINAL_DIR}/{city}_ripley_points_with_income.geojson',
                                     read_geometry=False, columns=['block_group_id', 'longitude', 'latitude'])
        t = t.merge(fin.groupby('block_group_id').size().rename('n_split_crowns_ge3m2_final'), left_on='block_group_id',
                    right_index=True, how='left')
        # same-tile context from the audit crown tables
        sp = pd.read_parquet(f'{DIAG}/crowns_city/{city}_split.parquet',
                             columns=['tile', 'parent_cc8', 'lon', 'lat', 'keep_unfiltered', 'keep_final'])
        sp = sp.merge(art[['tile', 'parent_cc8']].assign(art=True), on=['tile', 'parent_cc8'], how='left')
        sp['art'] = sp.art.eq(True)
        u = sp[sp.keep_unfiltered & ~sp.art]
        t = t.merge(count_in(bg, u.lon.values, u.lat.values, 'n_split_unfiltered'), left_on='block_group_id',
                    right_index=True, how='left')
        a = sp[sp.keep_final & sp.art]
        t = t.merge(count_in(bg, a.lon.values, a.lat.values, 'n_split_ge3m2_excluded_as_artifact'),
                    left_on='block_group_id', right_index=True, how='left')
        op = pd.read_parquet(f'{DIAG}/crowns_city/{city}_orig.parquet', columns=['lon', 'lat', 'keep_final'])
        op = op[op.keep_final]
        t = t.merge(count_in(bg, op.lon.values, op.lat.values, 'n_original_ge3m2_same_tiles'), left_on='block_group_id',
                    right_index=True, how='left')
        for c in ['n_original_detections', 'n_original_in_tile_coverage', 'n_original_in_nominal_tiles', 'n_split_crowns_ge3m2_final',
                  'n_split_unfiltered', 'n_split_ge3m2_excluded_as_artifact', 'n_original_ge3m2_same_tiles']:
            t[c] = t[c].fillna(0).astype(int)
        t = t[t.area_in_municipality_m2 > 1.0].copy()          # BGs that actually overlap the municipality
        t['in_original_dataset'] = t.n_original_detections > 0
        t['coverage_class'] = np.select([t.pct_covered < 1, t.pct_covered < 80], ['none', 'partial'], 'substantial')
        t['full_coverage_ge99pct'] = t.pct_covered >= 99
        t['has_retained_split_crowns'] = t.n_split_crowns_ge3m2_final > 0
        t['diagnosis'] = np.select(
            [t.has_retained_split_crowns, t.coverage_class == 'none', t.coverage_class == 'partial'],
            ['has crowns', 'no coverage', 'partial coverage, no crowns'], 'substantial coverage, no crowns')
        rows.append(t)
        city_geo[city] = dict(bg=bg.merge(t[['block_group_id', 'pct_covered', 'diagnosis']], left_on='bg_key',
                                          right_on='block_group_id', how='inner'),
                              bnd=bnd, cov=cov, eff=eff, art=artp, fin=fin)
        print(city, len(t), t.diagnosis.value_counts().to_dict(), flush=True)
    df = pd.concat(rows, ignore_index=True)
    # city-specific BG income quintiles (BGs with income)
    df['income_quintile'] = 'income missing'
    for city, g in df.groupby('city'):
        ok = g.median_household_income.notna()
        df.loc[g.index[ok], 'income_quintile'] = pd.qcut(g.median_household_income[ok], 5,
                                                         labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']).astype(str)
    df.to_csv(f'{OUT}/block_group_coverage.csv', index=False)

    def summ(g):
        return pd.Series({'block_groups': len(g), 'with_any_coverage': int((g.pct_covered >= 1).sum()),
                          'with_substantial_coverage_ge80pct': int((g.pct_covered >= 80).sum()),
                          'with_full_coverage_ge99pct': int(g.full_coverage_ge99pct.sum()),
                          'with_retained_crowns': int(g.has_retained_split_crowns.sum()),
                          'substantial_coverage_but_no_crowns': int(((g.pct_covered >= 80) & ~g.has_retained_split_crowns).sum()),
                          'with_any_nominal_tile_coverage': int((g.pct_covered_nominal_tiles >= 1).sum()),
                          'with_substantial_nominal_tile_coverage': int((g.pct_covered_nominal_tiles >= 80).sum()),
                          'area_weighted_pct_covered': 100 * g.area_covered_m2.sum() / g.area_in_municipality_m2.sum(),
                          'area_weighted_pct_covered_nominal_tiles': 100 * g.area_covered_nominal_tiles_m2.sum() / g.area_in_municipality_m2.sum(),
                          'original_detections': int(g.n_original_detections.sum()),
                          'original_in_tile_coverage': int(g.n_original_in_tile_coverage.sum()),
                          'split_crowns_ge3m2': int(g.n_split_crowns_ge3m2_final.sum())})
    cs = df.groupby('city').apply(summ, include_groups=False)
    cs.to_csv(f'{OUT}/city_summary.csv')
    qs = df.groupby(['city', 'income_quintile']).apply(
        lambda g: pd.concat([summ(g), pd.Series({'median_income_range': f"{g.median_household_income.min():,.0f}–{g.median_household_income.max():,.0f}",
                                                 'mean_pct_covered_unweighted': g.pct_covered.mean()})]),
        include_groups=False)
    qs.to_csv(f'{OUT}/income_quintile_coverage.csv')
    b = df[(df.city == 'bloomington') & ~df.has_retained_split_crowns].sort_values(['diagnosis', 'pct_covered'])
    b.to_csv(f'{OUT}/bloomington_missing_explained.csv', index=False)
    print(cs.to_string()); print(qs.to_string())
    draw_maps(city_geo)


def outline(g):
    """Boundary of the polygonal parts of g (a GeometryCollection from make_valid/difference has no boundary)."""
    parts = gpd.GeoSeries([g], crs=4326).explode(index_parts=False)
    return parts[parts.geom_type.isin(['Polygon', 'MultiPolygon'])].boundary


def draw_maps(city_geo):
    bounds = [0, 1, 20, 40, 60, 80, 99, 100.001]
    cmap = ListedColormap([NO_COV] + SEQ[1:])
    norm = BoundaryNorm(bounds, cmap.N)
    for city, d in city_geo.items():
        bg, fin = d['bg'], d['fin']
        two = city == 'bloomington'
        fig, axes = plt.subplots(1, 2 if two else 1, figsize=(16, 8.6) if two else (12.5, 9), facecolor='#fcfcfb')
        axes = np.atleast_1d(axes)
        ax = axes[0]
        bg.plot(column='pct_covered', cmap=cmap, norm=norm, ax=ax, edgecolor='white', linewidth=0.6)
        outline(d['cov']).plot(ax=ax, color='#b5b4ad', linewidth=0.5, linestyle=(0, (1, 2)))
        outline(d['eff']).plot(ax=ax, color=INK2, linewidth=0.6, linestyle=(0, (3, 2)))
        if not d['art'].is_empty:
            gpd.GeoSeries([d['art']], crs=4326).plot(ax=ax, facecolor='none', edgecolor='#e34948', hatch='////', linewidth=0.6)
        gpd.GeoSeries([d['bnd']], crs=4326).boundary.plot(ax=ax, color=INK, linewidth=1.0)
        n = len(fin)
        smp = fin.sample(min(n, 60000), random_state=0)
        ax.scatter(smp.longitude, smp.latitude, s=0.25 if n > 20000 else 0.6, c='#008300', alpha=0.35, linewidths=0)
        ax.set_title('Share of block-group area with supplied predictions (effective coverage)', fontsize=11, color=INK, loc='left')
        handles = [Patch(facecolor=NO_COV, label='< 1 % (no coverage)')] + \
                  [Patch(facecolor=SEQ[i + 1], label=f'{bounds[i + 1]:g}–{bounds[i + 2]:g} %' if i < 5 else '≥ 99 %')
                   for i in range(6)] + \
                  [Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label='edge of area with predictions'),
                   Line2D([], [], color='#b5b4ad', lw=0.8, ls=(0, (1, 2)), label='edge of tile rectangles'),
                   Line2D([], [], color=INK, lw=1, label='municipal boundary'),
                   Line2D([], [], marker='o', color='none', markerfacecolor='#008300', markersize=4,
                          label=f'retained crowns (n={n:,}{", 60k shown" if n > 60000 else ""})')]
        if not d['art'].is_empty:
            handles.append(Patch(facecolor='none', edgecolor='#e34948', hatch='////', label='excluded artefact blocks'))
        if two:
            ax.legend(handles=handles, loc='lower left', fontsize=7.5, frameon=True, framealpha=0.9)
        else:
            ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(1.01, 0.0), fontsize=8, frameon=False)
        if two:
            ax2 = axes[1]
            order = list(CAT)
            bg['diag'] = pd.Categorical(bg.diagnosis, categories=order)
            for k in order:
                s = bg[bg.diagnosis == k]
                if len(s):
                    s.plot(ax=ax2, facecolor=CAT[k], edgecolor='white', linewidth=0.8)
            outline(d['eff']).plot(ax=ax2, color=INK2, linewidth=0.6, linestyle=(0, (3, 2)))
            if not d['art'].is_empty:
                gpd.GeoSeries([d['art']], crs=4326).plot(ax=ax2, facecolor='none', edgecolor='#e34948', hatch='////', linewidth=0.6)
            gpd.GeoSeries([d['bnd']], crs=4326).boundary.plot(ax=ax2, color=INK, linewidth=1.0)
            for _, r in bg.iterrows():
                p = r.geometry.representative_point()
                ax2.annotate(f"{r.pct_covered:.0f}%", (p.x, p.y), ha='center', va='center', fontsize=6.5,
                             color='white' if r.diagnosis == 'has crowns' else INK)
            cnt = bg.diagnosis.value_counts()
            ax2.legend(handles=[Patch(facecolor=CAT[k], label=f'{k} ({cnt.get(k, 0)})') for k in order] +
                       [Patch(facecolor='none', edgecolor='#e34948', hatch='////', label='excluded artefact blocks')],
                       loc='lower left', fontsize=8, frameon=True, framealpha=0.9)
            ax2.set_title('Why block groups have or lack retained crowns (label = % covered)', fontsize=11,
                          color=INK, loc='left')
        x0, y0, x1, y1 = d['bnd'].bounds
        mx, my = (x1 - x0) * 0.04, (y1 - y0) * 0.04
        for a in axes:
            a.set_xlim(x0 - mx, x1 + mx); a.set_ylim(y0 - my, y1 + my)
            a.set_axis_off(); a.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        fig.suptitle(f'{city.title()} — block-group coverage of the supplied imagery', fontsize=13, color=INK, x=0.01, ha='left')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{OUT}/maps/{city}_block_group_coverage.png', dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == '__main__':
    main()
