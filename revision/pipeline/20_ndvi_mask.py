"""
20_ndvi_mask.py -- vegetation mask (option A of reports/ndvi_false_positive_diagnostic.md, applied after the channel-order fix
of reports/inference_channel_order.md): drops every crown whose mean NDVI (17_crown_ndvi.py: band 2 of the source mosaic,
nearest-warped onto the prediction grid, pixels with data only) is below the threshold. Calibration: 0.2 % of annotated
reference-crown pixels on the validation patches have NDVI < 0.2 (reports/validation_patch_ndvi.csv). Crowns with no
NDVI data at all (entirely on mosaic no-data) are also removed and counted separately. Nothing else changes:
the kept features are copied unchanged (same fields, same tree_id), points and polygons in the same order.
Writes <out>/<city>_ripley_{points,polygons}_with_income.geojson, <out>/ndvi_masked_crowns_<city>.csv (removed crowns) and
<out>/ndvi_mask_log.json; the build log of the input dataset is copied.
Usage: python 20_ndvi_mask.py <in_dataset> <work_dir> <out_dataset> <ndvi_threshold>
"""
import sys, os, json, shutil, glob
import pandas as pd, pyogrio

src, work, out, th = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
CITIES = ('austin', 'bloomington', 'cupertino', 'surrey')
os.makedirs(out, exist_ok=True)
log = dict(input=src, ndvi_threshold=th, rule='drop crown if mean NDVI < threshold', cities={})
for c in CITIES:
    nd = pd.read_parquet(f'{work}/diag/crown_ndvi_final_{c}.parquet')
    P = {k: pyogrio.read_dataframe(f'{src}/{c}_ripley_{k}_with_income.geojson') for k in ('points', 'polygons')}
    assert (P['points'].tree_id.values == P['polygons'].tree_id.values).all()
    assert len(nd) == len(P['points']) and (nd.source_tile.values == P['points'].source_tile.values).all() \
        and (nd.instance_label.values == P['points'].instance_label.values).all(), 'NDVI table not aligned with crowns'
    nodata = nd.ndvi_mean.isna().values  # crowns entirely on mosaic no-data (no NDVI): cannot be confirmed as vegetation
    drop = (nd.ndvi_mean < th).values | nodata
    for k, g in P.items():
        pyogrio.write_dataframe(g[~drop], f'{out}/{c}_ripley_{k}_with_income.geojson', driver='GeoJSON')
    rm = P['points'].loc[drop, ['tree_id', 'source_tile', 'instance_label', 'block_group_id', 'crown_area_m2']].assign(
        ndvi_mean=nd.ndvi_mean.values[drop])
    rm.to_csv(f'{out}/ndvi_masked_crowns_{c}.csv', index=False)
    a = P['points'].crown_area_m2
    log['cities'][c] = dict(n_in=len(drop), n_removed=int(drop.sum()), n_removed_no_ndvi_data=int(nodata.sum()), n_out=int((~drop).sum()),
                            share_removed=float(drop.mean()), area_share_removed=float(a[drop].sum() / a.sum()))
    print(c, log['cities'][c], flush=True)
for f in glob.glob(f'{src}/build_log_*.json') + glob.glob(f'{src}/excluded_artifact_instances_*.csv'):
    shutil.copy(f, out)
json.dump(log, open(f'{out}/ndvi_mask_log.json', 'w'), indent=1)
