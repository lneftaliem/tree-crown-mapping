"""
reproject_verbatim.py -- reproject one PV mosaic to EPSG:4326 with reproject_to_4326() copied VERBATIM from
sherlock_venv/notebooks/Auxiliary-2-SplitRasterToAnalyse.py (cell 1 of the notebook), and optionally compare
the result pixel-for-pixel with an existing original *_reprojected.tif.
Usage: python reproject_verbatim.py <mosaic.tif> <out_reprojected.tif> [original_reprojected.tif]
"""
import sys, shutil, json
import numpy as np
import rasterio
from rasterio import windows
from rasterio.warp import calculate_default_transform, reproject, Resampling
TARGET_CRS = 'EPSG:4326'

def reproject_to_4326(input_path, output_path):
    """
    Reproject a raster to EPSG:4326.
    """
    with rasterio.open(input_path) as src:
        # Check if already in target CRS
        if src.crs and str(src.crs).upper() == 'EPSG:4326':
            print(f"  Already in EPSG:4326, copying...")
            # Just copy the file
            shutil.copy(input_path, output_path)
            return output_path
        
        # Calculate transform for new CRS
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': TARGET_CRS,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        print(f"  Reprojecting from {src.crs} to {TARGET_CRS}...")
        print(f"  Original size: {src.width}x{src.height}, New size: {width}x{height}")
        
        # Reproject
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear
                )
    
    return output_path


if __name__ == '__main__':
    src_p, out_p = sys.argv[1:3]
    reproject_to_4326(src_p, out_p)
    if len(sys.argv) > 3:
        res = {'mosaic': src_p.split('/')[-1]}
        with rasterio.open(out_p) as a, rasterio.open(sys.argv[3]) as b:
            res.update(same_profile=(a.width, a.height, a.count, a.dtypes, a.crs) == (b.width, b.height, b.count, b.dtypes, b.crs),
                       same_transform=a.transform == b.transform, n_differ=0, max_abs=0)
            for _, w in a.block_windows(1):
                x, y = a.read(window=w).astype(np.int32), b.read(window=w).astype(np.int32)
                d = np.abs(x - y); res['n_differ'] += int((d > 0).sum()); res['max_abs'] = max(res['max_abs'], int(d.max()))
            res['share_differ'] = res['n_differ'] / (a.width * a.height * a.count)
        print('COMPARE', json.dumps(res), flush=True)
