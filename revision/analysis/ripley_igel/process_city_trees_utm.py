import numpy as np

import pandas as pd
import geopandas as gpd

import shapely
from shapely.geometry import Point, Polygon, MultiPolygon

from pointpats import k, l, g, f, k_test, l_test

from tqdm.notebook import tqdm
from datetime import datetime

from sklearn import metrics

import sys
import warnings
import logging
warnings.filterwarnings("ignore", message="n_simulations must be positive")
logging.basicConfig(level=logging.DEBUG)

def process_city(gdf_city,
                 gdf_trees,
                 id_name,
                 sup = None,
                 max_trees=100000,
                 min_trees=3):

    def polygon_iter(geoms):
        """Yield individual Polygon objects from Polygon or MultiPolygon."""
        if isinstance(geoms, Polygon):
            yield geoms
        elif isinstance(geoms, MultiPolygon):
            logging.info(f"MultiPolygon with {len(geoms.geoms)} polygons detected")
            for poly in geoms.geoms:
                yield poly
                
    # Default support
    if sup is None:
        sup = np.linspace(2,50,25)
    
    # Change coordinate system of block-group boundaries
    gdf_city = gdf_city.to_crs(epsg=PROJ_EPSG[city_name])

    # Count trees
    n_trees = gdf_trees.geometry.count()
    logging.debug(f"total number of trees: {n_trees}")
    # Change coordinate system of
    gdf_trees = gdf_trees.to_crs(epsg=PROJ_EPSG[city_name])

    # Bring block IDs to the same format
    gdf_city[id_name] = gdf_city[id_name].astype(int)
    gdf_trees["block_group_id"] = gdf_trees["block_group_id"].astype(int)
    assert len(gdf_city.geometry) == len(gdf_city)
    
    # Length of support for Ripley function computation
    n_sup = len(sup)

    # Loop over block-groups
    n_trees_polys = []  # number of trees in a polygon
    n_trees_aucs = []
    l_tests = []  # Ripley statistics
    aucs = []  # AUC values
    incomes = []  # household incomes
    areas = []  # processed areas
    # Loop over all entries
    for line in gdf_city.itertuples():
        geoms = line.geometry
        # look up income
        block_id = getattr(line, id_name)
        matches = gdf_trees.loc[
            gdf_trees["block_group_id"] == block_id, "median_household_income"
        ]
        if not matches.empty:
            income = matches.iloc[0]
        else:
            logging.warning(f"no match for id {block_id}")
            income = np.nan

        l = np.zeros(n_sup)  # for accumulating Ripley statistics
        n_trees_auc = 0  # trees used in AUC computation
        area_auc = 0  # area used in AUC computation
        n_trees_poly = 0  # number of trees in single polygon
        n_trees_poly_sum = 0  # number of trees in MultiPolygon
        # Loop over all polygons in a MultiPolygon
        for geom in polygon_iter(geoms):
            # Cut out trees belonging to current polygon
            # The following should be faster than gdf_trees_poly = gdf_trees[gdf_trees.within(geom)]
            gdf_trees_poly = gpd.sjoin(
                gdf_trees,
                gpd.GeoDataFrame(geometry=[geom], crs=gdf_trees.crs),
                predicate="within"
            )
            # Count trees in this polygon
            n_trees_poly = gdf_trees_poly.geometry.count()
            # Add up for MultiPolygon
            n_trees_poly_sum += n_trees_poly
       
            # Compute convex hull
            if n_trees_poly >= min_trees:
                coords = np.array(
                    [(pt.x, pt.y) for pt in gdf_trees_poly.geometry]
                )
                # Compute convex hull of the tree points
                hull_geom = shapely.geometry.MultiPoint(coords).convex_hull
                area = hull_geom.area
            else:
                area = 0

            # Compute Ripley statistics
            if area > 0:
                # If more than max_trees points, we sub-sample
                if n_trees_poly > max_trees:
                    logging.debug(f"processing {max_trees} trees sampled from a polygon with {n_trees_poly} trees in an area of {geom.area:,.2f}")
                    idx = np.random.choice(n_trees_poly, size=max_trees, replace=False)
                    l += area * l_test(coords[idx,:], sup, n_simulations=0, linearized=True, hull=hull_geom).statistic
                    n_trees_auc += max_trees
                else:
                    logging.debug(f"processing polygon with {n_trees_poly} trees in an area of {geom.area:,.2f}")
                    l += area * l_test(coords, sup, n_simulations=0, linearized=True, hull=hull_geom).statistic
                    n_trees_auc += n_trees_poly
                area_auc += area
            else:
                logging.info(f"skipping polygon with {n_trees_poly} trees in an area of {geom.area:,.2f}")
        # Turn sum into mean, not a number if no trees in block group
        if area_auc > 0:
            l /= area_auc
            a = metrics.auc(sup, l)
        else:
            l[:] = np.nan  # no trees in block group
            a = np.nan
        n_trees_polys.append(n_trees_poly_sum)
        n_trees_aucs.append(n_trees_auc)
        l_tests.append(l)
        aucs.append(a)
        areas.append(area_auc)
        incomes.append(income)

    gdf_city["L"] = l_tests
    gdf_city["auc"] = aucs
    gdf_city["trees"] = n_trees_polys
    gdf_city["sup"] = [sup] *  len(gdf_city)
    gdf_city["processed area"] = areas
    gdf_city["processed"] = n_trees_aucs
    gdf_city["income"] = incomes
    return gdf_city, sup
                                      

# Process cities

if len(sys.argv) > 1:
    cities = sys.argv[1:]
else:
    cities = ["cupertino", "austin", "bloomington", "surrey"]

sup = np.linspace(2,50,25)
min_trees=50
for city_name in cities:
    logging.info(f"Processing {city_name.title()}")
    # Load block-group boundaries
    fn = "census_boundaries/" + city_name + "_exact_block_groups.geojson"
    gdf_city = gpd.read_file(fn)

    # Load trees
    fn = city_name + "_ripley_points_with_income.geojson"
    gdf_trees = gpd.read_file(fn)

    if "DAUID" in gdf_city.columns:
        id_name = "DAUID"
    elif "GEOID" in gdf_city.columns:
        id_name = "GEOID"
    else:
        logging.error("unknow id")
        
    gdf, sup = process_city(gdf_city, gdf_trees, id_name, sup=sup, min_trees=min_trees)
    gdf.to_parquet(city_name + "_Ripley.parquet")

