"""
repro_data.py
-------------
Bridges the *committed* public datasets under ``data/`` to the directory layout
the analysis / figure scripts expect, so the published figures can be
reproduced directly from a fresh clone of this repository.

Why this exists
===============
The final, deduplicated per-tree datasets and study-area boundaries are
committed under::

    data/tree_points/<city>_ripley_points_with_income.geojson.gz         (single)
    data/tree_points/<city>_ripley_points_with_income.geojson.gz.part*   (split, reassembled)
    data/boundaries/<city>_exact_boundary.geojson
    data/boundaries/<city>_exact_block_groups.geojson

but the analysis scripts read tree points from
``$TREE_MAPPING_BASE_DIR/analysis_output/ripley_data/`` (uncompressed
``.geojson``) and boundaries from
``$TREE_MAPPING_BASE_DIR/census_boundaries/exact_boundaries/``.

Calling :func:`stage_all` (or running ``reproduce.sh``) materialises the
committed data into those locations, reassembling the split Austin/Surrey
archives and decompressing every city, so the figure scripts run unchanged.

The staging is idempotent: files already present at the destination are left
untouched, so re-running is cheap and never clobbers existing outputs.

Census-unit covariate tables (``data/covariates/``) and the Austin HOLC
polygons (``data/redlining/``) are staged the same way, into
``$TREE_MAPPING_BASE_DIR/analysis_output/equity_results/`` and
``$TREE_MAPPING_BASE_DIR/redlining_data/``.

Latitude-corrected crown areas
==============================
The ``*_corrected`` point datasets used by the revision analyses are a pure,
deterministic function of the committed point data: each tree's pixel count is
multiplied by the true ground area of one image pixel at that tree's latitude
(see :func:`apply_crown_area_correction`). They are therefore not committed;
:func:`load_corrected_points` rebuilds them on the fly from the committed
files, and ``run_crown_area_correction_pipeline.py --write-points`` writes
them to ``analysis_output/ripley_data_corrected/`` if you want them on disk.
"""

import os
import glob
import gzip
import shutil

# Location of THIS repository (where the committed data lives), independent of
# the caller's working directory.
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
COMMITTED_POINTS_DIR = os.path.join(REPO_DIR, "data", "tree_points")
COMMITTED_BOUNDARIES_DIR = os.path.join(REPO_DIR, "data", "boundaries")

# Where the analysis scripts read their inputs from.
BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
RIPLEY_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data")
BOUNDARIES_STAGE_DIR = os.path.join(BASE_DIR, "census_boundaries", "exact_boundaries")

CITIES = ("austin", "bloomington", "cupertino", "surrey")

POINTS_STEM = "{city}_ripley_points_with_income.geojson"
BOUNDARY_FILES = (
    "{city}_exact_boundary.geojson",
    "{city}_exact_block_groups.geojson",
)


def _decompress_gz(src_gz, dst):
    """Decompress a single .gz file to ``dst`` (streamed, low memory)."""
    with gzip.open(src_gz, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)


def _reassemble_and_decompress(parts, dst):
    """Concatenate split ``.gz.partNN`` chunks and decompress to ``dst``."""
    tmp_gz = dst + ".reassembled.gz"
    with open(tmp_gz, "wb") as fo:
        for part in parts:
            with open(part, "rb") as fi:
                shutil.copyfileobj(fi, fo, length=16 * 1024 * 1024)
    try:
        _decompress_gz(tmp_gz, dst)
    finally:
        if os.path.exists(tmp_gz):
            os.remove(tmp_gz)


def ensure_city_points(city, force=False):
    """Ensure ``<city>_ripley_points_with_income.geojson`` exists in the staging
    directory, materialising it from the committed compressed / split copy if
    needed. Returns the path to the decompressed GeoJSON, or ``None`` if no
    committed source could be found.
    """
    city = city.lower()
    fname = POINTS_STEM.format(city=city)
    dst = os.path.join(RIPLEY_STAGE_DIR, fname)

    if os.path.exists(dst) and not force:
        return dst

    os.makedirs(RIPLEY_STAGE_DIR, exist_ok=True)

    single_gz = os.path.join(COMMITTED_POINTS_DIR, fname + ".gz")
    parts = sorted(glob.glob(os.path.join(COMMITTED_POINTS_DIR, fname + ".gz.part*")))

    if os.path.exists(single_gz):
        print(f"[repro_data] decompressing {os.path.basename(single_gz)} -> {dst}")
        _decompress_gz(single_gz, dst)
        return dst
    if parts:
        print(f"[repro_data] reassembling {len(parts)} parts + decompressing -> {dst}")
        _reassemble_and_decompress(parts, dst)
        return dst

    print(f"[repro_data] WARNING: no committed point data found for '{city}' "
          f"under {COMMITTED_POINTS_DIR}")
    return None


def ensure_boundaries(city, force=False):
    """Copy the committed boundary + block-group GeoJSONs for ``city`` into the
    staging directory the scripts read from. Returns the list of destination
    paths that now exist.
    """
    city = city.lower()
    os.makedirs(BOUNDARIES_STAGE_DIR, exist_ok=True)
    out = []
    for tmpl in BOUNDARY_FILES:
        fname = tmpl.format(city=city)
        src = os.path.join(COMMITTED_BOUNDARIES_DIR, fname)
        dst = os.path.join(BOUNDARIES_STAGE_DIR, fname)
        if os.path.exists(dst) and not force:
            out.append(dst)
            continue
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            out.append(dst)
        else:
            print(f"[repro_data] WARNING: committed boundary missing: {src}")
    return out


def stage_all(cities=CITIES, points=True, boundaries=True, force=False):
    """Materialise all committed public data into the layout the analysis
    scripts expect. Safe to call at the top of any figure script; it is a no-op
    when the data is already staged.
    """
    for city in cities:
        if boundaries:
            ensure_boundaries(city, force=force)
        if points:
            ensure_city_points(city, force=force)


# ── Covariate tables, redlining polygons, corrected crown areas ──────────────

COMMITTED_COVARIATES_DIR = os.path.join(REPO_DIR, "data", "covariates")
COMMITTED_REDLINING_DIR = os.path.join(REPO_DIR, "data", "redlining")
EQUITY_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "equity_results")
CORRECTED_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data_corrected")
REDLINING_STAGE_DIR = os.path.join(BASE_DIR, "redlining_data")

CORRECTED_POINTS_STEM = "{city}_ripley_points_with_income_corrected.geojson"

# Native image pixel size in degrees for each city's mosaic (from tile_metadata.csv).
DEG_PER_PX = {
    "austin":      4.8827495976e-06,
    "bloomington": 5.0595793518e-06,
    "cupertino":   5.4267546470e-06,
    "surrey":      6.3230653393e-06,
}
# Square metres per square degree at the equator (111.32 km per degree)^2.
M2_PER_DEG2 = 111320.0 ** 2


def _copy_tree(src_dir, dst_dir, force=False):
    """Copy every file in ``src_dir`` into ``dst_dir`` without overwriting."""
    out = []
    if not os.path.isdir(src_dir):
        print(f"[repro_data] WARNING: committed directory missing: {src_dir}")
        return out
    os.makedirs(dst_dir, exist_ok=True)
    for fname in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.isfile(src):
            continue
        if force or not os.path.exists(dst):
            shutil.copyfile(src, dst)
        out.append(dst)
    return out


def stage_covariates(force=False):
    """Stage the committed census-unit covariate tables into
    ``analysis_output/equity_results/``."""
    return _copy_tree(COMMITTED_COVARIATES_DIR, EQUITY_STAGE_DIR, force=force)


def stage_redlining(force=False):
    """Stage the committed Austin HOLC polygons into ``redlining_data/``."""
    return _copy_tree(COMMITTED_REDLINING_DIR, REDLINING_STAGE_DIR, force=force)


def apply_crown_area_correction(df, city):
    """Add ``pixel_area_m2`` and ``crown_area_m2`` columns to a tree table.

    ``pixel_area_m2 = DEG_PER_PX[city]^2 * 111320^2 * cos(latitude)`` is the
    ground area of one image pixel at each tree's latitude, and
    ``crown_area_m2 = crown_area_px * pixel_area_m2``. ``df`` needs
    ``latitude`` and ``crown_area_px`` columns. Returns ``df`` (modified in place).
    """
    import numpy as np

    city = city.lower()
    lat_rad = np.radians(df["latitude"].to_numpy(dtype=float))
    df["pixel_area_m2"] = (DEG_PER_PX[city] ** 2) * M2_PER_DEG2 * np.cos(lat_rad)
    df["crown_area_m2"] = df["crown_area_px"] * df["pixel_area_m2"]
    return df


def load_corrected_points(city, columns=None, bbox=None):
    """Return the latitude-corrected tree points for ``city`` as a GeoDataFrame.

    Uses ``analysis_output/ripley_data_corrected/<city>_..._corrected.geojson``
    if it already exists; otherwise stages the committed point data and applies
    :func:`apply_crown_area_correction` in memory. ``columns`` and ``bbox`` are
    passed to ``pyogrio.read_dataframe`` so large cities can be read partially.
    Columns produced by the correction (``crown_area_m2``, ``pixel_area_m2``)
    may be requested even when reading the uncorrected source.
    """
    import pyogrio

    city = city.lower()
    corrected = os.path.join(CORRECTED_STAGE_DIR, CORRECTED_POINTS_STEM.format(city=city))
    if os.path.exists(corrected):
        return pyogrio.read_dataframe(corrected, columns=columns, bbox=bbox)

    src = ensure_city_points(city)
    if src is None:
        raise FileNotFoundError(f"No committed point data for {city}")
    derived = {"crown_area_m2", "pixel_area_m2"}
    read_cols = None
    if columns is not None:
        read_cols = [c for c in columns if c not in derived]
        for need in ("latitude", "crown_area_px"):
            if need not in read_cols:
                read_cols.append(need)
    gdf = pyogrio.read_dataframe(src, columns=read_cols, bbox=bbox)
    apply_crown_area_correction(gdf, city)
    if columns is not None:
        keep = list(columns) + (["geometry"] if "geometry" in gdf.columns else [])
        gdf = gdf[[c for c in keep if c in gdf.columns]]
    return gdf


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Stage committed public datasets for reproducing the figures.")
    ap.add_argument("--cities", default=",".join(CITIES),
                    help="comma-separated subset of cities to stage")
    ap.add_argument("--force", action="store_true",
                    help="re-materialise even if the destination already exists")
    args = ap.parse_args()

    stage_all(cities=tuple(c.strip().lower() for c in args.cities.split(",") if c.strip()),
              force=args.force)
    stage_covariates(force=args.force)
    stage_redlining(force=args.force)
    print("[repro_data] staging complete.")
    print(f"[repro_data]   tree points  -> {RIPLEY_STAGE_DIR}")
    print(f"[repro_data]   boundaries   -> {BOUNDARIES_STAGE_DIR}")
    print(f"[repro_data]   covariates   -> {EQUITY_STAGE_DIR}")
    print(f"[repro_data]   redlining    -> {REDLINING_STAGE_DIR}")
