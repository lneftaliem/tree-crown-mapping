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
    print("[repro_data] staging complete.")
    print(f"[repro_data]   tree points  -> {RIPLEY_STAGE_DIR}")
    print(f"[repro_data]   boundaries   -> {BOUNDARIES_STAGE_DIR}")
