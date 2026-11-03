"""
repro_data.py  (ADAPTED for the v1 consistently processed, instance-separated dataset)
-------------------------------------------------------------------------------------
Replaces the original staging module. The original staged the committed historical tree points
(data/tree_points/) and the committed census-unit tables whose tree columns were computed from them. Here:

  * tree points are served ONLY from the dataset directory given explicitly in TREE_MAPPING_DATASET_DIR
    (the final <city>_ripley_points_with_income.geojson / _polygons_ files); every file served is checksummed
    into $TREE_MAPPING_BASE_DIR/provenance/inputs_used.csv;
  * the census-unit tables (all_cities_urban_covariates[_corrected].csv) are NOT copied from the repository;
    they must be produced for this dataset by build_unit_table.py (tree columns recomputed from the dataset,
    non-tree covariates verified against the committed tables);
  * crown areas are NOT latitude-corrected again: the dataset's crown_area_m2 is the projected (UTM) area of the
    crown polygon. apply_crown_area_correction() keeps it and only adds pixel_area_m2 = crown_area_m2 / crown_area_px
    for reference.
The public API (names, return values, directory layout) is unchanged so the original scripts run unmodified.
"""
import os, hashlib, shutil, csv, time

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repository root
COMMITTED_BOUNDARIES_DIR = os.path.join(REPO_DIR, "data", "boundaries")
COMMITTED_REDLINING_DIR = os.path.join(REPO_DIR, "data", "redlining")
COMMITTED_PARK_CSV = os.path.join(REPO_DIR, "data", "covariates", "all_cities_park_covariates.csv")

BASE_DIR = os.environ.get("TREE_MAPPING_BASE_DIR", os.getcwd())
DATASET_DIR = os.environ.get("TREE_MAPPING_DATASET_DIR")
RIPLEY_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data")
BOUNDARIES_STAGE_DIR = os.path.join(BASE_DIR, "census_boundaries", "exact_boundaries")
EQUITY_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "equity_results")
CORRECTED_STAGE_DIR = os.path.join(BASE_DIR, "analysis_output", "ripley_data_corrected")
REDLINING_STAGE_DIR = os.path.join(BASE_DIR, "redlining_data")
PROV_CSV = os.path.join(BASE_DIR, "provenance", "inputs_used.csv")

CITIES = ("austin", "bloomington", "cupertino", "surrey")
POINTS_STEM = "{city}_ripley_points_with_income.geojson"
POLYGONS_STEM = "{city}_ripley_polygons_with_income.geojson"
CORRECTED_POINTS_STEM = "{city}_ripley_points_with_income_corrected.geojson"
BOUNDARY_FILES = ("{city}_exact_boundary.geojson", "{city}_exact_block_groups.geojson")


def _require_dataset():
    if not DATASET_DIR or not os.path.isdir(DATASET_DIR):
        raise RuntimeError("TREE_MAPPING_DATASET_DIR must point at the final dataset directory "
                           "(no fallback to historical tree data)")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def record_input(path, role):
    """Append (time, role, path, size, sha256) to provenance/inputs_used.csv (once per path and role)."""
    os.makedirs(os.path.dirname(PROV_CSV), exist_ok=True)
    real = os.path.realpath(path)
    seen = set()
    if os.path.exists(PROV_CSV):
        with open(PROV_CSV) as f:
            seen = {(r["role"], r["path"]) for r in csv.DictReader(f)}
    if (role, real) in seen:
        return
    new = not os.path.exists(PROV_CSV)
    with open(PROV_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "role", "path", "bytes", "sha256"])
        w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), role, real, os.path.getsize(real), _sha256(real)])


def _stage_link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        if os.path.realpath(dst) != os.path.realpath(src):
            raise RuntimeError(f"{dst} exists and does not point at the dataset ({os.path.realpath(dst)})")
        return dst
    os.symlink(os.path.realpath(src), dst)
    return dst


def ensure_city_points(city, force=False):
    """Path of the dataset's point GeoJSON for ``city``, staged (symlink) where the scripts read it."""
    _require_dataset()
    src = os.path.join(DATASET_DIR, POINTS_STEM.format(city=city.lower()))
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    record_input(src, f"tree_points:{city}")
    return _stage_link(src, os.path.join(RIPLEY_STAGE_DIR, POINTS_STEM.format(city=city.lower())))


def ensure_city_polygons(city):
    _require_dataset()
    src = os.path.join(DATASET_DIR, POLYGONS_STEM.format(city=city.lower()))
    record_input(src, f"tree_polygons:{city}")
    return src


def ensure_boundaries(city, force=False):
    """Census/municipal geometry (non-tree inputs) from the repository, unchanged."""
    city = city.lower()
    os.makedirs(BOUNDARIES_STAGE_DIR, exist_ok=True)
    out = []
    for tmpl in BOUNDARY_FILES:
        fname = tmpl.format(city=city)
        src, dst = os.path.join(COMMITTED_BOUNDARIES_DIR, fname), os.path.join(BOUNDARIES_STAGE_DIR, fname)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        record_input(src, "boundaries")
        out.append(dst)
    return out


def stage_all(cities=CITIES, points=True, boundaries=True, force=False):
    for city in cities:
        if boundaries:
            ensure_boundaries(city)
        if points:
            ensure_city_points(city)


def stage_covariates(force=False):
    """The census-unit tables must already have been generated for this dataset by build_unit_table.py."""
    need = ["all_cities_urban_covariates.csv", "all_cities_urban_covariates_corrected.csv", "all_cities_park_covariates.csv"]
    missing = [n for n in need if not os.path.exists(os.path.join(EQUITY_STAGE_DIR, n))]
    if missing:
        raise RuntimeError(f"census-unit tables not built for this dataset: {missing} (run build_unit_table.py)")
    return [os.path.join(EQUITY_STAGE_DIR, n) for n in need]


def stage_redlining(force=False):
    os.makedirs(REDLINING_STAGE_DIR, exist_ok=True)
    out = []
    for fname in sorted(os.listdir(COMMITTED_REDLINING_DIR)):
        src, dst = os.path.join(COMMITTED_REDLINING_DIR, fname), os.path.join(REDLINING_STAGE_DIR, fname)
        if os.path.isfile(src):
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            record_input(src, "redlining")
            out.append(dst)
    return out


# Original per-city pixel size (degrees) of the historical latitude correction. Used ONLY when
# TREE_MAPPING_LEGACY_CORRECTION=1, i.e. when re-running the analyses on the ORIGINAL unsplit dataset (which stores
# pixel counts only) to reproduce the published numbers. Never applied to datasets that carry crown_area_m2.
DEG_PER_PX = {"austin": 4.8827495976e-06, "bloomington": 5.0595793518e-06,
              "cupertino": 5.4267546470e-06, "surrey": 6.3230653393e-06}
M2_PER_DEG2 = 111320.0 ** 2


def apply_crown_area_correction(df, city):
    """Keep the dataset's projected crown_area_m2 (no second latitude correction)."""
    if "crown_area_m2" in df.columns and df["crown_area_m2"].notna().all():
        if os.environ.get("TREE_MAPPING_LEGACY_CORRECTION") == "1":
            raise RuntimeError("legacy latitude correction requested for a dataset that already has crown_area_m2")
        df["pixel_area_m2"] = df["crown_area_m2"] / df["crown_area_px"]
        return df
    if os.environ.get("TREE_MAPPING_LEGACY_CORRECTION") != "1":
        raise RuntimeError("dataset points carry no crown_area_m2; the historical latitude correction is only "
                           "allowed for the original-dataset reproduction (TREE_MAPPING_LEGACY_CORRECTION=1)")
    import numpy as np
    lat = np.radians(df["latitude"].to_numpy(dtype=float))
    df["pixel_area_m2"] = (DEG_PER_PX[city.lower()] ** 2) * M2_PER_DEG2 * np.cos(lat)
    df["crown_area_m2"] = df["crown_area_px"] * df["pixel_area_m2"]
    return df


def load_corrected_points(city, columns=None, bbox=None):
    """Dataset points with their projected crown_area_m2 (nothing recomputed)."""
    import pyogrio
    src = ensure_city_points(city)
    read_cols = None
    if columns is not None:
        avail = set(pyogrio.read_info(src)["fields"])
        read_cols = [c for c in columns if c != "pixel_area_m2" and c in avail]
        for need in ("crown_area_m2", "crown_area_px", "latitude"):
            if need in avail and need not in read_cols:
                read_cols.append(need)
    gdf = pyogrio.read_dataframe(src, columns=read_cols, bbox=bbox)
    apply_crown_area_correction(gdf, city)
    if columns is not None:
        keep = list(columns) + (["geometry"] if "geometry" in gdf.columns else [])
        gdf = gdf[[c for c in keep if c in gdf.columns]]
    return gdf


if __name__ == "__main__":
    stage_all()
    stage_redlining()
    print("[repro_data] staged dataset", DATASET_DIR, "->", BASE_DIR)
