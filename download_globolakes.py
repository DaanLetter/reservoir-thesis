"""
download_globolakes.py
----------------------
Downloads GloboLakes LSWT v4.0 NetCDF files from the CEDA archive for all
GeoDAR reservoirs that spatially match a GloboLakes lake.

Authentication follows the CEDA token workflow:
  - token obtained from https://services.ceda.ac.uk/account/token/

Steps:
  1. Fetch (or reuse) a CEDA download token
  2. Load the GloboLakes lake metadata (GLWD IDs + centroids + bounding boxes)
  3. Load GeoDAR dam locations
  4. Spatial join: for each dam find the nearest GloboLakes lake using a
     KD-tree, with an adaptive distance threshold scaled by lake area
  5. Filter to bounding-box-confirmed matches only
  6. Download the corresponding CEDA NetCDF file for each confirmed match;
     404s are silently skipped (not every GLWD ID has LSWT data)

Usage:
    python download_globolakes.py

Set DRY_RUN = True to print what would be downloaded without fetching anything.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from scipy.spatial import cKDTree

# ── USER CONFIG ────────────────────────────────────────────────────────────────

# Base URL for the GloboLakes LSWT v4.0 archive on CEDA
# Verify this path by browsing to the dataset on the CEDA archive
CEDA_BASE_URL = (
    "https://dap.ceda.ac.uk/neodc/globolakes/data/lake-surface-temp/per-lake/"
)

# Where to save the downloaded NetCDF files
OUTPUT_DIR = Path("Globolakes/nc_files")

# Path to the GloboLakes lake metadata CSV (BADC-CSV format from CEDA)
GLOBOLAKES_META = "Globolakes/globolakes-static_lake_centre_fv1.csv"

# Path to GeoDAR dam attributes (includes lat/lon, Lake_area, Grand_id)
GEODAR_CSV = "Data/RF/geodar_hydrolakes.csv"

# Adaptive spatial threshold: sqrt(Lake_area_km2) / 111 degrees
# Clipped so small lakes still get at least 0.1° and no lake exceeds 0.5°
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.5

# Set True to list files that would be downloaded without fetching anything
DRY_RUN = False

# Polite pause between requests (seconds) to avoid hammering the server
REQUEST_DELAY = 0.5

# ── END CONFIG ─────────────────────────────────────────────────────────────────


# ── TOKEN HANDLING ─────────────────────────────────────────────────────────────
# The CEDA token API is unreliable, so we use a manually generated token instead.
# To get your token:
#   1. Go to https://services.ceda.ac.uk and log in
#   2. Navigate to "MFA / Access Tokens" (or similar)
#   3. Generate a new token and paste it into CEDA_TOKEN below

CEDA_TOKEN = "YOUR_TOKEN_HERE"

def get_token():
    """
    Return the CEDA download token.
    Raises an error if the placeholder value has not been replaced.
    """
    if CEDA_TOKEN == "YOUR_TOKEN_HERE":
        raise RuntimeError(
            "Please replace CEDA_TOKEN in the script with your actual token.\n"
            "Generate one at: https://services.ceda.ac.uk"
        )
    return CEDA_TOKEN


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def parse_globolakes_meta(filepath):
    """
    Parse the GloboLakes BADC-CSV metadata file.
    The file has a multi-line header block, a 'data' marker, a column-number
    row, then the lake records, ending with 'end_data'.
    Returns a DataFrame: glwd_id, name, country, lat, lon, lat_min, lat_max,
                         lon_min, lon_max
    """
    with open(filepath, encoding="latin-1") as f:
        lines = f.readlines()

    # Locate the start of the data block
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("data"))

    # Skip the 'data' marker and the column-number header row
    data_lines = [
        l for l in lines[start + 2:]
        if not l.strip().startswith("end_data")
    ]

    df = pd.read_csv(
        StringIO("".join(data_lines)),
        header=None,
        names=["glwd_id", "name", "country", "lat", "lon",
               "lat_min", "lat_max", "lon_min", "lon_max"],
    )
    df["name"]    = df["name"].str.strip()
    df["glwd_id"] = df["glwd_id"].astype(int)
    return df


# ── SPATIAL MATCHING ───────────────────────────────────────────────────────────

def build_matches(df_globo, geodar):
    """
    Match each GeoDAR dam to its nearest GloboLakes lake using a KD-tree.

    Adaptive threshold: larger lakes get a wider search radius because the
    GeoDAR dam location (the dam wall) can be far from the lake centroid.
        threshold = clip(sqrt(Lake_area_km2) / 111, THRESHOLD_MIN, THRESHOLD_MAX)

    Bounding box filter: a match is 'confirmed' when the dam's lat/lon falls
    inside the GloboLakes lake's bounding box, providing geometric verification.

    Returns a DataFrame of confirmed matches only.
    """
    # Restrict to dams with a GRanD ID (better-documented reservoirs)
    geodar = geodar[geodar["Grand_id"].notna()].copy().reset_index(drop=True)

    # Compute adaptive threshold per dam
    geodar["threshold"] = np.sqrt(geodar["Lake_area"].fillna(1)) / 111
    geodar["threshold"] = geodar["threshold"].clip(THRESHOLD_MIN, THRESHOLD_MAX)

    # KD-tree on GloboLakes centroids, query with GeoDAR dam coordinates
    tree = cKDTree(df_globo[["lat", "lon"]].values)
    dists, idxs = tree.query(geodar[["lat", "lon"]].values, k=1)

    # Keep pairs within the adaptive distance threshold
    mask = dists < geodar["threshold"].values

    matches        = geodar[mask].copy().reset_index(drop=True)
    globo_matched  = df_globo.iloc[idxs[mask]].reset_index(drop=True)

    matches["globo_name"]    = globo_matched["name"].values
    matches["globo_glwd_id"] = globo_matched["glwd_id"].values
    matches["dist_deg"]      = dists[mask].round(4)

    # Bounding box confirmation: dam must lie inside the matched lake's bbox
    matches["inside_bbox"] = (
        (matches["lat"] >= globo_matched["lat_min"].values) &
        (matches["lat"] <= globo_matched["lat_max"].values) &
        (matches["lon"] >= globo_matched["lon_min"].values) &
        (matches["lon"] <= globo_matched["lon_max"].values)
    )

    # Return confirmed matches and the total spatial match count before bbox filter
    confirmed = matches[matches["inside_bbox"]].copy().reset_index(drop=True)
    return confirmed, int(mask.sum())


# ── DOWNLOAD ───────────────────────────────────────────────────────────────────

def glwd_to_filename(glwd_id):
    """Convert a GLWD integer ID to the CEDA filename convention."""
    return f"LAKE{glwd_id:08d}-GloboLakes-L3S-LSWT-v4.0-fv01.0.nc"


def download_file(glwd_id, token, output_dir, dry_run=False):
    """
    Download a single GloboLakes NetCDF file and save it to output_dir.

    Returns one of:
        'downloaded'  — file fetched and saved successfully
        'exists'      — file already on disk from a previous run
        'skipped'     — GLWD ID has no data file on CEDA (HTTP 404)
        'error'       — unexpected HTTP error
        'dry_run'     — dry-run mode, nothing fetched
    """
    filename   = glwd_to_filename(glwd_id)
    local_path = output_dir / filename

    # Skip files already downloaded in a previous run
    if local_path.exists():
        return "exists"

    url = CEDA_BASE_URL + filename + "?download=1"

    if dry_run:
        print(f"  [DRY RUN] {url}")
        return "dry_run"

    # Attach the CEDA token as a Bearer header (same pattern as CEDA example)
    headers  = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True, timeout=60)

    if response.status_code == 200:
        # Write to disk in 1 MB chunks to handle larger files efficiently
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        return "downloaded"

    elif response.status_code == 404:
        # Not every GLWD ID has LSWT data — this is expected for many lakes
        return "skipped"

    else:
        print(f"  [ERROR] HTTP {response.status_code} for GLWD {glwd_id}")
        return "error"


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Authenticate
    token = get_token()

    # 2. Load GloboLakes lake catalogue
    print("\nLoading GloboLakes metadata...")
    df_globo = parse_globolakes_meta(GLOBOLAKES_META)
    print(f"  {len(df_globo)} lakes in catalogue")

    # 3. Load GeoDAR dam locations
    print("Loading GeoDAR dam locations...")
    geodar = pd.read_csv(GEODAR_CSV)
    print(f"  {len(geodar)} dams, {geodar['Grand_id'].notna().sum()} with GRanD ID")

    # 4. Spatial join + bounding box filter
    print("Matching dams to GloboLakes lakes...")
    confirmed, n_spatial = build_matches(df_globo, geodar)
    print(f"  {n_spatial} spatial matches → {len(confirmed)} confirmed by bounding box")

    # Save match table for reference and later joining
    match_path = OUTPUT_DIR / "dam_globolakes_matches.csv"
    confirmed[["id_v11", "Grand_id", "Lake_name", "lat", "lon",
               "globo_name", "globo_glwd_id", "dist_deg"]].to_csv(
        match_path, index=False
    )
    print(f"  Match table saved to {match_path}")

    # 5. Download
    glwd_ids = confirmed["globo_glwd_id"].unique()
    print(f"\nDownloading {len(glwd_ids)} files "
          f"{'(DRY RUN) ' if DRY_RUN else ''}to {OUTPUT_DIR}...\n")

    counts = {"downloaded": 0, "exists": 0, "skipped": 0, "error": 0, "dry_run": 0}

    for i, glwd_id in enumerate(glwd_ids, 1):
        name   = confirmed.loc[confirmed["globo_glwd_id"] == glwd_id, "globo_name"].iloc[0]
        result = download_file(glwd_id, token, OUTPUT_DIR, DRY_RUN)
        counts[result] += 1

        label = {
            "downloaded": "✓ downloaded",
            "exists":     "  already exists",
            "skipped":    "  no data on CEDA (404)",
            "error":      "✗ error",
            "dry_run":    "  dry run",
        }[result]
        print(f"  [{i:3d}/{len(glwd_ids)}] GLWD {glwd_id:6d}  {name[:45]:<45}  {label}")

        if not DRY_RUN and result != "exists":
            time.sleep(REQUEST_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    actually_available = counts['downloaded'] + counts['exists']

    print(f"\n{'='*60}")
    print(f"  MATCHING SUMMARY")
    print(f"{'='*60}")
    print(f"  GeoDAR dams with GRanD ID         : {geodar['Grand_id'].notna().sum()}")
    print(f"  Spatial matches (adaptive thresh.) : {n_spatial}")
    print(f"  Confirmed by bounding box          : {len(confirmed)}")
    print(f"  Unique GloboLakes lakes to fetch   : {len(glwd_ids)}")
    print(f"{'='*60}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded                         : {counts['downloaded']}")
    print(f"  Already on disk                    : {counts['exists']}")
    print(f"  Not in GloboLakes (404)            : {counts['skipped']}")
    print(f"  Errors                             : {counts['error']}")
    print(f"{'='*60}")
    print(f"  Lakes with usable LSWT data        : {actually_available}")
    print(f"  (= downloaded + already on disk)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
