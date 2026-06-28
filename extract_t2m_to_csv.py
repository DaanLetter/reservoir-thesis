"""
Merge per-lake per-year ERA5 NetCDFs into one CSV per lake.
Run after download_era5_t2m.py completes (or partway through — partial is fine).

Output: Data/ERA5_T2M/csv/t2m_<id_v11>.csv  with columns [date, t2m_degC]
Also writes: Data/ERA5_T2M/lake_depths.csv  (Depth_avg per lake, for air2water priors)

USAGE:
  python extract_t2m_to_csv.py
"""

import xarray as xr
import pandas as pd
from pathlib import Path

LAKES_CSV = Path("LWSTlakes.csv")
NC_DIR    = Path("Data/ERA5_T2M/lakes")
CSV_DIR   = Path("Data/ERA5_T2M/csv")
CSV_DIR.mkdir(parents=True, exist_ok=True)

lakes = pd.read_csv(LAKES_CSV)
depths_out = []

for _, row in lakes.iterrows():
    lake_id = int(row["id_v11"])
    lat, lon = row["lat"], row["lon"]
    name  = str(row["Lake_name"]) if pd.notna(row["Lake_name"]) else f"lake_{lake_id}"
    depth = row["Depth_avg"]

    nc_files = sorted(NC_DIR.glob(f"t2m_{lake_id}_*.nc"))  # matches both yearly and 5-yr chunks
    if not nc_files:
        print(f"[SKIP] {name}: no files downloaded yet")
        continue

    out_csv = CSV_DIR / f"t2m_{lake_id}.csv"

    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    t2m_var = "t2m" if "t2m" in ds else list(ds.data_vars)[0]
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"

    ts = (ds[t2m_var]
          .sel(latitude=lat, longitude=lon, method="nearest")
          - 273.15)
    df = ts.to_dataframe(name="t2m_degC").reset_index()[[time_dim, "t2m_degC"]]
    df.rename(columns={time_dim: "date"}, inplace=True)
    df = df.sort_values("date").drop_duplicates("date")
    df.to_csv(out_csv, index=False)
    ds.close()

    depths_out.append({
        "id_v11"    : lake_id,
        "Lake_name" : name,
        "lat"       : lat,
        "lon"       : lon,
        "Depth_avg" : depth,
        "n_days"    : len(df),
    })
    print(f"[OK] {name:<30s} {len(nc_files)} yr-files → {len(df)} days → {out_csv.name}")

pd.DataFrame(depths_out).to_csv(Path("Data/ERA5_T2M/lake_depths.csv"), index=False)
print(f"\nWrote lake_depths.csv  ({len(depths_out)} lakes)")
print(f"Per-lake CSVs in {CSV_DIR}")
