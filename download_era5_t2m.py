"""
Download ERA5 daily mean 2m air temperature for 56 lakes via Open-Meteo.

Open-Meteo (https://open-meteo.com) provides ERA5 reanalysis data as a free
REST API — no account, no API key, no queue, instant response.
56 lakes × 1 request each = done in ~30 seconds.

Output: Data/ERA5_T2M/csv/t2m_<id_v11>.csv  columns: [date, t2m_degC]
Also:   Data/ERA5_T2M/lake_depths.csv        (Depth_avg per lake, for air2water priors)

USAGE:
  pip install requests pandas
  python download_era5_t2m.py
"""

import time
import requests
import pandas as pd
from pathlib import Path

LAKES_CSV = Path("LWSTlakes.csv")
OUT_DIR   = Path("Data/ERA5_T2M/csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1981-01-01"
END_DATE   = "2021-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"

lakes = pd.read_csv(LAKES_CSV)
depths_out = []
n_done = n_skip = n_fail = 0

for i, row in lakes.iterrows():
    lake_id = int(row["id_v11"])
    lat, lon = row["lat"], row["lon"]
    name  = str(row["Lake_name"]) if pd.notna(row["Lake_name"]) else f"lake_{lake_id}"
    depth = row["Depth_avg"]

    out_csv = OUT_DIR / f"t2m_{lake_id}.csv"
    if out_csv.exists():
        print(f"[SKIP] {name}")
        n_skip += 1
        depths_out.append({"id_v11": lake_id, "Lake_name": name,
                            "lat": lat, "lon": lon, "Depth_avg": depth})
        continue

    params = {
        "latitude"   : lat,
        "longitude"  : lon,
        "start_date" : START_DATE,
        "end_date"   : END_DATE,
        "daily"      : "temperature_2m_mean",
        "timezone"   : "UTC",
    }

    # Retry with exponential backoff (handles 429 rate limits)
    for attempt in range(5):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/5...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()

            df = pd.DataFrame({
                "date"     : data["daily"]["time"],
                "t2m_degC" : data["daily"]["temperature_2m_mean"],
            })
            df.to_csv(out_csv, index=False)

            n_done += 1
            print(f"[{n_done+n_skip:02d}/56] {name:<30s} {len(df)} days → {out_csv.name}")
            depths_out.append({"id_v11": lake_id, "Lake_name": name,
                                "lat": lat, "lon": lon, "Depth_avg": depth})
            break

        except Exception as e:
            if attempt == 4:
                n_fail += 1
                print(f"[FAIL] {name}: {e}")
            else:
                time.sleep(5 * (attempt + 1))

    time.sleep(2)   # 2s between requests to stay under rate limit

# Write companion depths file for air2water parameter priors
pd.DataFrame(depths_out).to_csv(Path("Data/ERA5_T2M/lake_depths.csv"), index=False)

print(f"\nDone. {n_done} downloaded, {n_skip} skipped, {n_fail} failed.")
print(f"CSVs → {OUT_DIR}")
print(f"Depths → Data/ERA5_T2M/lake_depths.csv")
