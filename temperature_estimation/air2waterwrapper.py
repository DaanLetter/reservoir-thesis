import logging
import os
from collections import defaultdict
import xarray as xr
import numpy as np
import pcraster as pcr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
import subprocess
import json



#set up logging and correct file path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
file_path = os.path.join(os.getcwd(), 'air2water') + os.sep


#build all the file paths within the repo
HERE      = Path(__file__).resolve().parent        # .../temperature_estimation
REPO_ROOT = HERE.parent                            # .../resevoir_study   <- the main folder
DAMS_DIR  = HERE / 'air2water' / 'dams'
DATA_DIR  = REPO_ROOT / 'Data' / 'ERA5_T2M' / 'csv'
GLOBO_DIR = REPO_ROOT / 'Globolakes' / 'nc_files'
LWST_CSV  = REPO_ROOT / 'LWSTlakes.csv'

MASTER_PSO = Path(__file__).resolve().parent / 'air2water' / 'PSO.txt' #needed to copy the PSO file to each lake folder


def make_dirs(dam_id):

    #creates the folders in the right directory needed for the air2water run

    dam_dir = DAMS_DIR / str(dam_id)
    inner = dam_dir / str(dam_id)       # the per-lake subfolder air2water needs as the data lives one level deeper than the PSE and Input file i also need to make
    inner.mkdir(parents=True, exist_ok=True)
    logger.info('Directory created at: %s', dam_dir)
    return dam_dir

def write_input_txt(dam_id, air_id="era5", wat_id="globo",
                    version=3, n_run=2000):
    
    #Writes the input.txt file in the correct folder with the correct name (dam ID). change air and water ID based on the datasets used. Standard 2000 runs and 8 parameters

    dam_dir = dam_dir = DAMS_DIR / str(dam_id)
    rows = [
        ("! Input file", None),                       # header line (no value)
        (str(dam_id),  "name of the lake/subfolder"),
        (air_id,       "name/ID of the air station"),
        (wat_id,       "name/ID of the water station"),
        ("c",          "type of series: c=continuous, m=mean year"),
        ("1d",         "time resolution: 1d=daily, nw=n weeks, 1m=monthly"),
        (str(version), "version: 1=4par, 2=6par, 3=8par"),
        ("0",          "Threshold temperature for ice formation"),
        ("RMS",        "objective function: KGE, NSE, RMS"),
        ("CRN",        "numerical model: CRN, RK2, RK4, EUL"),
        ("PSO",        "PSO, FORWARD, LATHYP"),
        ("0.01",       "minimum percentage of data: 0...1"),
        (str(n_run),   "nrun"),
        ("-2",         "mineff_index"),
        ("1",          "log_flag"),
        ("0.9",        "Courant number"),
    ]
    lines = [val if comment is None else f"{val}\t\t! {comment}"
             for val, comment in rows]
    path = dam_dir / "input.txt"
    path.write_text("\n".join(lines) + "\n", encoding="cp1252")
    return path

def write_pso_txt(dam_id):

    #copies the PSO.txt file to the correct folder so that the a2w can run in that folder. 
    #The PSO setting are well tested defaults that control how the optimizer searches, not the quality of the results
    #changing them risks destabalizing convergence for no real added benefit. 
    #RECCOMMENDED DO NOT CHANGE

    dam_dir = DAMS_DIR / str(dam_id)
    dest = dam_dir / 'PSO.txt'
    shutil.copy(MASTER_PSO, dest)          # copies into dams/<id>/ next to input.txt
    logger.info('PSO.txt copied to: %s', dest)
    return dest

def load_lake_series(dam_id, quality_min=3):

    #id_v11 -> daily frame [date, t2m_degC, lswt_degC] (water NaN where no pass).

    #get metadata, we need the GLOBOLAKES ID to build the NetCDF filename
    meta = pd.read_csv(LWST_CSV)
    row = meta.loc[meta['id_v11'].astype(int) == int(dam_id)].iloc[0]
    glwd = int(row['globo_glwd_id'])

    #load air temperature. Turn date column to datetime
    air  = pd.read_csv(DATA_DIR / f't2m_{int(dam_id)}.csv', parse_dates=['date'])

    air = air.sort_values('date').reset_index(drop=True)

    #load and extract water temperature
    ds   = xr.open_dataset(GLOBO_DIR / f'LAKE{glwd:08d}-GloboLakes-L3S-LSWT-v4.0-fv01.0.nc')

    wat = (ds['lake_surface_water_temperature']
           .where(ds['quality_level'] >= quality_min)
           .mean(dim=['lat', 'lon'], skipna=True)
           .to_series()
           .subtract(273.15))
    wat.index = wat.index.normalize()          # strip any time-of-day -> midnight
    wat = wat.groupby(level=0).mean()           # collapse any same-day duplicates

    # allign water to air
    df = air.copy()
    df['lswt_degC'] = df['date'].map(wat)        # NaN where no pass
    return df


PERIODS = {'cal': (1994, 2005), 'val': (2005, 2016)} #calibration and validation periods as set in the paper. Overlap is fine as the model needs a warm up year

def write_data_file(dam_id, df, period='cal', air_id="era5", wat_id="globo"):

    #Write one cc/cv file. period in {'cal','val'} -> _cc / _cv.

    suffix = {"cal": "cc", "val": "cv"}[period]
    start_year, end_year = PERIODS[period]

    #slice to the periods year
    sub = df[(df['date'].dt.year >= start_year) &
             (df['date'].dt.year <= end_year)].copy()
    


    #safety: air must be gap-free and start on 1 Jan (model warm-up year)
    first = sub['date'].iloc[0]
    assert (first.month, first.day) == (1, 1), f"{dam_id}: must start 1 Jan"
    assert sub['t2m_degC'].notna().all(), f"{dam_id}: air has gaps"

    #format the way the air2water executable wants
    lines = []
    for _, r in sub.iterrows():
        d = r['date']
        w = r['lswt_degC']
        w_str = "-999" if pd.isna(w) else f"{w:.3f}"
        lines.append(f"{d.year}\t{d.month}\t{d.day}\t{r['t2m_degC']:.3f}\t{w_str}")

    lake_dir = DAMS_DIR / str(dam_id) / str(dam_id) #go down to the place the files should live
    path = lake_dir / f"{air_id}_{wat_id}_{suffix}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="cp1252")
    return path

import numpy as np

def air2water_param_bounds(depth, n=2):
    """
    Port of air2water's pre_processing.m + parameters.m.
    Returns an (8, 2) array: row p = parameter p, col 0 = min bound, col 1 = max bound.
    Validated: reproduces parameters_depth=147m.txt exactly. n=2 matches the MATLAB default
    (each physical quantity is sampled at min / mid / max -> 3 points).
    """

    #IMPORTANT
    # Python port of pre_processing.m / parameters.m from the air2water v2.0
    # distribution (Piccolroaz & Toffolon, 2017; depth-scaling after Piccolroaz 2016
    # and Toffolon et al. 2014). MATLAB->Python translation done with assistance from
    # a generative-AI tool (Anthropic Claude) and validated by reproducing the
    # reference output parameters_depth=147m.txt to 7 significant figures.

    g = np.linspace                              # a:step:b in MATLAB == linspace(a, b, n+1) here

    # --- parameters.m: a-priori ranges of the underlying physical quantities ---
    maxrad  = g(200, 450, n+1)                   # max net radiation
    minrad  = g(0,   250, n+1)                   # min net radiation
    rs      = g(0.04, 0.20, n+1)                 # shortwave reflectivity (albedo)
    Tr      = g(0, 30, n+1)                      # reference temperature [degC]
    aa      = 0.97 * g(0.6, 0.9, n+1)            # atmospheric emissivity factor
    bb      = 0.97                               # water emissivity
    alphac  = g(3, 15, n+1)                      # sensible-heat transfer coeff
    Kelvin0 = 273.15 + Tr
    Dalphac = g(0.1, 15, n+1)                    # seasonal amplitude of alphac
    ea      = g(5, 15, n+1)                      # vapor pressure
    Dea     = g(0.1, 10, n+1)                    # seasonal amplitude of ea
    DT      = g(0, 30, n+1)                      # seasonal air-temp amplitude
    sb      = 5.67e-8                            # Stefan-Boltzmann

    # Rad_a(i,j)=(maxrad_i - minrad_j)/2.  MATLAB's Rad_a(i) single-index = first column.
    Rad_a_full = (maxrad[:, None] - minrad[None, :]) / 2
    minRad_a, maxRad_a = Rad_a_full.min(), Rad_a_full.max()
    Rad_b_diag = minrad + maxrad / 2             # Rad_b(i,i) = minrad_i + maxrad_i/2

    # ew = range of saturation vapor pressure over the Tr range
    ew_par = 6.112 * np.exp(Tr * 17.67 / (Tr + 243.5))
    ew = np.array([max(0, ew_par.min()), ew_par.max()])

    alphae  = alphac / 0.61                      # latent-heat transfer coeff (Bowen ratio)
    Dalphae = Dalphac

    # --- pre_processing.m: depth sets the "reactive volume" and its heat capacity pp ---
    min_D = 1 + depth / 1000 * 50
    max_D = max(10, depth)
    D  = g(min_D, max_D, n+1)
    pp = 1000 * 4186 * D / 86400                 # rho*cp*D / seconds-per-day

    # temp(Tr): linearised saturation vapor pressure used inside p1
    temp = 6.112 * np.exp(Tr*17.67/(Tr+243.5)) * (1 - 17.67*243.5/(Tr+243.5)**2 * Tr)

    # --- p1: min/max of a physical expression over 7 orthogonal axes ---
    # axes: Rad_b_diag, rs, (Tr/Kelvin0/temp share one axis), aa, alphae, ea, pp
    i  = Rad_b_diag[:, None, None, None, None, None, None]
    p  = rs       [None, :, None, None, None, None, None]
    K  = Kelvin0  [None, None, :, None, None, None, None]
    Tm = Tr       [None, None, :, None, None, None, None]
    tm = temp     [None, None, :, None, None, None, None]
    j  = aa       [None, None, None, :, None, None, None]
    k  = alphae   [None, None, None, None, :, None, None]
    l  = ea       [None, None, None, None, None, :, None]
    o  = pp       [None, None, None, None, None, None, :]
    # NB the alphae*(ea - temp) sign is intentional (the .m flags it as a fix to Piccolroaz 2013)
    p1_par = ((1-p)*i + sb*K**3*(j-bb)*(273.15 - 3*Tm) + k*(l-tm)) / o
    p1 = [p1_par.min(), p1_par.max()]

    # --- p2: closed-form min/max ---
    p2 = [(4*sb*aa[0]*Kelvin0[0]**3  + alphac[0])  / pp[-1],
          (4*sb*aa[-1]*Kelvin0[-1]**3 + alphac[-1]) / pp[0]]

    # --- p3: min/max over 6 axes (aa, Kelvin0, alphac, alphae, Tr, pp) ---
    ai = aa     [:, None, None, None, None, None]
    Ko = Kelvin0[None, :, None, None, None, None]
    ap = alphac [None, None, :, None, None, None]
    aj = alphae [None, None, None, :, None, None]
    Tk = Tr     [None, None, None, None, :, None]
    pl = pp     [None, None, None, None, None, :]
    p3_par = (4*sb*ai*Ko**3*(1 - (ai-bb)/ai) + ap
              + aj*6.112*np.exp(Tk*17.67/(Tk+243.5))*17.67*243.5/(Tk+243.5)**2) / pl
    p3 = [p3_par.min(), p3_par.max()]

    # --- p4..p8 ---
    p4 = [1, 100 * depth**(-0.35)]               # depth-explicit (the dominant timescale)
    p5 = [((1-rs[-1])*minRad_a + alphae[0]*Dea[0]  + Dalphae[0]*(ea[0]-ew[-1]+Dea[0])   + Dalphac[0]*DT[0])  / pp[-1],
          ((1-rs[0])*maxRad_a  + alphae[-1]*Dea[-1] + Dalphae[-1]*(ea[-1]-ew[0]+Dea[-1]) + Dalphac[-1]*DT[-1]) / pp[0]]
    p6, p7, p8 = [0, 1], [0, 150], [0, 0.5]      # fixed (stratification / ice timescales)

    par = np.array([p1, p2, p3, p4, p5, p6, p7, p8], float)
    par[0, 1] = min(2, par[0, 1])                # cap p1 max at 2
    par[4, 0] = max(0, par[0, 0])                # p5 min <- max(0, p1 min)  (a .m quirk)
    return par


def make_parameters_txt(dam_id, depth=None):
    
    #Write the depth-derived parameters.txt into dams/<id>/<id>/.
    
    if depth is None:                            # look up Depth_avg if not passed
        meta = pd.read_csv(LWST_CSV)
        depth = float(meta.loc[meta['id_v11'].astype(int) == int(dam_id), 'Depth_avg'].iloc[0])

    par = air2water_param_bounds(depth)          # (8,2): col0 min, col1 max
    fmt = lambda col: "   ".join(f"{v:.7e}" for v in col)
    text = fmt(par[:, 0]) + "\n" + fmt(par[:, 1]) + "\n"   # row1=min, row2=max

    lake_dir = DAMS_DIR / str(dam_id) / str(dam_id)
    path = lake_dir / "parameters.txt"
    path.write_text(text, encoding="cp1252")     # \n -> CRLF on Windows
    return path


EXE = HERE / 'air2water' / 'air2water_v2.0.exe'     # the single shared binary

def run_air2water(dam_id, timeout=1800):

    #Run the exe for one dam. It reads input.txt/PSO.txt from cwd=dam_dir and
    #writes results into dams/<id>/<id>/output_3/.

    dam_dir = DAMS_DIR / str(dam_id)
    if not EXE.exists():
        raise FileNotFoundError(f"executable not found: {EXE}")
    proc = subprocess.run(
        [str(EXE)],
        cwd=str(dam_dir),            # KEY: exe opens input.txt from its working dir
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:         # surface the Fortran's own error text
        raise RuntimeError(f"air2water failed for {dam_id} (exit {proc.returncode})\n"
                           f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    logger.info("air2water finished for %s", dam_id)
    return proc

def read_output(dam_id, model_version=3):

    #Parse the air2water outputs -> dict (params, cal/val efficiency, series).
    #Also writes dams/<id>/params.json for reuse without re-running PSO.

    out_dir = DAMS_DIR / str(dam_id) / str(dam_id) / f"output_{model_version}"

    def _find(prefix):                                   # filenames carry a run-stamp
        hits = sorted(out_dir.glob(f"{prefix}_*.out"), key=lambda p: p.stat().st_mtime)
        if not hits:
            raise FileNotFoundError(f"no {prefix}_*.out in {out_dir}")
        return hits[-1]                                   # newest if several

    # 1_PSO: line1 = 8 best params, line2 = calibration eff, line3 = validation eff
    raw = _find("1").read_text(encoding="cp1252").strip().splitlines()
    params  = [float(x) for x in raw[0].split()]
    eff_cal = float(raw[1].split()[0]) if len(raw) > 1 else np.nan
    eff_val = float(raw[2].split()[0]) if len(raw) > 2 else np.nan

    # 2_/3_PSO: year month day obs_air obs_water sim_water [obs_agg sim_agg]
    def _series(prefix):
        cols = ["year","month","day","obs_air","obs_water",
                "sim_water","obs_water_agg","sim_water_agg"]
        df = pd.read_csv(_find(prefix), sep=r"\s+", header=None, encoding="cp1252")
        df.columns = cols[:df.shape[1]]                  # handle 6- or 8-column output
        df = df[df["year"] != -999].copy()               # drop replicated warm-up year
        df["date"] = pd.to_datetime(df[["year","month","day"]])
        df = df.replace(-999, np.nan)                    # missing observed -> NaN
        return df

    result = {"dam_id": int(dam_id), "params": params,
              "eff_cal": eff_cal, "eff_val": eff_val,
              "cal_series": _series("2"), "val_series": _series("3")}

    (DAMS_DIR / str(dam_id) / "params.json").write_text(      # small, reusable record
        json.dumps({k: result[k] for k in ("dam_id","params","eff_cal","eff_val")},
                   indent=2), encoding="utf-8")
    return result