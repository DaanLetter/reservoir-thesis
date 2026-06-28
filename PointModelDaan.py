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
import resevoir_functions_Daan as rf

#set up logging and correct file path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.join(os.getcwd(), 'Data', 'POINTDATA') + os.sep

def load_nc(file_path):
    """Open a NetCDF file and log success or failure."""
    try:
        ds = xr.open_dataset(file_path)
        logger.info("Loaded %s", os.path.basename(file_path))
        return ds
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
        return None


def get_data(dam_id, plot=True):

    #Dam identification, Bridge links GeoDAR object IDs to GRanD IDs and stores the coordinates, design capacity and primary use type
    bridge  = pd.read_excel('Data/GeoDAR_PCRGLOBWB.xlsx')
    dam_row = bridge[bridge['geodar_v11'] == dam_id].iloc[0]

    latitude   =  dam_row['geodar_lat']     #pour-point latitude  [°N]
    longitude  =  dam_row['geodar_long']    #pour-point longitude [°E]
    capacity_design = dam_row['capacity'] * 1e6 #design capacity: million m³ → m³

    # STARFIT use type: starfit_release() routes through Irrigation_release()
    # only when use == 'Irrigation' or 'Water Supply'; all other values follow the Hydroelectricity path.
    # The bridge table 'use' column uses this exact string for hydroelectric dams.
    use_type = str(dam_row['use'])

    logger.info("Dam %d: lat=%.4f, lon=%.4f, design capacity=%.3e m³, use=%s",
                dam_id, latitude, longitude, capacity_design, use_type)

    #Grid-cell area
    # PCR-GLOBWB runs on a 5-arcminute regular lat/lon grid.
    # Cell area shrinks with latitude (cosine factor). We need this to convert
    # water depth variables (m/month) to volume (m³/month).
    # Formula: (cell side in radians)² × R_earth² × cos(lat)
    cell_area_m2 = (5/60 * np.pi/180)**2 * 6_371_000**2 * np.cos(np.radians(latitude))
    logger.info("Grid-cell area at dam: %.3e m²", cell_area_m2)

    #Choose the PCR output directory for this dam
    # Two disjoint PCR runs: original western/central (Data/POINTDATA) and the
    # Mississippi basin (Data/M44/netcdf). A dam lives in exactly one of them, so we
    # test which run has reservoir output at this cell (the domains overlap spatially
    # but not in their reservoirs, so a longitude cutoff would misroute overlap dams)
    # Static files (capacity, id-map, RF bounds) are GLOBAL → always POINTDATA.
    out_dir  = os.path.join('Data', 'POINTDATA') + os.sep
    east_dir = os.path.join('Data', 'M44', 'netcdf') + os.sep 
    _test = xr.open_dataset(out_dir + 'sos_resout_final_monthAvg_output.nc')['sos_reservoir_outflow_end']  # <
    if not bool(np.isfinite(_test.sel(lat=latitude, lon=longitude, method='nearest')).any()):              # <
        out_dir = east_dir                                                            # <
        logger.info("Dam %d not in western run → using Mississippi (M44) output", dam_id)  # <

    #Dam Capacity
    #more robust way of computing capacity
    #PCR-GLOBWB parameter file for 5 arcmin
    #query using lat/lon so no bridge table needed.
    #Also stores resevoir type as waterBodyTyp 
    #capacity is stored as million m3, convert to m3
    ds_params = load_nc(file_path + 'lakes_and_reservoirs_05min_geodar_2023.nc')
    res_params = ds_params.sel(lat=latitude, lon=longitude, method='nearest')
    capacity = float(res_params['resMaxCapInp'].item()) * 1e6
    logger.info(f"capacity of dam {dam_id} : {capacity} m3")

    #1. Reservoir outflow  [m³/month]
    # File stores the monthly average of the daily outflow total.
    # The unit label in the file ("m3.s-1") is wrong; values are m³/day.
    # Multiply by days_in_month to recover the monthly total volume.
    ds_dis = load_nc(out_dir + 'sos_resout_final_monthAvg_output.nc')
    df_dis = (ds_dis
            .sel(lat=latitude, lon=longitude, method='nearest')
            .to_dataframe()
            .reset_index()[['time', 'sos_reservoir_outflow_end']])
    df_dis['discharge_m3'] = df_dis['sos_reservoir_outflow_end'] * df_dis['time'].dt.days_in_month
    df_dis = df_dis[['time', 'discharge_m3']]

    #2. Reservoir storage  [m³]
    # End-of-month storage volume; already in m³, no conversion needed.
    # We use the monthEnd file (not dailyTot, which is corrupted at this location).
    ds_stor = load_nc(out_dir + 'waterBodyStorage_monthEnd_output.nc')
    df_stor = (ds_stor
            .sel(lat=latitude, lon=longitude, method='nearest')
            .to_dataframe()
            .reset_index()[['time', 'lake_and_reservoir_storage']]
            .rename(columns={'lake_and_reservoir_storage': 'storage_m3'}))

    #3. Reservoir inflow  [m³/month]
    # The daily file gives actual inflow to the reservoir in m³/s.
    # Multiplying each daily value by 86 400 (s/day) gives m³/day.
    # Summing within each calendar month gives m³/month with real monthly variability,
    # rather than distributing an annual figure uniformly across months.
    ds_inf = load_nc(out_dir + 'lake_and_reservoir_inflow_dailyTot_output.nc')
    df_inf_daily = (ds_inf
                    .sel(lat=latitude, lon=longitude, method='nearest')
                    .to_dataframe()
                    .reset_index()[['time', 'lake_and_reservoir_inflow']])
    df_inf_daily['inflow_m3_day'] = df_inf_daily['lake_and_reservoir_inflow'] * 86_400
    df_inf_daily['month_key']     = df_inf_daily['time'].dt.to_period('M')
    df_inf = (df_inf_daily
            .groupby('month_key')['inflow_m3_day']
            .sum()
            .reset_index()
            .rename(columns={'inflow_m3_day': 'inflow_m3'}))

    #4. Environmental flow  [m³/month]
    # monthTot file already stores the full monthly volume — no conversion needed.
    ds_ef = load_nc(out_dir + 'soswater_env_flow_monthTot_output.nc')
    df_ef = (ds_ef
            .sel(lat=latitude, lon=longitude, method='nearest')
            .to_dataframe()
            .reset_index()[['time', 'soswater_env_flow']]
            .rename(columns={'soswater_env_flow': 'env_flow_m3'}))

    #5. Water demand  [m³/month]
    # Each sector file stores water withdrawal as a depth (m/month) over the grid cell.
    # Multiplying by cell_area_m2 converts depth to volume (m³/month).
    # We sum the five PCR-GLOBWB sectors to get total demand on this cell.
    df_catch_wd = load_nc(file_path + '250_id_map_geodar_final 1.nc')
    mask_wd = df_catch_wd['area'] == dam_id

    withdrawal_sectors = {
        'domestic':     ('domesticWaterWithdrawal_monthTot_output.nc',    'domestic_water_withdrawal'),
        'industry':     ('industryWaterWithdrawal_monthTot_output.nc',    'industry_water_withdrawal'),
        'irr_nonpaddy': ('irrNonPaddyWaterWithdrawal_monthTot_output.nc', 'non_paddy_irrigation_withdrawal'),
        'irr_paddy':    ('irrPaddyWaterWithdrawal_monthTot_output.nc',    'paddy_irrigation_withdrawal'),
        'livestock':    ('livestockWaterWithdrawal_monthTot_output.nc',   'livestock_water_withdrawal'),
        'abstraction': ('surfaceWaterAbstraction_monthTot_output.nc', 'surface_water_abstraction')
    }

    sector_series = {}
    for sector, (fname, var) in withdrawal_sectors.items():
        ds_wd = load_nc(out_dir + fname)
        ts_wd = ds_wd[var].where(mask_wd).sum(dim=['lat', 'lon']) * cell_area_m2
        sector_series[f'{sector}_m3'] = ts_wd.values   # numpy array, length 540

    df_withdrawal = pd.DataFrame(sector_series,
                                index=pd.to_datetime(ds_wd['time'].values))
    df_withdrawal.index.name = 'time'
    df_withdrawal['total_m3'] = df_withdrawal.sum(axis=1)
    df_withdrawal = df_withdrawal.reset_index()

    logger.info("Withdrawal DataFrame: %d rows x %d cols", len(df_withdrawal), len(df_withdrawal.columns))
    df_demand = df_withdrawal[['time', 'total_m3']].copy().rename(columns={'total_m3' : 'demand_m3'})
    if plot == True:
        plt.figure(figsize=(10, 6))
        df_catch_wd['area'].where(df_catch_wd['area'] == dam_id).sel(
            lat=slice(latitude+5, latitude-5), lon=slice(longitude-5, longitude+5)
        ).plot(cmap='Reds')
        plt.scatter(longitude, latitude, color='blue', zorder=5, label=dam_id)
        plt.legend()
        plt.title(f'250km command area for {dam_id}')
        plt.show()


    #6. STARFIT flood & conservation bounds  [m³]
    # The RF bounds directory has 366 files (one per day-of-year, all labelled 2000-*).
    # Each file stores STARFIT flood and conservation storage targets as a percentage
    # of capacity for every grid cell.
    # We average the daily percentages within each calendar month to get a smooth
    # monthly bound, then convert from % to m³ by multiplying by capacity/100.

    rf_dir = os.path.join('Data', 'POINTDATA', '10_param_RF_bounds_final')
    monthly_flood = defaultdict(list)   # month → list of daily flood % values
    monthly_cons  = defaultdict(list)   # month → list of daily conservation % values

    for fname in os.listdir(rf_dir):
        if not fname.endswith('.nc'):
            continue
        month = int(fname.replace('.nc', '').split('-')[1])   # filename: "2000-<m>-<d>.nc"
        ds_b  = xr.open_dataset(os.path.join(rf_dir, fname))
        row_b = ds_b.sel(latitude=latitude, longitude=longitude, method='nearest')
        monthly_flood[month].append(float(row_b['flood'].values[0]))
        monthly_cons[month].append(float(row_b['conservation'].values[0]))

    df_bounds = pd.DataFrame({
        'month':            sorted(monthly_flood.keys()),
        'flood_m3':         [np.mean(monthly_flood[m]) / 100 * capacity for m in sorted(monthly_flood.keys())],
        'conservation_m3':  [np.mean(monthly_cons[m])  / 100 * capacity for m in sorted(monthly_cons.keys())],
    })

    #7. Merge into one dataframe 
    # Start from discharge (sets the monthly time index), then left-join everything.
    # Inflow comes from a daily file so it needs a Period key to join on month.
    df = df_dis.copy()
    df = pd.merge(df, df_stor,   on='time')
    df['month_key'] = df['time'].dt.to_period('M')
    df = pd.merge(df, df_inf,    on='month_key')            # inflow keyed by period
    df = pd.merge(df, df_ef,     on='time')
    df = pd.merge(df, df_demand, on='time')
    df['month'] = df['time'].dt.month
    df = pd.merge(df, df_bounds, on='month')                # bounds repeat each year by month
    df = df.drop(columns=['month_key']).sort_values('time').reset_index(drop=True)

    logger.info("Final dataframe: %d rows × %d cols, %s → %s",
                len(df), len(df.columns),
                df['time'].min().date(), df['time'].max().date())



    return df, capacity, use_type


def run_pointmodel(dam_id, df, capacity, use_type, plot=True):
    #  STARFIT point-model simulation 
    # We step forward month-by-month from 1979-01 to 2023-12 using the cleaned
    # dataframe built above.  Every volume is already in m³/month, so
    # no unit conversion is needed inside the loop.
    #
    # Column reference (all m³ unless noted):
    #   discharge_m3      – observed monthly outflow (PCR-GLOBWB SOS)
    #   inflow_m3         – monthly inflow aggregated from daily totals
    #   env_flow_m3       – minimum environmental flow constraint
    #   demand_m3         – total sectoral water withdrawal on the grid cell
    #   flood_m3          – STARFIT upper storage bound (flood zone)
    #   conservation_m3   – STARFIT lower storage bound (conservation zone)

    df_simulation = df.copy()

    #Monthly average outflow
    monthly_avg_outflow = df.groupby('month')['discharge_m3'].mean()

    # Initialise output columns so the DataFrame has the right shape before the loop.
    # operational_release stores the pre-spillage release so it can be carried forward
    # as prev_release without inflating the carry-forward with one-off spill events.
    for col in ['modelled_storage', 'model_release', 'operational_release',
                'model_current_storage', 'reduction_factor_model']:
        df_simulation[col] = 0.0

    logger.info('Long-run average monthly outflow: %.3e m³/month (no inter-month variability)', df['discharge_m3'].mean())

    for t, row in df_simulation.iterrows():

        #  Storage state 
        # Start at the observed PCR storage for the first timestep.
        # Every subsequent step carries forward the previous end-of-month storage.
        current_storage = df_simulation['storage_m3'][0] if t == 0 else df_simulation.loc[t - 1, 'modelled_storage']

        #  Previous release 
        # irrigation_release() and Hydroelectricity_release() both compare current_release
        # to demand to decide which branch to take, so we must carry it forward.
        # Use operational_release (pre-spillage) so a one-off spill event does not
        # lock in an inflated release for subsequent drought months.
        prev_release = df_simulation.loc[t - 1, 'operational_release'] if t > 0 else 0.0

        #  Monthly storage bounds [m³] 
        max_storage = row['flood_m3']         # upper bound: do not store above this
        min_storage = row['conservation_m3']  # lower bound: do not draw below this

        #  Reduction factor 
        # RF ∈ [0, 1]: how full the active storage zone currently is.
        # RF = 0 → at or below conservation level; RF = 1 → at or above flood level.
        reduction_factor_val = rf.reduction_factor(
            current_storage=current_storage,
            min_storage=min_storage,
            max_storage=max_storage,
            storage_capacity=capacity
        )

        #average outflow for this calendar month
        avg_outflow_m3 = monthly_avg_outflow[row['month']]

        #STARFIT release decision [m³/month]
        # avg_outflow : long-run mean monthly outflow used to set the baseline
        #               release proportional to current filling state.
        # env_flow    : hard lower bound on release (ecological minimum).
        # demand      : sectoral water-withdrawal target for this grid cell.
        # use_type    : derived from bridge table 'use' column for this dam.
        release = rf.starfit_release(
            current_storage=current_storage,
            storage_capacity=capacity,
            max_storage=max_storage,
            min_storage=min_storage,
            avg_outflow=avg_outflow_m3,
            env_flow=row['env_flow_m3'],
            demand=avg_outflow_m3,
            current_release=prev_release,
            use=use_type,
        )

        #  Water balance [m³] 
        # new_storage = S_t + inflow - release
        # Capped at storage_capacity: any surplus above full pool is spilled.
        new_storage_val = rf.new_storage(
            release=release,
            current_storage=current_storage,
            inflow=row['inflow_m3'],
            storage_capacity=capacity,
        )
        
        #  Spillage: excess inflow above full capacity released in this timestep 
        # If current_storage + inflow - release > capacity, new_storage() already
        # caps storage at capacity, but the surplus water would be deleted.
        # We compute it explicitly and add it to model_release so mass is conserved.
        uncapped = current_storage + row['inflow_m3'] - release
        spillage = max(0.0, uncapped - capacity)

        #  Write results back to the simulation dataframe 
        df_simulation.loc[t, 'operational_release']    = release          # pre-spillage
        df_simulation.loc[t, 'model_release']          = release + spillage
        df_simulation.loc[t, 'model_current_storage']  = current_storage
        df_simulation.loc[t, 'modelled_storage']       = new_storage_val
        df_simulation.loc[t, 'reduction_factor_model'] = reduction_factor_val

    if plot == True:
        plt.hist(df_simulation['reduction_factor_model'])
        plt.title('reduction factor values')
        plt.xlabel('rf')
        plt.ylabel('counts')
        plt.show()

        #  Time-series comparison: PCR-GLOBWB storage vs modelled 
        # Normalise all series by capacity so they are expressed as a fraction of the
        # design capacity (makes the flood and conservation bounds easy to read off).
        file_path = os.path.join(os.getcwd(), 'images') + os.sep

        plot_df = df_simulation[['time', 'storage_m3', 'modelled_storage',
                                'model_release', 'inflow_m3', 'conservation_m3', 'flood_m3']].copy()

        # Normalise by design capacity so every series is dimensionless [0, 1]
        plot_df['PCR storage']      = plot_df['storage_m3']       / capacity
        plot_df['modelled storage'] = plot_df['modelled_storage'] / capacity
        plot_df['modelled release'] = plot_df['model_release']    / capacity
        plot_df['inflow']           = plot_df['inflow_m3']        / capacity
        plot_df['conservation']     = plot_df['conservation_m3'] / capacity
        plot_df['flood']            = plot_df['flood_m3']        / capacity
        plot_long = plot_df.melt(
            id_vars='time',
            value_vars=['PCR storage', 'modelled storage', 'modelled release', 'inflow', 'conservation', 'flood'],
            var_name='series',
            value_name='fraction of capacity',
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=plot_long, x='time', y='fraction of capacity',
                    hue='series', ax=ax)
        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8, label='full capacity')
        ax.set_title(f'Reservoir storage — PCR-GLOBWB vs STARFIT modelled (dam id: {dam_id})')
        ax.set_xlabel('Time')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(file_path + f'StorageCurve{dam_id}.pdf')
        plt.show()

        file_path = os.path.join(os.getcwd(), 'Data', 'POINTDATA') + os.sep    


    logger.info('Simulation complete: %d timesteps, %s → %s',
                len(df_simulation),
                df_simulation['time'].iloc[0].date(),
                df_simulation['time'].iloc[-1].date())
    return df_simulation









def main():
    dam_id = 17393
    df, capacity, use_type = get_data(dam_id)

    df_simulation = run_pointmodel(dam_id, df, capacity, use_type, plot=True)

if __name__ == '__main__':
    main()