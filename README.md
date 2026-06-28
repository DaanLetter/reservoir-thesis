# Reservoir operations and lake-temperature modelling in PCR-GLOBWB 2

Code accompanying the MSc thesis *"Quantity before quality: validating data-derived
reservoir operations and a low-input lake-temperature model in PCR-GLOBWB 2"*
(D. Letter, Utrecht University).

The project has two components:

1. **Point model**: a standalone reconstruction of the STARFIT data-derived
   reservoir operating scheme as implemented in PCR-GLOBWB 2, validated against
   observed storage and outflow for 357 CONUS reservoirs (ResOpsUS). Introduces
   the **inflow-to-band ratio (IBR)** as an a-priori predictor of where the scheme
   fails.
2. **air2water**: calibration and validation of the lumped air2water model for
   daily lake surface water temperature from air temperature alone, across 56
   globally distributed, reservoir-dominated lakes.

## Repository structure

```
PointModelDaan.py            Point model (STARFIT reconstruction, mass balance, operating rules)
resevoir_functions_Daan.py   Helper functions used by the point model
Run_RF_Model.py              Random-forest model for STARFIT operating bounds
pointmodelvalidation.ipynb   Main analysis & validation notebook (metrics, IBR, figures)
download_era5_t2m.py         Fetch ERA5 2 m air temperature (Open-Meteo API)
download_globolakes.py       Fetch GloboLakes LSWT
extract_t2m_to_csv.py        Post-process ERA5 forcing to CSV
temperature_estimation/
    air2waterwrapper.py      Python wrapper around the air2water executable
    make_results_table.py    Build the air2water results tables
    air2watertest.ipynb      air2water calibration/validation analysis
    *.csv / *.tex            Result tables (KGE, validation metrics)
```

## Getting started

```bash
pip install -r requirements.txt
# plus pcraster:  conda install -c conda-forge pcraster
```

Notebooks were committed with outputs cleared; run them top to bottom to reproduce
the figures and tables.

## Data availability

Input datasets are **not** included (size / third-party licensing). Obtain them from:

- **ResOpsUS** (observed reservoir operations) — Steyaert et al. (2022), https://doi.org/10.1038/s41597-022-01134-7
- **GeoDAR / Global Dam Watch** (dam inventory) — https://www.globaldamwatch.org/geodar
- **GloboLakes** LSWT — Carrea & Merchant (2019), CEDA archive
- **ERA5** 2 m air temperature — retrieved per lake via the Open-Meteo historical API (`download_era5_t2m.py`)
- **PCR-GLOBWB 2** simulated outputs — Sutanudjaja et al. (2018), https://doi.org/10.5194/gmd-11-2429-2018
- **air2water** model (the compiled executable + pre/post-processing, ~3 GB) — Piccolroaz et al., https://github.com/spiccolroaz/air2water

## Citation

If you use this code, please cite the thesis. The underlying datasets and models
should be cited via the references above.
