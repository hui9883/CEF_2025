# CEF_code — Energy system whole-system cost-emission impact: trade-off or co-benefits

This project runs a full workflow on PyPSA networks in four stages:

1) **Load data & save preprocessed networks**  
2) **Optimize** (minimize cost or CO₂)  
3) **Analyze** (KPIs, CSV exports, MEF / cycle analysis)  
4) **Sensitivity analyses**

- **Data** live in `./data/`  
- **Helper functions** live in `./function/`  
- **Main runnable scripts are in the project root**  
- **Optimization results** go to `./results/…`  
- **Sensitivity outputs** go to subfolders in `./sensitive_analyse/`

---

## Folder map (key paths)

```
CEF_code/
├─ data/                         # input networks & excel helpers
│  ├─ network_2023.nc
│  ├─ network_2030.nc
│  ├─ network_eu_2023.nc
│  ├─ network_eu_2030.nc
│  └─ geo_data/ …
├─ function/                     # domain utilities (MEF, rain_flow, etc.)
│  ├─ mef_energy_log_eu.py
│  ├─ mef_energy_log_uk.py
│  ├─ mef_national_energy_log_eu.py
│  ├─ mef_national_energy_log_uk.py
│  └─ cyclic_data_preprocessing.py
├─ results/                      # created by optimization scripts
│  ├─ eu_min_CO2_2023/
│  ├─ eu_min_CO2_2030/
│  ├─ eu_min_cost_2023/
│  ├─ eu_min_cost_2030/
│  ├─ min_CO2_2023/
│  ├─ min_CO2_2030/
│  ├─ min_cost_2023/
│  └─ min_cost_2030/
├─ sensitive_analyse/            # sensitivity outputs go here
│  ├─ ldes_duration/
│  ├─ monte_carlo/
│  └─ round_trip_efficiency/
└─ (root scripts listed below)
```

> Windows Explorer may hide “.py” — all script names below are Python files.

---

## Environment

```bash
# Create / activate a venv or conda env first, then:
pip install -r requirements.txt
```

- **Solver**: scripts support HiGHS (`highspy`) or Gurobi (`gurobipy`).  
  Set `solver_name` and options inside the optimization scripts.
- For headless servers use `MPLBACKEND=Agg` (or change `TkAgg` to `Agg`).

---

## Run order (root scripts)

### 1) Load data & save preprocessed networks
```bash
python dataload_and_save_eu.py
python dataload_and_save_uk.py
```

### 2) Optimization (writes solved networks into `./results/…`)
```bash
python optimisation_eu.py   # EU: min_cost &/or min_CO2 for 2023/2030
python optimisation_uk.py   # UK: min_cost &/or min_CO2 for 2023/2030
```

### 3) Analysis (CSV exports, MEF / cycle KPIs)
```bash
python mef_energy_analysis_eu.py
python mef_energy_analysis_uk.py
```
Typical basecase artifact for sensitivity:
```
results/min_CO2_2023/analysis_output/national_cycle_output.pkl
```

### 4) Sensitivity analyses (outputs under `./sensitive_analyse/…`)
```bash
python sensitive_analyse_ldes_duration.py     # writes to sensitive_analyse/ldes_duration/
python sensitive_analyse_uk_monte_carlo.py    # writes to sensitive_analyse/monte_carlo/
python sensitive_analyse_RTE.py               # writes to its own subfolder
```

---

## Where to find outputs

- **Optimization**: `./results/<objective>_<year>/`  
  e.g., `results/min_CO2_2023/network_ff_constrained_time.nc` plus CSVs.
- **Analysis**: CSV summaries and `analysis_output/national_cycle_output.pkl`
  inside the corresponding `results/...` folder.
- **Sensitivity**: each script creates its own subfolder under
  `./sensitive_analyse/` (e.g., `ldes_duration/`, `monte_carlo/`, `round_trip_efficiency/`)
  containing per-iteration CSVs, pickles, and figures.

---

## Tips & troubleshooting

- **Gurobi missing** → switch to HiGHS in the scripts or install Gurobi with a license.  
- **NetCDF write errors** → install at least one backend (`netCDF4` or `h5netcdf`).  
- **Matplotlib backend errors** → use `Agg` on servers without a GUI.  
- **Long Windows paths** → enable long path support or shorten directory names.

---

## Quick recap (all in order)

```bash
pip install -r requirements.txt
python dataload_and_save_eu.py
python dataload_and_save_uk.py
python optimisation_eu.py
python optimisation_uk.py
python mef_energy_analysis_eu.py
python mef_energy_analysis_uk.py
python sensitive_analyse_ldes_duration.py
python sensitive_analyse_uk_monte_carlo.py
python sensitive_analyse_RTE.py
```

That’s it — run from the project **root**, and find results under `results/` and sensitivity outputs under `sensitive_analyse/`.
