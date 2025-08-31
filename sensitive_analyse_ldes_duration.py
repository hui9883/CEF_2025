# -*- coding: utf-8 -*-
"""
LDES duration sensitivity analysis.
All outputs are stored under: ./sensitive_analyse/ldes_duration/
Basecase metrics are loaded from: results/min_CO2_2023/analysis_output/national_cycle_output.pkl
"""

import os
import pickle
import datetime
import shutil
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use a GUI backend suitable for your environment
import matplotlib.pyplot as plt
import pypsa
from function.mef_national_energy_log_uk import national_cycle_analysis
from function.cyclic_data_preprocessing import (
    cycle_extraction,
    redistribute_generation,
    modify_and_redistribute,
)

# ----------------------------- Paths & setup -----------------------------
# Root folder for this analysis
ROOT_DIR = "./sensitive_analyse"
SUB_DIR = os.path.join(ROOT_DIR, "ldes_duration")
os.makedirs(SUB_DIR, exist_ok=True)

# Subfolders
NETWORK_DIR = os.path.join(SUB_DIR, "network_files")       # solved networks per LDES duration
ITER_DIR_ROOT = os.path.join(SUB_DIR, "iteration_files")    # per-iteration CSVs & pkl outputs
FIG_DIR = os.path.join(SUB_DIR, "figures")                  # plots
os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(ITER_DIR_ROOT, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Basecase pickle produced by your min_CO2_2023 run (OS-independent path)
BASECASE_PKL_PATH = os.path.join("results", "min_CO2_2023", "analysis_output", "national_cycle_output.pkl")

# Input baseline network for sensitivity (unsolved template)
ORIGINAL_NETWORK_PATH = "./data/network_2023.nc"

# -------------------------- Parameter sampling --------------------------
# LDES duration in days (snap interval to force zero SOC on LDES)
ldes_duration_days = [1, 2, 3, 4, 5, 7, 14, 30, 61, 92, 183, 365]
#
# # --------------------- Part 1: Optimize & save networks -----------------
# results_summary = []
#
# for d in ldes_duration_days:
#     # Load a fresh network each iteration
#     network = pypsa.Network(ORIGINAL_NETWORK_PATH)
#
#     # Identify Stores by naming convention
#     battery_ids = [sid for sid in network.stores.index if "battery" in sid.lower()]
#     ldes_ids = [sid for sid in network.stores.index if "other" in sid.lower()]
#
#     try:
#         # Build the optimization model
#         mdl = network.optimize.create_model()
#
#         # Handle store energy variable and available snapshots as Timestamps (avoid string indexing)
#         store_e = mdl.variables["Store-e"]
#         snaps = pd.DatetimeIndex(network.snapshots)
#
#         # Battery: force end-of-day (23:00) zero SOC for each day in 2030, only if those snapshots exist
#         snaps_2030_23 = [ts for ts in snaps if ts.year == 2030 and ts.hour == 23]
#         if battery_ids and snaps_2030_23:
#             eod_zero = store_e.sel(snapshot=snaps_2030_23, Store=battery_ids) == 0
#             mdl.add_constraints(eod_zero, name="EndOfDayZeroBattery")
#
#         # LDES: force zero SOC every d days at 23:00 in 2030, only if those snapshots exist
#         if ldes_ids:
#             periodic_targets = pd.date_range(
#                 start=pd.Timestamp("2030-01-01 23:00:00"),
#                 end=pd.Timestamp("2030-12-31 23:00:00"),
#                 freq=f"{d}D",
#             )
#             periodic_snaps = [ts for ts in periodic_targets if ts in snaps]
#             if periodic_snaps:
#                 ldes_zero = store_e.sel(snapshot=periodic_snaps, Store=ldes_ids) == 0
#                 mdl.add_constraints(ldes_zero, name="PeriodicZeroLDES")
#
#         # All stores: force zero SOC at the end of 2030 (only if that snapshot exists)
#         end_2030 = pd.Timestamp(2030, 12, 31, 23)
#         if end_2030 in snaps:
#             final_zero = store_e.sel(snapshot=[end_2030], Store=list(network.stores.index)) == 0
#             mdl.add_constraints(final_zero, name="FinalZeroAllStores")
#
#         # Objective: minimize emissions with a tiny cost tie-breaker
#         Gen_p = mdl.variables["Generator-p"]
#         gen_dim = Gen_p.coords["Generator"].to_index()
#         co2 = network.generators["co2_emissions"].reindex(gen_dim, fill_value=0)
#         mc = network.generators["marginal_cost"].reindex(gen_dim, fill_value=0)
#         aligned_em = co2 + 1e-6 * mc
#         mdl.objective = (Gen_p * aligned_em).sum(["snapshot", "Generator"])
#
#         # Solve with a time limit and return status for downstream checks
#         status, _ = network.optimize.solve_model(
#             solver_name="gurobi",
#             solver_options={"Threads": 1},
#         )
#
#         if status == "ok":
#             # KPIs (generator-based, consistent with this objective)
#             gen_profiles = network.generators_t.p
#             total_cost = (
#                 gen_profiles * network.generators["marginal_cost"]
#                 .reindex(gen_profiles.columns)
#                 .fillna(0)
#             ).sum().sum()
#             total_emissions = (
#                 gen_profiles * network.generators["co2_emissions"]
#                 .reindex(gen_profiles.columns)
#                 .fillna(0)
#             ).sum().sum()
#
#             # Save solved network
#             save_path = os.path.join(NETWORK_DIR, f"network_mc_{d:03d}.nc")
#             network.export_to_netcdf(save_path)
#
#             results_summary.append({
#                 "LDES duration (day)": d,
#                 "success": True,
#                 "solve_status": status,
#                 "total_cost": float(total_cost),
#                 "total_emissions": float(total_emissions),
#                 "network_file": save_path,
#             })
#         else:
#             results_summary.append({
#                 "LDES duration (day)": d,
#                 "success": False,
#                 "solve_status": status,
#                 "total_cost": None,
#                 "total_emissions": None,
#                 "network_file": "",
#             })
#
#     except Exception as e:
#         results_summary.append({
#             "LDES duration (day)": d,
#             "success": False,
#             "solve_status": "exception",
#             "total_cost": None,
#             "total_emissions": None,
#             "network_file": "",
#             "error": str(e),
#         })
#
# # Save run summary CSV
# summary_csv = os.path.join(SUB_DIR, "summary_results.csv")
# pd.DataFrame(results_summary).to_csv(summary_csv, index=False)
# Path to your solved base NetCDF (adjust if your base lives elsewhere)
# BASECASE_NC_PATH = os.path.join("results", "min_CO2_2023", "network_ff_constrained_time.nc")
# # Which duration to inject as base
# INJECT_DAY = 365
# # Target file path that Part 2 expects
# target_nc_365 = os.path.join(NETWORK_DIR, f"network_mc_{INJECT_DAY:03d}.nc")
# os.makedirs(NETWORK_DIR, exist_ok=True)
# shutil.copy(BASECASE_NC_PATH, target_nc_365)
# print(f"[Insert] Basecase network created -> {target_nc_365} for the {INJECT_DAY}-day case.")
#
#
# # --------------- Part 2: Export & post-process each iteration -----------
# output_summary = []
#
# for d in ldes_duration_days:
#     network_file = os.path.join(NETWORK_DIR, f"network_mc_{d:03d}.nc")
#     if not os.path.isfile(network_file):
#         print(f"[Iteration {d:03d}] Network file not found: {network_file}. Skipping.")
#         continue
#
#     # Per-iteration folder
#     iter_dir = os.path.join(ITER_DIR_ROOT, f"iteration_{d:03d}")
#     os.makedirs(iter_dir, exist_ok=True)
#     save_dir = iter_dir
#
#     # Load solved network
#     network = pypsa.Network(network_file)
#
#     # Basic exports
#     network.loads_t.p.to_csv(os.path.join(save_dir, f"demand_p_{d:03d}.csv"), header=True)
#     network.links_t.p1.to_csv(os.path.join(save_dir, f"links_p1_results_{d:03d}.csv"), header=True)
#     network.stores_t.e.to_csv(os.path.join(save_dir, f"stores_e_{d:03d}.csv"), header=True)
#
#     # Storage energy aggregated by carrier
#     store_by_carrier = network.stores_t.e.T.groupby(network.stores.carrier).sum().T
#     store_by_carrier.to_csv(os.path.join(save_dir, f"store_e_carrier_results_{d:03d}.csv"), header=True)
#
#     # Generation aggregated by carrier
#     p_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
#     p_by_carrier.to_csv(os.path.join(save_dir, f"gen_p_carrier_results_{d:03d}.csv"), header=True)
#
#     # Dynamic generator max output and remaining headroom
#     p_max_pu_full = network.generators_t.p_max_pu.reindex(columns=network.generators.index)
#     missing_generators = network.generators.index.difference(network.generators_t.p_max_pu.columns)
#     for gen in missing_generators:
#         p_max_pu_full[gen] = network.generators.at[gen, "p_max_pu"]
#     snapshot_max_output = p_max_pu_full.multiply(network.generators["p_nom"], axis=1)
#
#     network.generators_t["max_output"] = snapshot_max_output
#     network.generators_t["remain_output"] = (
#         (network.generators_t["max_output"] - network.generators_t.p)
#         * network.generators.get("efficiency", pd.Series(1.0, index=network.generators.index))
#           .reindex(network.generators_t.p.columns)
#           .fillna(1.0)
#     )
#
#     # Remaining headroom aggregated by bus+carrier
#     rem = network.generators_t["remain_output"].T
#     rem.index = rem.index.astype(str)
#     rem["bus"] = rem.index.map(network.generators["bus"])
#     rem["carrier"] = rem.index.map(network.generators["carrier"])
#     rem["bus_carrier"] = rem["bus"] + "_" + rem["carrier"]
#     p_by_bus_carrier = (
#         rem.groupby("bus_carrier").sum()
#            .drop(columns=["bus", "carrier"], errors="ignore")
#            .T
#     )
#     p_by_bus_carrier.to_csv(os.path.join(save_dir, f"p_by_bus_carrier_{d:03d}.csv"), index=True)
#
#     # Remaining headroom aggregated by carrier
#     re_p_by_carrier = (
#         network.generators_t["remain_output"].T
#         .groupby(network.generators.carrier).sum().T
#     )
#     re_p_by_carrier.to_csv(os.path.join(save_dir, f"re_p_carrier_results_{d:03d}.csv"), header=True)
#
#     # Actual generation aggregated by bus+carrier
#     generators_tp = network.generators_t.p.T
#     generators_tp.index = generators_tp.index.astype(str)
#     generators_tp["bus"] = generators_tp.index.map(network.generators["bus"])
#     generators_tp["carrier"] = generators_tp.index.map(network.generators["carrier"])
#     generators_tp["bus_carrier"] = generators_tp["bus"] + "_" + generators_tp["carrier"]
#     gen_by_bus_carrier = (
#         generators_tp.groupby("bus_carrier").sum()
#                      .drop(columns=["bus", "carrier"], errors="ignore")
#                      .T
#     )
#     gen_by_bus_carrier.to_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{d:03d}.csv"), index=True)
#
#     # ------------------------ Downstream analysis ------------------------
#     df = pd.read_csv(os.path.join(save_dir, f"store_e_carrier_results_{d:03d}.csv"))
#     df_capacity = pd.read_csv(os.path.join(save_dir, f"stores_e_{d:03d}.csv"))
#     df_gen = pd.read_csv(os.path.join(save_dir, f"gen_p_carrier_results_{d:03d}.csv"))
#     df_gen_remain = pd.read_csv(os.path.join(save_dir, f"p_by_bus_carrier_{d:03d}.csv"))
#     df_gen_remain_carrier = pd.read_csv(os.path.join(save_dir, f"re_p_carrier_results_{d:03d}.csv"))
#     df_storage_links = pd.read_csv(os.path.join(save_dir, f"links_p1_results_{d:03d}.csv"))
#     df_gen_bus_carrier_region = pd.read_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{d:03d}.csv"))
#     load = pd.read_csv(os.path.join(save_dir, f"demand_p_{d:03d}.csv"))
#
#     # Clean negatives to zero for selected numeric tables
#     num_cols = df.select_dtypes(include=["number"]).columns
#     df[num_cols] = df[num_cols].clip(lower=0)
#     num_cols = df_capacity.select_dtypes(include=["number"]).columns
#     df_capacity[num_cols] = df_capacity[num_cols].clip(lower=0)
#
#     print("Line columns:", network.lines.columns.tolist())
#
#     # Resource maps and ordering
#     regions = network.generators["bus"].unique().tolist()
#     carriers = [c for c in network.generators["carrier"].unique() if pd.notna(c)]
#     resource_usage = {c: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for c in carriers}
#     resource_usage.setdefault("Others", {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []})
#
#     # Region-level buses for battery and LDES
#     battery_bus = [s + "_Battery" for s in regions]
#     ES_bus = [s + "_OtherStorage" for s in regions]
#
#     # Normalize SOC to [0, 1] for cycle extraction
#     df["soc_batt"] = df["Battery"] / df["Battery"].max()
#     df["soc_ldes"] = df["ES"] / df["ES"].max()
#
#     # Build link-name lists for charging/discharging
#     battery_charger = [s + "_charger" for s in battery_bus]
#     battery_discharger = [s + "_discharger" for s in battery_bus]
#     ES_charger = [s + "_charger" for s in ES_bus]
#     ES_discharger = [s + "_discharger" for s in ES_bus]
#
#     # Make charging positive for readability
#     df_storage_links[battery_charger] = -df_storage_links[battery_charger]
#     df_storage_links[ES_charger] = -df_storage_links[ES_charger]
#
#     # Aggregate chargers/dischargers by base bus (strip suffix)
#     charger = (
#         df_storage_links
#         .filter(regex="_charger$")
#         .T.groupby(lambda col: col.rsplit("_", 2)[0]).sum()
#         .T
#     )
#     discharger = (
#         df_storage_links
#         .filter(regex="_discharger$")
#         .T.groupby(lambda col: col.rsplit("_", 2)[0]).sum()
#         .T
#     )
#     charger = pd.concat([df_storage_links[["snapshot"]], charger], axis=1)
#     discharger = pd.concat([df_storage_links[["snapshot"]], -discharger], axis=1)
#
#     # Convenience totals (optional diagnostics)
#     df_storage_links["bus_charger"] = df_storage_links[battery_charger].sum(axis=1)
#     df_storage_links["bus_discharger"] = df_storage_links[battery_discharger].sum(axis=1)
#     df_storage_links["es_bus_charger"] = df_storage_links[ES_charger].sum(axis=1)
#     df_storage_links["es_bus_discharger"] = df_storage_links[ES_discharger].sum(axis=1)
#
#     # Generation aggregated by bus (for redistribution)
#     df_gen_bus = network.generators_t.p.T.groupby(network.generators.bus).sum().T
#
#     # Emission & cost factors by carrier (use first value per carrier)
#     agg = network.generators.groupby("carrier").agg({"co2_emissions": "first", "marginal_cost": "first"})
#     CO2_FACTORS = {c: (row.co2_emissions, row.marginal_cost) for c, row in agg.iterrows()}
#     CO2_FACTORS = dict(sorted(CO2_FACTORS.items(), key=lambda x: (x[1][0], x[1][1])))
#
#     resources = list(CO2_FACTORS.keys())
#
#     # External processing (provided by your own modules)
#     df_gen_bus_carrier_region_updated, flow_df, flows_by_res, df_gen_charging = redistribute_generation(
#         df_gen_bus=df_gen_bus,
#         load=load,
#         charger=charger,
#         discharger=discharger,
#         df_gen_bus_carrier_region=df_gen_bus_carrier_region,
#         regions=regions,
#         resources=resources,
#     )
#
#     df_gen_remain_new, flows_by_res_dis = modify_and_redistribute(
#         df_gen_bus_carrier_region,
#         df_gen_remain,
#         discharger,
#         regions,
#         resources,
#     )
#
#     # Prepare SOC series for cycle extraction (prepend a zero row for boundary handling)
#     df_copy = df.copy()
#     df_copy["snapshot"] = pd.to_datetime(df_copy["snapshot"])
#     new_snapshot = df_copy["snapshot"].iloc[0] - pd.Timedelta(hours=1)
#     new_row = pd.DataFrame({col: [0] if col != "snapshot" else [new_snapshot] for col in df_copy.columns})
#     df_copy = pd.concat([new_row, df_copy], ignore_index=True)
#
#     process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy["soc_batt"])
#     process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy["soc_ldes"])
#
#     (cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
#      unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
#      unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
#      cost_bat, cost_es, emissions_bat, emissions_es,
#      cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
#      energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
#      emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
#      resource_usage) = national_cycle_analysis(
#         process_times_bat, process_ratios_bat,
#         process_times_es, process_ratios_es,
#         df_gen,
#         df_gen_bus_carrier_region_updated,
#         df_storage_links,
#         df_gen_remain_new.copy(),
#         resource_usage, CO2_FACTORS, resources, regions
#     )
#
#     # Save per-iteration pickle
#     out_pkl = os.path.join(iter_dir, f"national_cycle_output_{d:03d}.pkl")
#     with open(out_pkl, "wb") as f:
#         pickle.dump((
#             cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
#             unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
#             unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
#             cost_bat, cost_es, emissions_bat, emissions_es,
#             cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
#             energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
#             emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
#             resource_usage, resources
#         ), f)
#
#     output_summary.append(out_pkl)
#
# # Save list of all iteration pickle outputs
# summary_txt = os.path.join(SUB_DIR, "analysis_outputs_summary.txt")
# with open(summary_txt, "w") as f:
#     f.write("\n".join(output_summary))
# print(f"Saved analysis outputs for {len(output_summary)} iterations. Summary at {summary_txt}")

# ----------------------- Part 3: Aggregations & plots -------------------
# Collect all iteration pickles
pkl_paths = glob(os.path.join(ITER_DIR_ROOT, "iteration_*", "national_cycle_output_*.pkl"))

counts_bat = {"Z1": [], "Z2": [], "Z3": [], "Z4": []}
counts_es  = {"Z1": [], "Z2": [], "Z3": [], "Z4": []}
avg_ed_bat = {"Z1": [], "Z2": [], "Z3": [], "Z4": []}
avg_ed_es  = {"Z1": [], "Z2": [], "Z3": [], "Z4": []}
mean_pts_bat = []  # (ACEF, ACCF)
mean_pts_es  = []

for path in pkl_paths:
    with open(path, "rb") as f:
        data = pickle.load(f)

    uc_bat = np.array(data[8])
    uc_es  = np.array(data[9])
    co2_bat = np.array(data[10])
    co2_es  = np.array(data[11])
    ed_bat  = np.array(data[22])
    ed_es   = np.array(data[23])
    tcost_bat = np.array(data[12])
    tcost_es  = np.array(data[13])
    tco2_bat  = np.array(data[14])
    tco2_es   = np.array(data[15])

    # Quadrant masks
    masks_bat = [
        (uc_bat > 0) & (co2_bat > 0),
        (uc_bat <= 0) & (co2_bat > 0),
        (uc_bat > 0) & (co2_bat <= 0),
        (uc_bat <= 0) & (co2_bat <= 0),
    ]
    masks_es = [
        (uc_es > 0) & (co2_es > 0),
        (uc_es <= 0) & (co2_es > 0),
        (uc_es > 0) & (co2_es <= 0),
        (uc_es <= 0) & (co2_es <= 0),
    ]

    for i_q, q in enumerate(["Z1", "Z2", "Z3", "Z4"]):
        cnt_bat = np.sum(masks_bat[i_q]);  cnt_bat and counts_bat[q].append(cnt_bat)
        cnt_es  = np.sum(masks_es[i_q]);   cnt_es and counts_es[q].append(cnt_es)

        if masks_bat[i_q].any():
            avg_bat = ed_bat[masks_bat[i_q]].mean()
            avg_bat and avg_ed_bat[q].append(avg_bat)
        if masks_es[i_q].any():
            avg_es = ed_es[masks_es[i_q]].mean()
            avg_es and avg_ed_es[q].append(avg_es)

    # Mean intersection points (ACEF, ACCF)
    mean_pts_bat.append((tco2_bat / np.sum(ed_bat), tcost_bat / np.sum(ed_bat)))
    mean_pts_es.append((tco2_es / np.sum(ed_es), tcost_es / np.sum(ed_es)))

mean_pts_bat = np.array(mean_pts_bat)
mean_pts_es  = np.array(mean_pts_es)

# Load BASECASE directly from your existing pickle (no NetCDF work here)
with open(BASECASE_PKL_PATH, "rb") as f:
    data_base = pickle.load(f)

uc_bat_base = np.array(data_base[8])
uc_es_base  = np.array(data_base[9])
co2_bat_base = np.array(data_base[10])
co2_es_base  = np.array(data_base[11])
ed_bat_base  = np.array(data_base[22])
ed_es_base   = np.array(data_base[23])
tcost_bat_base = data_base[12]
tcost_es_base  = data_base[13]
tco2_bat_base  = data_base[14]
tco2_es_base   = data_base[15]

masks_bat_base = [
    (uc_bat_base > 0) & (co2_bat_base > 0),
    (uc_bat_base <= 0) & (co2_bat_base > 0),
    (uc_bat_base > 0) & (co2_bat_base <= 0),
    (uc_bat_base <= 0) & (co2_bat_base <= 0),
]
masks_es_base = [
    (uc_es_base > 0) & (co2_es_base > 0),
    (uc_es_base <= 0) & (co2_es_base > 0),
    (uc_es_base > 0) & (co2_es_base <= 0),
    (uc_es_base <= 0) & (co2_es_base <= 0),
]

baseline_counts_bat = {q: masks_bat_base[i].sum() for i, q in enumerate(["Z1", "Z2", "Z3", "Z4"])}
baseline_counts_es  = {q: masks_es_base[i].sum()  for i, q in enumerate(["Z1", "Z2", "Z3", "Z4"])}
baseline_avg_ed_bat = {q: (ed_bat_base[masks_bat_base[i]].mean() if masks_bat_base[i].any() else None)
                       for i, q in enumerate(["Z1", "Z2", "Z3", "Z4"])}
baseline_avg_ed_es  = {q: (ed_es_base[masks_es_base[i]].mean() if masks_es_base[i].any() else None)
                       for i, q in enumerate(["Z1", "Z2", "Z3", "Z4"])}