# -*- coding: utf-8 -*-
"""
Sensitivity analysis for round-trip efficiency.
All outputs are stored under: ./sensitive_analyse/round_trip_efficiency/
"""

import os
import re
import pickle
import datetime
import shutil
from glob import glob
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from function.mef_national_energy_log_uk import national_cycle_analysis
from function.cyclic_data_preprocessing import (
    cycle_extraction,
    redistribute_generation,
    modify_and_redistribute,
)


# ----------------------------- Paths & setup -----------------------------
# Root folder for this analysis
ROOT_DIR = "./sensitive_analyse"
SUB_DIR = os.path.join(ROOT_DIR, "round_trip_efficiency")
os.makedirs(SUB_DIR, exist_ok=True)

# Where solved network files will be saved
NETWORK_DIR = os.path.join(SUB_DIR, "network_files")
os.makedirs(NETWORK_DIR, exist_ok=True)

# Where per-iteration exports will be saved
ITER_DIR_ROOT = os.path.join(SUB_DIR, "iteration_files")
os.makedirs(ITER_DIR_ROOT, exist_ok=True)

# Input baseline network
original_network_path = "./data/network_2023.nc"


# -------------------------- Parameter sampling --------------------------
# Efficiency samples (in percent). You may adjust the lists as needed.
eff_bat_samples = [90, 90, 90, 80, 70, 70]
eff_ldes_samples = [60, 50, 40, 70, 70, 40]
n_samples = len(eff_bat_samples)


# # --------------------- Part 1: Optimize & save networks ------------------
# results_summary = []
#
# for i in range(n_samples):
#     # Load fresh network each iteration
#     network = pypsa.Network(original_network_path)
#
#     # Identify store groups by name tokens
#     battery_ids = [sid for sid in network.stores.index if "battery" in sid.lower()]
#     ldes_ids = [sid for sid in network.stores.index if "other" in sid.lower()]
#
#     try:
#         # Update discharge efficiencies (store-dispatch is implemented via links *_discharger)
#         for store_id in battery_ids:
#             eff_bat = eff_bat_samples[i]
#             network.links.at[f"{store_id}_discharger", "efficiency"] = eff_bat / 100.0
#
#         for store_id in ldes_ids:
#             eff_ldes = eff_ldes_samples[i]
#             network.links.at[f"{store_id}_discharger", "efficiency"] = eff_ldes / 100.0
#
#         # ===== Build optimization model =====
#         mdl = network.optimize.create_model()
#
#         # ===== Inline: add zero-SOC constraints =====
#         # Identify battery stores: prefer "carrier" field; fall back to name containing "Battery"
#         if "carrier" in network.stores.columns:
#             battery_ids = network.stores.index[
#                 network.stores.carrier.astype(str).str.contains("Battery", case=False, na=False)
#             ].tolist()
#         else:
#             battery_ids = [s for s in network.stores.index if "Battery" in str(s)]
#
#         store_e = mdl.variables["Store-e"]
#         snaps = pd.DatetimeIndex(network.snapshots)
#         Link_p = mdl.variables["Link-p"]
#         link_index = Link_p.coords["Link"].to_index()
#         store_links = []
#         for s in battery_ids + ldes_ids:
#             for suf in ["_charger", "_discharger"]:
#                 name = f"{s}{suf}"
#                 if name in link_index:
#                     store_links.append(name)
#         p_store = Link_p.sel(Link=store_links)
#         t_weights = xr.DataArray(np.arange(1, len(snaps) + 1, dtype=float), coords=[snaps], dims=["snapshot"])
#
#         # End-of-day snapshots: take the last snapshot of each calendar day (not necessarily 23:00)
#         eod_snaps = snaps.to_series().groupby(snaps.normalize()).max().tolist()
#
#         # Enforce SOC=0 for batteries at the end of each day
#         for store in battery_ids:
#             for ts in eod_snaps:
#                 mdl.add_constraints(
#                     store_e.sel(snapshot=ts, Store=store) == 0,
#                     name=f"EndOfDayZeroStorage_{store}_{ts.date()}"
#                 )
#
#         # Enforce SOC=0 for all stores at the final snapshot
#         last_snap = snaps.max()
#         for store in network.stores.index:
#             mdl.add_constraints(
#                 store_e.sel(snapshot=last_snap, Store=store) == 0,
#                 name=f"FinalZeroStorage_{store}_{last_snap}"
#             )
#
#         # ===== Objective =====
#         Gen_p = mdl.variables["Generator-p"]
#         gen_index = Gen_p.coords["Generator"].to_index()
#         mc = network.generators["marginal_cost"].reindex(gen_index, fill_value=0)
#         co2 = network.generators["co2_emissions"].reindex(gen_index, fill_value=0)
#         aligned = co2 + 1e-6 * mc
#         mdl.objective = (Gen_p * aligned).sum(["snapshot", "Generator"]) + 1e-6 * (t_weights * p_store).sum(["snapshot","Link"])
#
#         # ===== Solve =====
#         status, _ = network.optimize.solve_model(
#             solver_name="gurobi",
#             solver_options = {"Threads": 10}
#         )
#
#         if status == "ok":
#             # KPIs (Generators only — keep consistent with chosen objective above)
#             gen_profiles = network.generators_t.p
#             total_cost = (
#                 gen_profiles
#                 * network.generators["marginal_cost"].reindex(gen_profiles.columns).fillna(0)
#             ).sum().sum()
#             total_emissions = (
#                 gen_profiles
#                 * network.generators["co2_emissions"].reindex(gen_profiles.columns).fillna(0)
#             ).sum().sum()
#
#             # Save solved network
#             save_path = os.path.join(
#                 NETWORK_DIR,
#                 f"network_mc_bat{eff_bat_samples[i]}ldes{eff_ldes_samples[i]}.nc",
#             )
#             network.export_to_netcdf(save_path)
#
#             # Record run summary
#             results_summary.append({
#                 "index": i,
#                 "success": True,
#                 "solve_status": status,
#                 "eff_bat_percent": eff_bat_samples[i],
#                 "eff_ldes_percent": eff_ldes_samples[i],
#                 "total_cost": float(total_cost),
#                 "total_emissions": float(total_emissions),
#                 "network_file": save_path,
#             })
#         else:
#             results_summary.append({
#                 "index": i,
#                 "success": False,
#                 "solve_status": status,
#                 "eff_bat_percent": eff_bat_samples[i],
#                 "eff_ldes_percent": eff_ldes_samples[i],
#                 "total_cost": None,
#                 "total_emissions": None,
#                 "network_file": "",
#             })
#
#     except Exception as e:
#         # Keep efficiencies in percent to stay consistent with successful cases
#         results_summary.append({
#             "index": i,
#             "success": False,
#             "solve_status": "exception",
#             "eff_bat_percent": eff_bat_samples[i],
#             "eff_ldes_percent": eff_ldes_samples[i],
#             "total_cost": None,
#             "total_emissions": None,
#             "network_file": "",
#             "error": str(e),
#         })
#
# # Save run summary CSV inside the analysis folder
# summary_path = os.path.join(SUB_DIR, "summary_results.csv")
# pd.DataFrame(results_summary).to_csv(summary_path, index=False)
# BASECASE_NC_PATH = os.path.join("results", "min_CO2_2023", "network_ff_constrained_time.nc")
#
# # The target suffix/name we want to add:
# INSERT_SUFFIX = "bat90ldes70"
# target_nc = os.path.join(NETWORK_DIR, f"network_mc_{INSERT_SUFFIX}.nc")
#
# if not os.path.isfile(target_nc):
#     if not os.path.isfile(BASECASE_NC_PATH):
#         raise FileNotFoundError(
#             f"Base solved network not found at: {BASECASE_NC_PATH}\n"
#             "Set BASECASE_NC_PATH to your solved base NetCDF."
#         )
#     os.makedirs(NETWORK_DIR, exist_ok=True)
#     shutil.copy(BASECASE_NC_PATH, target_nc)
#     print(f"[Insert] Basecase network copied to {target_nc} as the 9070 case.")
# else:
#     print(f"[Insert] 9070 case already present: {target_nc}")

# ---------------- Part 2: Export results from each solved network --------
# Find all solved networks produced above
pattern = os.path.join(NETWORK_DIR, "network_mc_*.nc")

# Keep paths to all per-iteration pickle outputs for the final aggregation
output_summary = []

for network_path in sorted(glob(pattern)):
    basename = os.path.basename(network_path)
    # Example: "network_mc_bat90ldes70.nc" -> suffix "bat90ldes70"
    suffix = basename.replace("network_mc_", "").replace(".nc", "")
    print(f"[Iteration {suffix}] Found network file: {basename}")

    # Create per-iteration folder
    iter_dir = os.path.join(ITER_DIR_ROOT, f"iteration_{suffix}")
    os.makedirs(iter_dir, exist_ok=True)
    save_dir = iter_dir

    # Load solved network
    network = pypsa.Network(network_path)

    # Basic exports
    network.loads_t.p.to_csv(os.path.join(save_dir, f"demand_p_{suffix}.csv"), header=True)
    network.links_t.p1.to_csv(os.path.join(save_dir, f"links_p1_results_{suffix}.csv"), header=True)
    network.stores_t.e.to_csv(os.path.join(save_dir, f"stores_e_{suffix}.csv"), header=True)

    # Storage energy aggregated by carrier
    store_by_carrier = network.stores_t.e.T.groupby(network.stores.carrier).sum().T
    store_by_carrier.to_csv(os.path.join(save_dir, f"store_e_carrier_results_{suffix}.csv"), header=True)

    # Generator power aggregated by carrier
    p_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
    p_by_carrier.to_csv(os.path.join(save_dir, f"gen_p_carrier_results_{suffix}.csv"), header=True)

    # Dynamic generator max output and remaining headroom
    p_max_pu_full = network.generators_t.p_max_pu.reindex(columns=network.generators.index)
    missing_generators = network.generators.index.difference(network.generators_t.p_max_pu.columns)
    for gen in missing_generators:
        p_max_pu_full[gen] = network.generators.at[gen, "p_max_pu"]
    snapshot_max_output = p_max_pu_full.multiply(network.generators["p_nom"], axis=1)

    network.generators_t["max_output"] = snapshot_max_output
    remain_output = (network.generators_t["max_output"] - network.generators_t.p)

    # Multiply by efficiency if available; default to 1.0
    if "efficiency" in network.generators.columns:
        eff_series = network.generators["efficiency"].reindex(remain_output.columns).fillna(1.0)
    else:
        eff_series = pd.Series(1.0, index=remain_output.columns)
    network.generators_t["remain_output"] = remain_output.multiply(eff_series, axis=1)

    # Remaining headroom aggregated by bus+carrier
    rem = network.generators_t["remain_output"].T
    rem.index = rem.index.astype(str)
    rem["bus"] = rem.index.map(network.generators["bus"])
    rem["carrier"] = rem.index.map(network.generators["carrier"])
    rem["bus_carrier"] = rem["bus"] + "_" + rem["carrier"]
    p_by_bus_carrier = (
        rem.groupby("bus_carrier").sum()
           .drop(columns=["bus", "carrier"], errors="ignore")
           .T
    )
    p_by_bus_carrier.to_csv(os.path.join(save_dir, f"p_by_bus_carrier_{suffix}.csv"), index=True)

    # Remaining headroom aggregated by carrier
    re_p_by_carrier = (
        network.generators_t["remain_output"].T
        .groupby(network.generators.carrier).sum().T
    )
    re_p_by_carrier.to_csv(os.path.join(save_dir, f"re_p_carrier_results_{suffix}.csv"), header=True)

    # Actual generation aggregated by bus+carrier
    generators_tp = network.generators_t.p.T
    generators_tp.index = generators_tp.index.astype(str)
    generators_tp["bus"] = generators_tp.index.map(network.generators["bus"])
    generators_tp["carrier"] = generators_tp.index.map(network.generators["carrier"])
    generators_tp["bus_carrier"] = generators_tp["bus"] + "_" + generators_tp["carrier"]
    gen_by_bus_carrier = (
        generators_tp.groupby("bus_carrier").sum()
                     .drop(columns=["bus", "carrier"], errors="ignore")
                     .T
    )
    gen_by_bus_carrier.to_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{suffix}.csv"), index=True)

    # ------------------------ Post-processing & analysis ------------------------
    # Load exported CSVs for downstream analysis
    df = pd.read_csv(os.path.join(save_dir, f"store_e_carrier_results_{suffix}.csv"))
    df_capacity = pd.read_csv(os.path.join(save_dir, f"stores_e_{suffix}.csv"))
    df_gen = pd.read_csv(os.path.join(save_dir, f"gen_p_carrier_results_{suffix}.csv"))
    df_gen_remain = pd.read_csv(os.path.join(save_dir, f"p_by_bus_carrier_{suffix}.csv"))
    df_gen_remain_carrier = pd.read_csv(os.path.join(save_dir, f"re_p_carrier_results_{suffix}.csv"))
    df_storage_links = pd.read_csv(os.path.join(save_dir, f"links_p1_results_{suffix}.csv"))
    df_gen_bus_carrier_region = pd.read_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{suffix}.csv"))
    load = pd.read_csv(os.path.join(save_dir, f"demand_p_{suffix}.csv"))

    # Non-negative cleaning for select numeric tables
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].clip(lower=0)
    num_cols = df_capacity.select_dtypes(include=["number"]).columns
    df_capacity[num_cols] = df_capacity[num_cols].clip(lower=0)

    print("Line columns:", network.lines.columns.tolist())

    # Build helper maps and resource ordering
    all_carriers = df_gen.columns.tolist()
    regions = network.generators["bus"].unique().tolist()
    carriers = [c for c in network.generators["carrier"].unique() if pd.notna(c)]
    resource_usage = {c: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for c in carriers}
    if "Others" not in resource_usage:
        resource_usage["Others"] = {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}

    # Region-to-bus naming for battery and LDES
    battery_bus = [s + "_Battery" for s in regions]
    ES_bus = [s + "_OtherStorage" for s in regions]

    # Normalize SOC to [0, 1] for cycle extraction
    df["soc_batt"] = df["Battery"] / df["Battery"].max()
    df["soc_ldes"] = df["ES"] / df["ES"].max()

    # Build link-name lists for charging/discharging
    battery_charger = [s + "_charger" for s in battery_bus]
    battery_discharger = [s + "_discharger" for s in battery_bus]
    ES_charger = [s + "_charger" for s in ES_bus]
    ES_discharger = [s + "_discharger" for s in ES_bus]

    # Sign convention: make charging positive for clarity
    df_storage_links[battery_charger] = -df_storage_links[battery_charger]
    df_storage_links[ES_charger] = -df_storage_links[ES_charger]

    # Aggregate charger/discharger by base bus (strip the suffix)
    charger = (
        df_storage_links
        .filter(regex="_charger$")
        .T.groupby(lambda col: col.rsplit("_", 2)[0]).sum()
        .T
    )
    discharger = (
        df_storage_links
        .filter(regex="_discharger$")
        .T.groupby(lambda col: col.rsplit("_", 2)[0]).sum()
        .T
    )
    charger = pd.concat([df_storage_links[["snapshot"]], charger], axis=1)
    discharger = pd.concat([df_storage_links[["snapshot"]], -discharger], axis=1)

    # Convenience totals
    df_storage_links["bus_charger"] = df_storage_links[battery_charger].sum(axis=1)
    df_storage_links["bus_discharger"] = df_storage_links[battery_discharger].sum(axis=1)
    df_storage_links["es_bus_charger"] = df_storage_links[ES_charger].sum(axis=1)
    df_storage_links["es_bus_discharger"] = df_storage_links[ES_discharger].sum(axis=1)

    # Generator power aggregated by bus (for redistribution)
    df_gen_bus = network.generators_t.p.T.groupby(network.generators.bus).sum().T

    # Build per-carrier CO2 and cost factors (first value per carrier)
    agg = network.generators.groupby("carrier").agg({"co2_emissions": "first", "marginal_cost": "first"})
    CO2_FACTORS = {c: (row.co2_emissions, row.marginal_cost) for c, row in agg.iterrows()}
    CO2_FACTORS = dict(sorted(CO2_FACTORS.items(), key=lambda x: (x[1][0], x[1][1])))

    # Derive preferred resource order; adjust as needed for your workflow
    # Here we keep the default ranking by carrier factors
    resources = list(CO2_FACTORS.keys())

    # --- External processing ---
    df_gen_bus_carrier_region_updated, flow_df, flows_by_res, df_gen_charging = redistribute_generation(
        df_gen_bus=df_gen_bus,
        load=load,
        charger=charger,
        discharger=discharger,
        df_gen_bus_carrier_region=df_gen_bus_carrier_region,
        regions=regions,
        resources=resources,
    )

    # df_gen_remain_new, flows_by_res_dis = ...
    df_gen_remain_new, flows_by_res_dis = modify_and_redistribute(
        df_gen_bus_carrier_region,
        df_gen_remain,
        discharger,
        regions,
        resources,
    )

    # Prepare SOC series for cycle extraction (prepend a zero row for boundary handling)
    df_copy = df.copy()
    df_copy["snapshot"] = pd.to_datetime(df_copy["snapshot"])
    new_snapshot = df_copy["snapshot"].iloc[0] - pd.Timedelta(hours=1)
    new_row = pd.DataFrame({col: [0] if col != "snapshot" else [new_snapshot] for col in df_copy.columns})
    df_copy = pd.concat([new_row, df_copy], ignore_index=True)

    process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy["soc_batt"])
    process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy["soc_ldes"])

    # National-level cycle analysis
    (cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
     unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
     unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
     cost_bat, cost_es, emissions_bat, emissions_es,
     cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
     energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
     emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
     resource_usage) = national_cycle_analysis(
        process_times_bat, process_ratios_bat,
        process_times_es, process_ratios_es,
        df_gen,
        df_gen_bus_carrier_region_updated,
        df_storage_links,
        df_gen_remain_new.copy(),
        resource_usage, CO2_FACTORS, resources, regions
    )

    # Save iteration pickle
    output_path = os.path.join(iter_dir, f"national_cycle_output_{suffix}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((
            cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
            unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
            unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
            cost_bat, cost_es, emissions_bat, emissions_es,
            cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
            energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
            emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
            resource_usage, resources
        ), f)

    output_summary.append(output_path)
# Save list of all iteration pickle outputs
summary_txt = os.path.join(SUB_DIR, "analysis_outputs_summary.txt")
with open(summary_txt, "w") as f:
    f.write("\n".join(output_summary))
print(f"Saved analysis outputs for {len(output_summary)} iterations. Summary at {summary_txt}")

# -------------------- Part 3: Aggregate iteration outputs ----------------
# Collect all pickles from per-iteration folders
pkl_paths = glob(os.path.join(ITER_DIR_ROOT, "iteration_*", "national_cycle_output_*.pkl"))

count_bat = []
count_es = []
charged_energy_bat = []
charged_energy_es = []
discharged_energy_bat = []
discharged_energy_es = []
ACEF_bat = []
ACEF_es = []
ACCF_bat = []
ACCF_es = []
proportions_bat = []
proportions_es = []

for path in pkl_paths:
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Indices based on your national_cycle_analysis return
    uc_bat = np.array(data[8])   # unit_cost_bat_cycle
    uc_es  = np.array(data[9])   # unit_cost_es_cycle
    co2_bat = np.array(data[10]) # co2_emissions_bat_cycle
    co2_es  = np.array(data[11]) # co2_emissions_es_cycle
    ec_bat  = np.array(data[20]) # energy_charge_cycle_bat
    ec_es   = np.array(data[21]) # energy_charge_cycle_es
    ed_bat  = np.array(data[22]) # energy_discharge_cycle_bat
    ed_es   = np.array(data[23]) # energy_discharge_cycle_es
    tcost_bat = np.array(data[12]) # total cost of battery (sum over cycles)
    tcost_es  = np.array(data[13]) # total cost of LDES
    tco2_bat  = np.array(data[14]) # total CO2 of battery
    tco2_es   = np.array(data[15]) # total CO2 of LDES
    resources_0carbon = data[29][0:7]

    df_bat_cha = pd.DataFrame({res: data[28][res]["bat_cha"] for res in resources_0carbon})
    df_es_cha  = pd.DataFrame({res: data[28][res]["es_cha"]  for res in resources_0carbon})

    # Counts
    count_bat.append(len(data[2]))  # number of battery cycles
    count_es.append(len(data[3]))   # number of LDES cycles

    # Charged energy [GWh]
    charged_energy_bat.append(ec_bat.sum() / 1e3)
    charged_energy_es.append(ec_es.sum() / 1e3)

    # Discharged energy [GWh]
    discharged_energy_bat.append(ed_bat.sum() / 1e3)
    discharged_energy_es.append(ed_es.sum() / 1e3)

    # Average carbon emission factor [tCO2/MWh]
    ACEF_bat.append(tco2_bat / np.sum(ed_bat))
    ACEF_es.append(tco2_es / np.sum(ed_es))

    # Average cost [£/MWh]
    ACCF_bat.append(tcost_bat / np.sum(ed_bat))
    ACCF_es.append(tcost_es / np.sum(ed_es))

    # Proportions of charged energy sourced from the first 7 “zero-carbon” resources
    proportions_bat.append(df_bat_cha.sum().sum() / ec_bat.sum() * 100)
    proportions_es.append(df_es_cha.sum().sum() / ec_es.sum() * 100)

# Column labels from file suffixes
suffixes = [
    re.search(r"national_cycle_output_(.+)\.pkl$", os.path.basename(p)).group(1)
    for p in pkl_paths
]

# Row labels (fixed typos: £/MWh)
metrics = [
    "count_bat [Times]", "count_es [Times]",
    "charged_energy_bat [GWh]", "charged_energy_es [GWh]",
    "discharged_energy_bat [GWh]", "discharged_energy_es [GWh]",
    "ACEF_bat [tCO2/MWh]", "ACEF_es [tCO2/MWh]",
    "ACCF_bat [£/MWh]", "ACCF_es [£/MWh]",
    "proportions_bat [%]", "proportions_es [%]",
]

# Assemble results table
data_mat = np.vstack([
    count_bat,
    count_es,
    charged_energy_bat,
    charged_energy_es,
    discharged_energy_bat,
    discharged_energy_es,
    ACEF_bat,
    ACEF_es,
    ACCF_bat,
    ACCF_es,
    proportions_bat,
    proportions_es,
])

final_df = pd.DataFrame(data_mat, index=metrics, columns=suffixes).round(2)

# Save final Excel in the analysis subfolder
excel_out = os.path.join(SUB_DIR, "efficiency_analyse_outcome.xlsx")
final_df.to_excel(excel_out, index=True)
print(final_df)
