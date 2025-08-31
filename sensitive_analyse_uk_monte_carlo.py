# -*- coding: utf-8 -*-
"""
Monte Carlo sensitivity analysis for storage parameters (efficiency/scale/max-hours).
All outputs are stored under: ./sensitive_analyse/monte_carlo/
Basecase metrics are loaded from: results/min_CO2_2023/analysis_output/national_cycle_output.pkl
"""

import os
import pickle
import datetime
from glob import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Replace with a GUI backend appropriate for your environment
import matplotlib.pyplot as plt
import pypsa
from function.mef_national_energy_log_uk import national_cycle_analysis
from function.cyclic_data_preprocessing import (
    cycle_extraction,
    redistribute_generation,
    modify_and_redistribute,
)

# ----------------------------- Paths & setup -----------------------------
ROOT_DIR = "./sensitive_analyse"
SUB_DIR = os.path.join(ROOT_DIR, "monte_carlo")
NETWORK_DIR = os.path.join(SUB_DIR, "network_files")       # solved networks per MC draw
ITER_DIR_ROOT = os.path.join(SUB_DIR, "iteration_files")    # per-iteration CSVs & pkl outputs
FIG_DIR = os.path.join(SUB_DIR, "figures")                  # plots

os.makedirs(NETWORK_DIR, exist_ok=True)
os.makedirs(ITER_DIR_ROOT, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Basecase pickle produced by your min_CO2_2023 pipeline
BASECASE_PKL_PATH = os.path.join("results", "min_CO2_2023", "analysis_output", "national_cycle_output.pkl")

# Unsolved template network for each Monte Carlo iteration
ORIGINAL_NETWORK_PATH = "./data/network_2023.nc"

# ------------------------- Monte Carlo sampling --------------------------
n_samples = 1000

def generate_skewed(param_range, mode, size):
    """Triangular sampling on [left, right] with 'mode' as the peak."""
    left, right = param_range
    return np.random.triangular(left=left, mode=mode, right=right, size=size)

# Efficiency (round-trip on discharger link), scale (power multiplier), max-hours (energy duration multiplier)
eff_bat_samples     = generate_skewed([0.75, 0.95], 0.90, n_samples)
eff_ldes_samples    = generate_skewed([0.50, 0.80], 0.70, n_samples)
scale_bat_samples   = generate_skewed([0.50, 2.00], 1.00, n_samples)
scale_ldes_samples  = generate_skewed([0.50, 2.00], 1.00, n_samples)
maxhour_bat_samples = generate_skewed([0.50, 2.00], 1.00, n_samples)
maxhour_ldes_samples= generate_skewed([0.50, 2.00], 1.00, n_samples)

# Container to log per-run info
results_summary = []

# --------------------- Part 1: Optimize & save networks -----------------
for i in range(n_samples):
    network = pypsa.Network(ORIGINAL_NETWORK_PATH)
    battery_ids = [sid for sid in network.stores.index if "battery" in sid.lower()]
    ldes_ids    = [sid for sid in network.stores.index if "other"   in sid.lower()]

    try:
        # ---- Update storage parameters (Battery) ----
        for store_id in battery_ids:
            scale = float(scale_bat_samples[i])
            maxhour_factor = float(maxhour_bat_samples[i])
            eff = float(eff_bat_samples[i])

            old_p_nom = float(network.links.at[store_id + "_charger", "p_nom"])
            new_p_nom = old_p_nom * scale
            network.links.at[store_id + "_charger",   "p_nom"] = new_p_nom
            network.links.at[store_id + "_discharger","p_nom"] = new_p_nom

            # Update max_hours then re-derive e_nom from new p_nom * max_hours
            network.stores.at[store_id, "max_hours"] = float(network.stores.at[store_id, "max_hours"]) * maxhour_factor
            network.stores.at[store_id, "e_nom"] = new_p_nom * float(network.stores.at[store_id, "max_hours"])

            # Set discharger efficiency
            network.links.at[store_id + "_discharger", "efficiency"] = eff

        # ---- Update storage parameters (LDES/Other) ----
        for store_id in ldes_ids:
            scale = float(scale_ldes_samples[i])
            maxhour_factor = float(maxhour_ldes_samples[i])
            eff = float(eff_ldes_samples[i])

            old_p_nom = float(network.links.at[store_id + "_charger", "p_nom"])
            new_p_nom = old_p_nom * scale
            network.links.at[store_id + "_charger",   "p_nom"] = new_p_nom
            network.links.at[store_id + "_discharger","p_nom"] = new_p_nom

            network.stores.at[store_id, "max_hours"] = float(network.stores.at[store_id, "max_hours"]) * maxhour_factor
            network.stores.at[store_id, "e_nom"] = new_p_nom * float(network.stores.at[store_id, "max_hours"])

            network.links.at[store_id + "_discharger", "efficiency"] = eff

        # ===== Build optimization model =====
        mdl = network.optimize.create_model()

        # ===== Inline: add zero-SOC constraints =====
        # Identify battery stores: prefer "carrier" field; fall back to name containing "Battery"
        if "carrier" in network.stores.columns:
            battery_ids = network.stores.index[
                network.stores.carrier.astype(str).str.contains("Battery", case=False, na=False)
            ].tolist()
        else:
            battery_ids = [s for s in network.stores.index if "Battery" in str(s)]

        store_e = mdl.variables["Store-e"]
        snaps = pd.DatetimeIndex(network.snapshots)

        # End-of-day snapshots: take the last snapshot of each calendar day (not necessarily 23:00)
        eod_snaps = snaps.to_series().groupby(snaps.normalize()).max().tolist()

        # Enforce SOC=0 for batteries at the end of each day
        for store in battery_ids:
            for ts in eod_snaps:
                mdl.add_constraints(
                    store_e.sel(snapshot=ts, Store=store) == 0,
                    name=f"EndOfDayZeroStorage_{store}_{ts.date()}"
                )

        # Enforce SOC=0 for all stores at the final snapshot
        last_snap = snaps.max()
        for store in network.stores.index:
            mdl.add_constraints(
                store_e.sel(snapshot=last_snap, Store=store) == 0,
                name=f"FinalZeroStorage_{store}_{last_snap}"
            )

        # ===== Objective =====
        Gen_p = mdl.variables["Generator-p"]
        gen_index = Gen_p.coords["Generator"].to_index()
        mc = network.generators["marginal_cost"].reindex(gen_index, fill_value=0)
        co2 = network.generators["co2_emissions"].reindex(gen_index, fill_value=0)
        aligned = co2 + 1e-6 * mc
        mdl.objective = (Gen_p * aligned).sum(["snapshot", "Generator"])

        # ===== Solve =====
        status, _ = network.optimize.solve_model(
            solver_name="gurobi",
            solver_options={"Threads": 1},
        )

        # ---- Record results ----
        if status == "ok":
            gen_profiles = network.generators_t.p
            total_cost = (
                gen_profiles * network.generators["marginal_cost"]
                .reindex(gen_profiles.columns).fillna(0.0)
            ).sum().sum()
            total_emissions = (
                gen_profiles * network.generators["co2_emissions"]
                .reindex(gen_profiles.columns).fillna(0.0)
            ).sum().sum()

            save_path = os.path.join(NETWORK_DIR, f"network_mc_{i:03d}.nc")
            network.export_to_netcdf(save_path)

            results_summary.append({
                "index": i,
                "success": True,
                "solve_status": status,
                "eff_bat": eff_bat_samples[i],
                "eff_ldes": eff_ldes_samples[i],
                "scale_bat": scale_bat_samples[i],
                "scale_ldes": scale_ldes_samples[i],
                "maxhour_bat": maxhour_bat_samples[i],
                "maxhour_ldes": maxhour_ldes_samples[i],
                "total_cost": float(total_cost),
                "total_emissions": float(total_emissions),
                "network_file": save_path,
            })
        else:
            results_summary.append({
                "index": i,
                "success": False,
                "solve_status": status,
                "eff_bat": eff_bat_samples[i],
                "eff_ldes": eff_ldes_samples[i],
                "scale_bat": scale_bat_samples[i],
                "scale_ldes": scale_ldes_samples[i],
                "maxhour_bat": maxhour_bat_samples[i],
                "maxhour_ldes": maxhour_ldes_samples[i],
                "total_cost": None,
                "total_emissions": None,
                "network_file": "",
            })

    except Exception as e:
        results_summary.append({
            "index": i,
            "success": False,
            "solve_status": "exception",
            "eff_bat": eff_bat_samples[i],
            "eff_ldes": eff_ldes_samples[i],
            "scale_bat": scale_bat_samples[i],
            "scale_ldes": scale_ldes_samples[i],
            "maxhour_bat": maxhour_bat_samples[i],
            "maxhour_ldes": maxhour_ldes_samples[i],
            "total_cost": None,
            "total_emissions": None,
            "network_file": "",
            "error": str(e),
        })

# Save summary CSV for the Monte Carlo solve phase
summary_csv = os.path.join(SUB_DIR, "summary_results.csv")
pd.DataFrame(results_summary).to_csv(summary_csv, index=False)

# --------------- Part 2: Export & post-process each iteration -----------
output_summary = []

for i in range(n_samples):
    network_file = os.path.join(NETWORK_DIR, f"network_mc_{i:03d}.nc")
    if not os.path.isfile(network_file):
        print(f"[Iteration {i:03d}] Network file not found: {network_file}. Skipping.")
        continue

    # Per-iteration folder
    iter_dir = os.path.join(ITER_DIR_ROOT, f"iteration_{i:03d}")
    os.makedirs(iter_dir, exist_ok=True)
    save_dir = iter_dir

    # Load solved network
    network = pypsa.Network(network_file)

    # Basic exports
    network.loads_t.p.to_csv(os.path.join(save_dir, f"demand_p_{i:03d}.csv"), header=True)
    network.links_t.p1.to_csv(os.path.join(save_dir, f"links_p1_results_{i:03d}.csv"), header=True)
    network.stores_t.e.to_csv(os.path.join(save_dir, f"stores_e_{i:03d}.csv"), header=True)

    # Storage energy aggregated by carrier
    store_by_carrier = network.stores_t.e.T.groupby(network.stores.carrier).sum().T
    store_by_carrier.to_csv(os.path.join(save_dir, f"store_e_carrier_results_{i:03d}.csv"), header=True)

    # Generation aggregated by carrier
    p_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
    p_by_carrier.to_csv(os.path.join(save_dir, f"gen_p_carrier_results_{i:03d}.csv"), header=True)

    # Dynamic generator max output and remaining headroom
    p_max_pu_full = network.generators_t.p_max_pu.reindex(columns=network.generators.index)
    missing_generators = network.generators.index.difference(network.generators_t.p_max_pu.columns)
    for g in missing_generators:
        p_max_pu_full[g] = network.generators.at[g, "p_max_pu"]
    snapshot_max_output = p_max_pu_full.multiply(network.generators["p_nom"], axis=1)

    network.generators_t["max_output"] = snapshot_max_output
    # Safe efficiency reindex/fill for remain_output
    eff_series = network.generators.get("efficiency", pd.Series(1.0, index=network.generators.index))
    eff_series = eff_series.reindex(network.generators_t.p.columns).fillna(1.0)
    network.generators_t["remain_output"] = (network.generators_t["max_output"] - network.generators_t.p) * eff_series

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
    p_by_bus_carrier.to_csv(os.path.join(save_dir, f"p_by_bus_carrier_{i:03d}.csv"), index=True)

    # Remaining headroom aggregated by carrier
    re_p_by_carrier = (
        network.generators_t["remain_output"].T
        .groupby(network.generators.carrier).sum().T
    )
    re_p_by_carrier.to_csv(os.path.join(save_dir, f"re_p_carrier_results_{i:03d}.csv"), header=True)

    # Actual generation aggregated by bus+carrier
    gen_tp = network.generators_t.p.T
    gen_tp.index = gen_tp.index.astype(str)
    gen_tp["bus"] = gen_tp.index.map(network.generators["bus"])
    gen_tp["carrier"] = gen_tp.index.map(network.generators["carrier"])
    gen_tp["bus_carrier"] = gen_tp["bus"] + "_" + gen_tp["carrier"]
    gen_by_bus_carrier = (
        gen_tp.groupby("bus_carrier").sum()
              .drop(columns=["bus", "carrier"], errors="ignore")
              .T
    )
    gen_by_bus_carrier.to_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{i:03d}.csv"), index=True)

    # ------------------------ Downstream analysis ------------------------
    df = pd.read_csv(os.path.join(save_dir, f"store_e_carrier_results_{i:03d}.csv"))
    df_capacity = pd.read_csv(os.path.join(save_dir, f"stores_e_{i:03d}.csv"))
    df_gen = pd.read_csv(os.path.join(save_dir, f"gen_p_carrier_results_{i:03d}.csv"))
    df_gen_remain = pd.read_csv(os.path.join(save_dir, f"p_by_bus_carrier_{i:03d}.csv"))
    df_gen_remain_carrier = pd.read_csv(os.path.join(save_dir, f"re_p_carrier_results_{i:03d}.csv"))
    df_storage_links = pd.read_csv(os.path.join(save_dir, f"links_p1_results_{i:03d}.csv"))
    df_gen_bus_carrier_region = pd.read_csv(os.path.join(save_dir, f"gen_by_bus_carrier_{i:03d}.csv"))
    load = pd.read_csv(os.path.join(save_dir, f"demand_p_{i:03d}.csv"))

    # Clean negatives to zero where appropriate
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].clip(lower=0)
    num_cols = df_capacity.select_dtypes(include=["number"]).columns
    df_capacity[num_cols] = df_capacity[num_cols].clip(lower=0)

    print("Line columns:", network.lines.columns.tolist())

    # Regions/carriers
    regions = network.generators["bus"].unique().tolist()
    carriers = [c for c in network.generators["carrier"].unique() if pd.notna(c)]
    resource_usage = {c: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for c in carriers}
    resource_usage.setdefault("Others", {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []})

    # Region-level batteries and other storage bus names
    battery_bus = [s + "_Battery" for s in regions]
    ES_bus = [s + "_OtherStorage" for s in regions]

    # Normalize SOC for cycle extraction
    df["soc_batt"] = df["Battery"] / df["Battery"].max()
    df["soc_ldes"] = df["ES"] / df["ES"].max()

    # Link lists
    battery_charger = [s + "_charger" for s in battery_bus]
    battery_discharger = [s + "_discharger" for s in battery_bus]
    ES_charger = [s + "_charger" for s in ES_bus]
    ES_discharger = [s + "_discharger" for s in ES_bus]

    # Make charging positive
    df_storage_links[battery_charger] = -df_storage_links[battery_charger]
    df_storage_links[ES_charger] = -df_storage_links[ES_charger]

    # Aggregate charger/discharger flows to base bus (strip suffix)
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

    # Convenience totals (optional)
    df_storage_links["bus_charger"] = df_storage_links[battery_charger].sum(axis=1)
    df_storage_links["bus_discharger"] = df_storage_links[battery_discharger].sum(axis=1)
    df_storage_links["es_bus_charger"] = df_storage_links[ES_charger].sum(axis=1)
    df_storage_links["es_bus_discharger"] = df_storage_links[ES_discharger].sum(axis=1)

    # Generation aggregated by bus (for redistribution)
    df_gen_bus = network.generators_t.p.T.groupby(network.generators.bus).sum().T

    # Emission & cost factors by carrier (first value per carrier)
    agg = network.generators.groupby("carrier").agg({"co2_emissions": "first", "marginal_cost": "first"})
    CO2_FACTORS = {c: (row.co2_emissions, row.marginal_cost) for c, row in agg.iterrows()}
    CO2_FACTORS = dict(sorted(CO2_FACTORS.items(), key=lambda x: (x[1][0], x[1][1])))
    resources = list(CO2_FACTORS.keys())

    # External processing (your modules)
    df_gen_bus_carrier_region_updated, flow_df, flows_by_res, df_gen_charging = redistribute_generation(
        df_gen_bus=df_gen_bus,
        load=load,
        charger=charger,
        discharger=discharger,
        df_gen_bus_carrier_region=df_gen_bus_carrier_region,
        regions=regions,
        resources=resources,
    )

    df_gen_remain_new, flows_by_res_dis = modify_and_redistribute(
        df_gen_bus_carrier_region,
        df_gen_remain,
        discharger,
        regions,
        resources,
    )
    df_gen_remain_new_copy = df_gen_remain_new.copy()

    # Prepare SOC series (prepend a zero row for boundary handling)
    df_copy = df.copy()
    df_copy["snapshot"] = pd.to_datetime(df_copy["snapshot"])
    new_snapshot = df_copy["snapshot"].iloc[0] - pd.Timedelta(hours=1)
    new_row = pd.DataFrame({col: [0] if col != "snapshot" else [new_snapshot] for col in df_copy.columns})
    df_copy = pd.concat([new_row, df_copy], ignore_index=True)

    process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy["soc_batt"])
    process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy["soc_ldes"])

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
        df_gen_remain_new_copy,
        resource_usage, CO2_FACTORS, resources, regions
    )

    # Save per-iteration pickle
    out_pkl = os.path.join(iter_dir, f"national_cycle_output_{i:03d}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump((
            cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
            unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
            unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
            cost_bat, cost_es, emissions_bat, emissions_es,
            cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
            energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
            emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
            resource_usage
        ), f)
    output_summary.append(out_pkl)

# Save list of all iteration pickle outputs
summary_txt = os.path.join(SUB_DIR, "analysis_outputs_summary.txt")
with open(summary_txt, "w") as f:
    f.write("\n".join(output_summary))
print(f"Saved analysis outputs for {len(output_summary)} iterations. Summary at {summary_txt}")

# ----------------------- Part 3: Aggregations & plots -------------------
# Collect all iteration pickles
pkl_paths = glob(os.path.join(ITER_DIR_ROOT, "iteration_*", "national_cycle_output_*.pkl"))

# Containers for per-iteration metrics
counts_bat = {'Z1': [], 'Z2': [], 'Z3': [], 'Z4': []}
counts_es  = {'Z1': [], 'Z2': [], 'Z3': [], 'Z4': []}
avg_ed_bat = {'Z1': [], 'Z2': [], 'Z3': [], 'Z4': []}
avg_ed_es  = {'Z1': [], 'Z2': [], 'Z3': [], 'Z4': []}
mean_pts_bat = []  # (ACEF, ACCF)
mean_pts_es  = []

# Process each iteration file
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

    # Define quadrant masks
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

    for i_q, q in enumerate(['Z1', 'Z2', 'Z3', 'Z4']):
        cnt_bat = int(np.sum(masks_bat[i_q]));  cnt_bat and counts_bat[q].append(cnt_bat)
        cnt_es  = int(np.sum(masks_es[i_q]));   cnt_es  and counts_es[q].append(cnt_es)

        if masks_bat[i_q].any():
            avg_bat = float(ed_bat[masks_bat[i_q]].mean())
            avg_bat and avg_ed_bat[q].append(avg_bat)
        if masks_es[i_q].any():
            avg_es = float(ed_es[masks_es[i_q]].mean())
            avg_es and avg_ed_es[q].append(avg_es)

    # Mean intersection points (ACEF, ACCF)
    mean_pts_bat.append((tco2_bat / np.sum(ed_bat), tcost_bat / np.sum(ed_bat)))
    mean_pts_es.append((tco2_es / np.sum(ed_es), tcost_es / np.sum(ed_es)))

mean_pts_bat = np.array(mean_pts_bat)
mean_pts_es  = np.array(mean_pts_es)

# ---- Load BASECASE directly from existing pickle ----
with open(BASECASE_PKL_PATH, "rb") as f:
    data_base = pickle.load(f)

uc_bat_base   = np.array(data_base[8])
uc_es_base    = np.array(data_base[9])
co2_bat_base  = np.array(data_base[10])
co2_es_base   = np.array(data_base[11])
ed_bat_base   = np.array(data_base[22])
ed_es_base    = np.array(data_base[23])
tcost_bat_base= data_base[12]
tcost_es_base = data_base[13]
tco2_bat_base = data_base[14]
tco2_es_base  = data_base[15]

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

baseline_counts_bat = {q: int(masks_bat_base[i].sum()) for i, q in enumerate(['Z1','Z2','Z3','Z4'])}
baseline_counts_es  = {q: int(masks_es_base[i].sum())  for i, q in enumerate(['Z1','Z2','Z3','Z4'])}
baseline_avg_ed_bat = {q: (float(ed_bat_base[masks_bat_base[i]].mean()) if masks_bat_base[i].any() else None)
                       for i, q in enumerate(['Z1','Z2','Z3','Z4'])}
baseline_avg_ed_es  = {q: (float(ed_es_base[masks_es_base[i]].mean()) if masks_es_base[i].any() else None)
                       for i, q in enumerate(['Z1','Z2','Z3','Z4'])}