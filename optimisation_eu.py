import os
from typing import Optional

import numpy as np
import pandas as pd
import pypsa

EPS = 1e-6  # tiny tie-breaker weight for objectives
print(pypsa.__version__)

# ---------- Loaders & helpers ----------
def load_eu_network(nc_path: str) -> pypsa.Network:
    """Load a pre-built EU PyPSA network from NetCDF and backfill required fields if missing."""
    n = pypsa.Network(nc_path)

    # Ensure CO2 factors on generators
    if "co2_emissions" not in n.generators.columns:
        co2_by_carrier = {
            'Lignite': 1.06, 'Hard coal': 0.867, 'CCGT': 0.36, 'SCGT': 0.46, 'Oil': 0.65,
            'Biomass': 0.0, 'Biogas': 0.19656, 'BECCS': 0, 'Wind': 0, 'Wind offshore': 0,
            'PV': 0, 'Hydro': 0, 'Other RES': 0, 'Battery': 0, 'Other storage': 0, 'Nuclear': 0, 'DSR': 0,
        }
        n.generators["co2_emissions"] = n.generators["carrier"].map(co2_by_carrier).fillna(0)

    # Ensure marginal cost on generators
    if "marginal_cost" not in n.generators.columns:
        n.generators["marginal_cost"] = 0.0

    # Ensure CO2 factors on storage units
    if "co2_emissions" not in n.storage_units.columns:
        co2_by_carrier_s = {
            'Lignite': 1.06, 'Hard coal': 0.867, 'CCGT': 0.36, 'SCGT': 0.46, 'Oil': 0.65,
            'Biomass': 0.0, 'Biogas': 0.19656, 'BECCS': 0, 'Wind': 0, 'Wind offshore': 0,
            'PV': 0, 'Hydro': 0, 'Other RES': 0, 'Battery': 0, 'Other storage': 0, 'Nuclear': 0, 'DSR': 0,
        }
        n.storage_units["co2_emissions"] = n.storage_units["carrier"].map(co2_by_carrier_s).fillna(0)

    # Ensure marginal cost on storage units
    if "marginal_cost" not in n.storage_units.columns:
        n.storage_units["marginal_cost"] = 0.0

    return n


def add_zero_soc_constraints(network, n, year=None,
                                           st_battery_names=None,
                                           st_other_names=None):
    """
    Mirror the user's original logic exactly:
      - For each day of `year`, if "DD/MM/YYYY 23:00:00" exists in snapshots,
        set StorageUnit-state_of_charge == 0 for Battery units.
      - For the year's final "DD/MM/YYYY 23:00:00", set SOC == 0 only for Other storage.
    Everything is done with string timestamps; no datetime conversion on coords.
    """

    # Use the model variable's snapshot coordinate for membership checks (strings)
    var = n.variables["StorageUnit-state_of_charge"]
    snap_idx_str = var.coords["snapshot"].to_index().astype(str)
    snap_set = set(snap_idx_str)

    # Infer year from the first snapshot string if not provided (expects DD/MM/YYYY ...)
    if year is None:
        first_str = snap_idx_str[0]
        # split "DD/MM/YYYY HH:MM:SS" -> "DD/MM/YYYY" -> YYYY
        year = int(first_str.split(" ")[0].split("/")[-1])

    # Resolve StorageUnit names if not provided
    su_names = var.coords["StorageUnit"].to_index().astype(str)
    if st_battery_names is None:
        # Prefer carrier tag if present; else fallback to name pattern
        if "carrier" in network.storage_units.columns:
            batt_mask = network.storage_units.carrier.astype(str).str.contains("Battery", case=False, na=False)
            st_battery_names = [su for su in su_names if su in network.storage_units.index[batt_mask]]
        else:
            st_battery_names = [su for su in su_names if "battery" in su.lower()]
    if st_other_names is None:
        if "carrier" in network.storage_units.columns:
            other_mask = network.storage_units.carrier.astype(str).str.contains("Other", case=False, na=False)
            st_other_names = [su for su in su_names if su in network.storage_units.index[other_mask]]
        else:
            st_other_names = [su for su in su_names if "other" in su.lower() or "other_storage" in su.lower()]

    # ---- Batteries: end-of-day 23:00 each day ----
    for store in st_battery_names:
        for day in pd.date_range("2030-01-01", "2030-12-31", freq="D"):
            ts = (day + pd.Timedelta(hours=23)).strftime("%m/%d/%Y %H:%M:%S")
            if ts in network.snapshots:
                n.add_constraints(
                    n.variables["StorageUnit-state_of_charge"].loc[ts, store] == 0,
                    name=f"ZeroSOC_Battery_{store}_{ts}"
                )

    # ---- Other storage: final day @ 23:00 only ----
    ts_final = (pd.Timestamp("2030-12-31 23:00:00")).strftime("%m/%d/%Y %H:%M:%S")
    if ts_final in network.snapshots:
        for store in st_other_names:
            n.add_constraints(
                n.variables["StorageUnit-state_of_charge"].loc[ts_final, store] == 0,
                name=f"ZeroSOC_Other_{store}_{ts_final}"
            )

def solve_and_save_eu(
    n: pypsa.Network,
    objective: str,
    save_dir: str,
    nc_filename: str = "network_ff_constrained_time.nc",
) -> str:
    """
    Build the model, set the objective (min_cost or min_CO2) including both
    Generator and StorageUnit terms, solve, and save the solved network to NetCDF.
    No CSV exports here.
    Returns the absolute path to the saved NetCDF file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create model and add SOC constraints
    mdl = n.optimize.create_model()
    add_zero_soc_constraints(n, mdl, year=2030)

    # Variables
    g_p = mdl.variables["Generator-p"]
    su_p_disp = mdl.variables["StorageUnit-p_dispatch"]

    # Indices aligned to variables
    g_idx = g_p.coords["Generator"].to_index()
    su_idx = su_p_disp.coords["StorageUnit"].to_index()

    # Parameters (aligned)
    mc_g = n.generators["marginal_cost"].reindex(g_idx, fill_value=0)
    co2_g = n.generators["co2_emissions"].reindex(g_idx, fill_value=0)

    mc_s = n.storage_units["marginal_cost"].reindex(su_idx, fill_value=0)
    co2_s = n.storage_units["co2_emissions"].reindex(su_idx, fill_value=0)

    # Objective
    if objective == "min_cost":
        obj_g = (g_p * (mc_g + EPS * co2_g)).sum().sum()
        obj_s = (su_p_disp * (mc_s + EPS * co2_s)).sum().sum()
    elif objective == "min_CO2":
        obj_g = (g_p * (co2_g + EPS * mc_g)).sum(["snapshot", "Generator"])
        obj_s = (su_p_disp * (co2_s + EPS * mc_s)).sum(["snapshot", "StorageUnit"])
    else:
        raise ValueError("objective must be 'min_cost' or 'min_CO2'")

    mdl.objective = obj_g + obj_s

    # Solve
    n.optimize.solve_model(solver_name="gurobi", solver_options={"Threads": 1})
    # n.optimize.solve_model(solver_name="HIGHS", solver_options={"threads": 10})

    # Save solved network
    nc_path = os.path.join(save_dir, nc_filename)
    n.export_to_netcdf(nc_path)
    return os.path.abspath(nc_path)


def export_results_from_nc_eu(
    nc_path: str,
    save_dir: Optional[str] = None,
    objective: Optional[str] = None,
):
    """
    Load the solved network from NetCDF, compute KPIs and all aggregations,
    and export the required CSV files (including StorageUnit and hydro logic).
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(nc_path))
    os.makedirs(save_dir, exist_ok=True)

    # Load network
    n = pypsa.Network(nc_path)

    # KPIs: Generators + StorageUnits (discharge only for storage)
    gen_p = n.generators_t.p
    su_p = n.storage_units_t.p  # >0 = discharge to bus, <0 = charging

    # Align parameters
    mc_g = n.generators["marginal_cost"].reindex(gen_p.columns).fillna(0)
    co2_g = n.generators["co2_emissions"].reindex(gen_p.columns).fillna(0)
    mc_s = n.storage_units["marginal_cost"].reindex(su_p.columns).fillna(0)
    co2_s = n.storage_units["co2_emissions"].reindex(su_p.columns).fillna(0)

    # Use only the discharge part for storage to be consistent with the model objective
    su_dispatch = su_p.clip(lower=0)

    gen_cost = (gen_p * mc_g).sum().sum()
    su_cost = (su_dispatch * mc_s).sum().sum()
    gen_emis = (gen_p * co2_g).sum().sum()
    su_emis = (su_dispatch * co2_s).sum().sum()

    total_cost = gen_cost + su_cost
    total_emis = gen_emis + su_emis

    prefix = f"[{objective}] " if objective else ""
    print(f"{prefix}Total Operation Cost (Generators + Storage): {total_cost}")
    print(f"{prefix}Total Emissions (Generators + Storage): {total_emis}")

    # NetCDF-derived CSV exports
    n.loads_t.p.to_csv(os.path.join(save_dir, "demand_p.csv"), header=True)
    n.buses_t.p.to_csv(os.path.join(save_dir, "buses_p_results.csv"), header=True)

    # StorageUnit SOC and pseudo link flows (charge/discharge breakdown)
    n.storage_units_t.state_of_charge.to_csv(os.path.join(save_dir, "stores_e.csv"), header=True)

    p = n.storage_units_t.p  # positive = dispatch to bus; negative = charging from bus
    charge = -p.clip(upper=0)  # positive charging
    charge.columns = [f"{c}_charger" for c in charge.columns]
    discharge_raw = -p.clip(lower=0)  # raw (sign-flipped) dispatch
    discharge_copy = -discharge_raw.copy()  # used for hydro post-processing
    hydro = discharge_copy.filter(regex='_Reservoir$').copy()
    hydro.columns = hydro.columns.str.replace('_Reservoir', '_Hydro', regex=False)
    discharge_raw.columns = [f"{c}_discharger" for c in discharge_raw.columns]
    pd.concat([charge, discharge_raw], axis=1).to_csv(
        os.path.join(save_dir, "links_p1_results.csv"), header=True
    )

    # Aggregations by carrier
    store_e_by_carrier = (
        n.storage_units_t.state_of_charge.T.groupby(n.storage_units.carrier).sum().T
    )
    store_e_by_carrier.to_csv(os.path.join(save_dir, "store_e_carrier_results.csv"), header=True)

    store_p_by_carrier = (
        n.storage_units_t.p.abs().T.groupby(n.storage_units.carrier).sum().T
    )
    store_p_by_carrier.to_csv(os.path.join(save_dir, "store_p_carrier_results.csv"), header=True)

    p_by_carrier = n.generators_t.p.T.groupby(n.generators.carrier).sum().T
    p_by_carrier.to_csv(os.path.join(save_dir, "gen_p_carrier_results.csv"), header=True)

    # Dynamic generator max output and remaining output
    p_max_pu_full = n.generators_t.p_max_pu.reindex(columns=n.generators.index)
    missing_g = n.generators.index.difference(n.generators_t.p_max_pu.columns)
    for g in missing_g:
        p_max_pu_full[g] = n.generators.at[g, "p_max_pu"]
    snapshot_max_output = p_max_pu_full.multiply(n.generators["p_nom"], axis=1)

    n.generators_t["max_output"] = snapshot_max_output
    n.generators_t["remain_output"] = (n.generators_t["max_output"] - n.generators_t.p)
    if "efficiency" in n.generators.columns:
        eff = n.generators["efficiency"].reindex(n.generators_t["remain_output"].columns).fillna(1.0)
        n.generators_t["remain_output"] *= eff

    # Remaining output by bus+carrier
    remain_output = n.generators_t["remain_output"].T
    remain_output.index = remain_output.index.astype(str)
    remain_output["bus"] = remain_output.index.map(n.generators["bus"])
    remain_output["carrier"] = remain_output.index.map(n.generators["carrier"])
    remain_output["bus_carrier"] = remain_output["bus"] + "_" + remain_output["carrier"]
    p_by_bus_carrier = remain_output.groupby("bus_carrier").sum().drop(columns=["bus", "carrier"], errors="ignore").T

    # Hydro remaining headroom calculation (based on installed power and SOC trajectory)
    res_units = [u for u in n.storage_units.index if u.endswith("_Reservoir")]
    if len(res_units) > 0:
        p_nom = n.storage_units.p_nom.loc[res_units]
        soc = n.storage_units_t.state_of_charge[res_units]
        n_steps = len(soc.index)
        P_inst = pd.DataFrame([p_nom.values] * n_steps, columns=p_nom.index, index=soc.index)
        P_inst_cumsum = P_inst.cumsum()
        soc_eff = (soc - P_inst_cumsum).clip(lower=0)

        new_cols = [c.replace("_Reservoir", "_Hydro") for c in P_inst.columns]
        P_eff = pd.DataFrame(np.minimum(P_inst.values, soc_eff.values), index=soc.index, columns=new_cols)
        remaining = (P_eff - hydro.reindex_like(P_eff).fillna(0)).clip(lower=0)
        for col in remaining.columns:
            p_by_bus_carrier[col] = p_by_bus_carrier.get(col, 0) + remaining[col]

    p_by_bus_carrier.to_csv(os.path.join(save_dir, "p_by_bus_carrier.csv"), index=True)

    # Remaining output by carrier
    re_p_by_carrier = n.generators_t["remain_output"].T.groupby(n.generators.carrier).sum().T
    re_p_by_carrier.to_csv(os.path.join(save_dir, "re_p_carrier_results.csv"), header=True)

    # Generator output by bus+carrier (with hydro addition)
    gen_tp = n.generators_t.p.T
    gen_tp.index = gen_tp.index.astype(str)
    gen_tp["bus"] = gen_tp.index.map(n.generators["bus"])
    gen_tp["carrier"] = gen_tp.index.map(n.generators["carrier"])
    gen_tp["bus_carrier"] = gen_tp["bus"] + "_" + gen_tp["carrier"]
    gen_by_bus_carrier = gen_tp.groupby("bus_carrier").sum().drop(columns=["bus", "carrier"], errors="ignore").T
    if 'hydro' in locals():
        for col in hydro.columns:
            gen_by_bus_carrier[col] = gen_by_bus_carrier.get(col, 0) + hydro[col]
    gen_by_bus_carrier.to_csv(os.path.join(save_dir, "gen_by_bus_carrier.csv"), index=True)

    # Sorted generators_t.p by (carrier, marginal_cost)
    generators_t_p = n.generators_t.p.copy()
    generators = n.generators.copy()
    generators_t_p.loc["carrier"] = generators_t_p.columns.map(generators["carrier"])
    generators_t_p.loc["marginal_cost"] = generators_t_p.columns.map(generators["marginal_cost"])
    sorted_columns = generators[["carrier", "marginal_cost"]].sort_values(
        by=["carrier", "marginal_cost"], ascending=[True, True]
    ).index
    generators_t_p = generators_t_p[sorted_columns]
    for row_name in generators_t_p.index:
        if row_name != "carrier":
            generators_t_p.loc[row_name] = pd.to_numeric(generators_t_p.loc[row_name], errors="coerce")
    generators_t_p.to_csv(os.path.join(save_dir, "sorted_generators_t_p.csv"))


# # ---------- Run: 2 years × 2 objectives ----------
# import os
#
# RESULTS_ROOT = "./results/"  # root save directory as requested
# os.makedirs(RESULTS_ROOT, exist_ok=True)
#
# eu_paths = {
#     2023: "./data/network_eu_2023.nc",
#     2030: "./data/network_eu_2030.nc",
# }
#
# for year, nc in eu_paths.items():
#     net = load_eu_network(nc)
#
#     # Min cost
#     save_dir_cost = os.path.join(RESULTS_ROOT, f"eu_min_cost_{year}")
#     os.makedirs(save_dir_cost, exist_ok=True)
#     solved_nc_cost = solve_and_save_eu(net.copy(), "min_cost", save_dir_cost)
#     export_results_from_nc_eu(solved_nc_cost, save_dir=save_dir_cost, objective="min_cost")
#
#     # Min CO2
#     save_dir_co2 = os.path.join(RESULTS_ROOT, f"eu_min_CO2_{year}")
#     os.makedirs(save_dir_co2, exist_ok=True)
#     solved_nc_co2 = solve_and_save_eu(net.copy(), "min_CO2", save_dir_co2)
#     export_results_from_nc_eu(solved_nc_co2, save_dir=save_dir_co2, objective="min_CO2")