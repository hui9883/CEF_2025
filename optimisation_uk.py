import os
import pandas as pd
import pypsa

EPS = 1e-6  # tiny tie-breaker weight
print(pypsa.__version__)

# ---------- Utilities ----------
def load_network(nc_path: str) -> pypsa.Network:
    """Load a pre-built PyPSA network from NetCDF and ensure required fields exist."""
    n = pypsa.Network(nc_path)

    # Ensure 'co2_emissions' exists on generators (fallback by carrier if missing)
    if "co2_emissions" not in n.generators.columns:
        co2_by_carrier = {
            'Wind': 0, 'PV': 0, 'Wind offshore': 0, 'Hydro': 0, 'Nuclear': 0,
            'Biogas_CHP': 0.19656, 'Biogas': 0.19656, 'CCGT': 0.36,
            'Biomass': 0, 'Biomass_CHP': 0, 'SCGT': 0.46, 'SCGT_CHP': 0.46, 'Oil': 0.65
        }
        n.generators["co2_emissions"] = n.generators["carrier"].map(co2_by_carrier).fillna(0)

    # Ensure 'marginal_cost' exists on generators
    if "marginal_cost" not in n.generators.columns:
        n.generators["marginal_cost"] = 0.0

    return n


def add_zero_soc_constraints(network: pypsa.Network, model):
    """Force battery store SOC to zero at the end of each day and all stores to zero at the final snapshot."""
    # Identify battery stores primarily via carrier; fallback to name contains "Battery"
    if "carrier" in network.stores.columns:
        battery_ids = network.stores.index[
            network.stores.carrier.astype(str).str.contains("Battery", case=False, na=False)
        ].tolist()
    else:
        battery_ids = [s for s in network.stores.index if "Battery" in str(s)]

    snaps = pd.DatetimeIndex(network.snapshots)
    # End-of-day snapshots (the last snapshot for each calendar day)
    eod_snaps = snaps.to_series().groupby(snaps.normalize()).max().tolist()

    for store in battery_ids:
        for ts in eod_snaps:
            model.add_constraints(
                model.variables["Store-e"].loc[ts, store] == 0,
                name=f"EndOfDayZeroStorage_{store}_{ts.date()}"
            )

    # Final snapshot constraint for all stores
    last_snap = snaps.max()
    for store in network.stores.index:
        model.add_constraints(
            model.variables["Store-e"].loc[last_snap, store] == 0,
            name=f"FinalZeroStorage_{store}_{last_snap}"
        )

def solve_and_save_network(
    network: pypsa.Network,
    objective: str,
    save_dir: str,
    nc_filename: str = "network_ff_constrained_time.nc",
) -> str:
    """
    Build the optimization model, set the objective, solve it, and save the solved
    network to NetCDF. No CSV exports here.
    Returns the absolute path to the saved NetCDF file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build model and add SOC constraints
    mdl = network.optimize.create_model()
    add_zero_soc_constraints(network, mdl)

    # Align variables/parameters
    gen_p = mdl.variables["Generator-p"]  # xarray: dims ["snapshot", "Generator"]
    gen_index = gen_p.coords["Generator"].to_index()
    mc = network.generators["marginal_cost"].reindex(gen_index, fill_value=0)
    co2 = network.generators["co2_emissions"].reindex(gen_index, fill_value=0)

    # Objective
    if objective == "min_cost":
        aligned = mc + EPS * co2
        mdl.objective = (gen_p * aligned).sum().sum()
    elif objective == "min_CO2":
        aligned = co2 + EPS * mc
        mdl.objective = (gen_p * aligned).sum(["snapshot", "Generator"])
    else:
        raise ValueError("objective must be 'min_cost' or 'min_CO2'")

    # Solve
    network.optimize.solve_model(solver_name="gurobi", solver_options={"Threads": 10})

    # Save solved network
    nc_path = os.path.join(save_dir, nc_filename)
    network.export_to_netcdf(nc_path)
    return os.path.abspath(nc_path)


def export_results_from_nc(
    nc_path: str,
    save_dir: str | None = None,
    objective: str | None = None,
):
    """
    Load the solved network from NetCDF, compute KPIs and all aggregations,
    and export the required CSV files.
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(nc_path))
    os.makedirs(save_dir, exist_ok=True)

    # Load network
    network = pypsa.Network(nc_path)

    # KPIs
    gen_power = network.generators_t.p
    mc_full = network.generators["marginal_cost"].reindex(gen_power.columns).fillna(0)
    total_cost = (gen_power * mc_full).sum().sum()
    total_emis = (
        gen_power
        * network.generators["co2_emissions"].reindex(gen_power.columns).fillna(0)
    ).sum().sum()
    prefix = f"[{objective}] " if objective else ""
    print(f"{prefix}Total Operation Cost: {total_cost}")
    print(f"{prefix}Total Emissions: {total_emis}")

    # Bulk CSV exports
    network.loads_t.p.to_csv(os.path.join(save_dir, "demand_p.csv"), header=True)
    network.links_t.p0.to_csv(os.path.join(save_dir, "links_p0_results.csv"), header=True)
    network.links_t.p1.to_csv(os.path.join(save_dir, "links_p1_results.csv"), header=True)
    network.stores_t.p.to_csv(os.path.join(save_dir, "store_p_results.csv"), header=True)
    network.buses_t.p.to_csv(os.path.join(save_dir, "buses_p_results.csv"), header=True)

    # p_nom_opt fallback if not present
    links_p_nom = getattr(network.links, "p_nom_opt", None)
    if links_p_nom is None or not hasattr(links_p_nom, "to_csv"):
        links_p_nom = network.links.p_nom
    links_p_nom.to_csv(os.path.join(save_dir, "links_p_nom_opt.csv"), header=True)

    # stores_t.e and e_nom_opt fallback
    network.stores_t.e.to_csv(os.path.join(save_dir, "stores_e.csv"), header=True)
    stores_e_nom = getattr(network.stores, "e_nom_opt", None)
    if stores_e_nom is None or not hasattr(stores_e_nom, "to_csv"):
        stores_e_nom = network.stores.e_nom
    stores_e_nom.to_csv(os.path.join(save_dir, "stores_e_nom_opt.csv"), header=True)

    # Aggregations by carrier
    store_e_by_carrier = network.stores_t.e.T.groupby(network.stores.carrier).sum().T
    store_e_by_carrier.to_csv(os.path.join(save_dir, "store_e_carrier_results.csv"), header=True)

    p_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
    p_by_carrier.to_csv(os.path.join(save_dir, "gen_p_carrier_results.csv"), header=True)

    store_p_by_carrier = network.stores_t.p.T.groupby(network.stores.carrier).sum().T
    store_p_by_carrier.to_csv(os.path.join(save_dir, "store_p_carrier_results.csv"), header=True)

    # Per-snapshot max output and remaining output
    p_max_pu_full = network.generators_t.p_max_pu.reindex(columns=network.generators.index)
    missing = network.generators.index.difference(network.generators_t.p_max_pu.columns)
    for g in missing:
        p_max_pu_full[g] = network.generators.at[g, "p_max_pu"]
    snapshot_max_output = p_max_pu_full.multiply(network.generators["p_nom"], axis=1)

    network.generators_t["max_output"] = snapshot_max_output
    network.generators_t["remain_output"] = (network.generators_t["max_output"] - network.generators_t.p)
    if "efficiency" in network.generators.columns:
        eff = network.generators["efficiency"].reindex(network.generators_t["remain_output"].columns).fillna(1.0)
        network.generators_t["remain_output"] *= eff

    # Remaining output by bus+carrier
    rem = network.generators_t["remain_output"].T
    rem.index = rem.index.astype(str)
    rem["bus"] = rem.index.map(network.generators["bus"])
    rem["carrier"] = rem.index.map(network.generators["carrier"])
    rem["bus_carrier"] = rem["bus"] + "_" + rem["carrier"]
    p_by_bus_carrier = rem.groupby("bus_carrier").sum().drop(columns=["bus", "carrier"], errors="ignore").T
    p_by_bus_carrier.to_csv(os.path.join(save_dir, "p_by_bus_carrier.csv"), index=True)

    # Remaining output by carrier
    re_p_by_carrier = network.generators_t["remain_output"].T.groupby(network.generators.carrier).sum().T
    re_p_by_carrier.to_csv(os.path.join(save_dir, "re_p_carrier_results.csv"), header=True)

    # Generator output grouped by bus+carrier
    gen_tp = network.generators_t.p.T
    gen_tp.index = gen_tp.index.astype(str)
    gen_tp["bus"] = gen_tp.index.map(network.generators["bus"])
    gen_tp["carrier"] = gen_tp.index.map(network.generators["carrier"])
    gen_tp["bus_carrier"] = gen_tp["bus"] + "_" + gen_tp["carrier"]
    gen_by_bus_carrier = gen_tp.groupby("bus_carrier").sum().drop(columns=["bus", "carrier"], errors="ignore").T
    gen_by_bus_carrier.to_csv(os.path.join(save_dir, "gen_by_bus_carrier.csv"), index=True)

    # generators_t.p sorted by (carrier, marginal_cost)
    generators_t_p = network.generators_t.p.copy()
    generators = network.generators.copy()
    generators_t_p.loc["carrier"] = generators_t_p.columns.map(generators["carrier"])
    generators_t_p.loc["marginal_cost"] = generators_t_p.columns.map(generators["marginal_cost"])
    sorted_cols = generators[["carrier", "marginal_cost"]].sort_values(
        by=["carrier", "marginal_cost"], ascending=[True, True]
    ).index
    generators_t_p = generators_t_p[sorted_cols]
    for row in generators_t_p.index:
        if row != "carrier":
            generators_t_p.loc[row] = pd.to_numeric(generators_t_p.loc[row], errors="coerce")
    generators_t_p.to_csv(os.path.join(save_dir, "sorted_generators_t_p.csv"))


# ---------- Main: 2 years × 2 objectives = 4 result folders ----------
import os

RESULTS_ROOT = "./results/"
os.makedirs(RESULTS_ROOT, exist_ok=True)

paths = {
    2023: "./data/network_2023.nc",
    2030: "./data/network_2030.nc",
}

for year in sorted(paths):
    nc = paths[year]
    net = load_network(nc)

    # Min cost
    dir_cost = os.path.join(RESULTS_ROOT, f"min_cost_{year}/")
    solved_nc_cost = solve_and_save_network(net.copy(), "min_cost", dir_cost)
    export_results_from_nc(solved_nc_cost, save_dir=dir_cost, objective="min_cost")

    # Min CO2
    dir_co2 = os.path.join(RESULTS_ROOT, f"min_CO2_{year}/")
    solved_nc_co2 = solve_and_save_network(net.copy(), "min_CO2", dir_co2)
    export_results_from_nc(solved_nc_co2, save_dir=dir_co2, objective="min_CO2")
