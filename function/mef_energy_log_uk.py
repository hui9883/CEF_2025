import numpy as np
import pandas as pd

def cef_bat_discharge_log(
    idx: int,
    region: str,
    regions: list[str],
    values: pd.DataFrame,
    cef_bat: list,
    cost_bat: list,
    df_battery_links_charge: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res_dis: np.ndarray,
    flow_matrix_dis: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict
) -> tuple:
    """
    UK (single-region view) — attribute CO₂ and cost for **battery DISCHARGE** at one time step.

    What this does
    --------------
    For the given time step `idx` and target `region`, meet the region's battery
    *discharge* requirement by drawing (1) local residual generation and then
    (2) external residual generation routed to this region. The function:
      - deducts used energy from `values` (local + external residual),
      - deducts used external residual from `flows_by_res_dis` (T, U, R_src, R_dst),
      - logs the source→region substitutions into `flow_matrix_dis` (R×R),
      - accumulates per-source usage in `resource_usage[src]["bat_dis"]`,
      - appends step CO₂ and cost to `cef_bat` and `cost_bat`.

    Parameters
    ----------
    idx : int
        Time-step (row) index.
    region : str
        Target region for which we allocate discharge.
    regions : list[str]
        All regions, used to index `flows_by_res_dis` and `flow_matrix_dis`.
    values : pd.DataFrame
        Residual generation at each step with columns like f"{region}_{src}".
        This frame is DECREMENTED in place for energy taken.
    cef_bat, cost_bat : list
        Collectors for step-level CO₂ (mass) and cost assigned to battery discharge.
    df_battery_links_charge : pd.DataFrame
        Must contain f"{region}_Battery_discharger" (negative for discharge).
    ratio : float
        Scaling (e.g., to convert power to energy or include efficiency).
    resource_usage : dict
        Accumulates per-source usage; this function appends to "bat_dis".
    flows_by_res_dis : np.ndarray
        Shape (T, U, R, R): external residual available per (time, resource, src_region, dst_region).
        This array is DECREMENTED for energy taken externally.
    flow_matrix_dis : pd.DataFrame
        R×R matrix (index/columns=regions) logging how much discharge substitutes
        residual generation from each source region to the target region.
    resources : list[str]
        Resource names in the intended DISPATCH order (e.g., high-carbon → low-carbon).
    CO2_FACTORS : dict[str, tuple[float, float]]
        {resource: (co2_intensity, marginal_cost)}.

    Returns
    -------
    tuple
        (cef_bat, cost_bat, values, resource_usage, flows_by_res_dis, flow_matrix_dis)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # Ensure a per-step slot exists for all sources
    for src in resource_usage:
        resource_usage[src].setdefault("bat_dis", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0
    ordered = resources  # discharge uses the order as-is

    # Positive energy need for allocation (discharger is negative by convention)
    need = -df_battery_links_charge.at[t, f"{region}_Battery_discharger"] * ratio
    if need <= 0:
        cef_bat.append(0.0)
        cost_bat.append(0.0)
        return cef_bat, cost_bat, values, resource_usage, flows_by_res_dis, flow_matrix_dis

    for src in ordered:
        if need <= 0:
            break

        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # External residual available to this region for this resource
        avail_ext = float(flows_by_res_dis[t, j, :, k].sum())
        # Total residual recorded locally for this resource
        total_here = float(values.at[values.index[t], col])
        # Local-only residual (excludes what is earmarked as external inflow)
        avail_loc = total_here - avail_ext

        # 1) Use local residual first
        used_loc = min(avail_loc, need)
        if used_loc > 0:
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["bat_dis"][-1] += used_loc
            flow_matrix_dis.loc[region, region] += used_loc

        # 2) Use external residual next (proportional to source-region contributions)
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res_dis[t, j, :, k] / avail_ext  # safe since avail_ext > 0
            for i, src_reg in enumerate(regions):
                amt = float(used_ext * prop[i])
                flows_by_res_dis[t, j, i, k] -= amt  # consume external inflow
                flow_matrix_dis.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext  # keep residual accounting consistent
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["bat_dis"][-1] += used_ext

    # 3) Fallback: unmet need → map to default thermal (no flow matrix record)
    if need > 0:
        co2_o, cst_o = CO2_FACTORS['CCGT']
        cef_sum  += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["bat_dis"][-1] += need
        need = 0

    cef_bat.append(cef_sum)
    cost_bat.append(cost_sum)
    return cef_bat, cost_bat, values, resource_usage, flows_by_res_dis, flow_matrix_dis


def cef_es_discharge_log(
    idx: int,
    region: str,
    regions: list[str],
    values: pd.DataFrame,
    cef_es: list,
    cost_es: list,
    df_storage_links: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res_dis: np.ndarray,
    flow_matrix_dis: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict
) -> tuple:
    """
    UK (single-region view) — attribute CO₂ and cost for **LDES/ES DISCHARGE** at one time step.

    Same logic as battery discharge, but using ES (OtherStorage) series.

    Returns
    -------
    tuple
        (cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # Ensure a per-step slot exists for all sources
    for src in resource_usage:
        resource_usage[src].setdefault("es_dis", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0
    ordered = resources  # discharge: use order as-is

    # Positive energy need for allocation (discharger is negative)
    need = -df_storage_links.at[t, f"{region}_OtherStorage_discharger"] * ratio
    if need <= 0:
        cef_es.append(0.0)
        cost_es.append(0.0)
        return cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis

    for src in ordered:
        if need <= 0:
            break

        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        avail_ext  = float(flows_by_res_dis[t, j, :, k].sum())
        total_here = float(values.at[values.index[t], col])
        avail_loc  = total_here - avail_ext

        # 1) Local residual
        used_loc = min(avail_loc, need)
        if used_loc > 0:
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["es_dis"][-1] += used_loc
            flow_matrix_dis.loc[region, region] += used_loc

        # 2) External residual
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res_dis[t, j, :, k] / avail_ext
            for i, src_reg in enumerate(regions):
                amt = float(used_ext * prop[i])
                flows_by_res_dis[t, j, i, k] -= amt
                flow_matrix_dis.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["es_dis"][-1] += used_ext

    # Fallback to default thermal
    if need > 0:
        co2_o, cst_o = CO2_FACTORS['CCGT']
        cef_sum  += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["es_dis"][-1] += need
        need = 0

    cef_es.append(cef_sum)
    cost_es.append(cost_sum)
    return cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis


def cef_bat_log(
    idx: int,
    region: str,
    regions: list[str],
    others: dict,
    values: pd.DataFrame,
    cef_bat: list,
    cost_bat: list,
    df_battery_links_charge: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res: np.ndarray,
    flow_matrix: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict
) -> tuple:
    """
    UK (single-region view) — attribute CO₂ and cost for **battery CHARGING** at one time step.

    Charging rule:
      - Use the REVERSED merit (`reversed(resources)`) so that if `resources`
        is high-carbon → low-carbon (or high-cost → low-cost), charging prefers
        the cleaner/cheaper end first.
      - First consume *external* inflows proportionally (flows_by_res),
        then consume local residual.

    Returns
    -------
    tuple
        (cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # Ensure a per-step slot exists for all sources
    for src in resource_usage:
        resource_usage[src].setdefault("bat_cha", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0

    # Charging: reverse the dispatch order
    ordered = list(reversed(resources))

    # Positive charging need
    need = df_battery_links_charge.at[t, f"{region}_Battery_charger"] * ratio
    if need <= 0:
        cef_bat.append(0.0)
        cost_bat.append(0.0)
        return cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix

    for src in ordered:
        if need <= 0:
            break

        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # External inflow available to this region for this resource
        avail_ext  = float(flows_by_res[t, j, :, k].sum())
        total_here = float(values.at[values.index[t], col])
        avail_loc  = total_here - avail_ext  # local residual excluding external earmark

        # 1) Take external inflow proportionally
        used_ext = min(avail_ext, need)
        if used_ext > 0:
            prop = (flows_by_res[t, j, :, k] / avail_ext) if avail_ext > 0 else np.zeros(R)
            for i, src_reg in enumerate(regions):
                amt = float(used_ext * prop[i])
                flows_by_res[t, j, i, k] -= amt
                flow_matrix.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["bat_cha"][-1] += used_ext

        # 2) Then take local residual
        if need > 0 and avail_loc > 0:
            used_loc = min(avail_loc, need)
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["bat_cha"][-1] += used_loc
            flow_matrix.loc[region, region] += used_loc

    # 3) Fallback: unmet charging need uses region-specific 'others' mapping
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum  += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["bat_cha"][-1] += need
        flow_matrix.loc[region, region] += need
        need = 0

    cef_bat.append(cef_sum)
    cost_bat.append(cost_sum)
    return cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix


def cef_es_log(
    idx: int,
    region: str,
    regions: list[str],
    others: dict,
    values: pd.DataFrame,
    cef_es: list,
    cost_es: list,
    df_storage_links: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res: np.ndarray,
    flow_matrix: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict
) -> tuple:
    """
    UK (single-region view) — attribute CO₂ and cost for **LDES/ES CHARGING** at one time step.

    Same charging rule as batteries (use reversed(resources); external first, then local).

    Returns
    -------
    tuple
        (cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # Ensure a per-step slot exists for all sources
    for src in resource_usage:
        resource_usage[src].setdefault("es_cha", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0
    ordered = list(reversed(resources))

    # Positive charging need for ES
    need = df_storage_links.at[t, f"{region}_OtherStorage_charger"] * ratio
    if need <= 0:
        cef_es.append(0.0)
        cost_es.append(0.0)
        return cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix

    for src in ordered:
        if need <= 0:
            break

        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        avail_ext  = float(flows_by_res[t, j, :, k].sum())
        total_here = float(values.at[values.index[t], col])
        avail_loc  = total_here - avail_ext

        # 1) External inflow
        used_ext = min(avail_ext, need)
        if used_ext > 0:
            prop = (flows_by_res[t, j, :, k] / avail_ext) if avail_ext > 0 else np.zeros(R)
            for i, src_reg in enumerate(regions):
                amt = float(used_ext * prop[i])
                flows_by_res[t, j, i, k] -= amt
                flow_matrix.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["es_cha"][-1] += used_ext

        # 2) Local residual
        if need > 0 and avail_loc > 0:
            used_loc = min(avail_loc, need)
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["es_cha"][-1] += used_loc
            flow_matrix.loc[region, region] += used_loc

    # 3) Fallback: region-specific 'others'
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum  += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["es_cha"][-1] += need
        flow_matrix.loc[region, region] += need
        need = 0

    cef_es.append(cef_sum)
    cost_es.append(cost_sum)
    return cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix


def cycle_analysis(
    process_times_bat, process_ratios_bat,
    process_times_es,  process_ratios_es,
    others, gen_bus_carrier, df_storage_links, df_gen_remain,
    region, regions, CO2_FACTORS, resource_usage,
    flows_by_res, flow_matrices, flows_by_res_dis, resources
):
    """
    UK (single region) — iterate over detected charge/discharge cycles and compute
    per-cycle CO₂ metrics and costs for BAT and ES.

    Notes
    -----
    * `resources` is the baseline merit order (e.g., high-carbon → low-carbon or
      high-cost → low-cost). Discharge uses it as-is; charge uses reversed order.
    * Sign convention:
        - f"{region}_..._charger"  : positive for charging energy
        - f"{region}_..._discharger": negative for discharging energy
    * `flows_by_res` / `flows_by_res_dis` have shape (T, U, R_src, R_dst).
      They are decremented when external energy is used.
    * `flow_matrices` is a dict with DataFrames for 'bat_cha', 'bat_dis', 'es_cha', 'es_dis'
      (R×R, index/columns = regions) to track source→sink substitutions.

    Returns
    -------
    tuple
        (
          unit_cost_bat_cycle, unit_cost_es_cycle,
          co2_bat_charge_cycle, co2_es_charge_cycle,
          energy_bat_charge_cycle, energy_es_charge_cycle,
          energy_bat_discharged_cycle, energy_es_discharged_cycle,
          emissions_bat_charged_cycle, emissions_es_charged_cycle,
          emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
          co2_delta_bat_emissions, co2_delta_es_emissions,
          cost_delta_bat, cost_delta_es,
          resource_usage
        )
    """
    # ---------------- per-cycle collectors ----------------
    energy_bat_charge_cycle, energy_es_charge_cycle = [], []
    energy_bat_discharged_cycle, energy_es_discharged_cycle = [], []
    co2_bat_charge_cycle, co2_es_charge_cycle = [], []
    unit_cost_bat_cycle, unit_cost_es_cycle = [], []
    emissions_bat_charged_cycle, emissions_es_charged_cycle = [], []
    emissions_bat_discharged_cycle, emissions_es_discharged_cycle = [], []
    co2_delta_bat_emissions, co2_delta_es_emissions = [], []
    cost_delta_bat, cost_delta_es = [], []

    gen_bus_carrier_0 = gen_bus_carrier  # working copy decremented by charging allocators

    # ================= Battery cycles =================
    for cycle_number, times in process_times_bat.items():
        energy_bat_charge = (df_storage_links[f"{region}_Battery_charger"].iloc[times] *
                             process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links[f"{region}_Battery_discharger"].iloc[times] *
                                process_ratios_bat[cycle_number]).sum()

        cef_bat_charge, cef_bat_discharge = [], []
        cost_bat_charge, cost_bat_discharge = [], []

        resource_bat = {src: {"bat_cha": [], "bat_dis": []} for src in resource_usage.keys()}

        # Step allocation over the cycle window
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]

            # Battery CHARGE (external then local; reversed order)
            cef_bat_charge, cost_bat_charge, gen_bus_carrier_0, resource_bat, \
            flows_by_res, flow_matrices['bat_cha'] = cef_bat_log(
                idx, region, regions, others, gen_bus_carrier_0,
                cef_bat_charge, cost_bat_charge, df_storage_links, ratio,
                resource_bat, flows_by_res, flow_matrices['bat_cha'],
                resources, CO2_FACTORS
            )

            # Battery DISCHARGE (local then external; order as-is)
            cef_bat_discharge, cost_bat_discharge, df_gen_remain, resource_bat, \
            flows_by_res_dis, flow_matrices['bat_dis'] = cef_bat_discharge_log(
                idx, region, regions, df_gen_remain,
                cef_bat_discharge, cost_bat_discharge, df_storage_links, ratio,
                resource_bat, flows_by_res_dis, flow_matrices['bat_dis'],
                resources, CO2_FACTORS
            )

        # Aggregate per-cycle deltas
        emissions_bat_charged    = float(np.sum(cef_bat_charge))
        emissions_bat_discharged = float(np.sum(cef_bat_discharge))
        cost_bat_charged         = float(np.sum(cost_bat_charge))
        cost_bat_discharged      = float(np.sum(cost_bat_discharge))

        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged
        delta_bat_cost      = -cost_bat_discharged + cost_bat_charged

        # Record cycle KPIs
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        unit_cost_bat_cycle.append(delta_bat_cost / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        cost_delta_bat.append(delta_bat_cost)

        # Roll per-source usage from this cycle window
        for src in resource_usage.keys():
            resource_usage[src]["bat_cha"].append(float(np.sum(resource_bat[src]["bat_cha"])))
            resource_usage[src]["bat_dis"].append(float(np.sum(resource_bat[src]["bat_dis"])))

    # ================= ES/LDES cycles =================
    for cycle_number, times in process_times_es.items():
        energy_es_charge = (df_storage_links[f"{region}_OtherStorage_charger"].iloc[times] *
                            process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links[f"{region}_OtherStorage_discharger"].iloc[times] *
                               process_ratios_es[cycle_number]).sum()

        cef_es_charge, cef_es_discharge = [], []
        cost_es_charge, cost_es_discharge = [], []

        resource_es = {src: {"es_cha": [], "es_dis": []} for src in resource_usage.keys()}

        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]

            # ES CHARGE
            cef_es_charge, cost_es_charge, gen_bus_carrier_0, resource_es, \
            flows_by_res, flow_matrices['es_cha'] = cef_es_log(
                idx, region, regions, others, gen_bus_carrier_0,
                cef_es_charge, cost_es_charge, df_storage_links, ratio,
                resource_es, flows_by_res, flow_matrices['es_cha'],
                resources, CO2_FACTORS
            )

            # ES DISCHARGE
            cef_es_discharge, cost_es_discharge, df_gen_remain, resource_es, \
            flows_by_res_dis, flow_matrices['es_dis'] = cef_es_discharge_log(
                idx, region, regions, df_gen_remain,
                cef_es_discharge, cost_es_discharge, df_storage_links, ratio,
                resource_es, flows_by_res_dis, flow_matrices['es_dis'],
                resources, CO2_FACTORS
            )

        emissions_es_charged    = float(np.sum(cef_es_charge))
        emissions_es_discharged = float(np.sum(cef_es_discharge))
        cost_es_charged         = float(np.sum(cost_es_charge))
        cost_es_discharged      = float(np.sum(cost_es_discharge))

        delta_es_emissions = -emissions_es_discharged + emissions_es_charged
        delta_es_cost      = -cost_es_discharged + cost_es_charged

        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        unit_cost_es_cycle.append(delta_es_cost / energy_es_discharge if energy_es_discharge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        cost_delta_es.append(delta_es_cost)

        for src in resource_usage.keys():
            resource_usage[src]["es_cha"].append(float(np.sum(resource_es[src]["es_cha"])))
            resource_usage[src]["es_dis"].append(float(np.sum(resource_es[src]["es_dis"])))

    return (
        unit_cost_bat_cycle, unit_cost_es_cycle,
        co2_bat_charge_cycle, co2_es_charge_cycle,
        energy_bat_charge_cycle, energy_es_charge_cycle,
        energy_bat_discharged_cycle, energy_es_discharged_cycle,
        emissions_bat_charged_cycle, emissions_es_charged_cycle,
        emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
        co2_delta_bat_emissions, co2_delta_es_emissions,
        cost_delta_bat, cost_delta_es,
        resource_usage
    )