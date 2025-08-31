import numpy as np

def cef_bat_discharge_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                          resource_usage, CO2_FACTORS, order, regions):
    """
    EU case — allocate emissions/cost for BATTERY DISCHARGE at a single time step.

    This function attributes the discharged battery energy at row `idx` to available
    regional generation by walking the given `order` (e.g., high-carbon→low-carbon or
    high-cost→low-cost). It subtracts the allocated energy from `values` (per region,
    per source), accumulates CO₂ and cost, and records per-source usage.

    Args:
        idx : int
            Time-step index (row) to process.
        values : pd.DataFrame
            Available generation by region/source at each time step. Must contain
            columns like f"{region}_{source}" for all `regions` and sources in `order`.
            This frame is decremented in place (copy beforehand if needed).
        cef_bat : list
            Collector for allocated CO₂ (mass) at this step (battery discharge).
        cost_bat : list
            Collector for allocated cost at this step (battery discharge).
        df_battery_links_charge : pd.DataFrame
            Link flows. Needs:
              - "bus_discharger" (fleet total, sign convention per your data)
              - f"{region}_Battery_discharger" for each region.
            Note: code multiplies by `-1` to turn discharger power into positive energy.
        ratio : float
            Scaling factor (e.g., step duration in hours) applied to link flows.
        resource_usage : dict
            Per-source usage accumulator; must contain keys for sources and an "Others"
            bucket if you rely on the default fallback. This function appends to:
              - resource_usage[source]["bat_dis"]
        CO2_FACTORS : dict[str, tuple[float, float]]
            {source: (co2_factor, marginal_cost)}.
        order : Sequence[str]
            Resource index order for **discharge** allocation (iterated as is).
            Typically high-carbon→low-carbon or high-cost→low-cost.
        regions : Sequence[str]
            Regions to allocate across; also used to access per-region columns.

    Returns:
        tuple: (cef_bat, cost_bat, values, resource_usage)
    """
    # Start per-step usage log for all sources
    for source in resource_usage.keys():
        resource_usage[source]["bat_dis"].append(0)

    # If no fleet battery discharge (after scaling), record zeros and return
    if df_battery_links_charge.at[idx, "bus_discharger"] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0
        other_energy = 0.0

        # Allocate discharge independently within each region
        for region in regions:
            # Make discharger energy positive for allocation
            remaining_energy = -df_battery_links_charge.at[idx, f"{region}_Battery_discharger"] * ratio

            # Walk the priority order (as given) to find supply
            for source in order:
                co2, cost = CO2_FACTORS[source]
                if remaining_energy <= 0:
                    break

                # Pull from available generation for this region/source
                available = values[f"{region}_{source}"].iloc[idx]
                used_energy = min(available, remaining_energy)

                # Accumulate CO₂ and cost
                total_emissions += used_energy * co2
                total_cost      += used_energy * cost

                # Decrement availability and remaining demand
                remaining_energy -= used_energy
                values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["bat_dis"][-1] = used_energy

            # Anything left is counted as "other" (fallback stack)
            other_energy += remaining_energy

        # Fallback to a default source (e.g., CCGT) if not fully met
        if other_energy > 0:
            total_emissions += other_energy * CO2_FACTORS['CCGT'][0]
            total_cost      += other_energy * CO2_FACTORS['CCGT'][1]
            resource_usage['Others']["bat_dis"][-1] = other_energy

        # Store step totals
        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage


def cef_es_discharge_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                         resource_usage, CO2_FACTORS, order, regions):
    """
    EU case — allocate emissions/cost for OTHER STORAGE (ES/LDES) DISCHARGE at one step.

    Notes:
      - Uses columns:
          "es_bus_discharger" and f"{region}_Other_storage_discharger"
      - Iterates `order` as provided (not reversed) for discharge.
      - Decrements `values` per region/source when energy is taken.

    Returns:
        tuple: (cef_bat, cost_bat, resource_usage)
        (Matches your EU function’s original return signature.)
    """
    # Initialize per-step usage
    for source in resource_usage.keys():
        resource_usage[source]["es_dis"].append(0)

    # Skip if no ES discharge
    if df_battery_links_charge.at[idx, "es_bus_discharger"] == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0
        other_energy = 0.0

        for region in regions:
            remaining_energy = -df_battery_links_charge.at[idx, f"{region}_Other_storage_discharger"] * ratio

            for source in order:
                co2, cost = CO2_FACTORS[source]
                if remaining_energy <= 0:
                    break

                available = values[f"{region}_{source}"].iloc[idx]
                used_energy = min(available, remaining_energy)

                total_emissions += used_energy * co2
                total_cost      += used_energy * cost

                remaining_energy -= used_energy
                values.loc[idx, f"{region}_{source}"] -= used_energy
                resource_usage[source]["es_dis"][-1] = used_energy

            other_energy += remaining_energy

        if other_energy > 0:
            total_emissions += other_energy * CO2_FACTORS['CCGT'][0]
            total_cost      += other_energy * CO2_FACTORS['CCGT'][1]
            resource_usage['Others']["es_dis"][-1] = other_energy

        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    # EU version returns without `values` to match your original function
    return cef_bat, cost_bat, resource_usage


def cef_bat_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                resource_usage, CO2_FACTORS, order, regions):
    """
    EU case — allocate emissions/cost for BATTERY CHARGING at a single time step.

    Notes:
      - Uses columns:
          "bus_charger" and f"{region}_Battery_charger"
      - Iterates **reversed(order)** for charging (i.e., opposite direction to discharge),
        matching your UK/EU design where charging prefers cleaner/cheaper first if
        `order` is high-carbon→low-carbon or high-cost→low-cost.

    Returns:
        tuple: (cef_bat, cost_bat, values, resource_usage)
    """
    # Initialize per-step usage
    for source in resource_usage.keys():
        resource_usage[source]["bat_cha"].append(0)

    # Skip if no battery charging
    if df_battery_links_charge.at[idx, "bus_charger"] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0

        for region in regions:
            remaining_energy = df_battery_links_charge.at[idx, f"{region}_Battery_charger"] * ratio

            # Charging: walk reversed merit (cleaner/cheaper first if order is dirty/expensive→clean/cheap)
            for source in reversed(order):
                co2, cost = CO2_FACTORS[source]
                if remaining_energy <= 0:
                    break

                available = values[f"{region}_{source}"].iloc[idx]
                used_energy = min(available, remaining_energy)

                total_emissions += used_energy * co2
                total_cost      += used_energy * cost

                remaining_energy -= used_energy
                values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["bat_cha"][-1] = used_energy

        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage


def cef_es_log(idx, values, cef_es, cost_es, df_ldes_links_charge, ratio,
               resource_usage, CO2_FACTORS, order, regions):
    """
    EU case — allocate emissions/cost for OTHER STORAGE (ES/LDES) CHARGING at one step.

    Notes:
      - Uses columns:
          "es_bus_charger" and f"{region}_Other_storage_charger"
      - Iterates **reversed(order)** for charging (same reasoning as batteries).
      - This function does not decrement `values` in the original EU code for ES charging;
        if you want symmetric behavior with battery charging, you can subtract like above.
        (Here we keep your EU logic intact.)

    Returns:
        tuple: (cef_es, cost_es, resource_usage)
    """
    # Initialize per-step usage
    for source in resource_usage.keys():
        resource_usage[source]["es_cha"].append(0)

    # Skip if no ES charging
    if df_ldes_links_charge.at[idx, "es_bus_charger"] * ratio == 0:
        cef_es.append(0)
        cost_es.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0

        for region in regions:
            remaining_energy = df_ldes_links_charge.at[idx, f"{region}_Other_storage_charger"] * ratio

            for source in reversed(order):
                co2, cost = CO2_FACTORS[source]
                if remaining_energy <= 0:
                    break

                available = values[f"{region}_{source}"].iloc[idx]
                used_energy = min(available, remaining_energy)

                total_emissions += used_energy * co2
                total_cost      += used_energy * cost

                remaining_energy -= used_energy
                # NOTE: EU version did not decrement `values` here originally.
                # If desired, mirror battery with:
                # values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["es_cha"][-1] = used_energy

        cef_es.append(total_emissions)
        cost_es.append(total_cost)

    return cef_es, cost_es, resource_usage


def national_cycle_analysis(
    process_times_bat, process_ratios_bat,
    process_times_es,  process_ratios_es,
    df_gen, df_gen_0, df_storage_links, df_gen_remain,
    resource_usage, CO2_FACTORS, order, regions
):
    """
    Analyze charge/discharge cycles and compute cycle-level CO₂ and cost metrics
    for batteries and LDES (EU case logic preserved).

    Key note on `order`:
        `order` is the resource index order constructed per your optimization
        objective — either **high-carbon → low-carbon** or **high-cost → low-cost**.
        The allocator functions you call here follow your original design:
          - DISCHARGE walks `order` as-is.
          - CHARGE walks `reversed(order)`.
        If you flip the objective (e.g., cost vs. carbon), rebuild or reverse
        `order` accordingly to keep the intended marginal behavior.

    Args:
        process_times_bat (dict[int, list[int]]): For each battery cycle id, the list
            of time-step indices in that cycle.
        process_ratios_bat (dict[int, list[float]]): Per-cycle ratios aligned with
            `process_times_bat[cycle]` (used as step weights, e.g., to scale power→energy).
        process_times_es (dict[int, list[int]]): Same as above for LDES cycles.
        process_ratios_es (dict[int, list[float]]): Same as above for LDES cycles.
        df_gen (pd.DataFrame): Per-step generation by source (for cycle-average CI).
        df_gen_0 (pd.DataFrame): Baseline generation snapshot used by *charging* allocators;
            copied to a working frame and decremented during charging allocation.
        df_storage_links (pd.DataFrame): Storage link series used by allocators. Required columns:
            Battery: 'bus_charger', 'bus_discharger',
                     f"{region}_Battery_charger", f"{region}_Battery_discharger"
            LDES:    'es_bus_charger', 'es_bus_discharger',
                     f"{region}_Other_storage_charger", f"{region}_Other_storage_discharger"
        df_gen_remain (pd.DataFrame): Remaining generation used by *discharging* allocators;
            copied to a working frame and decremented during discharge allocation.
        resource_usage (dict): Accumulators per source; receives per-cycle sums in keys:
            'bat_cha', 'bat_dis', 'es_cha', 'es_dis'.
        CO2_FACTORS (Mapping[str, Tuple[float, float]]): {source: (co2_factor, marginal_cost)}.
        order (Sequence[str]): See “Key note on `order`” above.
        regions (Sequence[str]): Regions to iterate and to resolve per-region columns.

    Returns:
        tuple:
          (
            cef_bat_t, cef_es_t,                                    # per-cycle stepwise (charge - discharge)
            carbon_intensity_bat_cycle, carbon_intensity_es_cycle,  # cycle-average CI from df_gen
            unit_ccost_bat_cycle, unit_ccost_es_cycle,              # charge unit cost per cycle
            unit_dcost_bat_cycle, unit_dcost_es_cycle,              # discharge unit cost per cycle
            unit_cost_bat_cycle, unit_cost_es_cycle,                # (Δcost / discharged energy) per cycle
            co2_bat_charge_cycle, co2_es_charge_cycle,              # (ΔCO2 / discharged energy) per cycle
            total_cost_bat, total_cost_es,                          # sums over all cycles (Δcost)
            total_emissions_bat, total_emissions_es,                # sums over all cycles (ΔCO2)
            cost_bat_charged_cycle, cost_es_charged_cycle,          # cost components per cycle
            cost_bat_discharged_cycle, cost_es_discharged_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle,        # energies per cycle
            energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle,# CO2 components per cycle
            emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
            resource_usage                                          # per-source usage rollups
          )
    """
    # ---------------- accumulators ----------------
    energy_bat_charge_cycle, energy_es_charge_cycle = [], []
    energy_bat_discharged_cycle, energy_es_discharged_cycle = [], []
    co2_bat_charge_cycle, co2_bat_charge_energy_cycle = [], []
    co2_es_charge_cycle,  co2_es_charge_energy_cycle  = [], []
    emissions_bat_charged_cycle, emissions_es_charged_cycle = [], []
    emissions_bat_discharged_cycle, emissions_es_discharged_cycle = [], []
    cost_bat_charged_cycle, cost_bat_discharged_cycle = [], []
    cost_es_charged_cycle,  cost_es_discharged_cycle  = [], []
    co2_delta_bat_emissions, co2_delta_es_emissions = [], []
    cost_delta_bat_emissions, cost_delta_es_emissions = [], []
    carbon_intensity_bat_cycle, carbon_intensity_es_cycle = [], []
    unit_ccost_bat_cycle, unit_ccost_es_cycle = [], []
    unit_dcost_bat_cycle, unit_dcost_es_cycle = [], []
    unit_cost_bat_cycle,  unit_cost_es_cycle  = [], []

    # Working copies decremented by allocation routines
    df_gen_1 = df_gen_0.copy()
    df_gen_remain_1 = df_gen_remain.copy()

    cef_bat_t, cef_es_t = {}, {}

    # ================= BATTERY cycles =================
    for cycle_number, times in process_times_bat.items():
        # Per-cycle energies (apply step ratios)
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()

        # Stepwise logs filled by allocators
        cef_bat_charge, cef_bat_discharge = [], []
        cost_bat_charge, cost_bat_discharge = [], []

        # Local per-source tracker for this cycle; rolled into `resource_usage` after the loop
        resource_bat = {src: {"bat_cha": [], "bat_dis": []} for src in resource_usage.keys()}

        # Cycle-average CI from df_gen over the cycle
        values_sum = df_gen.iloc[times].sum()
        total_emissions = total_energy = 0.0
        for src, (co2, _cost) in CO2_FACTORS.items():
            total_emissions += values_sum[src] * co2
            total_energy    += values_sum[src]
        carbon_intensity_bat = (total_emissions / total_energy) if total_energy != 0 else 0

        # Step allocation: CHARGE (reversed(order)) then DISCHARGE (order)
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]  # step weight

            # Charging allocator: decrements df_gen_1
            cef_bat_charge, cost_bat_charge, df_gen_1, resource_bat = cef_bat_log(
                idx, df_gen_1, cef_bat_charge, cost_bat_charge,
                df_storage_links, ratio, resource_bat, CO2_FACTORS, order, regions
            )

            # Discharging allocator: decrements df_gen_remain_1
            cef_bat_discharge, cost_bat_discharge, df_gen_remain_1, resource_bat = cef_bat_discharge_log(
                idx, df_gen_remain_1, cef_bat_discharge, cost_bat_discharge,
                df_storage_links, ratio, resource_bat, CO2_FACTORS, order, regions
            )

        # Stepwise delta and per-cycle aggregates
        cef_bat_t[cycle_number] = [c - d for c, d in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged    = float(np.sum(cef_bat_charge))
        emissions_bat_discharged = float(np.sum(cef_bat_discharge))
        cost_bat_charged         = float(np.sum(cost_bat_charge))
        cost_bat_discharged      = float(np.sum(cost_bat_discharge))

        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged
        delta_bat_cost      = -cost_bat_discharged + cost_bat_charged

        # Unit costs
        unit_ccost_bat  = (cost_bat_charged   / energy_bat_charge)    if energy_bat_charge    != 0 else 0
        unit_dcost_bat  = (cost_bat_discharged/ energy_bat_discharge) if energy_bat_discharge != 0 else 0
        unit_cost_bat   = (delta_bat_cost     / energy_bat_discharge) if energy_bat_discharge != 0 else 0

        # Record cycle results
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        cost_bat_charged_cycle.append(cost_bat_charged)
        cost_bat_discharged_cycle.append(cost_bat_discharged)

        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        co2_bat_charge_energy_cycle.append(delta_bat_emissions / energy_bat_charge if energy_bat_charge != 0 else 0)

        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        cost_delta_bat_emissions.append(delta_bat_cost)

        unit_ccost_bat_cycle.append(unit_ccost_bat)
        unit_dcost_bat_cycle.append(unit_dcost_bat)
        unit_cost_bat_cycle.append(unit_cost_bat)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

        # Roll per-source cycle usage into the global accumulator
        for src in resource_usage.keys():
            resource_usage[src]["bat_cha"].append(float(np.sum(resource_bat[src]["bat_cha"])))
            resource_usage[src]["bat_dis"].append(float(np.sum(resource_bat[src]["bat_dis"])))

    total_emissions_bat = float(np.sum(co2_delta_bat_emissions))
    total_cost_bat      = float(np.sum(cost_delta_bat_emissions))

    # ================= LDES / ES cycles =================
    for cycle_number, times in process_times_es.items():
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()

        cef_es_charge, cef_es_discharge = [], []
        cost_es_charge, cost_es_discharge = [], []

        resource_es = {src: {"es_cha": [], "es_dis": []} for src in resource_usage.keys()}

        # Cycle-average CI from df_gen
        values_sum = df_gen.iloc[times].sum()
        total_emissions = total_energy = 0.0
        for src, (co2, _cost) in CO2_FACTORS.items():
            total_emissions += values_sum[src] * co2
            total_energy    += values_sum[src]
        carbon_intensity_es = (total_emissions / total_energy) if total_energy != 0 else 0

        # Step allocation: CHARGE (reversed(order)) then DISCHARGE (order)
        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]

            cef_es_charge, cost_es_charge, resource_es = cef_es_log(
                idx, df_gen_1, cef_es_charge, cost_es_charge,
                df_storage_links, ratio, resource_es, CO2_FACTORS, order, regions
            )
            cef_es_discharge, cost_es_discharge, resource_es = cef_es_discharge_log(
                idx, df_gen_remain_1, cef_es_discharge, cost_es_discharge,
                df_storage_links, ratio, resource_es, CO2_FACTORS, order, regions
            )

        cef_es_t[cycle_number] = [c - d for c, d in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged    = float(np.sum(cef_es_charge))
        emissions_es_discharged = float(np.sum(cef_es_discharge))
        cost_es_charged         = float(np.sum(cost_es_charge))
        cost_es_discharged      = float(np.sum(cost_es_discharge))

        delta_es_emissions = -emissions_es_discharged + emissions_es_charged
        delta_es_cost      = -cost_es_discharged + cost_es_charged

        unit_ccost_es = (cost_es_charged    / energy_es_charge)    if energy_es_charge    != 0 else 0
        unit_dcost_es = (cost_es_discharged / energy_es_discharge) if energy_es_discharge != 0 else 0
        unit_cost_es  = (delta_es_cost      / energy_es_discharge) if energy_es_discharge != 0 else 0

        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        cost_es_charged_cycle.append(cost_es_charged)
        cost_es_discharged_cycle.append(cost_es_discharged)

        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        co2_es_charge_energy_cycle.append(delta_es_emissions / energy_es_charge if energy_es_charge != 0 else 0)

        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        cost_delta_es_emissions.append(delta_es_cost)

        unit_ccost_es_cycle.append(unit_ccost_es)
        unit_dcost_es_cycle.append(unit_dcost_es)
        unit_cost_es_cycle.append(unit_cost_es)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

        for src in resource_usage.keys():
            resource_usage[src]["es_cha"].append(float(np.sum(resource_es[src]["es_cha"])))
            resource_usage[src]["es_dis"].append(float(np.sum(resource_es[src]["es_dis"])))

    total_emissions_es = float(np.sum(co2_delta_es_emissions))
    total_cost_es      = float(np.sum(cost_delta_es_emissions))

    # ---------------- return everything ----------------
    return (
        cef_bat_t, cef_es_t,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
        unit_ccost_bat_cycle, unit_ccost_es_cycle,
        unit_dcost_bat_cycle, unit_dcost_es_cycle,
        unit_cost_bat_cycle, unit_cost_es_cycle,
        co2_bat_charge_cycle, co2_es_charge_cycle,
        total_cost_bat, total_cost_es,
        total_emissions_bat, total_emissions_es,
        cost_bat_charged_cycle, cost_es_charged_cycle,
        cost_bat_discharged_cycle, cost_es_discharged_cycle,
        energy_bat_charge_cycle, energy_es_charge_cycle,
        energy_bat_discharged_cycle, energy_es_discharged_cycle,
        emissions_bat_charged_cycle, emissions_es_charged_cycle,
        emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
        resource_usage
    )