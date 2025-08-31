import numpy as np

def cef_bat_discharge_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                          resource_usage, CO2_FACTORS, order, regions):
    """
    Compute and append the Cycle Emission Factor (CEF) and cost for BATTERY DISCHARGE at one time step.

    Args:
        idx (int): Time-step index.
        values (pd.DataFrame): Available per-source generation by region (columns like "{region}_{source}").
        cef_bat (list): Output list to append the battery discharge CEF (allocated emissions, mass).
        cost_bat (list): Output list to append the allocated cost.
        df_battery_links_charge (pd.DataFrame): Battery link flows; uses columns like
            "bus_discharger" and "{region}_Battery_discharger".
        ratio (float): Scaling for link values (e.g., convert power→energy).
        resource_usage (dict): Accumulator per source; updates key ["bat_dis"] at this step.
        CO2_FACTORS (dict): {source: (co2_factor, marginal_cost)}.
        order (list[str]): Merit order for allocating discharge to sources (high→low).
        regions (list[str]): Regions to iterate over.

    Returns:
        tuple: (cef_bat, cost_bat, values, resource_usage)
    """
    # Initialize per-source tracker at this step
    for source in resource_usage.keys():
        resource_usage[source]["bat_dis"].append(0)

    # If no battery discharge this step (after scaling), append zeros and exit
    if df_battery_links_charge.at[idx, "bus_discharger"] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0
        other_energy = 0.0

        # Allocate regional discharge to sources by merit order
        for region in regions:
            remaining_energy = -df_battery_links_charge.at[idx, f"{region}_Battery_discharger"] * ratio
            for source in order:
                if remaining_energy <= 0:
                    break
                co2, cost = CO2_FACTORS[source]
                used_energy = min(values[f"{region}_{source}"].iloc[idx], remaining_energy)
                total_emissions += used_energy * co2
                total_cost += used_energy * cost
                remaining_energy -= used_energy
                # Update available generation and usage bookkeeping
                values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["bat_dis"][-1] = used_energy
            other_energy += remaining_energy

        # Any remaining unmet discharge is allocated to a default source (e.g., CCGT)
        if other_energy > 0:
            total_emissions += other_energy * CO2_FACTORS['CCGT'][0]
            total_cost += other_energy * CO2_FACTORS['CCGT'][1]
            resource_usage['Others']["bat_dis"][-1] = other_energy

        cef_bat.append(total_emissions)  # allocated emissions (mass)
        cost_bat.append(total_cost)      # allocated cost

    return cef_bat, cost_bat, values, resource_usage


def cef_es_discharge_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                         resource_usage, CO2_FACTORS, order, regions):
    """
    Compute and append the Cycle Emission Factor (CEF) and cost for OTHER STORAGE DISCHARGE at one time step.

    Args:
        idx (int): Time-step index.
        values (pd.DataFrame): Available per-source generation by region (columns like "{region}_{source}").
        cef_bat (list): Output list to append the ES (other storage) discharge CEF (allocated emissions, mass).
        cost_bat (list): Output list to append the allocated cost.
        df_battery_links_charge (pd.DataFrame): Storage link flows; uses columns like
            "es_bus_discharger" and "{region}_OtherStorage_discharger".
        ratio (float): Scaling for link values (e.g., convert power→energy).
        resource_usage (dict): Accumulator per source; updates key ["es_dis"] at this step.
        CO2_FACTORS (dict): {source: (co2_factor, marginal_cost)}.
        order (list[str]): Merit order for allocating discharge to sources (high→low).
        regions (list[str]): Regions to iterate over.

    Returns:
        tuple: (cef_bat, cost_bat, values, resource_usage)
    """
    # Initialize per-source tracker at this step
    for source in resource_usage.keys():
        resource_usage[source]["es_dis"].append(0)

    # If no ES discharge this step, append zeros and exit
    if df_battery_links_charge.at[idx, "es_bus_discharger"] == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0
        other_energy = 0.0

        for region in regions:
            remaining_energy = -df_battery_links_charge.at[idx, f"{region}_OtherStorage_discharger"] * ratio
            for source in order:
                if remaining_energy <= 0:
                    break
                co2, cost = CO2_FACTORS[source]
                used_energy = min(values[f"{region}_{source}"].iloc[idx], remaining_energy)
                total_emissions += used_energy * co2
                total_cost += used_energy * cost
                remaining_energy -= used_energy
                values.loc[idx, f"{region}_{source}"] -= used_energy
                resource_usage[source]["es_dis"][-1] = used_energy
            other_energy += remaining_energy

        if other_energy > 0:
            total_emissions += other_energy * CO2_FACTORS['CCGT'][0]
            total_cost += other_energy * CO2_FACTORS['CCGT'][1]
            resource_usage['Others']["es_dis"][-1] = other_energy

        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage


def cef_bat_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio,
                resource_usage, CO2_FACTORS, order, regions):
    """
    Compute and append the Cycle Emission Factor (CEF) and cost for BATTERY CHARGING at one time step.

    Args:
        idx (int): Time-step index.
        values (pd.DataFrame): Available per-source generation by region (columns like "{region}_{source}").
        cef_bat (list): Output list to append the battery charging CEF (allocated emissions, mass).
        cost_bat (list): Output list to append the allocated cost.
        df_battery_links_charge (pd.DataFrame): Battery link flows; uses columns like
            "bus_charger" and "{region}_Battery_charger".
        ratio (float): Scaling for link values (e.g., convert power→energy).
        resource_usage (dict): Accumulator per source; updates key ["bat_cha"] at this step.
        CO2_FACTORS (dict): {source: (co2_factor, marginal_cost)}.
        order (list[str]): Merit order for allocating charging from sources (low→high if reversed below).
        regions (list[str]): Regions to iterate over.

    Returns:
        tuple: (cef_bat, cost_bat, values, resource_usage)
    """
    # Initialize per-source tracker at this step
    for source in resource_usage.keys():
        resource_usage[source]["bat_cha"].append(0)

    # If no battery charging this step, append zeros and exit
    if df_battery_links_charge.at[idx, "bus_charger"] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0

        for region in regions:
            remaining_energy = df_battery_links_charge.at[idx, f"{region}_Battery_charger"] * ratio
            # For charging allocation, iterate reversed order (low→high emission first if your order is high→low)
            for source in reversed(order):
                if remaining_energy <= 0:
                    break
                co2, cost = CO2_FACTORS[source]
                used_energy = min(values[f"{region}_{source}"].iloc[idx], remaining_energy)
                total_emissions += used_energy * co2
                total_cost += used_energy * cost
                remaining_energy -= used_energy
                values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["bat_cha"][-1] = used_energy

        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage


def cef_es_log(idx, values, cef_es, cost_es, df_ldes_links_charge, ratio,
               resource_usage, CO2_FACTORS, order, regions):
    """
    Compute and append the Cycle Emission Factor (CEF) and cost for OTHER STORAGE (LDES) CHARGING at one time step.

    Args:
        idx (int): Time-step index.
        values (pd.DataFrame): Available per-source generation by region (columns like "{region}_{source}").
        cef_es (list): Output list to append the ES charging CEF (allocated emissions, mass).
        cost_es (list): Output list to append the allocated cost.
        df_ldes_links_charge (pd.DataFrame): ES link flows; uses columns like
            "es_bus_charger" and "{region}_OtherStorage_charger".
        ratio (float): Scaling for link values (e.g., convert power→energy).
        resource_usage (dict): Accumulator per source; updates key ["es_cha"] at this step.
        CO2_FACTORS (dict): {source: (co2_factor, marginal_cost)}.
        order (list[str]): Merit order for allocating charging from sources (low→high if reversed below).
        regions (list[str]): Regions to iterate over.

    Returns:
        tuple: (cef_es, cost_es, values, resource_usage)
    """
    # Initialize per-source tracker at this step
    for source in resource_usage.keys():
        resource_usage[source]["es_cha"].append(0)

    # If no ES charging this step, append zeros and exit
    if df_ldes_links_charge.at[idx, "es_bus_charger"] * ratio == 0:
        cef_es.append(0)
        cost_es.append(0)
    else:
        total_emissions = 0.0
        total_cost = 0.0

        for region in regions:
            remaining_energy = df_ldes_links_charge.at[idx, f"{region}_OtherStorage_charger"] * ratio
            for source in reversed(order):
                if remaining_energy <= 0:
                    break
                co2, cost = CO2_FACTORS[source]
                used_energy = min(values[f"{region}_{source}"].iloc[idx], remaining_energy)
                total_emissions += used_energy * co2
                total_cost += used_energy * cost
                remaining_energy -= used_energy
                values.loc[values.index[idx], f"{region}_{source}"] -= used_energy
                resource_usage[source]["es_cha"][-1] = used_energy

        cef_es.append(total_emissions)
        cost_es.append(total_cost)

    return cef_es, cost_es, values, resource_usage

def aef_log(idx, values, aef, storage_power, ratio, order, CO2_FACTORS):
    """
    Compute and append the average emission factor (AEF) at a single time step.

    Args:
        idx (int): Row index (time step) in `values`.
        values (pd.DataFrame): Per-source generation; must contain columns in `order`.
        aef (list): Output list; the AEF for this time step is appended here.
        storage_power (pd.Series): Storage charge/discharge at each time step
            (sign per your convention; only zero/non-zero is used here).
        ratio (float): Scaling factor applied to `storage_power` (e.g., step hours).
        order (Iterable[str]): Source column names to include in the AEF.
        CO2_FACTORS (Mapping[str, Tuple[float, float]]): {source: (co2_factor, marginal_cost)}.

    Returns:
        list: The same `aef` list with the appended AEF (e.g., tCO₂/MWh) for row `idx`.
    """
    # If no storage activity at this step, record AEF as 0 for convenience
    if storage_power.iloc[idx] * ratio == 0:
        aef.append(0.0)
        return aef

    # Sum per-source emissions at this step: Σ (generation_i * EF_i)
    emissions_sum = 0.0
    for source in order:
        co2_factor, _ = CO2_FACTORS[source]  # marginal_cost not used
        emissions_sum += float(values[source].iloc[idx]) * float(co2_factor)

    # Total generation of the selected sources at this step (denominator for AEF)
    total_gen = float(values.loc[values.index[idx], order].sum())

    # Guard against zero/negative total generation
    if total_gen <= 0:
        aef.append(0.0)
        return aef

    # Average Emission Factor (e.g., tCO2/MWh)
    aef_step = emissions_sum / total_gen
    aef.append(aef_step)
    return aef

def mef_log(idx, values, mef, storage_power, ratio, order, CO2_FACTORS):
    """
    Compute and append the marginal emission factor (MEF) at a single time step.

    Args:
        idx (int): Row index (time step) in `values`.
        values (pd.DataFrame): Per-source generation; must contain columns in `order`.
        mef (list): Output list; the MEF for this time step is appended here.
        storage_power (pd.Series): Storage charge/discharge at each time step
            (used to skip zero-activity steps).
        ratio (float): Scaling factor applied to `storage_power` (e.g., step hours).
        order (Iterable[str]): Source column names ordered by merit; the marginal
            source is searched from the end (reversed order).
        CO2_FACTORS (Mapping[str, Tuple[float, float]]): {source: (co2_factor, marginal_cost)};
            only `co2_factor` is used.

    Returns:
        list: The same `mef` list with the appended MEF (e.g., tCO₂/MWh) for row `idx`.
    """
    # If no storage activity at this step, log MEF as 0 for convenience.
    if storage_power.iloc[idx] * ratio == 0:
        mef.append(0.0)
        return mef

    # Find the marginal source: first with positive generation when scanning reversed merit order.
    marginal_co2 = 0.0
    for source in reversed(order):
        if float(values[source].iloc[idx]) > 0:
            marginal_co2 = float(CO2_FACTORS[source][0])  # take its CO₂ factor (tCO2/MWh)
            break

    # Append the marginal emission factor (not mass). To get allocated emissions (mass),
    # multiply `marginal_co2` by `storage_power.iloc[idx] * ratio` outside this function.
    mef.append(marginal_co2)
    return mef




def national_cycle_analysis(
    process_times_bat, process_ratios_bat,
    process_times_es,  process_ratios_es,
    df_gen, df_gen_0, df_storage_links, df_gen_remain,
    resource_usage, CO2_FACTORS, order, regions
):
    """
    Analyze battery and LDES charge/discharge cycles and compute cycle-level
    CO₂ and cost metrics. Logic is unchanged; comments and arg docs are updated.

    Args:
        process_times_bat (dict[int, list[int]]):
            For each battery cycle id -> list of time-step indices in that cycle.
        process_ratios_bat (dict[int, list[float]]):
            Per-cycle list of ratios aligned with `process_times_bat[cycle]`.
        process_times_es (dict[int, list[int]]):
            For each LDES/other-storage cycle id -> list of time-step indices.
        process_ratios_es (dict[int, list[float]]):
            Per-cycle list of ratios aligned with `process_times_es[cycle]`.

        df_gen (pd.DataFrame):
            Generation by source at each time step (used for cycle-average CI).
            Columns must include the keys in `CO2_FACTORS`.
        df_gen_0 (pd.DataFrame):
            Baseline generation snapshot used by allocators; will be copied to
            `df_gen_1` and decremented during *charging* allocations.
        df_storage_links (pd.DataFrame):
            Storage link series (power/energy per step). Required columns include:
              - Battery:   'bus_charger', 'bus_discharger',
                           f'{region}_Battery_charger', f'{region}_Battery_discharger'
              - LDES/ES:   'es_bus_charger', 'es_bus_discharger',
                           f'{region}_OtherStorage_charger', f'{region}_OtherStorage_discharger'
        df_gen_remain (pd.DataFrame):
            Remaining generation used by *discharge* allocations; copied to
            `df_gen_remain_1` and decremented during discharge allocation.

        resource_usage (dict[str, dict[str, list]]):
            Accumulators per source, with keys like 'bat_cha', 'bat_dis', 'es_cha', 'es_dis'.
        CO2_FACTORS (Mapping[str, Tuple[float, float]]):
            {source: (co2_factor, marginal_cost)} used for allocation accounting.
        order (Sequence[str]):
            **Resource index order constructed per optimization objective** —
            either **high-carbon → low-carbon** or **high-cost → low-cost**.
            The allocator iterates `order` (or `reversed(order)`), so pass/reverse
            it to match your intended marginal/merit logic.
        regions (Sequence[str]):
            Region labels used to access per-region columns in `df_storage_links`
            and per-region source columns in generation frames.

    Returns:
        tuple:
            (
              # time-series (per cycle) Δemissions list (charge − discharge) at step level
              cef_bat_t, cef_es_t,
              # cycle-average carbon intensities from `df_gen` over each cycle
              carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
              # unit costs (charge-only, discharge-only, and net Δ) per cycle
              unit_ccost_bat_cycle, unit_ccost_es_cycle,
              unit_dcost_bat_cycle, unit_dcost_es_cycle,
              unit_cost_bat_cycle,  unit_cost_es_cycle,
              # per-cycle Δemissions normalized by discharged/charged energy
              co2_bat_charge_cycle, co2_es_charge_cycle,
              # total Δcost and Δemissions across all cycles
              total_cost_bat, total_cost_es,
              total_emissions_bat, total_emissions_es,
              # raw cost components per cycle
              cost_bat_charged_cycle, cost_es_charged_cycle,
              cost_bat_discharged_cycle, cost_es_discharged_cycle,
              # energies per cycle
              energy_bat_charge_cycle, energy_es_charge_cycle,
              energy_bat_discharged_cycle, energy_es_discharged_cycle,
              # emissions components per cycle
              emissions_bat_charged_cycle, emissions_es_charged_cycle,
              emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
              # Δemissions lists (battery, ES) across cycles
              resource_usage
            )
    """
    # ------------------- accumulators -------------------
    energy_bat_charge_cycle = []
    energy_es_charge_cycle = []
    energy_bat_discharged_cycle = []
    energy_es_discharged_cycle = []

    co2_bat_charge_cycle = []
    co2_bat_charge_energy_cycle = []
    co2_es_charge_cycle = []
    co2_es_charge_energy_cycle = []

    emissions_bat_charged_cycle = []
    emissions_es_charged_cycle = []
    cost_bat_charged_cycle = []
    cost_bat_discharged_cycle = []
    emissions_bat_discharged_cycle = []
    emissions_es_discharged_cycle = []
    cost_es_charged_cycle = []
    cost_es_discharged_cycle = []

    co2_delta_bat_emissions = []
    co2_delta_es_emissions = []
    cost_delta_bat_emissions = []
    cost_delta_es_emissions = []

    carbon_intensity_bat_cycle = []
    carbon_intensity_es_cycle = []

    unit_ccost_bat_cycle = []
    unit_ccost_es_cycle = []
    unit_dcost_bat_cycle = []
    unit_dcost_es_cycle = []
    unit_cost_bat_cycle = []
    unit_cost_es_cycle = []

    # Working copies that will be decremented by allocation routines
    df_gen_1 = df_gen_0.copy()
    df_gen_remain_1 = df_gen_remain.copy()

    cef_bat_t = {}
    cef_es_t = {}

    # =================== Battery cycles ===================
    for cycle_number, times in process_times_bat.items():
        # Per-cycle energies (apply step ratios)
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()

        # Stepwise emission/cost logs to be filled by allocators
        cef_bat_charge = []
        cef_bat_discharge = []
        cost_bat_charge = []
        cost_bat_discharge = []

        # Local tracker that allocators fill for this cycle; then rolled up into `resource_usage`
        resource_bat = {source: {"bat_cha": [], "bat_dis": []} for source in resource_usage.keys()}

        # Cycle-average CI based on total generation in `df_gen` over the cycle
        values = df_gen.iloc[times].sum()
        total_emissions = 0.0
        total_energy = 0.0
        for source, (co2, cost) in CO2_FACTORS.items():
            total_emissions += values[source] * co2
            total_energy    += values[source]
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Stepwise allocation (charging first, then discharging), respecting `order`
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]  # step-level ratio within the cycle

            # Charging allocation: consumes from df_gen_1 (decremented inside)
            cef_bat_charge, cost_bat_charge, df_gen_1, resource_bat = cef_bat_log(
                idx, df_gen_1, cef_bat_charge, cost_bat_charge, df_storage_links,
                ratio, resource_bat, CO2_FACTORS, order, regions
            )

            # Discharging allocation: consumes from df_gen_remain_1 (decremented inside)
            cef_bat_discharge, cost_bat_discharge, df_gen_remain_1, resource_bat = cef_bat_discharge_log(
                idx, df_gen_remain_1, cef_bat_discharge, cost_bat_discharge, df_storage_links,
                ratio, resource_bat, CO2_FACTORS, order, regions
            )

        # Per-step delta list and cycle aggregates
        cef_bat_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged   = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        cost_bat_charged        = np.sum(cost_bat_charge)
        cost_bat_discharged     = np.sum(cost_bat_discharge)

        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged
        delta_bat_cost      = -cost_bat_discharged + cost_bat_charged

        # Unit costs (charge-only, discharge-only, and net per discharged energy)
        unit_charged_cost_bat   = cost_bat_charged   / energy_bat_charge    if energy_bat_charge    != 0 else 0
        unit_discharged_cost_bat= cost_bat_discharged/ energy_bat_discharge if energy_bat_discharge != 0 else 0
        unit_cha_dis_cost_bat   = delta_bat_cost     / energy_bat_discharge if energy_bat_discharge != 0 else 0

        # Record battery cycle results
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

        unit_ccost_bat_cycle.append(unit_charged_cost_bat)
        unit_dcost_bat_cycle.append(unit_discharged_cost_bat)
        unit_cost_bat_cycle.append(unit_cha_dis_cost_bat)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

        # Roll up per-source usage for this cycle into the main accumulator
        for source in resource_usage.keys():
            resource_usage[source]["bat_cha"].append(np.sum(resource_bat[source]["bat_cha"]))
            resource_usage[source]["bat_dis"].append(np.sum(resource_bat[source]["bat_dis"]))

    total_emissions_bat = np.sum(co2_delta_bat_emissions)
    total_cost_bat      = np.sum(cost_delta_bat_emissions)

    # =================== LDES / ES cycles ===================
    for cycle_number, times in process_times_es.items():
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()

        cef_es_charge = []
        cef_es_discharge = []
        cost_es_charge = []
        cost_es_discharge = []

        resource_es = {source: {"es_cha": [], "es_dis": []} for source in resource_usage.keys()}

        # Cycle-average CI from `df_gen` over the cycle
        values = df_gen.iloc[times].sum()
        total_emissions = 0.0
        total_energy = 0.0
        for source, (co2, cost) in CO2_FACTORS.items():
            total_emissions += values[source] * co2
            total_energy    += values[source]
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        # Stepwise allocation (charging then discharging)
        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]

            cef_es_charge, cost_es_charge, df_gen_1, resource_es = cef_es_log(
                idx, df_gen_1, cef_es_charge, cost_es_charge, df_storage_links,
                ratio, resource_es, CO2_FACTORS, order, regions
            )
            cef_es_discharge, cost_es_discharge, df_gen_remain_1, resource_es = cef_es_discharge_log(
                idx, df_gen_remain_1, cef_es_discharge, cost_es_discharge, df_storage_links,
                ratio, resource_es, CO2_FACTORS, order, regions
            )

        cef_es_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged   = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        cost_es_charged        = np.sum(cost_es_charge)
        cost_es_discharged     = np.sum(cost_es_discharge)

        delta_es_emissions = -emissions_es_discharged + emissions_es_charged
        delta_es_cost      = -cost_es_discharged + cost_es_charged

        unit_charged_cost_es    = cost_es_charged    / energy_es_charge    if energy_es_charge    != 0 else 0
        unit_discharged_cost_es = cost_es_discharged / energy_es_discharge if energy_es_discharge != 0 else 0
        unit_cha_dis_cost_es    = delta_es_cost      / energy_es_discharge if energy_es_discharge != 0 else 0

        # Record ES cycle results
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

        unit_ccost_es_cycle.append(unit_charged_cost_es)
        unit_dcost_es_cycle.append(unit_discharged_cost_es)
        unit_cost_es_cycle.append(unit_cha_dis_cost_es)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

        for source in resource_usage.keys():
            resource_usage[source]["es_cha"].append(np.sum(resource_es[source]["es_cha"]))
            resource_usage[source]["es_dis"].append(np.sum(resource_es[source]["es_dis"]))

    total_emissions_es = np.sum(co2_delta_es_emissions)
    total_cost_es      = np.sum(cost_delta_es_emissions)

    # ------------------- return all results -------------------
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


def national_aef_analysis(
    process_times_bat, process_ratios_bat,
    process_times_es,  process_ratios_es,
    df_gen_0, df_storage_links, order, CO2_FACTORS
):
    """
    Analyze battery and LDES charge/discharge cycles and compute cycle-level
    CO2 metrics and energies.

    Inputs
    -------
    process_times_bat : dict[int, list[int]]
        For each battery cycle id -> list of time-step indices in that cycle.
    process_ratios_bat : dict[int, list[float]]
        Per-cycle list of weights/ratios aligned with `process_times_bat[cycle]`.
        Each ratio is applied to the corresponding time step in that cycle.
    process_times_es : dict[int, list[int]]
        Same as above but for LDES cycles.
    process_ratios_es : dict[int, list[float]]
        Same as above but for LDES cycles.
    df_gen_0 : pd.DataFrame
        Generation by source at each time step. Columns must include all keys in
        `CO2_FACTORS` (e.g., 'CCGT', 'Wind', 'PV', ...). Used for AEF and totals.
    df_storage_links : pd.DataFrame
        Storage link powers/energies by column, including:
          - 'bus_charger', 'bus_discharger'            (battery)
          - 'es_bus_charger', 'es_bus_discharger'      (LDES/other storage)
        The per-region columns referenced below must also exist, e.g.:
          - f"{region}_Battery_charger", f"{region}_Battery_discharger"
          - f"{region}_OtherStorage_charger", f"{region}_OtherStorage_discharger"
    order (Sequence[str]): **Resource index order constructed per optimization objective**.
            - **high-carbon → low-carbon** or **high-cost → low-cost**.
            Adjust `order` (or reverse it before passing) to match the intended marginal rule.
    CO2_FACTORS : dict[str, tuple[float, float]]
        Mapping source -> (co2_factor, marginal_cost). Only names/keys are assumed
        to exist in df_gen_0's columns.

    Returns
    -------
    tuple :
        (
          cef_bat_t, cef_es_t,                            # dict[cycle_id] -> list(step_emissions_charge - step_emissions_discharge)
          carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
          total_emissions_bat, total_emissions_es,
          co2_emissions_factor_bat, co2_emissions_factor_es,  # overall factors (Δemissions / discharged energy)
          co2_bat_charge_cycle, co2_es_charge_cycle,          # per-cycle Δemissions / discharged energy
          energy_bat_charge_cycle, energy_es_charge_cycle,    # per-cycle energies
          energy_bat_discharged_cycle, energy_es_discharged_cycle,
          emissions_bat_charged_cycle, emissions_es_charged_cycle,
          emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
          co2_delta_bat_emissions, co2_delta_es_emissions
        )

    Notes
    -----
    - This function calls `aef_log(...)` for per-step AEF-based values exactly as in the original code.
      If your `aef_log` now returns an emission *factor* (tCO2/MWh), summing it directly yields a
      non-energy-weighted sum; if you need emission *mass*, multiply by energy at each step before summing.
    """
    # --- initialize accumulators ---
    energy_bat_charge_cycle = []
    energy_es_charge_cycle = []
    energy_bat_discharged_cycle = []
    energy_es_discharged_cycle = []

    co2_bat_charge_cycle = []
    co2_es_charge_cycle = []

    emissions_bat_charged_cycle = []
    emissions_es_charged_cycle = []
    emissions_bat_discharged_cycle = []
    emissions_es_discharged_cycle = []

    co2_delta_bat_emissions = []
    co2_delta_es_emissions = []

    carbon_intensity_bat_cycle = []
    carbon_intensity_es_cycle = []

    cef_bat_t = {}
    cef_es_t = {}

    # ========== Battery cycles ==========
    for cycle_number, times in process_times_bat.items():
        # Cycle energies (apply per-step ratio)
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()

        # Hold per-step values (same as original)
        cef_bat_charge = []
        cef_bat_discharge = []

        # Cycle-level average carbon intensity from total generation in this cycle
        values = df_gen_0.iloc[times].sum()
        total_emissions = 0.0
        total_energy = 0.0
        for source, (co2, cost) in CO2_FACTORS.items():
            total_emissions += values[source] * co2
            total_energy    += values[source]
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Per-step AEF calls (unchanged logic)
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]  # per-step ratio within this cycle
            cef_bat_charge = aef_log(
                idx, df_gen_0, cef_bat_charge,
                df_storage_links['bus_charger'],
                ratio, order, CO2_FACTORS
            )
            cef_bat_discharge = aef_log(
                idx, df_gen_0, cef_bat_discharge,
                -df_storage_links['bus_discharger'],
                ratio, order, CO2_FACTORS
            )

        # Difference list per step (charge - discharge)
        cef_bat_t[cycle_number] = [
            charge - discharge for charge, discharge in zip(cef_bat_charge, cef_bat_discharge)
        ]

        # Cycle sums (same as original)
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged

        # Store battery cycle outputs
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(
            delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0
        )
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

    # Overall battery factor and total
    co2_emissions_factor_bat = (
        np.sum(co2_delta_bat_emissions) / np.sum(energy_bat_discharged_cycle)
        if np.sum(energy_bat_discharged_cycle) != 0 else 0
    )
    total_emissions_bat = np.sum(co2_delta_bat_emissions)

    # ========== LDES cycles ==========
    for cycle_number, times in process_times_es.items():
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()

        cef_es_charge = []
        cef_es_discharge = []

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0.0
        total_energy = 0.0
        for source, (co2, cost) in CO2_FACTORS.items():
            total_emissions += values[source] * co2
            total_energy    += values[source]
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]
            cef_es_charge = aef_log(
                idx, df_gen_0, cef_es_charge,
                df_storage_links['es_bus_charger'],
                ratio, order, CO2_FACTORS
            )
            cef_es_discharge = aef_log(
                idx, df_gen_0, cef_es_discharge,
                -df_storage_links['es_bus_discharger'],
                ratio, order, CO2_FACTORS
            )

        cef_es_t[cycle_number] = [
            charge - discharge for charge, discharge in zip(cef_es_charge, cef_es_discharge)
        ]

        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged

        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(
            delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0
        )
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

    co2_emissions_factor_es = (
        np.sum(co2_delta_es_emissions) / np.sum(energy_es_discharged_cycle)
        if np.sum(energy_es_discharged_cycle) != 0 else 0
    )
    total_emissions_es = np.sum(co2_delta_es_emissions)

    return (
        cef_bat_t, cef_es_t,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
        total_emissions_bat, total_emissions_es,
        co2_emissions_factor_bat, co2_emissions_factor_es,
        co2_bat_charge_cycle, co2_es_charge_cycle,
        energy_bat_charge_cycle, energy_es_charge_cycle,
        energy_bat_discharged_cycle, energy_es_discharged_cycle,
        emissions_bat_charged_cycle, emissions_es_charged_cycle,
        emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
        co2_delta_bat_emissions, co2_delta_es_emissions
    )


def national_mef_analysis(
    process_times_bat, process_ratios_bat,
    process_times_es,  process_ratios_es,
    df_gen_0, df_storage_links, order, CO2_FACTORS
):
    """
    Analyze battery and LDES charge/discharge cycles using *marginal emission factor* (MEF).

    This follows your original logic: per cycle, it calls `mef_log` at each step to
    obtain the marginal CO₂ factor of the system (based on the resource order) and
    aggregates per-cycle metrics.

    Args:
        process_times_bat (dict[int, list[int]]): For each battery cycle id -> list of time-step indices.
        process_ratios_bat (dict[int, list[float]]): Per-cycle weights aligned with `process_times_bat[cycle]`.
        process_times_es (dict[int, list[int]]): For each LDES/other-storage cycle id -> list of time-step indices.
        process_ratios_es (dict[int, list[float]]): Per-cycle weights aligned with `process_times_es[cycle]`.
        df_gen_0 (pd.DataFrame): Per-step generation by source (columns must include keys in `CO2_FACTORS`).
        df_storage_links (pd.DataFrame): Storage link series with columns:
            'bus_charger', 'bus_discharger', 'es_bus_charger', 'es_bus_discharger'.
        order (Sequence[str]): **Resource index order constructed per optimization objective**.
            - **high-carbon → low-carbon** or **high-cost → low-cost**.
            Adjust `order` (or reverse it before passing) to match the intended marginal rule.
        CO2_FACTORS (Mapping[str, Tuple[float, float]]): {source: (co2_factor, marginal_cost)}.

    Returns:
        tuple: (
            cef_bat_t, cef_es_t,                             # per-cycle lists of (charge - discharge) step values
            carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
            total_emissions_bat, total_emissions_es,
            co2_emissions_factor_bat, co2_emissions_factor_es,  # Δemissions / discharged energy (per fleet)
            co2_bat_charge_cycle, co2_es_charge_cycle,          # per-cycle Δemissions / discharged energy
            energy_bat_charge_cycle, energy_es_charge_cycle,
            energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle,
            emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
            co2_delta_bat_emissions, co2_delta_es_emissions
        )
    """
    # --- accumulators ---
    energy_bat_charge_cycle, energy_es_charge_cycle = [], []
    energy_bat_discharged_cycle, energy_es_discharged_cycle = [], []
    co2_bat_charge_cycle, co2_es_charge_cycle = [], []
    emissions_bat_charged_cycle, emissions_es_charged_cycle = [], []
    emissions_bat_discharged_cycle, emissions_es_discharged_cycle = [], []
    co2_delta_bat_emissions, co2_delta_es_emissions = [], []
    carbon_intensity_bat_cycle, carbon_intensity_es_cycle = [], []
    df_gen_1 = df_gen_0.copy()  # kept for parity with original structure (unused below)
    cef_bat_t, cef_es_t = {}, {}

    # ===== Battery cycles =====
    for cycle_number, times in process_times_bat.items():
        # Cycle energies (apply per-step ratio)
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()

        # Per-step MEF lists (naming kept for backward compatibility)
        cef_bat_charge, cef_bat_discharge = [], []
        cost_bat_charge, cost_bat_discharge = [], []  # placeholders to mirror original (unused)

        # Cycle-average carbon intensity from total generation over the cycle
        values = df_gen_0.iloc[times].sum()
        total_emissions = total_energy = 0.0
        for src, (co2, _cost) in CO2_FACTORS.items():
            total_emissions += values[src] * co2
            total_energy    += values[src]
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Per-step MEF: `mef_log` internally scans `reversed(order)` to find the marginal source
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]
            cef_bat_charge = mef_log(idx, df_gen_0, cef_bat_charge, df_storage_links['bus_charger'],    ratio, order, CO2_FACTORS)
            cef_bat_discharge = mef_log(idx, df_gen_0, cef_bat_discharge, -df_storage_links['bus_discharger'], ratio, order, CO2_FACTORS)

        # Step-wise difference (charge - discharge) and cycle aggregates
        cef_bat_t[cycle_number] = [c - d for c, d in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged

        # Store outputs
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

    co2_emissions_factor_bat = (
        np.sum(co2_delta_bat_emissions) / np.sum(energy_bat_discharged_cycle)
        if np.sum(energy_bat_discharged_cycle) != 0 else 0
    )
    total_emissions_bat = np.sum(co2_delta_bat_emissions)

    # ===== LDES / Other storage cycles =====
    for cycle_number, times in process_times_es.items():
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()

        cef_es_charge, cef_es_discharge = [], []
        cost_es_charge, cost_es_discharge = [], []  # placeholders (unused)

        values = df_gen_0.iloc[times].sum()
        total_emissions = total_energy = 0.0
        for src, (co2, _cost) in CO2_FACTORS.items():
            total_emissions += values[src] * co2
            total_energy    += values[src]
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]
            cef_es_charge = mef_log(idx, df_gen_0, cef_es_charge, df_storage_links['es_bus_charger'],    ratio, order, CO2_FACTORS)
            cef_es_discharge = mef_log(idx, df_gen_0, cef_es_discharge, -df_storage_links['es_bus_discharger'], ratio, order, CO2_FACTORS)

        cef_es_t[cycle_number] = [c - d for c, d in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged

        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

    co2_emissions_factor_es = (
        np.sum(co2_delta_es_emissions) / np.sum(energy_es_discharged_cycle)
        if np.sum(energy_es_discharged_cycle) != 0 else 0
    )
    total_emissions_es = np.sum(co2_delta_es_emissions)

    return (
        cef_bat_t, cef_es_t,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
        total_emissions_bat, total_emissions_es,
        co2_emissions_factor_bat, co2_emissions_factor_es,
        co2_bat_charge_cycle, co2_es_charge_cycle,
        energy_bat_charge_cycle, energy_es_charge_cycle,
        energy_bat_discharged_cycle, energy_es_discharged_cycle,
        emissions_bat_charged_cycle, emissions_es_charged_cycle,
        emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
        co2_delta_bat_emissions, co2_delta_es_emissions
    )
