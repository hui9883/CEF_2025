import numpy as np

# Dictionary mapping energy sources to their respective CO2 emission factors (in tons/MWh).
CO2_FACTORS = {
    'PV': (0, 9),
    'Wind': (0, 13),
    'Wind offshore': (0, 17),
    'Hydro': (0, 21),
    'Nuclear': (0, 21),
    'Biomass': (0, 59),
    'Biomass_CHP': (0, 107),
    'Biogas': (0.19656, 73),
    'Biogas_CHP': (0.19656, 79),
    'CCGT': (0.36, 48),
    'SCGT_CHP': (0.46, 81),
    'SCGT': (0.46, 82),
    'Oil': (0.65, 144)
}

def cef_bat_discharge_log(idx, values, gen_by_carrier, cef_bat, cost_bat, df_battery_links_charge, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery storage charging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    cef_bat (list): List to accumulate the calculated CEF values for battery charging.
    df_battery_links_charge (Series): Series indicating the battery charging energy at each time step.

    Returns:
    tuple: Updated CEF list (`cef_bat`) and the updated energy generation DataFrame (`values`).
    """
    # If no energy is used for charging at the current time step, append 0 to CEF list.
    for source in resource_usage.keys():
        resource_usage[source]["bat_dis"].append(0)
    if df_battery_links_charge.iloc[idx] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        start_key = 'Wind'  # Default start source

        # Identify the starting energy source for calculation
        for source in reversed(sorted(CO2_FACTORS.keys(), key=lambda x: CO2_FACTORS[x][
            0])):  # Check sources in reverse order of CO2 factors
            try:
                if gen_by_carrier[source].iloc[idx] != 0:
                    start_key = source
                    break
            except KeyError:
                continue

        remaining_energy = -df_battery_links_charge.iloc[idx] * ratio
        total_emissions = 0
        total_cost = 0
        start = False

        # Iterate through each energy source sorted by their CO2 factors in descending order.
        for source, (co2, cost) in CO2_FACTORS.items():
            if source == start_key:
                start = True
            if not start:
                continue
            if remaining_energy <= 0:
                break

            # Use energy from the current source without exceeding the remaining energy.
            used_energy = min(values[source].iloc[idx], remaining_energy)
            total_emissions += used_energy * co2  # Accumulate emissions for the used energy.
            total_cost += used_energy * cost
            remaining_energy -= used_energy  # Reduce remaining energy for the next source.
            values.loc[values.index[idx], source] -= used_energy  # Update the available energy from the current source.
            resource_usage[source]["bat_dis"][-1] = used_energy  # Accumulate used energy for the source

        # Handle any remaining energy with a default emission factor
        if remaining_energy > 0:
            total_emissions += remaining_energy * 0.36  # Default CO2 factor for CCGT
            total_cost += 48 * remaining_energy
            resource_usage['Others']["bat_dis"][-1] = remaining_energy
        # Append total emissions for the current time step to the CEF list.
        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage

def cef_es_discharge_log(idx, values, gen_by_carrier, cef_bat, cost_bat, df_battery_links_charge, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery storage charging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    cef_bat (list): List to accumulate the calculated CEF values for battery charging.
    df_battery_links_charge (Series): Series indicating the battery charging energy at each time step.

    Returns:
    tuple: Updated CEF list (`cef_bat`) and the updated energy generation DataFrame (`values`).
    """
    # If no energy is used for charging at the current time step, append 0 to CEF list.
    for source in resource_usage.keys():
        resource_usage[source]["es_dis"].append(0)
    if df_battery_links_charge.iloc[idx] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        start_key = 'Wind'  # Default start source

        # Identify the starting energy source for calculation
        for source in reversed(sorted(CO2_FACTORS.keys(), key=lambda x: CO2_FACTORS[x][
            0])):  # Check sources in reverse order of CO2 factors
            try:
                if gen_by_carrier[source].iloc[idx] != 0:
                    start_key = source
                    break
            except KeyError:
                continue

        remaining_energy = -df_battery_links_charge.iloc[idx] * ratio
        total_emissions = 0
        total_cost = 0
        start = False

        # Iterate through each energy source sorted by their CO2 factors in descending order.
        for source, (co2, cost) in CO2_FACTORS.items():
            if source == start_key:
                start = True
            if not start:
                continue
            if remaining_energy <= 0:
                break

            # Use energy from the current source without exceeding the remaining energy.
            used_energy = min(values[source].iloc[idx], remaining_energy)
            total_emissions += used_energy * co2  # Accumulate emissions for the used energy.
            total_cost += used_energy * cost
            remaining_energy -= used_energy  # Reduce remaining energy for the next source.
            values.loc[idx, source] -= used_energy  # Update the available energy from the current source.
            resource_usage[source]["es_dis"][-1] = used_energy  # Accumulate used energy for the source

        # Handle any remaining energy with a default emission factor
        if remaining_energy > 0:
            total_emissions += remaining_energy * 0.36
            total_cost += 48 * remaining_energy
            resource_usage['Others']["es_dis"][-1] = remaining_energy
        # Append total emissions for the current time step to the CEF list.
        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, resource_usage


def cef_bat_log(idx, values, cef_bat, cost_bat, df_battery_links_charge, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery storage charging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    cef_bat (list): List to accumulate the calculated CEF values for battery charging.
    df_battery_links_charge (Series): Series indicating the battery charging energy at each time step.

    Returns:
    tuple: Updated CEF list (`cef_bat`) and the updated energy generation DataFrame (`values`).
    """
    # If no energy is used for charging at the current time step, append 0 to CEF list.
    for source in resource_usage.keys():
        resource_usage[source]["bat_cha"].append(0)
    if df_battery_links_charge.iloc[idx] * ratio == 0:
        cef_bat.append(0)
        cost_bat.append(0)
    else:
        remaining_energy = df_battery_links_charge.iloc[idx] * ratio
        total_emissions = 0
        total_cost = 0

        # Iterate through each energy source sorted by their CO2 factors in descending order.
        for source, (co2, cost) in reversed(sorted(CO2_FACTORS.items(), key=lambda x: x[1][0])):
            if remaining_energy <= 0:
                break

            # Use energy from the current source without exceeding the remaining energy.
            used_energy = min(values[source].iloc[idx], remaining_energy)
            total_emissions += used_energy * co2  # Accumulate emissions for the used energy.
            total_cost += used_energy * cost
            remaining_energy -= used_energy  # Reduce remaining energy for the next source.
            values.loc[values.index[idx], source] -= used_energy  # Update the available energy from the current source.
            resource_usage[source]["bat_cha"][-1] = used_energy  # Accumulate used energy for the source

        # Append total emissions for the current time step to the CEF list.
        cef_bat.append(total_emissions)
        cost_bat.append(total_cost)

    return cef_bat, cost_bat, values, resource_usage

def cef_es_log(idx, values, cef_es, cost_es, df_ldes_links_charge, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for long-duration energy storage (LDES) charging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    cef_es (list): List to accumulate the calculated CEF values for LDES charging.
    df_ldes_links_charge (Series): Series indicating the LDES charging energy at each time step.

    Returns:
    tuple: Updated CEF list (`cef_es`) and the updated energy generation DataFrame (`values`).
    """
    # If no energy is used for charging at the current time step, append 0 to CEF list.
    for source in resource_usage.keys():
        resource_usage[source]["es_cha"].append(0)
    if df_ldes_links_charge.iloc[idx] * ratio == 0:
        cef_es.append(0)
        cost_es.append(0)
    else:
        remaining_energy = df_ldes_links_charge.iloc[idx] *ratio
        total_emissions = 0
        total_cost = 0

        # Iterate through each energy source sorted by their CO2 factors in descending order.
        for source, (co2, cost) in reversed(sorted(CO2_FACTORS.items(), key=lambda x: x[1][0])):
            if remaining_energy <= 0:
                break

            # Use energy from the current source without exceeding the remaining energy.
            used_energy = min(values[source].iloc[idx], remaining_energy)
            total_emissions += used_energy * co2  # Accumulate emissions for the used energy.
            total_cost += used_energy * cost
            remaining_energy -= used_energy  # Reduce remaining energy for the next source.
            resource_usage[source]["es_cha"][-1] = used_energy  # Accumulate used energy for the source

        # Append total emissions for the current time step to the CEF list.
        cef_es.append(total_emissions)
        cost_es.append(total_cost)

    return cef_es, cost_es, resource_usage



def aef_log(idx, values, aef, df_battery_links_charge, ratio):
    """
    Calculate and append the marginal emission factor (MEF) for a specific index based on the available energy source data.

    Args:
    idx (int): Index for the current time step in the DataFrame.
    values (DataFrame): DataFrame containing the energy generation data for each source.
    mef (list): List to accumulate calculated MEF values.
    es_energy (Series): Series indicating the amount of energy stored or used at each time step.
    gen_update (DataFrame): DataFrame tracking the updated generation data after accounting for storage usage.

    Returns:
    tuple: Returns the updated mef list and gen_update DataFrame after calculations.
    """
    # Append zero to mef list if the energy storage at the current index is zero (no energy use).
    if df_battery_links_charge.iloc[idx] * ratio == 0:
        aef.append(0)
        return aef

    remaining_energy = df_battery_links_charge.iloc[idx] * ratio
    total_emissions = 0

    # Loop through each energy source available in the CO2_FACTORS dictionary.
    for source, (co2, cost) in CO2_FACTORS.items():
        # Calculate the energy used from the current source without exceeding the remaining energy.
        total_emissions += values[source].iloc[idx] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.

    #get aef (co2/MWh) then times the charged energy in MWh
    total_emissions = total_emissions * (df_battery_links_charge.iloc[idx] * ratio)/values.iloc[idx, 1:].sum()

    # Compute the MEF for the current index if there was any energy usage, otherwise set it to zero.

    aef.append(total_emissions)

    return aef




def mef_log(idx, values, mef, df_battery_links_charge, ratio):
    """
    Calculate and append the marginal emission factor (MEF) for a specific index based on the available energy source data.

    Args:
    idx (int): Index for the current time step in the DataFrame.
    values (DataFrame): DataFrame containing the energy generation data for each source.
    mef (list): List to accumulate calculated MEF values.
    es_energy (Series): Series indicating the amount of energy stored or used at each time step.
    gen_update (DataFrame): DataFrame tracking the updated generation data after accounting for storage usage.

    Returns:
    tuple: Returns the updated mef list and gen_update DataFrame after calculations.
    """
    # Append zero to mef list if the energy storage at the current index is zero (no energy use).
    if df_battery_links_charge.iloc[idx] * ratio == 0:
        mef.append(0)
        return mef

    selected_co2 = 0
    for source, (co2, cost) in reversed(sorted(CO2_FACTORS.items(), key=lambda x: x[1][0])):
        if values[source].iloc[idx] > 0:
            selected_co2 = co2
            break

    total_emissions = selected_co2 * df_battery_links_charge.iloc[idx] * ratio

    mef.append(total_emissions)

    return mef



def national_cycle_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, df_gen_0, df_storage_links, df_gen_remain, resource_usage):
    """
    Analyze charging and discharging cycles to calculate the CO2 impact and energy usage for batteries and long-duration energy storage (LDES).

    Args:
    charg_x (list): List of start indices for battery charging cycles.
    charg_y (list): List of end indices for battery charging cycles.
    charg_x_es (list): List of start indices for LDES charging cycles.
    charg_y_es (list): List of end indices for LDES charging cycles.
    df_gen_0 (DataFrame): Initial generation data.
    df_battery_links_charge (DataFrame): Battery charging energy data per time step.
    df_ldes_links_charge (DataFrame): LDES charging energy data per time step.
    df_storage_links (DataFrame): Storage discharging energy data per time step.
    df_gen_remain (DataFrame): Remaining generation data after accounting for discharging.
    gen_bus_carrier (DataFrame): Generation data by region and carrier.
    regions (str or list): Region(s) to consider in the analysis.

    Returns:
    tuple: Results including:
        - CO2 emissions factors for batteries and LDES.
        - Lists of CO2 impacts and energy usage per cycle.
        - Lists of emissions during charging and discharging for batteries and LDES.
        - Lists of delta CO2 emissions per cycle.
    """
    # Initialize lists to store results
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
    emissions_bat_discharged_cycle = []
    emissions_es_discharged_cycle = []
    cost_es_charged_cycle = []
    co2_delta_bat_emissions = []
    co2_delta_es_emissions = []
    carbon_intensity_bat_cycle = []
    carbon_intensity_es_cycle = []
    df_gen_1 = df_gen_0.copy()
    cef_bat_t = {}
    cef_es_t = {}
    unit_ccost_bat_cycle = []
    unit_ccost_es_cycle = []
    unit_dcost_bat_cycle = []
    unit_dcost_es_cycle = []
    unit_cost_bat_cycle = []
    unit_cost_es_cycle = []

    # Analyze battery charging and discharging cycles
    for cycle_number, times in process_times_bat.items():
        # Calculate total energy used during the cycle
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (-df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        cef_bat_charge = []  # Initialize list for CO2 emissions during charging
        cef_bat_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_bat_charge = []
        cost_bat_discharge = []
        resource_bat = {
            source: {"bat_cha": [], "bat_dis": []} for source in resource_usage.keys()
        }
        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_bat_charge, cost_bat_charge, df_gen_1, resource_bat = cef_bat_log(idx, df_gen_1, cef_bat_charge, cost_bat_charge, df_storage_links['bus_charger'], resource_bat)
        #     cef_bat_discharge, cost_bat_discharge, df_gen_remain, resource_bat = cef_bat_discharge_log(idx, df_gen_remain, df_gen_0, cef_bat_discharge, cost_bat_discharge, df_storage_links['bus_discharger'], resource_bat)
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]
            cef_bat_charge, cost_bat_charge, df_gen_1, resource_bat = cef_bat_log(
                idx,
                df_gen_1,
                cef_bat_charge,
                cost_bat_charge,
                df_storage_links['bus_charger'],
                ratio,
                resource_bat
            )
            cef_bat_discharge, cost_bat_discharge, df_gen_remain, resource_bat = cef_bat_discharge_log(
                idx,
                df_gen_remain,
                df_gen_0,
                cef_bat_discharge,
                cost_bat_discharge,
                df_storage_links['bus_discharger'],
                ratio,
                resource_bat
            )

        # Calculate emissions and deltas for the cycle
        cef_bat_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        cost_bat_charged = np.sum(cost_bat_charge)
        cost_bat_discharged = np.sum(cost_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged
        delta_bat_cost = -cost_bat_discharged + cost_bat_charged
        unit_charged_cost_bat = cost_bat_charged / energy_bat_charge if energy_bat_charge != 0 else 0
        unit_discharged_cost_bat = cost_bat_discharged / energy_bat_discharge if energy_bat_discharge != 0 else 0
        unit_cha_dis_cost_bat = delta_bat_cost / energy_bat_discharge if energy_bat_discharge != 0 else 0

        # Append results for batteries
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        cost_bat_charged_cycle.append(cost_bat_charged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        co2_bat_charge_energy_cycle.append(delta_bat_emissions / energy_bat_charge if energy_bat_charge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        unit_ccost_bat_cycle.append(unit_charged_cost_bat)
        unit_dcost_bat_cycle.append(unit_discharged_cost_bat)
        unit_cost_bat_cycle.append(unit_cha_dis_cost_bat)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)
        for source in resource_usage.keys():
            resource_usage[source]["bat_cha"].append(np.sum(resource_bat[source]["bat_cha"]))
            resource_usage[source]["bat_dis"].append(np.sum(resource_bat[source]["bat_dis"]))

    # Calculate overall CO2 emissions factor for batteries
    co2_emissions_factor_bat = np.sum(co2_delta_bat_emissions) / np.sum(energy_bat_discharged_cycle) if np.sum(energy_bat_discharged_cycle) != 0 else 0
    total_emissions_bat = np.sum(co2_delta_bat_emissions)
    total_cost_bat = np.sum(cost_bat_charged_cycle)

    # Analyze LDES charging and discharging cycles
    for cycle_number, times in process_times_es.items():
        # Calculate total energy used during the cycle
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (-df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        cef_es_charge = []  # Initialize list for CO2 emissions during charging
        cef_es_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_es_charge = []
        cost_es_discharge = []
        resource_es = {
            source: {"es_cha": [], "es_dis": []} for source in resource_usage.keys()
        }

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_es_charge, cost_es_charge, resource_es = cef_es_log(idx, df_gen_1, cef_es_charge, cost_es_charge, df_storage_links['es_bus_charger'], resource_es)
        #     cef_es_discharge, cost_es_discharge, resource_es = cef_es_discharge_log(idx, df_gen_remain, df_gen_0, cef_es_discharge, cost_es_discharge, df_storage_links['es_bus_discharger'], resource_es)
        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]  # 获取当前时间步对应的比例
            cef_es_charge, cost_es_charge, resource_es = cef_es_log(
                idx,
                df_gen_1,
                cef_es_charge,
                cost_es_charge,
                df_storage_links['es_bus_charger'],
                ratio,
                resource_es
            )
            cef_es_discharge, cost_es_discharge, resource_es = cef_es_discharge_log(
                idx,
                df_gen_remain,
                df_gen_0,
                cef_es_discharge,
                cost_es_discharge,
                df_storage_links['es_bus_discharger'],
                ratio,
                resource_es
            )

        # Calculate emissions and deltas for the cycle
        cef_es_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        cost_es_charged = np.sum(cost_es_charge)
        cost_es_discharged = np.sum(cost_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged
        delta_es_cost = -cost_es_discharged + cost_es_charged
        unit_charged_cost_es = cost_es_charged / energy_es_charge if energy_es_charge != 0 else 0
        unit_discharged_cost_es = cost_es_discharged / energy_es_discharge if energy_es_discharge != 0 else 0
        unit_cha_dis_cost_es = delta_es_cost / energy_es_discharge if energy_es_discharge != 0 else 0

        # Append results for LDES
        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        cost_es_charged_cycle.append(cost_es_charged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        co2_es_charge_energy_cycle.append(delta_es_emissions / energy_es_charge if energy_es_charge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        unit_ccost_es_cycle.append(unit_charged_cost_es)
        unit_dcost_es_cycle.append(unit_discharged_cost_es)
        unit_cost_es_cycle.append(unit_cha_dis_cost_es)
        carbon_intensity_es_cycle.append(carbon_intensity_es)
        for source in resource_usage.keys():
            resource_usage[source]["es_cha"].append(np.sum(resource_es[source]["es_cha"]))
            resource_usage[source]["es_dis"].append(np.sum(resource_es[source]["es_dis"]))

    # Calculate overall CO2 emissions factor for LDES
    co2_emissions_factor_es = np.sum(co2_delta_es_emissions) / np.sum(energy_es_discharged_cycle) if np.sum(energy_es_discharged_cycle) != 0 else 0
    total_emissions_es = np.sum(co2_delta_es_emissions)
    total_cost_es = np.sum(cost_es_charged_cycle)

    # Return all results as a tuple
    return (cef_bat_t, cef_es_t, unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
            unit_cost_bat_cycle, unit_cost_es_cycle,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle, total_cost_bat, total_cost_es, total_emissions_bat,
            total_emissions_es,co2_emissions_factor_bat, co2_emissions_factor_es, co2_bat_charge_cycle, co2_es_charge_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle, energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle, emissions_bat_discharged_cycle,
            emissions_es_discharged_cycle, co2_delta_bat_emissions, co2_delta_es_emissions, resource_usage)

def national_aef_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es,  df_gen_0, df_storage_links):
    """
    Analyze charging and discharging cycles to calculate the CO2 impact and energy usage for batteries and long-duration energy storage (LDES).

    Args:
    charg_x (list): List of start indices for battery charging cycles.
    charg_y (list): List of end indices for battery charging cycles.
    charg_x_es (list): List of start indices for LDES charging cycles.
    charg_y_es (list): List of end indices for LDES charging cycles.
    df_gen_0 (DataFrame): Initial generation data.
    df_battery_links_charge (DataFrame): Battery charging energy data per time step.
    df_ldes_links_charge (DataFrame): LDES charging energy data per time step.
    df_storage_links (DataFrame): Storage discharging energy data per time step.
    df_gen_remain (DataFrame): Remaining generation data after accounting for discharging.
    gen_bus_carrier (DataFrame): Generation data by region and carrier.
    regions (str or list): Region(s) to consider in the analysis.

    Returns:
    tuple: Results including:
        - CO2 emissions factors for batteries and LDES.
        - Lists of CO2 impacts and energy usage per cycle.
        - Lists of emissions during charging and discharging for batteries and LDES.
        - Lists of delta CO2 emissions per cycle.
    """
    # Initialize lists to store results
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
    df_gen_1 = df_gen_0.copy()
    cef_bat_t = {}
    cef_es_t = {}

    # Analyze battery charging and discharging cycles
    for cycle_number, times in process_times_bat.items():
        # Calculate total energy used during the cycle
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (
                    -df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        cef_bat_charge = []  # Initialize list for CO2 emissions during charging
        cef_bat_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_bat_charge = []
        cost_bat_discharge = []

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[
                                   source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_bat_charge = aef_log(idx, df_gen_0, cef_bat_charge, df_storage_links['bus_charger'])
        #     cef_bat_discharge = aef_log(idx, df_gen_0, cef_bat_discharge, -df_storage_links['bus_discharger'])
        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]
            cef_bat_charge = aef_log(
                idx,
                df_gen_0,
                cef_bat_charge,
                df_storage_links['bus_charger'],
                ratio
            )
            cef_bat_discharge = aef_log(
                idx,
                df_gen_0,
                cef_bat_discharge,
                -df_storage_links['bus_discharger'],
                ratio
            )

        # Calculate emissions and deltas for the cycle
        cef_bat_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged

        # Append results for batteries
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

    # Calculate overall CO2 emissions factor for batteries
    co2_emissions_factor_bat = np.sum(co2_delta_bat_emissions) / np.sum(energy_bat_discharged_cycle) if np.sum(energy_bat_discharged_cycle) != 0 else 0
    total_emissions_bat = np.sum(co2_delta_bat_emissions)

    # Analyze LDES charging and discharging cycles
    for cycle_number, times in process_times_es.items():
        # Calculate total energy used during the cycle
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (
                    -df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        cef_es_charge = []  # Initialize list for CO2 emissions during charging
        cef_es_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_es_charge = []
        cost_es_discharge = []

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_es_charge = aef_log(idx, df_gen_0, cef_es_charge, df_storage_links['es_bus_charger'])
        #     cef_es_discharge = aef_log(idx, df_gen_0, cef_es_discharge, -df_storage_links['es_bus_discharger'])
        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]
            cef_es_charge = aef_log(
                idx,
                df_gen_0,
                cef_es_charge,
                df_storage_links['es_bus_charger'],
                ratio
            )
            cef_es_discharge = aef_log(
                idx,
                df_gen_0,
                cef_es_discharge,
                -df_storage_links['es_bus_discharger'],
                ratio
            )

        # Calculate emissions and deltas for the cycle
        cef_es_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged

        # Append results for LDES
        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

    # Calculate overall CO2 emissions factor for LDES
    co2_emissions_factor_es = np.sum(co2_delta_es_emissions) / np.sum(energy_es_discharged_cycle) if np.sum(energy_es_discharged_cycle) != 0 else 0
    total_emissions_es = np.sum(co2_delta_es_emissions)

    # Return all results as a tuple
    return (cef_bat_t, cef_es_t,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle, total_emissions_bat, total_emissions_es,
            co2_emissions_factor_bat, co2_emissions_factor_es, co2_bat_charge_cycle, co2_es_charge_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle, energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle, emissions_bat_discharged_cycle,
            emissions_es_discharged_cycle, co2_delta_bat_emissions, co2_delta_es_emissions)


def national_mef_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es,  df_gen_0, df_storage_links):
    """
    Analyze charging and discharging cycles to calculate the CO2 impact and energy usage for batteries and long-duration energy storage (LDES).

    Args:
    charg_x (list): List of start indices for battery charging cycles.
    charg_y (list): List of end indices for battery charging cycles.
    charg_x_es (list): List of start indices for LDES charging cycles.
    charg_y_es (list): List of end indices for LDES charging cycles.
    df_gen_0 (DataFrame): Initial generation data.
    df_battery_links_charge (DataFrame): Battery charging energy data per time step.
    df_ldes_links_charge (DataFrame): LDES charging energy data per time step.
    df_storage_links (DataFrame): Storage discharging energy data per time step.
    df_gen_remain (DataFrame): Remaining generation data after accounting for discharging.
    gen_bus_carrier (DataFrame): Generation data by region and carrier.
    regions (str or list): Region(s) to consider in the analysis.

    Returns:
    tuple: Results including:
        - CO2 emissions factors for batteries and LDES.
        - Lists of CO2 impacts and energy usage per cycle.
        - Lists of emissions during charging and discharging for batteries and LDES.
        - Lists of delta CO2 emissions per cycle.
    """
    # Initialize lists to store results
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
    df_gen_1 = df_gen_0.copy()
    cef_bat_t = {}
    cef_es_t = {}

    # Analyze battery charging and discharging cycles
    for cycle_number, times in process_times_bat.items():
        # Calculate total energy used during the cycle
        energy_bat_charge = (df_storage_links['bus_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (
                    -df_storage_links['bus_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        cef_bat_charge = []  # Initialize list for CO2 emissions during charging
        cef_bat_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_bat_charge = []
        cost_bat_discharge = []

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[
                                   source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_bat = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_bat_charge = mef_log(idx, df_gen_0, cef_bat_charge, df_storage_links['bus_charger'])
        #     cef_bat_discharge = mef_log(idx, df_gen_0, cef_bat_discharge, -df_storage_links['bus_discharger'])
        for pos, idx in enumerate(times):
            # 获取对应位置的 process ratio
            ratio = process_ratios_bat[cycle_number][pos]
            # 使用乘积后的数据传递给 mef_log
            cef_bat_charge = mef_log(idx, df_gen_0, cef_bat_charge, df_storage_links['bus_charger'],ratio)
            cef_bat_discharge = mef_log(idx, df_gen_0, cef_bat_discharge, -df_storage_links['bus_discharger'], ratio)

        # Calculate emissions and deltas for the cycle
        cef_bat_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_bat_charge, cef_bat_discharge)]
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged

        # Append results for batteries
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        carbon_intensity_bat_cycle.append(carbon_intensity_bat)

    # Calculate overall CO2 emissions factor for batteries
    co2_emissions_factor_bat = np.sum(co2_delta_bat_emissions) / np.sum(energy_bat_discharged_cycle) if np.sum(energy_bat_discharged_cycle) != 0 else 0
    total_emissions_bat = np.sum(co2_delta_bat_emissions)

    # Analyze LDES charging and discharging cycles
    for cycle_number, times in process_times_es.items():
        # Calculate total energy used during the cycle
        energy_es_charge = (df_storage_links['es_bus_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (
                    -df_storage_links['es_bus_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        cef_es_charge = []  # Initialize list for CO2 emissions during charging
        cef_es_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_es_charge = []
        cost_es_discharge = []

        values = df_gen_0.iloc[times].sum()
        total_emissions = 0
        total_energy = 0
        for source, (co2, cost) in CO2_FACTORS.items():
            # Calculate the energy used from the current source without exceeding the remaining energy.
            total_emissions += values[source] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.
            total_energy += values[source]

        # get aef (co2/MWh) then times the charged energy in MWh
        carbon_intensity_es = total_emissions / total_energy if total_energy != 0 else 0

        # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_es_charge = mef_log(idx, df_gen_0, cef_es_charge, df_storage_links['es_bus_charger'])
        #     cef_es_discharge = mef_log(idx, df_gen_0, cef_es_discharge, -df_storage_links['es_bus_discharger'])
        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]
            cef_es_charge = mef_log(
                idx,
                df_gen_0,
                cef_es_charge,
                df_storage_links['es_bus_charger'], ratio
            )
            cef_es_discharge = mef_log(
                idx,
                df_gen_0,
                cef_es_discharge,
                -df_storage_links['es_bus_discharger'], ratio
            )

        # Calculate emissions and deltas for the cycle
        cef_es_t[cycle_number] = [charge - discharge for charge, discharge in zip(cef_es_charge, cef_es_discharge)]
        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged

        # Append results for LDES
        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        carbon_intensity_es_cycle.append(carbon_intensity_es)

    # Calculate overall CO2 emissions factor for LDES
    co2_emissions_factor_es = np.sum(co2_delta_es_emissions) / np.sum(energy_es_discharged_cycle) if np.sum(energy_es_discharged_cycle) != 0 else 0
    total_emissions_es = np.sum(co2_delta_es_emissions)

    # Return all results as a tuple
    return (cef_bat_t, cef_es_t,
        carbon_intensity_bat_cycle, carbon_intensity_es_cycle, total_emissions_bat, total_emissions_es,
            co2_emissions_factor_bat, co2_emissions_factor_es, co2_bat_charge_cycle, co2_es_charge_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle, energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle, emissions_bat_discharged_cycle,
            emissions_es_discharged_cycle, co2_delta_bat_emissions, co2_delta_es_emissions)