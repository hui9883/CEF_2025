import numpy as np
import pandas as pd

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
regions = ['EN_NorthEast','EN_NorthWest','EN_Yorkshire',
           'EN_EastMidlands','EN_WestMidlands',
           'EN_East','EN_London','EN_SouthEast',
           'EN_SouthWest','EN_Wales','Scotland',
           'NorthernIreland']
resource_usage = {
    source: {"bat_cha": 0, "es_cha": 0, "bat_dis": 0, "es_dis": 0} for source in CO2_FACTORS.keys()
}  # Initialize resource usage dictionary for each source


def adjust_gen_by_demand(values, regions, load):
    """
    Adjusts the values DataFrame based on the demand in the load DataFrame for each region, across all time indices.

    Parameters:
    values (DataFrame): DataFrame containing source values for different regions.
    regions (list): List of regions to process.
    load (DataFrame): DataFrame containing load demands for each region.

    Returns:
    DataFrame: The modified values DataFrame with updated entries based on demand.
    """
    # Iterate over all indices in the values DataFrame
    for idx in range(len(values)):
        for region in regions:
            source_demand = load[region].iloc[idx]
            for source, _ in CO2_FACTORS.items():  # Assuming CO2_FACTORS is a dictionary of tuples
                try:
                    use_source = values[region + '_' + source].iloc[idx]
                    values.loc[values.index[idx], region + '_' + source] = use_source - min(use_source, source_demand)
                    source_demand = max(source_demand - use_source, 0)
                    if source_demand == 0:
                        break
                except KeyError:
                    continue  # If the key does not exist in the DataFrame, skip to next source

    return values


def calculate_aef(df_gen, df_gen_bus_carrier_total):
    aef = {}
    for region in regions:
        aef_0 = 0
        denominator = 0
        for source, (co2, _) in CO2_FACTORS.items():
            column_name = f"{region}_{source}"
            # 提取数据并计算AEF的分子和分母
            try:
                generation_data = df_gen.loc[:, f"{source}"] - df_gen_bus_carrier_total.loc[:, column_name]
            except KeyError:
                # 如果列不存在，则默认使用df_gen的数据
                generation_data = df_gen.loc[:, f"{source}"]

            # 累加计算AEF的分子和分母
            aef_0 += generation_data * co2
            denominator += generation_data

        aef[region] = np.where(denominator != 0, aef_0 / denominator, 0.651)

    return aef

def cef_bat_discharge_log(idx, values, gen_bus_carrier, cef_bat, cost_bat, df_battery_links_charge, regions, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery discharging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    gen_bus_carrier (DataFrame): DataFrame containing the generation data per bus and carrier.
    cef_bat (list): List to accumulate the calculated CEF values for battery discharging.
    df_battery_links_charge (DataFrame): DataFrame indicating the battery discharging energy at each time step.
    regions (str or list): Region(s) to consider in the calculation.

    Returns:
    tuple: Updated CEF list (`cef_bat`) and the updated energy generation DataFrame (`values`).
    """
    for source in resource_usage.keys():
        resource_usage[source]["bat_dis"].append(0)
    cef_b = []
    cost_b = []
    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if df_battery_links_charge[region + '_Battery_discharger'].iloc[idx] * ratio == 0:
            cef_b.append(0)  # No discharging energy, append 0
            cost_b.append(0)
        else:
            start_key = 'Wind'  # Default start source

            # Identify the starting energy source for calculation
            for source in reversed(sorted(CO2_FACTORS.keys(), key=lambda x: CO2_FACTORS[x][0])):  # Check sources in reverse order of CO2 factors
                try:
                    if gen_bus_carrier[region + '_' + source].iloc[idx] != 0:
                        start_key = source
                        break
                except KeyError:
                    continue

            # Initialize emission calculation
            remaining_disenergy = -df_battery_links_charge[region + '_Battery_discharger'].iloc[idx] * ratio
            total_emissions = 0
            total_cost = 0
            start = False

            # Process each energy source starting from the identified source
            for source, (co2, cost) in CO2_FACTORS.items():
                if source == start_key:
                    start = True
                if not start:
                    continue

                if remaining_disenergy <= 0:
                    break

                # Use energy from the current source without exceeding remaining energy
                try:
                    used_energy = min(values[region + '_' + source].iloc[idx], remaining_disenergy)
                except KeyError:
                    used_energy = 0

                total_emissions += used_energy * co2  # Accumulate emissions
                total_cost += cost * used_energy
                remaining_disenergy -= used_energy  # Update remaining energy
                resource_usage[source]["bat_dis"][-1] = used_energy  # Accumulate used energy for the source

                # Update energy in the values DataFrame
                try:
                    values.loc[values.index[idx], region + '_' + source] -= used_energy
                except KeyError:
                    pass

            # Handle any remaining energy with a default emission factor
            if remaining_disenergy > 0:
                total_emissions += remaining_disenergy * 0.36
                total_cost += 48 * remaining_disenergy
                resource_usage['Others']["bat_dis"][-1] = remaining_disenergy

            cef_b.append(total_emissions)  # Append total emissions for the region
            cost_b.append(total_cost)

    cef_bat.append(sum(cef_b))  # Sum emissions for all regions
    cost_bat.append(sum(cost_b))
    return cef_bat, cost_bat, values, resource_usage


def cef_es_discharge_log(idx, values, gen_bus_carrier, cef_es, cost_es, df_battery_links_charge, regions, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for long-duration energy storage (LDES) discharging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    gen_bus_carrier (DataFrame): DataFrame containing the generation data per bus and carrier.
    cef_es (list): List to accumulate the calculated CEF values for LDES discharging.
    df_battery_links_charge (DataFrame): DataFrame indicating the LDES discharging energy at each time step.
    regions (str or list): Region(s) to consider in the calculation.

    Returns:
    tuple: Updated CEF list (`cef_es`) and the updated energy generation DataFrame (`values`).
    """
    for source in resource_usage.keys():
        resource_usage[source]["es_dis"].append(0)
    cef_l = []
    cost_l = []
    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if df_battery_links_charge[region + '_OtherStorage_discharger'].iloc[idx] * ratio == 0:
            cef_l.append(0)  # No discharging energy, append 0
            cost_l.append(0)
        else:
            start_key = 'Wind'  # Default start source

            # Identify the starting energy source for calculation
            for source in reversed(sorted(CO2_FACTORS.keys(), key=lambda x: CO2_FACTORS[x][0])):  # Check sources in reverse order of CO2 factors
                try:
                    if gen_bus_carrier[region + '_' + source].iloc[idx] != 0:
                        start_key = source
                        break
                except KeyError:
                    continue

            # Initialize emission calculation
            remaining_disenergy = -df_battery_links_charge[region + '_OtherStorage_discharger'].iloc[idx] * ratio
            total_emissions = 0
            total_cost = 0
            start = False

            # Process each energy source starting from the identified source
            for source, (co2, cost) in CO2_FACTORS.items():
                if source == start_key:
                    start = True
                if not start:
                    continue

                if remaining_disenergy <= 0:
                    break

                # Use energy from the current source without exceeding remaining energy
                try:
                    used_energy = min(values[region + '_' + source].iloc[idx], remaining_disenergy)
                except KeyError:
                    used_energy = 0

                total_emissions += used_energy * co2  # Accumulate emissions
                total_cost += used_energy * cost
                remaining_disenergy -= used_energy  # Update remaining energy
                resource_usage[source]["es_dis"][-1] = used_energy  # Accumulate used energy for the source

                # Update energy in the values DataFrame
                try:
                    values.loc[values.index[idx], region + '_' + source] -= used_energy
                except KeyError:
                    pass

            # Handle any remaining energy with a default emission factor
            if remaining_disenergy > 0:
                total_emissions += remaining_disenergy * 0.36
                total_cost += 48 * remaining_disenergy
                resource_usage['Others']["es_dis"][-1] = remaining_disenergy

            cef_l.append(total_emissions)  # Append total emissions for the region
            cost_l.append(total_cost)

    cef_es.append(sum(cef_l))  # Sum emissions for all regions
    cost_es.append(sum(cost_l))
    return cef_es, cost_es, resource_usage


def cef_bat_log(idx, aef, values, cef_bat, cost_bat, df_battery_links_charge,regions, ratio, resource_usage):
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery discharging at a specific time step.

    Args:
    idx (int): Current time step index in the DataFrame.
    values (DataFrame): DataFrame containing energy generation data for each source.
    gen_bus_carrier (DataFrame): DataFrame containing the generation data per bus and carrier.
    cef_bat (list): List to accumulate the calculated CEF values for battery discharging.
    df_battery_links_charge (DataFrame): DataFrame indicating the battery discharging energy at each time step.
    regions (str or list): Region(s) to consider in the calculation.

    Returns:
    tuple: Updated CEF list (`cef_bat`) and the updated energy generation DataFrame (`values`).
    """
    for source in resource_usage.keys():
        resource_usage[source]["bat_cha"].append(0)
    cef_b = []
    cost_b = []
    if isinstance(regions, str):
        regions = [regions]
    for region in regions:
        if df_battery_links_charge[region + '_Battery_charger'].iloc[idx] * ratio == 0:
            cef_b.append(0)  # No discharging energy, append 0
            cost_b.append(0)
        else:
            # Initialize emission calculation
            remaining_energy = df_battery_links_charge[region + '_Battery_charger'].iloc[idx] * ratio
            total_emissions = 0
            total_cost = 0

            # Process each energy source starting from the identified source
            for source, (co2, cost) in reversed(sorted(CO2_FACTORS.items(), key=lambda x: x[1][0])):
                if remaining_energy <= 0:
                    break

                # Use energy from the current source without exceeding remaining energy
                try:
                    used_energy = min(values[region + '_' + source].iloc[idx], remaining_energy)
                except KeyError:
                    used_energy = 0

                total_emissions += used_energy * co2  # Accumulate emissions
                total_cost += cost * used_energy
                remaining_energy -= used_energy  # Update remaining energy
                resource_usage[source]["bat_cha"][-1] = used_energy  # Accumulate used energy for the source

                # Update energy in the values DataFrame
                try:
                    values.loc[values.index[idx], region + '_' + source] -= used_energy
                except KeyError:
                    pass

            # Handle any remaining energy with a default emission factor
            if remaining_energy > 0:
                total_emissions += remaining_energy * aef[region][idx] #
                total_cost += 48 * remaining_energy
                resource_usage['Others']["bat_cha"][-1] = remaining_energy

            cef_b.append(total_emissions)  # Append total emissions for the region
            cost_b.append(total_cost)

    cef_bat.append(sum(cef_b))  # Sum emissions for all regions
    cost_bat.append(sum(cost_b))
    return cef_bat, cost_bat, values, resource_usage


def cef_es_log(idx, values, cef_es, cost_es, df_storage_links, regions, ratio, resource_usage, aef):
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
    cef_l = []
    cost_l = []
    if isinstance(regions, str):
        regions = [regions]

    for region in regions:
        if df_storage_links[region + '_OtherStorage_charger'].iloc[idx] * ratio == 0:
            cef_l.append(0)  # No discharging energy, append 0
            cost_l.append(0)
        else:
            # Initialize emission calculation
            remaining_energy = df_storage_links[region + '_OtherStorage_charger'].iloc[idx] * ratio
            total_emissions = 0
            total_cost = 0
            start = False

            # Process each energy source starting from the identified source
            for source, (co2, cost) in reversed(sorted(CO2_FACTORS.items(), key=lambda x: x[1][0])):
                if remaining_energy <= 0:
                    break

                # Use energy from the current source without exceeding remaining energy
                try:
                    used_energy = min(values[region + '_' + source].iloc[idx], remaining_energy)
                except KeyError:
                    used_energy = 0

                total_emissions += used_energy * co2  # Accumulate emissions
                total_cost += used_energy * cost
                remaining_energy -= used_energy  # Update remaining energy
                resource_usage[source]["es_cha"][-1] = used_energy  # Accumulate used energy for the source

                # Update energy in the values DataFrame
                try:
                    values.loc[values.index[idx], region + '_' + source] -= used_energy
                except KeyError:
                    pass

            # Handle any remaining energy with a default emission factor
            if remaining_energy > 0:
                total_emissions += remaining_energy * aef[region][idx]
                total_cost += 48 * remaining_energy
                resource_usage['Others']["es_cha"][-1] = remaining_energy

            cef_l.append(total_emissions)  # Append total emissions for the region
            cost_l.append(total_cost)

    cef_es.append(sum(cef_l))  # Sum emissions for all regions
    cost_es.append(sum(cost_l))
    return cef_es, cost_es, resource_usage



def aef_log(idx, values, aef, df_battery_links_charge,gen_update, ratio):
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
        return aef, gen_update

    remaining_energy = df_battery_links_charge.iloc[idx] * ratio
    total_emissions = 0

    # Loop through each energy source available in the CO2_FACTORS dictionary.
    for source, co2 in CO2_FACTORS.items():
        # Calculate the energy used from the current source without exceeding the remaining energy.
        total_emissions += values[source].iloc[idx] * co2  # Accumulate total emissions based on the used energy and its CO2 factor.

    #get aef (co2/MWh) then times the charged energy in MWh
    total_emissions = (total_emissions * df_battery_links_charge.iloc[idx] * ratio)/values.iloc[idx].sum()

    # Compute the MEF for the current index if there was any energy usage, otherwise set it to zero.

    aef.append(total_emissions)

    return aef




def mef_log(idx, values, mef, df_battery_links_charge,gen_update, ratio):
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
        return mef, gen_update

    total_emissions = 0

    if values["Hard coal"].iloc[idx]>0:
        co2 = CO2_FACTORS["Hard coal"]
    else:
        if values["Oil"].iloc[idx] > 0:
            co2 = CO2_FACTORS["Oil"]
        else:
            if values["SCGT"].iloc[idx] > 0:
                co2 = CO2_FACTORS["SCGT"]
            else:
                if values["CCGT"].iloc[idx] > 0:
                    co2 = CO2_FACTORS["CCGT"]
                else:
                    co2 = 0

    total_emissions = co2 * df_battery_links_charge.iloc[idx] * ratio

    mef.append(total_emissions)

    return mef



def cycle_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, aef, gen_bus_carrier, df_storage_links, df_gen_remain,  regions, resource_usage):
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
    gen_bus_carrier_0 = gen_bus_carrier

    # Analyze battery charging and discharging cycles
    for cycle_number, times in process_times_bat.items():
        # Calculate total energy used during the cycle
        energy_bat_charge = (df_storage_links[regions + '_Battery_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (
                    -df_storage_links[regions + '_Battery_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        cef_bat_charge = []  # Initialize list for CO2 emissions during charging
        cef_bat_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_bat_charge = []
        cost_bat_discharge = []
        resource_bat = {
            source: {"bat_cha": [], "bat_dis": []} for source in resource_usage.keys()
        }


        # # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_bat_charge, cost_bat_charge, gen_bus_carrier_0, resource_bat = cef_bat_log(idx, aef, gen_bus_carrier_0, cef_bat_charge, cost_bat_charge, df_storage_links, regions, resource_bat)
        #     cef_bat_discharge, cost_bat_discharge, df_gen_remain, resource_bat = cef_bat_discharge_log(idx, df_gen_remain, gen_bus_carrier, cef_bat_discharge, cost_bat_discharge, df_storage_links, regions, resource_bat)

        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]  # 获取当前时间步对应的比例
            cef_bat_charge, cost_bat_charge, gen_bus_carrier_0, resource_bat = cef_bat_log(
                idx,
                aef,
                gen_bus_carrier_0,
                cef_bat_charge,
                cost_bat_charge,
                df_storage_links,
                regions,
                ratio,
                resource_bat
            )
            cef_bat_discharge, cost_bat_discharge, df_gen_remain, resource_bat = cef_bat_discharge_log(
                idx,
                df_gen_remain,
                gen_bus_carrier,
                cef_bat_discharge,
                cost_bat_discharge,
                df_storage_links,
                regions,
                ratio,
                resource_bat
            )

        # Calculate emissions and deltas for the cycle
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
        for source in resource_usage.keys():
            resource_usage[source]["bat_cha"].append(np.sum(resource_bat[source]["bat_cha"]))
            resource_usage[source]["bat_dis"].append(np.sum(resource_bat[source]["bat_dis"]))

    # Calculate overall CO2 emissions factor for batteries
    co2_emissions_factor_bat = np.average(co2_bat_charge_cycle)
    total_emissions_bat = np.sum(co2_delta_bat_emissions)

    # Analyze LDES charging and discharging cycles
    for cycle_number, times in process_times_es.items():
        # Calculate total energy used during the cycle
        energy_es_charge = (df_storage_links[regions + '_OtherStorage_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (
                    -df_storage_links[regions + '_OtherStorage_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        cef_es_charge = []  # Initialize list for CO2 emissions during charging
        cef_es_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_es_charge = []
        cost_es_discharge = []
        resource_es = {
            source: {"es_cha": [], "es_dis": []} for source in resource_usage.keys()
        }

        # # Calculate CEF for each time step in the cycle
        # for idx in times:
        #     cef_es_charge, cost_es_charge, resource_es = cef_es_log(idx, gen_bus_carrier_0, cef_es_charge, cost_es_charge, df_storage_links, regions, resource_es, aef)
        #     cef_es_discharge, cost_es_discharge, resource_es = cef_es_discharge_log(idx, df_gen_remain, gen_bus_carrier, cef_es_discharge, cost_es_discharge, df_storage_links, regions, resource_es)

        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]  # 获取当前时间步对应的比例
            cef_es_charge, cost_es_charge, resource_es = cef_es_log(
                idx,
                gen_bus_carrier_0,
                cef_es_charge,
                cost_es_charge,
                df_storage_links,
                regions,
                ratio,
                resource_es,
                aef
            )
            cef_es_discharge, cost_es_discharge, resource_es = cef_es_discharge_log(
                idx,
                df_gen_remain,
                gen_bus_carrier,
                cef_es_discharge,
                cost_es_discharge,
                df_storage_links,
                regions,
                ratio,
                resource_es
            )

        # Calculate emissions and deltas for the cycle
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
        for source in resource_usage.keys():
            resource_usage[source]["es_cha"].append(np.sum(resource_es[source]["es_cha"]))
            resource_usage[source]["es_dis"].append(np.sum(resource_es[source]["es_dis"]))

    # Calculate overall CO2 emissions factor for LDES
    co2_emissions_factor_es = np.average(co2_es_charge_cycle)
    total_emissions_es = np.sum(co2_delta_es_emissions)

    # Return all results as a tuple
    return (total_emissions_bat, total_emissions_es,co2_emissions_factor_bat, co2_emissions_factor_es, co2_bat_charge_cycle, co2_es_charge_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle, energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle, emissions_bat_discharged_cycle,
            emissions_es_discharged_cycle, co2_delta_bat_emissions, co2_delta_es_emissions, resource_usage)
