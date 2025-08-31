import pickle
import sys
import numpy as np
import pandas as pd
import os
import pypsa
from function.mef_national_energy_log_eu import national_cycle_analysis
from function.cyclic_data_preprocessing import cycle_extraction, modify_and_redistribute, redistribute_generation
from function.mef_energy_log_eu import cycle_analysis


def analyze_one_scenario(pre_dic: str) -> None:
    """
    Run the full post-processing pipeline for a single optimization result folder.
    Notes:
      - Most logic is unchanged; only comments/messages are English now.
      - Extra CI/CEF artifacts are saved only for the '_min_CO2_2023' (emission-minimization 2023) case.
    """
    # Output dir for this scenario
    save_dir = f"{pre_dic}analysis_output/"
    os.makedirs(save_dir, exist_ok=True)

    # Identify scenario tag for file naming
    if '_min_cost_2023' in pre_dic:
        p = 'eu_min_cost_2023'
    elif '_min_CO2_2023' in pre_dic:
        p = 'eu_min_emission_2023'
    elif '_min_cost_2030' in pre_dic:
        p = 'eu_min_cost_2030'
    elif '_min_CO2' in pre_dic:
        p = 'eu_min_emission_2030'
    else:
        p = None
    print(f"Processing for: {p}  (source: {pre_dic})")

    # Load network & precomputed CSVs exported by the optimization step
    network = pypsa.Network(pre_dic + 'network_ff_constrained_time.nc')
    df = pd.read_csv(pre_dic + 'store_e_carrier_results.csv')
    df_capacity = pd.read_csv(pre_dic + 'stores_e.csv')
    df_gen = pd.read_csv(pre_dic + 'gen_p_carrier_results.csv')
    df_gen_remain = pd.read_csv(pre_dic + 'p_by_bus_carrier.csv')
    df_gen_remain_carrier = pd.read_csv(pre_dic + 're_p_carrier_results.csv')
    df_storage_links = pd.read_csv(pre_dic + 'links_p1_results.csv')
    df_gen_bus_carrier_region = pd.read_csv(pre_dic + 'gen_by_bus_carrier.csv')
    load = pd.read_csv(pre_dic + 'demand_p.csv')
    df_gen_remain_new = df_gen_remain.copy()

    # Aggregate generator power by bus (region) across carriers
    df0 = df_gen_bus_carrier_region.copy().set_index('snapshot')
    df0.columns = df0.columns.str.split('_', expand=True)
    df_gen_bus = df0.groupby(level=0, axis=1).sum().reset_index()

    # Make CCS zero-emission (if present)
    network.generators.loc[
        network.generators["carrier"] == 'CCGT CCS',
        "co2_emissions"
    ] = 0.0

    all_carriers = df_gen.columns.to_list()
    regions = network.generators['bus'].unique().tolist()
    carriers = [c for c in network.generators['carrier'].unique() if pd.notna(c)]

    # Build emission/cost map per carrier (sorted by CO2 then cost)
    agg = network.generators.groupby('carrier').agg({
        'co2_emissions': 'first',
        'marginal_cost': 'first'
    })
    CO2_FACTORS = {
        carrier: (row.co2_emissions, row.marginal_cost)
        for carrier, row in agg.iterrows()
    }
    CO2_FACTORS = dict(
        sorted(
            CO2_FACTORS.items(),
            key=lambda item: (item[1][0], item[1][1])  # sort by (CO2, cost)
        )
    )

    # Resource merit order by scenario (either cost-first or CO2-first)
    if '_min_cost' in pre_dic:
        sources = list(CO2_FACTORS.keys())
        resources = list(sorted(sources, key=lambda s: CO2_FACTORS[s][1]))  # by cost
    elif '_min_CO2' in pre_dic:
        resources = list(CO2_FACTORS.keys())  # by CO2 (already sorted above)
    else:
        resources = list(CO2_FACTORS.keys())
        print("Warning: cannot detect scenario order; defaulting to (CO2, cost).")

    # Build column names for storage link CSVs
    battery_bus = [s + '_Battery' for s in regions]
    ES_bus = [s + '_Other_storage' for s in regions]

    battery_charger = [s + '_charger' for s in battery_bus]
    battery_discharger = [s + '_discharger' for s in battery_bus]

    ES_charger = [s + '_charger' for s in ES_bus]
    ES_discharger = [s + '_discharger' for s in ES_bus]

    # Aggregate total (system-wide) battery/LDES charge/discharge
    df_storage_links['bus_charger'] = df_storage_links[battery_charger].sum(axis=1)
    df_storage_links['bus_discharger'] = df_storage_links[battery_discharger].sum(axis=1)
    df_storage_links['es_bus_charger'] = df_storage_links[ES_charger].sum(axis=1)
    df_storage_links['es_bus_discharger'] = df_storage_links[ES_discharger].sum(axis=1)

    # Build approximate SOC from charge/discharge series (assuming 1h resolution)
    # Note: dischargers are expected negative in the CSV; division by efficiency keeps signs.
    P_c_batt = df_storage_links[battery_charger].sum(axis=1)
    P_d_batt = df_storage_links[battery_discharger].sum(axis=1) / 0.9
    P_c_es = df_storage_links[ES_charger].sum(axis=1)
    P_d_es = df_storage_links[ES_discharger].sum(axis=1) / 0.7

    # ΔE = charge + discharge (discharge is negative), then cumulative sum
    df_storage_links['delta_E_batt'] = P_c_batt + P_d_batt
    df_storage_links['delta_E_es'] = P_c_es + P_d_es

    df_storage_links['soc_batt'] = df_storage_links['delta_E_batt'].cumsum().clip(lower=0)
    df_storage_links['soc_ldes'] = df_storage_links['delta_E_es'].cumsum().clip(lower=0)

    # Normalize SOC using the system-wide maxima
    df['Battery'] = df_storage_links['soc_batt']
    df['ES'] = df_storage_links['soc_ldes']
    df['soc_batt'] = df['Battery'] / df['Battery'].max()
    df['soc_ldes'] = df['ES'] / df['ES'].max()

    # Prepare SOC series with a leading dummy hour (aligns with downstream cycle logic)
    df_copy = df.copy()
    df_copy['snapshot'] = pd.to_datetime(df['snapshot'], format="%d/%m/%Y %H:%M:%S", dayfirst=True)
    new_snapshot = df_copy['snapshot'].iloc[0] - pd.Timedelta(hours=1)
    new_row = pd.DataFrame({col: [0] if col != 'snapshot' else [new_snapshot] for col in df_copy.columns})
    df_copy = pd.concat([new_row, df_copy], ignore_index=True)

    # Extract cycles for battery and LDES
    process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy['soc_batt'])
    process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy['soc_ldes'])

    # Split total charge/discharge into per-region totals (Battery + Other_storage)
    df_storage = df_storage_links.set_index('snapshot')
    charger = pd.DataFrame(index=df_storage.index)
    discharger = pd.DataFrame(index=df_storage.index)
    for region in regions:
        bat_c = f"{region}_Battery_charger"
        ldes_c = f"{region}_Other_storage_charger"
        bat_d = f"{region}_Battery_discharger"
        ldes_d = f"{region}_Other_storage_discharger"
        charger[region] = df_storage[bat_c] + df_storage[ldes_c]
        discharger[region] = df_storage[bat_d] + df_storage[ldes_d]
    charger = charger.reset_index()
    discharger = (-discharger).reset_index()  # make discharger positive (energy out)

    # Region-level annual generation (original)
    region_gen = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_bus_carrier_region.columns:
                region_gen.loc[reg, res] = df_gen_bus_carrier_region[col].sum()

    # Compute (original) regional carbon intensity
    carbon_intensities = {}
    total_energys1 = {}
    for reg in regions:
        total_emissions = sum(region_gen.loc[reg, res] * CO2_FACTORS[res][0] for res in resources)
        total_energy = region_gen.loc[reg].sum()
        ci = total_emissions / total_energy if total_energy != 0 else 0
        carbon_intensities[reg] = ci
        total_energys1[reg] = total_energy
    ci_df = pd.DataFrame.from_dict(
        carbon_intensities,
        orient='index',
        columns=['carbon_intensity (tCO2/MWh)']
    )
    ci_df.index.name = 'Region'

    # Reallocate generation after serving load/charge/discharge and compute flows
    df_gen_bus_carrier_region_updated, flow_df, flows_by_res, df_gen_charging = redistribute_generation(
        df_gen_bus=df_gen_bus,
        load=load,
        charger=charger,
        discharger=discharger,
        df_gen_bus_carrier_region=df_gen_bus_carrier_region,
        regions=regions,
        resources=resources
    )

    # Region-level annual generation (updated)
    region_gen_updated = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_bus_carrier_region_updated.columns:
                region_gen_updated.loc[reg, res] = df_gen_bus_carrier_region_updated[col].sum()

    # Compute (updated) regional carbon intensity
    carbon_intensities = {}
    total_energys2 = {}
    for reg in regions:
        total_emissions = sum(region_gen_updated.loc[reg, res] * CO2_FACTORS[res][0] for res in resources)
        total_energy = region_gen_updated.loc[reg].sum()
        ci = total_emissions / total_energy if total_energy != 0 else 0
        carbon_intensities[reg] = ci
        total_energys2[reg] = total_energy
    ci_df2 = pd.DataFrame.from_dict(
        carbon_intensities,
        orient='index',
        columns=['carbon_intensity (tCO2/MWh)']
    )
    ci_df2.index.name = 'Region'

    # Energy totals delta
    eg_df = pd.DataFrame({
        'total_energy1': total_energys1,
        'total_energy2': total_energys2,
    })
    eg_df['energy_diff'] = eg_df['total_energy2'] - eg_df['total_energy1']
    eg_df.index.name = 'region'

    # Refine "remaining generation" with resource-aware redistribution for discharging
    df_gen_remain_new, flows_by_res_dis = modify_and_redistribute(
        df_gen_bus_carrier_region,
        df_gen_remain,
        discharger,
        regions,
        resources
    )
    df_gen_remain_new_copy = df_gen_remain_new.copy()
    df_gen_remain_new_modified_copy = df_gen_remain_new.copy()

    # Region totals for remaining generation (original remain CSV)
    region_gen_remain = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_remain.columns:
                region_gen_remain.loc[reg, res] = df_gen_remain[col].sum()

    # Initialize resource-usage ledger
    resource_usage = {
        carrier: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}
        for carrier in carriers
    }
    if "Others" not in resource_usage:
        resource_usage["Others"] = {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}

    # National cycle analysis (system-wide cycles)
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
        df_gen, df_gen_bus_carrier_region_updated, df_storage_links,
        df_gen_remain_new_modified_copy,
        resource_usage, CO2_FACTORS, resources, regions
    )

    # Persist the main national cycle outputs for this scenario
    output_path = os.path.join(save_dir, f'national_cycle_output.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump((cef_bat_t, cef_es_t, carbon_intensity_bat_cycle, carbon_intensity_es_cycle,
                     unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle,
                     unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
                     cost_bat, cost_es, emissions_bat, emissions_es,
                     cost_charged_bat, cost_charged_es, cost_discharged_bat, cost_discharged_es,
                     energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
                     emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
                     resource_usage), f)
    print(f"Saved: {output_path}")

    # Convert lists to numpy arrays for metrics
    unit_ccost_bat_cycle = np.array(unit_ccost_bat_cycle)
    unit_ccost_es_cycle = np.array(unit_ccost_es_cycle)
    unit_dcost_bat_cycle = np.array(unit_dcost_bat_cycle)
    unit_dcost_es_cycle = np.array(unit_dcost_es_cycle)
    unit_cost_bat_cycle = np.array(unit_cost_bat_cycle)
    unit_cost_es_cycle = np.array(unit_cost_es_cycle)
    co2_emissions_bat_cycle = np.array(co2_emissions_bat_cycle)
    co2_emissions_es_cycle = np.array(co2_emissions_es_cycle)
    energy_discharge_cycle_bat = np.array(energy_discharge_cycle_bat)
    energy_discharge_cycle_es = np.array(energy_discharge_cycle_es)
    energy_charge_cycle_bat = np.array(energy_charge_cycle_bat)
    energy_charge_cycle_es = np.array(energy_charge_cycle_es)
    cost_charged_bat = np.array(cost_charged_bat)
    cost_charged_es = np.array(cost_charged_es)
    cost_discharged_bat = np.array(cost_discharged_bat)
    cost_discharged_es = np.array(cost_discharged_es)
    emissions_charged_bat = np.array(emissions_charged_bat)
    emissions_charged_es = np.array(emissions_charged_es)
    emissions_discharged_bat = np.array(emissions_discharged_bat)
    emissions_discharged_es = np.array(emissions_discharged_es)
    cost_bat = np.array(cost_bat)
    cost_es = np.array(cost_es)
    emissions_bat = np.array(emissions_bat)
    emissions_es = np.array(emissions_es)

    # Scenario-level aggregate indicators
    accf_bat = cost_bat / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) else 0.0
    accf_es = cost_es / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) else 0.0
    acef_bat = emissions_bat / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) else 0.0
    acef_es = emissions_es / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) else 0.0
    achcf_bat = np.sum(cost_charged_bat) / np.sum(energy_charge_cycle_bat) if np.sum(energy_charge_cycle_bat) else 0.0
    achcf_es = np.sum(cost_charged_es) / np.sum(energy_charge_cycle_es) if np.sum(energy_charge_cycle_es) else 0.0
    adicf_bat = np.sum(cost_discharged_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) else 0.0
    adicf_es = np.sum(cost_discharged_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) else 0.0

    # Assemble raw data tables
    metrics_dict = {
        'unit_ccost_bat_cycle [£/MWh]': unit_ccost_bat_cycle,
        'unit_ccost_es_cycle [£/MWh]': unit_ccost_es_cycle,
        'unit_dcost_bat_cycle [£/MWh]': unit_dcost_bat_cycle,
        'unit_dcost_es_cycle [£/MWh]': unit_dcost_es_cycle,
        'cost_bat_cycle [£/MWh]': unit_cost_bat_cycle,
        'cost_es_cycle [£/MWh]': unit_cost_es_cycle,
        'co2_bat_cycle [tCO₂/MWh]': co2_emissions_bat_cycle,
        'co2_es_cycle [tCO₂/MWh]': co2_emissions_es_cycle,
        'energy_charged_bat_cycle [MWh]': energy_charge_cycle_bat,
        'energy_charged_es_cycle [MWh]': energy_charge_cycle_es,
        'energy_discharged_bat_cycle [MWh]': energy_discharge_cycle_bat,
        'energy_discharged_es_cycle [MWh]': energy_discharge_cycle_es,
        'cost_charged_bat [£]': cost_charged_bat,
        'cost_charged_es [£]': cost_charged_es,
        'cost_discharged_bat [£]': cost_discharged_bat,
        'cost_discharged_es [£]': cost_discharged_es,
        'emissions_charged_bat [tCO₂]': emissions_charged_bat,
        'emissions_charged_es [tCO₂]': emissions_charged_es,
        'emissions_discharged_bat [tCO₂]': emissions_discharged_bat,
        'emissions_discharged_es [tCO₂]': emissions_discharged_es,
    }
    cycle_items = {k: v for k, v in metrics_dict.items() if "_cycle" in k}
    noncycle_items = {k: v for k, v in metrics_dict.items() if "_cycle" not in k}
    energy_cycle_keys = [
        'energy_charged_bat_cycle [MWh]', 'energy_charged_es_cycle [MWh]',
        'energy_discharged_bat_cycle [MWh]', 'energy_discharged_es_cycle [MWh]'
    ]
    for key in energy_cycle_keys:
        if key in metrics_dict:
            cycle_items[key] = metrics_dict[key]
            noncycle_items[key] = metrics_dict[key]

    df_cycle = pd.concat([pd.Series(arr, name=name) for name, arr in cycle_items.items()], axis=1)
    df_noncycle = pd.concat([pd.Series(arr, name=name) for name, arr in noncycle_items.items()], axis=1)

    # Statistics: for *_cycle → Max/Min/Mean; for others → Max/Min/Sum
    def _safe_agg(x, f):
        x = pd.Series(x).dropna()
        return getattr(x, f)() if len(x) else np.nan

    stat_records = []
    for name, arr in cycle_items.items():
        stat_records.append(
            {"Metric": name, "Max": _safe_agg(arr, "max"), "Min": _safe_agg(arr, "min"),
             "Mean": _safe_agg(arr, "mean"), "Sum": np.nan}
        )
    for name, arr in noncycle_items.items():
        stat_records.append(
            {"Metric": name, "Max": _safe_agg(arr, "max"), "Min": _safe_agg(arr, "min"),
             "Mean": np.nan, "Sum": _safe_agg(arr, "sum")}
        )
    df_stats = pd.DataFrame(stat_records).set_index("Metric")

    # Quadrant summary (cost vs CO2, by cycle)
    bat_q1 = (unit_cost_bat_cycle > 0) & (co2_emissions_bat_cycle > 0)
    bat_q2 = (unit_cost_bat_cycle <= 0) & (co2_emissions_bat_cycle > 0)
    bat_q3 = (unit_cost_bat_cycle > 0) & (co2_emissions_bat_cycle <= 0)
    bat_q4 = (unit_cost_bat_cycle <= 0) & (co2_emissions_bat_cycle <= 0)
    bat_counts = [bat_q1.sum(), bat_q2.sum(), bat_q3.sum(), bat_q4.sum()]
    bat_disch = [
        np.round(np.sum(energy_discharge_cycle_bat[bat_q1]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_bat[bat_q2]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_bat[bat_q3]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_bat[bat_q4]) / 1e3),
    ]

    es_q1 = (unit_cost_es_cycle > 0) & (co2_emissions_es_cycle > 0)
    es_q2 = (unit_cost_es_cycle <= 0) & (co2_emissions_es_cycle > 0)
    es_q3 = (unit_cost_es_cycle > 0) & (co2_emissions_es_cycle <= 0)
    es_q4 = (unit_cost_es_cycle <= 0) & (co2_emissions_es_cycle <= 0)
    es_counts = [es_q1.sum(), es_q2.sum(), es_q3.sum(), es_q4.sum()]
    es_disch = [
        np.round(np.sum(energy_discharge_cycle_es[es_q1]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_es[es_q2]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_es[es_q3]) / 1e3),
        np.round(np.sum(energy_discharge_cycle_es[es_q4]) / 1e3),
    ]

    df_quad = pd.DataFrame({
        'Quadrant': ['Q1', 'Q2', 'Q3', 'Q4'],
        'Bat_Count': bat_counts,
        'Bat_Discharge': bat_disch,
        'ES_Count': es_counts,
        'ES_Discharge': es_disch
    })

    # Average indicators summary
    avg_rows = [
        ("Average Cycle Cost Factor (Battery) [£/MWh]", float(accf_bat)),
        ("Average Cycle Cost Factor (LDES) [£/MWh]", float(accf_es)),
        ("Average Cycle Emission Factor (Battery) [tCO₂/MWh]", float(acef_bat)),
        ("Average Cycle Emission Factor (LDES) [tCO₂/MWh]", float(acef_es)),
        ("Average Charging Cost Factor (Battery) [£/MWh]", float(achcf_bat)),
        ("Average Charging Cost Factor (LDES) [£/MWh]", float(achcf_es)),
        ("Average Discharging Cost Factor (Battery) [£/MWh]", float(adicf_bat)),
        ("Average Discharging Cost Factor (LDES) [£/MWh]", float(adicf_es)),
    ]
    df_avg = pd.DataFrame(avg_rows, columns=["Average Indicator", "Value"])

    # Write scenario-level Excel outputs
    output_filename = f"metrics_statistics_{p}.xlsx"
    output_path = os.path.join(save_dir, output_filename)
    with pd.ExcelWriter(output_path) as writer:
        df_cycle.to_excel(writer, sheet_name="Raw data indicators", index=False)
        df_noncycle.to_excel(writer, sheet_name="Raw Data", index=False)
        df_stats.to_excel(writer, sheet_name="Statistics")
        df_quad.to_excel(writer, sheet_name="Quadrant Summary", index=False)
        df_avg.to_excel(writer, sheet_name="Average Indicators", index=False)
    print(f"Saved: {output_path}")

    # Resource-usage breakdown for cycle attribution (national totals)
    resources_ext = resources + ["Others"]
    df_bat_cha = pd.DataFrame({res: resource_usage[res]["bat_cha"] for res in resources_ext})
    df_bat_dis = pd.DataFrame({res: resource_usage[res]["bat_dis"] for res in resources_ext})
    df_es_cha = pd.DataFrame({res: resource_usage[res]["es_cha"] for res in resources_ext})
    df_es_dis = pd.DataFrame({res: resource_usage[res]["es_dis"] for res in resources_ext})

    # Attach per-cycle unit cost (ccf) and emissions (cef)
    df_bat_cha["ccf"] = unit_cost_bat_cycle
    df_bat_cha["cef"] = co2_emissions_bat_cycle
    df_bat_dis["ccf"] = unit_cost_bat_cycle
    df_bat_dis["cef"] = co2_emissions_bat_cycle

    df_es_cha["ccf"] = unit_cost_es_cycle
    df_es_cha["cef"] = co2_emissions_es_cycle
    df_es_dis["ccf"] = unit_cost_es_cycle
    df_es_dis["cef"] = co2_emissions_es_cycle

    # Keep resource-column order strictly as `resources` (exclude Others here)
    region_gen = region_gen[resources]
    region_gen_remain = region_gen_remain[resources]

    # Append "Total" row (column-wise sums)
    region_gen.loc["Total"] = region_gen.sum(axis=0)
    region_gen_remain.loc["Total"] = region_gen_remain.sum(axis=0)

    # Save usage summary
    output_path = f"{save_dir}resource_usage_summary_{p}.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_bat_cha.to_excel(writer, sheet_name="bat_cha", index=False)
        df_bat_dis.to_excel(writer, sheet_name="bat_dis", index=False)
        df_es_cha.to_excel(writer, sheet_name="es_cha", index=False)
        df_es_dis.to_excel(writer, sheet_name="es_dis", index=False)
        region_gen.to_excel(writer, sheet_name="region_generation")
        region_gen_remain.to_excel(writer, sheet_name="region_remain")
    print(f"Saved: {output_path}")

    # The remaining regional deep-dive (per-region flows/usage & CI/CEF persistence)
    # is performed only for the emission-minimization 2023 scenario.
    if '_min_CO2_2023' not in pre_dic:
        print("Skipping per-region deep-dive and CI/CEF pickles (only for _min_CO2_2023).")
        return

    # -------- Per-region deep-dive (only for _min_CO2_2023) --------
    valid_sources = [src for src, (factor, _) in CO2_FACTORS.items() if factor > 0]

    # Decide region-specific fallback "Others" carrier:
    #   1) If any emitting tech exists, pick the one with the largest installed capacity.
    #   2) Otherwise, among non-zero installed capacity, pick the highest-cost.
    #   3) If nothing is installed, warn and leave None.
    others = {}
    gen_capacity_by_bus = network.generators.groupby(['bus', 'carrier'])['p_nom'].sum().unstack(fill_value=0)
    for region in gen_capacity_by_bus.index:
        caps = gen_capacity_by_bus.loc[region, valid_sources]
        if caps.sum() > 0:
            chosen = caps.idxmax()
        else:
            nonzero = gen_capacity_by_bus.loc[region]
            nonzero = nonzero[nonzero > 0]
            if nonzero.empty:
                chosen = None
                print('Warning: no installed capacity in region:', region)
            else:
                cost_map = {src: CO2_FACTORS[src][1] for src in nonzero.index if src in CO2_FACTORS}
                chosen = max(cost_map, key=cost_map.get)
        others[region] = chosen

    results_dict = {}
    types = ['bat_cha', 'bat_dis', 'es_cha', 'es_dis']
    flow_matrices = {t: pd.DataFrame(0.0, index=regions, columns=regions) for t in types}

    # Precompute normalized SOC for each region’s Battery/Other_storage
    battery_soc = []
    ES_soc = []
    for i in regions:
        battery_i = i + '_Battery'
        ES_i = i + '_Other_storage'
        if '_ext' in pre_dic:
            battery_i_2030 = i + '_Battery_2030'
            df_capacity[battery_i + '_soc'] = (df_capacity[battery_i_2030] + df_capacity[battery_i]) / (
                df_capacity[battery_i_2030].max() + df_capacity[battery_i].max())
            ES_i_2030 = i + '_OtherStorage_2030'
            df_capacity[ES_i + '_soc'] = (df_capacity[ES_i_2030] + df_capacity[ES_i]) / (
                df_capacity[ES_i_2030].max() + df_capacity[ES_i].max())
        else:
            df_capacity[battery_i + '_soc'] = df_capacity[battery_i] / df_capacity[battery_i].max()
            df_capacity[ES_i + '_soc'] = df_capacity[ES_i] / df_capacity[ES_i].max()
        battery_soc.append(battery_i + '_soc')
        ES_soc.append(ES_i + '_soc')

        # Prepare region-specific generation-by-carrier over time
        region_gen_p = network.generators_t.p.T.groupby(network.generators.bus).get_group(i)
        region_gen_p_carrier = region_gen_p.groupby(network.generators.carrier).sum().T
        region_gen_p_carrier['demand'] = network.loads_t.p[i]
        file_name = i + '_carrier.csv'
        region_gen_p_carrier.to_csv(pre_dic + file_name, header=True)

    # Re-run cycle extraction per region (using region SOC time series)
    charging_avg_bat_list = []
    discharging_avg_bat_list = []
    charging_avg_es_list = []
    discharging_avg_es_list = []

    # Collect regional outputs
    dict_unit_cost_bat = {}
    dict_unit_cost_es = {}
    dict_co2_emissions_bat = {}
    dict_co2_emissions_es = {}
    dict_energy_discharge_cycle_bat = {}
    dict_energy_discharge_cycle_es = {}
    dict_acef_bat = {}
    dict_accf_bat = {}
    dict_acef_es = {}
    dict_accf_es = {}

    sorted_resources = sorted(
        [c for c in resources if c != "Others"],
        key=lambda c: (CO2_FACTORS[c][0], CO2_FACTORS[c][1])
    ) + ["Others"]

    for i in regions:
        # Region CSV (per-carrier power over time)
        file_name = i + '_carrier.csv'
        df_gen_bus_carrier = pd.read_csv(pre_dic + file_name)
        df_gen_bus_carrier = df_gen_bus_carrier.reindex(columns=all_carriers, fill_value=0)
        df_gen_bus_carrier_update = df_gen_bus_carrier.copy()

        # Region-local resource usage accumulator
        resource_usage_i = {carrier: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for carrier in carriers}
        resource_usage_i["Others"] = {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}
        print(f"Region: {i}")

        # Build a region SOC series with a leading dummy hour
        df_capacity_copy = df_capacity.copy()
        df_capacity_copy['snapshot'] = pd.to_datetime(df_capacity_copy['snapshot'], dayfirst=True)
        new_snapshot = df_capacity_copy['snapshot'].iloc[0] - pd.Timedelta(hours=1)
        new_row = pd.DataFrame({col: [0] if col != 'snapshot' else [new_snapshot] for col in df_capacity_copy.columns})
        df_capacity_copy = pd.concat([new_row, df_capacity_copy], ignore_index=True)

        battery_i = i + '_Battery'
        ES_i = i + '_Other_storage'
        process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_capacity_copy[battery_i + '_soc'])
        process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_capacity_copy[ES_i + '_soc'])

        # Region-level cycle analysis (uses national flows/redistribution tensors)
        (unit_cost_bat_cycle, unit_cost_es_cycle, co2_emissions_bat_cycle, co2_emissions_es_cycle,
         energy_charge_cycle_bat, energy_charge_cycle_es, energy_discharge_cycle_bat, energy_discharge_cycle_es,
         emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
         co2_delta_emissions, co2_delta_emissions_es, cost_delta_bat, cost_delta_es,
         resource_usage_i) = cycle_analysis(
            process_times_bat, process_ratios_bat, process_times_es, process_ratios_es,
            others, df_gen_bus_carrier_region_updated, df_storage_links, df_gen_remain_new,
            i, regions, CO2_FACTORS, resource_usage_i, flows_by_res, flow_matrices, flows_by_res_dis, resources
        )

        # Assemble per-region dataframes for export
        df_bat_cha = pd.DataFrame({res: resource_usage_i[res]["bat_cha"] for res in resources})
        df_bat_dis = pd.DataFrame({res: resource_usage_i[res]["bat_dis"] for res in resources})
        df_es_cha = pd.DataFrame({res: resource_usage_i[res]["es_cha"] for res in resources})
        df_es_dis = pd.DataFrame({res: resource_usage_i[res]["es_dis"] for res in resources})

        df_bat_extra = pd.DataFrame({
            "unit_cost_bat": unit_cost_bat_cycle,
            "co2_emissions_bat": co2_emissions_bat_cycle
        })
        df_es_extra = pd.DataFrame({
            "unit_cost_es": unit_cost_es_cycle,
            "co2_emissions_es": co2_emissions_es_cycle
        })

        df_bat_cha = pd.concat([df_bat_cha, df_bat_extra], axis=1)
        df_bat_dis = pd.concat([df_bat_dis, df_bat_extra], axis=1)
        df_es_cha = pd.concat([df_es_cha, df_es_extra], axis=1)
        df_es_dis = pd.concat([df_es_dis, df_es_extra], axis=1)

        # Keep region sheets for a single xlsx
        results_dict.setdefault(i, {})
        results_dict[i]["df_bat_cha"] = df_bat_cha
        results_dict[i]["df_bat_dis"] = df_bat_dis
        results_dict[i]["df_es_cha"] = df_es_cha
        results_dict[i]["df_es_dis"] = df_es_dis

        # Convert to arrays for regional summaries
        unit_cost_bat_cycle = np.array(unit_cost_bat_cycle)
        unit_cost_es_cycle = np.array(unit_cost_es_cycle)
        co2_emissions_bat_cycle = np.array(co2_emissions_bat_cycle)
        co2_emissions_es_cycle = np.array(co2_emissions_es_cycle)
        energy_discharge_cycle_bat = np.array(energy_discharge_cycle_bat)
        energy_discharge_cycle_es = np.array(energy_discharge_cycle_es)
        emissions_charged_bat = np.array(emissions_charged_bat)
        emissions_charged_es = np.array(emissions_charged_es)
        emissions_discharged_bat = np.array(emissions_discharged_bat)
        emissions_discharged_es = np.array(emissions_discharged_es)
        co2_delta_emissions_bat = np.array(co2_delta_emissions)
        co2_delta_emissions_es = np.array(co2_delta_emissions_es)
        cost_delta_bat = np.array(cost_delta_bat)
        cost_delta_es = np.array(cost_delta_es)

        # Store arrays by region
        dict_unit_cost_bat[i] = unit_cost_bat_cycle
        dict_unit_cost_es[i] = unit_cost_es_cycle
        dict_co2_emissions_bat[i] = co2_emissions_bat_cycle
        dict_co2_emissions_es[i] = co2_emissions_es_cycle
        dict_energy_discharge_cycle_bat[i] = energy_discharge_cycle_bat
        dict_energy_discharge_cycle_es[i] = energy_discharge_cycle_es

        # Mean charging/discharging intensities for lines
        avg_charging_bat = np.sum(emissions_charged_bat) / np.sum(energy_charge_cycle_bat) if np.sum(energy_charge_cycle_bat) else 0.0
        avg_discharging_bat = np.sum(emissions_discharged_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) else 0.0
        charging_avg_bat_list.append(avg_charging_bat)
        discharging_avg_bat_list.append(avg_discharging_bat)

        avg_charging_es = np.sum(emissions_charged_es) / np.sum(energy_charge_cycle_es) if np.sum(energy_charge_cycle_es) else 0.0
        avg_discharging_es = np.sum(emissions_discharged_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) else 0.0
        charging_avg_es_list.append(avg_charging_es)
        discharging_avg_es_list.append(avg_discharging_es)

        # Region-level cycle-average CEF/CCF
        cef_bat = np.sum(co2_delta_emissions_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) > 0 else 0.0
        cef_es = np.sum(co2_delta_emissions_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) > 0 else 0.0
        ccf_bat = np.sum(cost_delta_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) > 0 else 0.0
        ccf_es = np.sum(cost_delta_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) > 0 else 0.0

        dict_acef_bat[i] = cef_bat
        dict_accf_bat[i] = ccf_bat
        dict_acef_es[i] = cef_es
        dict_accf_es[i] = ccf_es

    # Export all per-region sheets to one Excel
    output_path = f"{save_dir}regional_resources_usage_{p}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for outer_key, inner_dict in results_dict.items():
            for inner_key, df_reg_sheet in inner_dict.items():
                sheet_name = f"{outer_key}_{inner_key}"[:31]  # Excel sheet name limit
                df_reg_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Saved: {output_path}")

    # Pack select dicts
    all_data = {
        'unit_cost_bat': dict_unit_cost_bat,
        'unit_cost_es': dict_unit_cost_es,
        'co2_bat': dict_co2_emissions_bat,
        'co2_es': dict_co2_emissions_es,
        'energy_discharge_bat': dict_energy_discharge_cycle_bat,
        'energy_discharge_es': dict_energy_discharge_cycle_es,
        'acef_bat': dict_acef_bat,
        'acef_es': dict_acef_es,
    }

    # Save CI/CEF artifacts only for the 2023 CO2-min case
    ci_save_dir = './results/CI_CEF_data/'
    os.makedirs(ci_save_dir, exist_ok=True)
    ci_df.to_pickle(f"{ci_save_dir}eu_ci_df.pkl")
    ci_df2.to_pickle(f"{ci_save_dir}eu_ci_df2.pkl")
    eg_df.to_pickle(f"{ci_save_dir}eu_energy_difference.pkl")
    with open(f"{ci_save_dir}eu_results.pkl", 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Write the flow matrices to an Excel file
    out_xlsx = os.path.join(save_dir, f"flow_matrices_{p}.xlsx")
    with pd.ExcelWriter(out_xlsx) as writer:
        # Sheet 1: annual inter-regional generation flow matrix
        flow_df.to_excel(
            writer,
            sheet_name="generation",
            index_label="generation"  # label for the index column
        )
        # Sheets 2–5: per-process flow matrices (e.g., bat_cha, bat_dis, es_cha, es_dis)
        for sheet_name, df in flow_matrices.items():
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index_label=sheet_name
            )
    print(f"Saved: {out_xlsx}")
    # Compute region-average CEFs from updated generation mix
    avg_cef_list = []
    for reg in regions:
        cols = [f"{reg}_{res}" for res in resources if f"{reg}_{res}" in df_gen_bus_carrier_region_updated.columns]
        if not cols:
            avg_cef_list.append(0.0)
            continue
        factors = [CO2_FACTORS[res][0] for res in resources if f"{reg}_{res}" in df_gen_bus_carrier_region_updated.columns]
        gen_df = df_gen_bus_carrier_region_updated[cols]
        numerator = (gen_df.values * factors).sum()
        denominator = gen_df.values.sum()
        avg_cef_list.append(numerator / denominator if denominator > 0 else 0.0)
    avg_cef_list = [round(x, 2) for x in avg_cef_list]

    data = {
        'avg_cef_list': avg_cef_list,
        'charging_avg_bat_list': charging_avg_bat_list,
        'discharging_avg_bat_list': discharging_avg_bat_list,
        'charging_avg_es_list': charging_avg_es_list,
        'discharging_avg_es_list': discharging_avg_es_list,
    }
    with open(f"{save_dir}saved_lists_{p}.pkl", 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved regional CI/CEF pickles for scenario: {p}")


def main():
    """
    Run for specified scenarios or all four by default.
    Usage:
        python analysis_main.py
        python analysis_main.py all
        python analysis_main.py eu_min_cost_2023 eu_min_emission_2030
    """
    import os
    RESULTS_ROOT = "./results/"
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    scenarios_map = {
        "eu_min_cost_2023":     os.path.join(RESULTS_ROOT, "eu_min_cost_2023/"),
        "eu_min_emission_2023": os.path.join(RESULTS_ROOT, "eu_min_CO2_2023/"),
        "eu_min_cost_2030":     os.path.join(RESULTS_ROOT, "eu_min_cost_2030/"),
        "eu_min_emission_2030": os.path.join(RESULTS_ROOT, "eu_min_CO2_2030/"),
    }

    # Parse CLI
    args = sys.argv[1:]
    if not args or (len(args) == 1 and args[0].lower() == "all"):
        to_run = list(scenarios_map.keys())
    else:
        to_run = [a for a in args if a in scenarios_map]
        if not to_run:
            print("No valid scenario keys provided. Valid options:")
            for k in scenarios_map:
                print(" -", k)
            return

    for key in to_run:
        pre_dic = scenarios_map[key]
        try:
            analyze_one_scenario(pre_dic)
        except Exception as e:
            print(f"[{key}] analysis failed: {e}", file=sys.stderr)


# if __name__ == "__main__":
#     main()


for path in [
    "./results/eu_min_cost_2023/",
    "./results/eu_min_CO2_2023/",
    "./results/eu_min_cost_2030/",
    "./results/eu_min_CO2_2030/",
]:
    print(f"Running: {path}")
    analyze_one_scenario(path)