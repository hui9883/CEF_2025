import sys
import os
import random
import pickle
import numpy as np
import pandas as pd
import pypsa
from function.mef_energy_log_uk import cycle_analysis
from function.mef_national_energy_log_uk import (
    national_cycle_analysis,
    national_mef_analysis,
    national_aef_analysis,
)
from function.cyclic_data_preprocessing import (
    cycle_extraction,
    redistribute_generation,
    modify_and_redistribute,
)

# =========================
# Core per-scenario runner
# =========================
def analyze_one_scenario(pre_dic: str) -> None:
    """
    Run the full UK analysis pipeline for a given scenario folder (pre_dic).

    Notes:
    - Keeps existing logic intact; only comments and I/O plumbing were adjusted.
    - Some summary artifacts are saved only for *_min_CO2_2023* (as requested).
    """

    # ------------------------------
    # Output folder under scenario
    # ------------------------------
    save_dir = f"{pre_dic}analysis_output/"
    os.makedirs(save_dir, exist_ok=True)

    # Scenario key suffix (used in filenames)
    if "min_cost_2023" in pre_dic:
        p = "min_cost_2023"
    elif "min_CO2_2023" in pre_dic:
        p = "min_emission_2023"
    elif "min_cost_2030" in pre_dic:
        p = "min_cost_2030"
    elif "min_CO2_2030" in pre_dic:
        p = "min_emission_2030"
    else:
        p = None
    print(f"Processing for: {p}")

    # ------------------------------
    # Load scenario data
    # ------------------------------
    network = pypsa.Network(pre_dic + "network_ff_constrained_time.nc")
    df = pd.read_csv(pre_dic + "store_e_carrier_results.csv")
    df_capacity = pd.read_csv(pre_dic + "stores_e.csv")
    df_gen = pd.read_csv(pre_dic + "gen_p_carrier_results.csv")
    df_gen_remain = pd.read_csv(pre_dic + "p_by_bus_carrier.csv")
    df_gen_remain_carrier = pd.read_csv(pre_dic + "re_p_carrier_results.csv")
    df_storage_links = pd.read_csv(pre_dic + "links_p1_results.csv")
    df_gen_bus_carrier_region = pd.read_csv(pre_dic + "gen_by_bus_carrier.csv")
    load = pd.read_csv(pre_dic + "demand_p.csv")

    # Clip negatives if any (robustness, preserves math)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].clip(lower=0)
    num_cols = df_capacity.select_dtypes(include=["number"]).columns
    df_capacity[num_cols] = df_capacity[num_cols].clip(lower=0)

    print(network.lines.columns.to_list())

    all_carriers = df_gen.columns.to_list()
    regions = network.generators["bus"].unique().tolist()
    carriers = [c for c in network.generators["carrier"].unique() if pd.notna(c)]

    # Resource usage accumulators
    resource_usage = {
        carrier: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for carrier in carriers
    }
    if "Others" not in resource_usage:
        resource_usage["Others"] = {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}

    # Normalize SOC series used for cycle extraction
    battery_bus = [s + "_Battery" for s in regions]
    ES_bus = [s + "_OtherStorage" for s in regions]
    df["soc_batt"] = df["Battery"] / df["Battery"].max()
    df["soc_ldes"] = df["ES"] / df["ES"].max()

    # Build (charger / discharger) column names per region
    battery_charger = [s + "_charger" for s in battery_bus]
    battery_discharger = [s + "_discharger" for s in battery_bus]
    ES_charger = [s + "_charger" for s in ES_bus]
    ES_discharger = [s + "_discharger" for s in ES_bus]

    # Positive = discharge, Negative = charge (sign as in original)
    df_storage_links[battery_charger] = -df_storage_links[battery_charger]
    df_storage_links[ES_charger] = -df_storage_links[ES_charger]

    # Aggregate per-bus charger/discharger to per-region frames
    charger = (
        df_storage_links.filter(regex="_charger$").T.groupby(lambda col: col.rsplit("_", 2)[0]).sum().T
    )
    discharger = (
        df_storage_links.filter(regex="_discharger$").T.groupby(lambda col: col.rsplit("_", 2)[0]).sum().T
    )
    charger = pd.concat([df_storage_links[["snapshot"]], charger], axis=1)
    discharger = pd.concat([df_storage_links[["snapshot"]], -discharger], axis=1)

    # Convenience totals
    df_storage_links["bus_charger"] = df_storage_links[battery_charger].sum(axis=1)
    df_storage_links["bus_discharger"] = df_storage_links[battery_discharger].sum(axis=1)
    df_storage_links["es_bus_charger"] = df_storage_links[ES_charger].sum(axis=1)
    df_storage_links["es_bus_discharger"] = df_storage_links[ES_discharger].sum(axis=1)

    # Region-level generation time series
    df_gen_bus = network.generators_t.p.T.groupby(network.generators.bus).sum().T

    # CO2 and cost factors per carrier
    agg = network.generators.groupby("carrier").agg(
        {"co2_emissions": "first", "marginal_cost": "first"}
    )
    CO2_FACTORS = {carrier: (row.co2_emissions, row.marginal_cost) for carrier, row in agg.iterrows()}
    CO2_FACTORS = dict(sorted(CO2_FACTORS.items(), key=lambda item: (item[1][0], item[1][1])))

    # Resource order:
    # - If "min_cost" scenario: sort by marginal cost
    # - If "min_CO2" scenario: sort by CO2 (already sorted above)
    if "min_cost" in pre_dic:
        sources = list(CO2_FACTORS.keys())
        resources = list(sorted(sources, key=lambda s: CO2_FACTORS[s][1]))
    elif "min_CO2" in pre_dic:
        resources = list(CO2_FACTORS.keys())
    else:
        resources = list(CO2_FACTORS.keys())
        print("cannot find the order")

    # ------------------------------
    # Region-year generation totals & carbon intensities (before re-distribution)
    # ------------------------------
    region_gen = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_bus_carrier_region.columns:
                region_gen.loc[reg, res] = df_gen_bus_carrier_region[col].sum()

    carbon_intensities = {}
    total_energys1 = {}
    for reg in regions:
        total_emissions = sum(region_gen.loc[reg, res] * CO2_FACTORS[res][0] for res in resources)
        total_energy = region_gen.loc[reg].sum()
        ci = total_emissions / total_energy if total_energy != 0 else 0
        carbon_intensities[reg] = ci
        total_energys1[reg] = total_energy

    ci_df = pd.DataFrame.from_dict(carbon_intensities, orient="index", columns=["carbon_intensity (tCO2/MWh)"])
    ci_df.index.name = "Region"

    # ------------------------------
    # Generation redistribution (surplus → deficits) and flows
    # ------------------------------
    df_gen_bus_carrier_region_updated, flow_df, flows_by_res, df_gen_charging = redistribute_generation(
        df_gen_bus=df_gen_bus,
        load=load,
        charger=charger,
        discharger=discharger,
        df_gen_bus_carrier_region=df_gen_bus_carrier_region,
        regions=regions,
        resources=resources,
    )

    # Post-redistribution CI
    region_gen_updated = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_bus_carrier_region_updated.columns:
                region_gen_updated.loc[reg, res] = df_gen_bus_carrier_region_updated[col].sum()

    carbon_intensities = {}
    total_energys2 = {}
    for reg in regions:
        total_emissions = sum(region_gen_updated.loc[reg, res] * CO2_FACTORS[res][0] for res in resources)
        total_energy = region_gen_updated.loc[reg].sum()
        ci = total_emissions / total_energy if total_energy != 0 else 0
        carbon_intensities[reg] = ci
        total_energys2[reg] = total_energy

    ci_df2 = pd.DataFrame.from_dict(carbon_intensities, orient="index", columns=["carbon_intensity (tCO2/MWh)"])
    ci_df2.index.name = "Region"

    eg_df = pd.DataFrame({"total_energy1": total_energys1, "total_energy2": total_energys2})
    eg_df["energy_diff"] = eg_df["total_energy2"] - eg_df["total_energy1"]
    eg_df.index.name = "region"

    # ------------------------------
    # Remaining generation after honoring demand/charging
    # ------------------------------
    df_gen_remain_new, flows_by_res_dis = modify_and_redistribute(
        df_gen_bus_carrier_region, df_gen_remain, discharger, regions, resources
    )
    df_gen_remain_new_modified_copy = df_gen_remain_new.copy()

    region_gen_remain = pd.DataFrame(0.0, index=regions, columns=resources)
    for reg in regions:
        for res in resources:
            col = f"{reg}_{res}"
            if col in df_gen_remain.columns:
                region_gen_remain.loc[reg, res] = df_gen_remain[col].sum()

    # ------------------------------
    # Extract cycles (SOC-based) for battery & LDES
    # ------------------------------
    df_copy = df.copy()
    df_copy["snapshot"] = pd.to_datetime(df["snapshot"])
    new_snapshot = df_copy["snapshot"].iloc[0] - pd.Timedelta(hours=1)
    new_row = pd.DataFrame({col: [0] if col != "snapshot" else [new_snapshot] for col in df_copy.columns})
    df_copy = pd.concat([new_row, df_copy], ignore_index=True)
    process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy["soc_batt"])
    process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy["soc_ldes"])

    # ------------------------------
    # National cycle analysis (CEF-based)
    # ------------------------------
    (
        cef_bat_t,
        cef_es_t,
        carbon_intensity_bat_cycle,
        carbon_intensity_es_cycle,
        unit_ccost_bat_cycle,
        unit_ccost_es_cycle,
        unit_dcost_bat_cycle,
        unit_dcost_es_cycle,
        unit_cost_bat_cycle,
        unit_cost_es_cycle,
        co2_emissions_bat_cycle,
        co2_emissions_es_cycle,
        cost_bat,
        cost_es,
        emissions_bat,
        emissions_es,
        cost_charged_bat,
        cost_charged_es,
        cost_discharged_bat,
        cost_discharged_es,
        energy_charge_cycle_bat,
        energy_charge_cycle_es,
        energy_discharge_cycle_bat,
        energy_discharge_cycle_es,
        emissions_charged_bat,
        emissions_charged_es,
        emissions_discharged_bat,
        emissions_discharged_es,
        resource_usage,
    ) = national_cycle_analysis(
        process_times_bat,
        process_ratios_bat,
        process_times_es,
        process_ratios_es,
        df_gen,
        df_gen_bus_carrier_region_updated,
        df_storage_links,
        df_gen_remain_new_modified_copy,
        resource_usage,
        CO2_FACTORS,
        resources,
        regions,
    )

    # Save CEF bundle
    with open(os.path.join(save_dir, "national_cycle_output.pkl"), "wb") as f:
        pickle.dump(
            (
                cef_bat_t,
                cef_es_t,
                carbon_intensity_bat_cycle,
                carbon_intensity_es_cycle,
                unit_ccost_bat_cycle,
                unit_ccost_es_cycle,
                unit_dcost_bat_cycle,
                unit_dcost_es_cycle,
                unit_cost_bat_cycle,
                unit_cost_es_cycle,
                co2_emissions_bat_cycle,
                co2_emissions_es_cycle,
                cost_bat,
                cost_es,
                emissions_bat,
                emissions_es,
                cost_charged_bat,
                cost_charged_es,
                cost_discharged_bat,
                cost_discharged_es,
                energy_charge_cycle_bat,
                energy_charge_cycle_es,
                energy_discharge_cycle_bat,
                energy_discharge_cycle_es,
                emissions_charged_bat,
                emissions_charged_es,
                emissions_discharged_bat,
                emissions_discharged_es,
                resource_usage,
            ),
            f,
        )

    # ------------------------------
    # National MEF & AEF analyses (added: save both as pkl)
    # ------------------------------
    (
        mef_bat_t,
        mef_es_t,
        carbon_intensity_bat_cycle_mef,
        carbon_intensity_es_cycle_mef,
        emissions_bat_mef,
        emissions_es_mef,
        co2_emissions_factor_bat_mef,
        co2_emissions_factor_es_mef,
        co2_emissions_bat_cycle_mef,
        co2_emissions_es_cycle_mef,
        energy_charge_cycle_bat_mef,
        energy_charge_cycle_es_mef,
        energy_discharge_cycle_bat_mef,
        energy_discharge_cycle_es_mef,
        emissions_charged_bat_mef,
        emissions_charged_es_mef,
        emissions_discharged_bat_mef,
        emissions_discharged_es_mef,
        co2_delta_emissions_mef,
        co2_delta_emissions_es_mef,
    ) = national_mef_analysis(
        process_times_bat,
        process_ratios_bat,
        process_times_es,
        process_ratios_es,
        df_gen,
        df_storage_links,
        resources,
        CO2_FACTORS,
    )

    with open(os.path.join(save_dir, "national_mef_output.pkl"), "wb") as f:
        pickle.dump(
            (
                mef_bat_t,
                mef_es_t,
                carbon_intensity_bat_cycle_mef,
                carbon_intensity_es_cycle_mef,
                emissions_bat_mef,
                emissions_es_mef,
                co2_emissions_factor_bat_mef,
                co2_emissions_factor_es_mef,
                co2_emissions_bat_cycle_mef,
                co2_emissions_es_cycle_mef,
                energy_charge_cycle_bat_mef,
                energy_charge_cycle_es_mef,
                energy_discharge_cycle_bat_mef,
                energy_discharge_cycle_es_mef,
                emissions_charged_bat_mef,
                emissions_charged_es_mef,
                emissions_discharged_bat_mef,
                emissions_discharged_es_mef,
                co2_delta_emissions_mef,
                co2_delta_emissions_es_mef,
            ),
            f,
        )

    (
        aef_bat_t,
        aef_es_t,
        carbon_intensity_bat_cycle_aef,
        carbon_intensity_es_cycle_aef,
        emissions_bat_aef,
        emissions_es_aef,
        co2_emissions_factor_bat_aef,
        co2_emissions_factor_es_aef,
        co2_emissions_bat_cycle_aef,
        co2_emissions_es_cycle_aef,
        energy_charge_cycle_bat_aef,
        energy_charge_cycle_es_aef,
        energy_discharge_cycle_bat_aef,
        energy_discharge_cycle_es_aef,
        emissions_charged_bat_aef,
        emissions_charged_es_aef,
        emissions_discharged_bat_aef,
        emissions_discharged_es_aef,
        co2_delta_emissions_aef,
        co2_delta_emissions_es_aef,
    ) = national_aef_analysis(
        process_times_bat,
        process_ratios_bat,
        process_times_es,
        process_ratios_es,
        df_gen,
        df_storage_links,
        resources,
        CO2_FACTORS,
    )

    with open(os.path.join(save_dir, "national_aef_output.pkl"), "wb") as f:
        pickle.dump(
            (
                aef_bat_t,
                aef_es_t,
                carbon_intensity_bat_cycle_aef,
                carbon_intensity_es_cycle_aef,
                emissions_bat_aef,
                emissions_es_aef,
                co2_emissions_factor_bat_aef,
                co2_emissions_factor_es_aef,
                co2_emissions_bat_cycle_aef,
                co2_emissions_es_cycle_aef,
                energy_charge_cycle_bat_aef,
                energy_charge_cycle_es_aef,
                energy_discharge_cycle_bat_aef,
                energy_discharge_cycle_es_aef,
                emissions_charged_bat_aef,
                emissions_charged_es_aef,
                emissions_discharged_bat_aef,
                emissions_discharged_es_aef,
                co2_delta_emissions_aef,
                co2_delta_emissions_es_aef,
            ),
            f,
        )

    # ------------------------------
    # Averages
    # ------------------------------
    mean_bat = emissions_bat / np.sum(energy_discharge_cycle_bat)
    mean_es = emissions_es / np.sum(energy_discharge_cycle_es)
    mean_bat_mef = co2_emissions_factor_bat_mef
    mean_es_mef = co2_emissions_factor_es_mef
    mean_bat_aef = co2_emissions_factor_bat_aef
    mean_es_aef = co2_emissions_factor_es_aef

    # ------------------------------
    # Convert lists to numpy for statistics & Excel
    # ------------------------------
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

    # ------------------------------
    # Indicators
    # ------------------------------
    accf_bat = cost_bat / np.sum(energy_discharge_cycle_bat)
    accf_es = cost_es / np.sum(energy_discharge_cycle_es)
    acef_bat = emissions_bat / np.sum(energy_discharge_cycle_bat)
    acef_es = emissions_es / np.sum(energy_discharge_cycle_es)
    achcf_bat = np.sum(cost_charged_bat) / np.sum(energy_charge_cycle_bat)
    achcf_es = np.sum(cost_charged_es) / np.sum(energy_charge_cycle_es)
    adicf_bat = np.sum(cost_discharged_bat) / np.sum(energy_discharge_cycle_bat)
    adicf_es = np.sum(cost_discharged_es) / np.sum(energy_discharge_cycle_es)

    # ------------------------------
    # Assemble raw metrics dict
    # ------------------------------
    metrics_dict = {
        "unit_ccost_bat_cycle [£/MWh]": unit_ccost_bat_cycle,
        "unit_ccost_es_cycle [£/MWh]": unit_ccost_es_cycle,
        "unit_dcost_bat_cycle [£/MWh]": unit_dcost_bat_cycle,
        "unit_dcost_es_cycle [£/MWh]": unit_dcost_es_cycle,
        "cost_bat_cycle [£/MWh]": unit_cost_bat_cycle,
        "cost_es_cycle [£/MWh]": unit_cost_es_cycle,
        "co2_bat_cycle [tCO₂/MWh]": co2_emissions_bat_cycle,
        "co2_es_cycle [tCO₂/MWh]": co2_emissions_es_cycle,
        "energy_charged_bat_cycle [MWh]": energy_charge_cycle_bat,
        "energy_charged_es_cycle [MWh]": energy_charge_cycle_es,
        "energy_discharged_bat_cycle [MWh]": energy_discharge_cycle_bat,
        "energy_discharged_es_cycle [MWh]": energy_discharge_cycle_es,
        "cost_charged_bat [£]": cost_charged_bat,
        "cost_charged_es [£]": cost_charged_es,
        "cost_discharged_bat [£]": cost_discharged_bat,
        "cost_discharged_es [£]": cost_discharged_es,
        "emissions_charged_bat [tCO₂]": emissions_charged_bat,
        "emissions_charged_es [tCO₂]": emissions_charged_es,
        "emissions_discharged_bat [tCO₂]": emissions_discharged_bat,
        "emissions_discharged_es [tCO₂]": emissions_discharged_es,
    }

    # Raw data tables (cycle vs non-cycle)
    cycle_items = {k: v for k, v in metrics_dict.items() if "_cycle" in k}
    noncycle_items = {k: v for k, v in metrics_dict.items() if "_cycle" not in k}
    energy_cycle_keys = [
        "energy_charged_bat_cycle [MWh]",
        "energy_charged_es_cycle [MWh]",
        "energy_discharged_bat_cycle [MWh]",
        "energy_discharged_es_cycle [MWh]",
    ]
    for key in energy_cycle_keys:
        if key in metrics_dict:
            cycle_items[key] = metrics_dict[key]
            noncycle_items[key] = metrics_dict[key]

    df_cycle = pd.concat([pd.Series(arr, name=name) for name, arr in cycle_items.items()], axis=1)
    df_noncycle = pd.concat([pd.Series(arr, name=name) for name, arr in noncycle_items.items()], axis=1)

    def _safe_agg(x, f):
        x = pd.Series(x).dropna()
        return getattr(x, f)() if len(x) else np.nan

    # Statistics:
    #   with "_cycle" → Max/Min/Mean
    #   without       → Max/Min/Sum
    stat_records = []
    for name, arr in cycle_items.items():
        stat_records.append(
            {
                "Metric": name,
                "Max": _safe_agg(arr, "max"),
                "Min": _safe_agg(arr, "min"),
                "Mean": _safe_agg(arr, "mean"),
                "Sum": np.nan,
            }
        )
    for name, arr in noncycle_items.items():
        stat_records.append(
            {"Metric": name, "Max": _safe_agg(arr, "max"), "Min": _safe_agg(arr, "min"), "Mean": np.nan, "Sum": _safe_agg(arr, "sum")}
        )
    df_stats = pd.DataFrame(stat_records).set_index("Metric")

    # Quadrants summary (unit cost vs CO2)
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

    df_quad = pd.DataFrame(
        {
            "Quadrant": ["Q1", "Q2", "Q3", "Q4"],
            "Bat_Count": bat_counts,
            "Bat_Discharge": bat_disch,
            "ES_Count": es_counts,
            "ES_Discharge": es_disch,
        }
    )

    # Average indicators (labels + values)
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

    # Write Excel metrics
    output_filename = f"metrics_statistics_{p}.xlsx"
    output_path_xlsx = os.path.join(save_dir, output_filename)
    with pd.ExcelWriter(output_path_xlsx) as writer:
        df_cycle.to_excel(writer, sheet_name="Raw data indicators", index=False)
        df_noncycle.to_excel(writer, sheet_name="Raw Data", index=False)
        df_stats.to_excel(writer, sheet_name="Statistics")
        df_quad.to_excel(writer, sheet_name="Quadrant Summary", index=False)
        df_avg.to_excel(writer, sheet_name="Average Indicators", index=False)
    print(f"Saved file: {output_path_xlsx}")

    # ------------------------------
    # Resource usage tables
    # ------------------------------
    resources_ext = resources + ["Others"]
    df_bat_cha = pd.DataFrame({res: resource_usage[res]["bat_cha"] for res in resources_ext})
    df_bat_dis = pd.DataFrame({res: resource_usage[res]["bat_dis"] for res in resources_ext})
    df_es_cha = pd.DataFrame({res: resource_usage[res]["es_cha"] for res in resources_ext})
    df_es_dis = pd.DataFrame({res: resource_usage[res]["es_dis"] for res in resources_ext})

    # Append per-cycle unit cost (ccf) and emission factor (cef)
    df_bat_cha["ccf"] = unit_cost_bat_cycle
    df_bat_cha["cef"] = co2_emissions_bat_cycle
    df_bat_dis["ccf"] = unit_cost_bat_cycle
    df_bat_dis["cef"] = co2_emissions_bat_cycle
    df_es_cha["ccf"] = unit_cost_es_cycle
    df_es_cha["cef"] = co2_emissions_es_cycle
    df_es_dis["ccf"] = unit_cost_es_cycle
    df_es_dis["cef"] = co2_emissions_es_cycle

    # Align region generation tables to resources order (excludes Others)
    region_gen = region_gen[resources]
    region_gen_remain = region_gen_remain[resources]
    region_gen.loc["Total"] = region_gen.sum(axis=0)
    region_gen_remain.loc["Total"] = region_gen_remain.sum(axis=0)

    out_xl = f"{save_dir}resource_usage_summary_{p}.xlsx"
    with pd.ExcelWriter(out_xl, engine="openpyxl") as writer:
        df_bat_cha.to_excel(writer, sheet_name="bat_cha", index=False)
        df_bat_dis.to_excel(writer, sheet_name="bat_dis", index=False)
        df_es_cha.to_excel(writer, sheet_name="es_cha", index=False)
        df_es_dis.to_excel(writer, sheet_name="es_dis", index=False)
        region_gen.to_excel(writer, sheet_name="region_generation")
        region_gen_remain.to_excel(writer, sheet_name="region_remain")
        return
    # If not CO2_2023, bail out early (kept original behavior)
    if "min_CO2_2023" not in pre_dic:
        print("Skipping per-region deep-dive and CI/CEF pickles (only for min_CO2_2023).")
        return

    # ------------------------------
    # Per-region breakdown & visuals prep (kept logic; only comments updated)
    # ------------------------------
    sorted_resources = resources + ["Others"]
    bat_cha_all = {res: [] for res in sorted_resources}
    bat_dis_all = {res: [] for res in sorted_resources}
    es_cha_all = {res: [] for res in sorted_resources}
    es_dis_all = {res: [] for res in sorted_resources}

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

    charging_avg_bat_list = []
    discharging_avg_bat_list = []
    charging_avg_es_list = []
    discharging_avg_es_list = []

    # Export per-region generator/carrier time series
    for region_i in regions:
        region_gen_p = network.generators_t.p.T.groupby(network.generators.bus).get_group(region_i)
        region_gen_p_carrier = region_gen_p.groupby(network.generators.carrier).sum().T
        region_gen_p_carrier["demand"] = network.loads_t.p[region_i]
        file_name = region_i + "_carrier.csv"
        region_gen_p_carrier.to_csv(pre_dic + file_name, header=True)

    battery_soc = []
    ES_soc = []

    # Choose fallback ("Others") per region
    valid_sources = [src for src, (factor, _) in CO2_FACTORS.items() if factor > 0]
    others = {}
    gen_capacity_by_bus = (
        network.generators.groupby(["bus", "carrier"])["p_nom"].sum().unstack(fill_value=0)
    )
    for region in gen_capacity_by_bus.index:
        caps = gen_capacity_by_bus.loc[region, valid_sources]
        if caps.sum() > 0:
            chosen = caps.idxmax()
        else:
            nonzero = gen_capacity_by_bus.loc[region]
            nonzero = nonzero[nonzero > 0]
            if nonzero.empty:
                chosen = None
                print("Warning: No installed capacity in region:", region)
            else:
                cost_map = {src: CO2_FACTORS[src][1] for src in nonzero.index if src in CO2_FACTORS}
                chosen = max(cost_map, key=cost_map.get)
        others[region] = chosen

    results_dict = {}
    types = ["bat_cha", "bat_dis", "es_cha", "es_dis"]
    flow_matrices = {t: pd.DataFrame(0.0, index=regions, columns=regions) for t in types}

    for i in regions:
        battery_i = i + "_Battery"
        ES_i = i + "_OtherStorage"

        # Build normalized SOC for this region (kept exactly)
        df_capacity[battery_i + "_soc"] = df_capacity[battery_i] / df_capacity[battery_i].max()
        df_capacity[ES_i + "_soc"] = df_capacity[ES_i] / df_capacity[ES_i].max()

        battery_soc.append(battery_i + "_soc")
        ES_soc.append(ES_i + "_soc")

        # Load region carrier TS
        file_name = i + "_carrier.csv"
        df_gen_bus_carrier = pd.read_csv(pre_dic + file_name)
        df_gen_bus_carrier = df_gen_bus_carrier.reindex(columns=all_carriers, fill_value=0)
        df_gen_bus_carrier_update = df_gen_bus_carrier.copy()

        resource_usage_i = {carrier: {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []} for carrier in carriers}
        resource_usage_i["Others"] = {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}
        print(i)

        # Region SOC copy with leading zero row for cycle extraction
        df_capacity_copy = df_capacity.copy()
        df_capacity_copy["snapshot"] = pd.to_datetime(df["snapshot"])
        new_snapshot = df_capacity_copy["snapshot"].iloc[0] - pd.Timedelta(hours=1)
        new_row = pd.DataFrame(
            {col: [0] if col != "snapshot" else [new_snapshot] for col in df_capacity_copy.columns}
        )
        df_capacity_copy = pd.concat([new_row, df_capacity_copy], ignore_index=True)
        process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_capacity_copy[battery_i + "_soc"])
        process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_capacity_copy[ES_i + "_soc"])

        (
            unit_cost_bat_cycle,
            unit_cost_es_cycle,
            co2_emissions_bat_cycle,
            co2_emissions_es_cycle,
            energy_charge_cycle_bat,
            energy_charge_cycle_es,
            energy_discharge_cycle_bat,
            energy_discharge_cycle_es,
            emissions_charged_bat,
            emissions_charged_es,
            emissions_discharged_bat,
            emissions_discharged_es,
            co2_delta_emissions_bat,
            co2_delta_emissions_es,
            cost_delta_bat,
            cost_delta_es,
            resource_usage_i,
        ) = cycle_analysis(
            process_times_bat,
            process_ratios_bat,
            process_times_es,
            process_ratios_es,
            others,
            df_gen_bus_carrier_region_updated,
            df_storage_links,
            df_gen_remain_new,
            i,
            regions,
            CO2_FACTORS,
            resource_usage_i,
            flows_by_res,
            flow_matrices,
            flows_by_res_dis,
            resources,
        )

        # Collect per-resource usage into data frames
        df_bat_cha = pd.DataFrame({res: resource_usage_i[res]["bat_cha"] for res in resources})
        df_bat_dis = pd.DataFrame({res: resource_usage_i[res]["bat_dis"] for res in resources})
        df_es_cha = pd.DataFrame({res: resource_usage_i[res]["es_cha"] for res in resources})
        df_es_dis = pd.DataFrame({res: resource_usage_i[res]["es_dis"] for res in resources})

        # Attach per-cycle unit cost/emissions
        df_bat_extra = pd.DataFrame({"unit_cost_bat": unit_cost_bat_cycle, "co2_emissions_bat": co2_emissions_bat_cycle})
        df_es_extra = pd.DataFrame({"unit_cost_es": unit_cost_es_cycle, "co2_emissions_es": co2_emissions_es_cycle})
        df_bat_cha = pd.concat([df_bat_cha, df_bat_extra], axis=1)
        df_bat_dis = pd.concat([df_bat_dis, df_bat_extra], axis=1)
        df_es_cha = pd.concat([df_es_cha, df_es_extra], axis=1)
        df_es_dis = pd.concat([df_es_dis, df_es_extra], axis=1)

        results_dict[i] = {
            "df_bat_cha": df_bat_cha,
            "df_bat_dis": df_bat_dis,
            "df_es_cha": df_es_cha,
            "df_es_dis": df_es_dis,
        }

        # Cast to numpy for convenient slicing
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
        co2_delta_emissions_bat = np.array(co2_delta_emissions_bat)
        co2_delta_emissions_es = np.array(co2_delta_emissions_es)
        cost_delta_bat = np.array(cost_delta_bat)
        cost_delta_es = np.array(cost_delta_es)

        # Store arrays per region
        dict_unit_cost_bat[i] = unit_cost_bat_cycle
        dict_unit_cost_es[i] = unit_cost_es_cycle
        dict_co2_emissions_bat[i] = co2_emissions_bat_cycle
        dict_co2_emissions_es[i] = co2_emissions_es_cycle
        dict_energy_discharge_cycle_bat[i] = energy_discharge_cycle_bat
        dict_energy_discharge_cycle_es[i] = energy_discharge_cycle_es

        # Aggregate usage for stacked visuals
        for res in resource_usage_i:
            bat_cha_all[res].append(np.sum(resource_usage_i[res]["bat_cha"]))
            bat_dis_all[res].append(-np.sum(resource_usage_i[res]["bat_dis"]))
            es_cha_all[res].append(np.sum(resource_usage_i[res]["es_cha"]))
            es_dis_all[res].append(-np.sum(resource_usage_i[res]["es_dis"]))

        # Averages per region (robust to zero)
        avg_charging_bat = (
            np.sum(emissions_charged_bat) / np.sum(energy_charge_cycle_bat) if np.sum(energy_charge_cycle_bat) else 0.0
        )
        avg_discharging_bat = (
            np.sum(emissions_discharged_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) else 0.0
        )
        charging_avg_bat_list.append(avg_charging_bat)
        discharging_avg_bat_list.append(avg_discharging_bat)

        avg_charging_es = (
            np.sum(emissions_charged_es) / np.sum(energy_charge_cycle_es) if np.sum(energy_charge_cycle_es) else 0.0
        )
        avg_discharging_es = (
            np.sum(emissions_discharged_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) else 0.0
        )
        charging_avg_es_list.append(avg_charging_es)
        discharging_avg_es_list.append(avg_discharging_es)

        cef_bat = (
            np.sum(co2_delta_emissions_bat) / np.sum(energy_discharge_cycle_bat)
            if np.sum(energy_discharge_cycle_bat) > 0
            else 0.0
        )
        cef_es = (
            np.sum(co2_delta_emissions_es) / np.sum(energy_discharge_cycle_es)
            if np.sum(energy_discharge_cycle_es) > 0
            else 0.0
        )
        ccf_bat = (
            np.sum(cost_delta_bat) / np.sum(energy_discharge_cycle_bat) if np.sum(energy_discharge_cycle_bat) > 0 else 0.0
        )
        ccf_es = (
            np.sum(cost_delta_es) / np.sum(energy_discharge_cycle_es) if np.sum(energy_discharge_cycle_es) > 0 else 0.0
        )
        dict_acef_bat[i] = cef_bat
        dict_accf_bat[i] = ccf_bat
        dict_acef_es[i] = cef_es
        dict_accf_es[i] = ccf_es

    # Persist per-region usage tables
    output_path = f"{save_dir}regional_resources_usage_{p}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        for outer_key, inner_dict in results_dict.items():
            for inner_key, df_ in inner_dict.items():
                sheet_name = f"{outer_key}_{inner_key}"[:31]
                df_.to_excel(writer, sheet_name=sheet_name, index=False)

    # Bundle selected arrays for downstream use (CO2_2023 only)
    all_data = {
        "unit_cost_bat": dict_unit_cost_bat,
        "unit_cost_es": dict_unit_cost_es,
        "co2_bat": dict_co2_emissions_bat,
        "co2_es": dict_co2_emissions_es,
        "energy_discharge_bat": dict_energy_discharge_cycle_bat,
        "energy_discharge_es": dict_energy_discharge_cycle_es,
        "acef_bat": dict_acef_bat,
        "acef_es": dict_acef_es,
    }
    ci_save_dir = "./results/CI_CEF_data/"
    os.makedirs(ci_save_dir, exist_ok=True)
    ci_df.to_pickle(f"{ci_save_dir}uk_ci_df.pkl")
    ci_df2.to_pickle(f"{ci_save_dir}uk_ci_df2.pkl")
    eg_df.to_pickle(f"{ci_save_dir}uk_energy_difference.pkl")
    with open(f"{ci_save_dir}uk_results.pkl", "wb") as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save flow matrices workbook
    with pd.ExcelWriter(f"{save_dir}flow_matrices{p}.xlsx") as writer:
        flow_df.to_excel(writer, sheet_name="generation", index_label="generation")
        for sheet_name, dfm in flow_matrices.items():
            dfm.to_excel(writer, sheet_name=sheet_name, index_label=sheet_name)

    # Region average CEF (post-redistribution)
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

    # Round lists for convenience
    avg_cef_list = [round(x, 2) for x in avg_cef_list]
    charging_avg_bat_list = [round(x, 2) for x in charging_avg_bat_list]
    discharging_avg_bat_list = [round(x, 2) for x in discharging_avg_bat_list]
    charging_avg_es_list = [round(x, 2) for x in charging_avg_es_list]
    discharging_avg_es_list = [round(x, 2) for x in discharging_avg_es_list]

    # Save the lists
    lists_payload = {
        "avg_cef_list": avg_cef_list,
        "charging_avg_bat_list": charging_avg_bat_list,
        "discharging_avg_bat_list": discharging_avg_bat_list,
        "charging_avg_es_list": charging_avg_es_list,
        "discharging_avg_es_list": discharging_avg_es_list,
    }
    with open(os.path.join(save_dir, f"saved_lists_{p}.pkl"), "wb") as f:
        pickle.dump(lists_payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# =========================
# CLI entry: run 1..N scenarios
# =========================
def main():
    """
    Run for specified scenarios or all four by default.

    Usage:
        python uk_analysis.py                       # run all four
        python uk_analysis.py all                  # run all four
        python uk_analysis.py uk_min_cost_2023 uk_min_emission_2030
    """
    RESULTS_ROOT = "./results/"
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    scenarios_map = {
        "uk_min_cost_2023": os.path.join(RESULTS_ROOT, "min_cost_2023/"),
        "uk_min_emission_2023": os.path.join(RESULTS_ROOT, "min_CO2_2023/"),
        "uk_min_cost_2030": os.path.join(RESULTS_ROOT, "min_cost_2030/"),
        "uk_min_emission_2030": os.path.join(RESULTS_ROOT, "min_CO2_2030/"),
    }

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
    "./results/min_CO2_2023/",
    "./results/min_cost_2023/",
    "./results/min_cost_2030/",
    "./results/min_CO2_2030/",
]:
    print(f"Running: {path}")
    analyze_one_scenario(path)