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
    CO2_FACTORS: dict,
    others: dict
) -> tuple:
    """
    针对时刻 idx 和单个地区 region，计算并更新该地区电池放电的碳排放和成本，并更新：
      - values: 扣减已使用的本地+外地可用发电剩余
      - flows_by_res_dis: 扣减已使用的外地可用发电剩余 (T, U, R, R)
      - flow_matrix_dis: (DataFrame) 累加本地区放电“替代”各源区发电剩余的容量
      - resource_usage[src]["bat_dis"]
      - cef_bat, cost_bat

    Args:
      idx:                    时间步索引
      region:                 单个地区名
      regions:                全部地区列表
      values:                 当前时刻各 region_resource 发电剩余量 DataFrame
      gen_bus_carrier:        含原始各 source 发电量的 DataFrame，用于选 start_key
      cef_bat, cost_bat:      累计放电碳排与成本的列表
      df_battery_links_charge:含放电需求的 DataFrame
      ratio:                  放电效率系数
      resource_usage:         记录各资源用量的 dict
      flows_by_res_dis:       ndarray (T, U, R, R)，按资源拆分的外地可用剩余
      flow_matrix_dis:        DataFrame (R×R)，索引/列均为 regions
      resources:              资源类型列表
      CO2_FACTORS:            dict, 各资源 CO2 和成本参数
      others:                 dict, fallback 其它资源名称映射

    Returns:
      cef_bat, cost_bat, values, resource_usage, flows_by_res_dis, flow_matrix_dis
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # 初始化本时刻用量
    for src in resource_usage:
        resource_usage[src].setdefault("bat_dis", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0

    # —— 构造 ordered ——
    ordered = resources

    # —— 计算放电需求 ——
    need = -df_battery_links_charge.at[t, f"{region}_Battery_discharger"] * ratio
    if need == 0:
        cef_bat.append(0.0)
        cost_bat.append(0.0)
        return cef_bat, cost_bat, values, resource_usage, flows_by_res_dis, flow_matrix_dis

    # —— 按资源顺序，先本地后外地 ——
    for src in ordered:
        if need <= 0:
            break
        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # 计算可用量
        avail_ext = flows_by_res_dis[t, j, :, k].sum()
        total_here = values.at[values.index[t], col]
        avail_loc = total_here - avail_ext

        # 1) 本地先用
        used_loc = min(avail_loc, need)
        if used_loc > 0:
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum += used_loc * cst
            resource_usage[src]["bat_dis"][-1] += used_loc
            flow_matrix_dis.loc[region, region] += used_loc

        # 2) 再用外地
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res_dis[t, j, :, k] / avail_ext
            for i, src_reg in enumerate(regions):
                amt = used_ext * prop[i]
                flows_by_res_dis[t, j, i, k] -= amt
                flow_matrix_dis.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum += used_ext * cst
            resource_usage[src]["bat_dis"][-1] += used_ext

    # —— 3) 兜底 Others ——
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["bat_dis"][-1] += need
        need = 0

    # —— 4) 更新累计并返回 ——
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
    CO2_FACTORS: dict,
    others: dict
) -> tuple:
    """
    针对时刻 idx 和单个地区 region，计算并更新该地区 LDES 放电的碳排放 & 成本，并更新：
      - values: 扣减本地+外地可用剩余发电
      - flows_by_res_dis: 扣减外地可用发电剩余 (T, U, R, R)
      - flow_matrix_dis: 累加本区放电“替代”各源区发电剩余量 (DataFrame)
      - resource_usage[src]["es_dis"]
      - cef_es, cost_es 列表

    Args:
      idx:                   时间步索引
      region:                单个地区名
      regions:               全部地区列表
      values:                当前时刻各 region_resource 发电剩余量 DataFrame
      cef_es, cost_es:       累计放电碳排 & 成本列表
      df_storage_links:      LDES 放电需求 DataFrame（列 "{region}_OtherStorage_discharger"）
      ratio:                 放电效率系数
      resource_usage:        每资源用量记录 dict
      flows_by_res_dis:      ndarray (T, U, R, R)，外地可用发电剩余
      flow_matrix_dis:       DataFrame (R×R)，索引/列均为 regions
      resources:             资源类型列表
      CO2_FACTORS:           各资源 CO2 & 成本参数 dict
      others:                fallback 其它资源名称映射 dict

    Returns:
      (cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # 初始化
    for src in resource_usage:
        resource_usage[src].setdefault("es_dis", []).append(0)
    cef_sum, cost_sum = 0.0, 0.0
    ordered = resources

    # 3) 放电需求
    need = -df_storage_links.at[t, f"{region}_Other_storage_discharger"] * ratio
    if need == 0:
        cef_es.append(0.0)
        cost_es.append(0.0)
        return cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis

    # 4) 依次消耗：本地 -> 外地
    for src in ordered:
        if need <= 0:
            break
        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # 可用量
        avail_ext = flows_by_res_dis[t, j, :, k].sum()
        total_here = values.at[values.index[t], col]
        avail_loc = total_here - avail_ext

        # a) 本地先用
        used_loc = min(avail_loc, need)
        if used_loc > 0:
            values.at[values.index[t], col] -= used_loc
            need -= used_loc
            cef_sum += used_loc * co2
            cost_sum += used_loc * cst
            resource_usage[src]["es_dis"][-1] += used_loc
            flow_matrix_dis.loc[region, region] += used_loc

        # b) 外地再用
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res_dis[t, j, :, k] / avail_ext
            for i, src_reg in enumerate(regions):
                amt = used_ext * prop[i]
                flows_by_res_dis[t, j, i, k] -= amt
                flow_matrix_dis.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need -= used_ext
            cef_sum += used_ext * co2
            cost_sum += used_ext * cst
            resource_usage[src]["es_dis"][-1] += used_ext

    # 5) 兜底 Others
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["es_dis"][-1] += need
        need = 0

    # 6) 更新列表
    cef_es.append(cef_sum)
    cost_es.append(cost_sum)

    return cef_es, cost_es, values, resource_usage, flows_by_res_dis, flow_matrix_dis

def cef_bat_log(
    idx: int,
    region: str,
    regions: list[str],
    values: pd.DataFrame,
    cef_bat: list,
    cost_bat: list,
    df_battery_links_charge: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res: np.ndarray,
    flow_matrix: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict,
    others: dict
) -> tuple:
    """
    Calculate and append the Carbon Emission Factor (CEF) for battery charging at a specific time step,
    updating energy usage and tracking inter-region flows.

    Args:
      idx:                   Current time step index
      region:                Single region name
      regions:               List of all regions
      values:                DataFrame of remaining energy per region_resource at this step
      cef_bat:               List accumulating CEF values for battery charging
      cost_bat:              List accumulating costs for battery charging
      df_battery_links_charge: DataFrame of battery charge demand per region
      ratio:                 Charging efficiency ratio
      resource_usage:        Dict recording per-source usage
      flows_by_res:          ndarray (T,U,R,R) inter-region, per-resource inflow volumes
      flow_matrix:           DataFrame (R×R) tracking cumulative source→dest flows
      resources:             List of resource names in order matching flows_by_res
      CO2_FACTORS:           Dict mapping resource→(CO2, cost)
      others:                Dict mapping region→fallback resource name

    Returns:
      (cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # initialize usage for this timestep
    for src in resource_usage:
        resource_usage[src].setdefault("bat_cha", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0
    need = df_battery_links_charge.at[t, f"{region}_Battery_charger"] * ratio
    if need <= 0:
        cef_bat.append(0.0)
        cost_bat.append(0.0)
        return cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix
    # resources ordered by CO2 descending for charging
    ordered = list(reversed(resources))

    for src in ordered:
        if need <= 0:
            break
        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # compute available ext and loc (without modifying)
        avail_ext = flows_by_res[t, j, :, k].sum()
        total_here = values.at[values.index[t], col]
        avail_loc = total_here - avail_ext

        # 1) external inflow first
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res[t, j, :, k] / avail_ext
            for i, src_reg in enumerate(regions):
                amt = used_ext * prop[i]
                flows_by_res[t, j, i, k] -= amt
                flow_matrix.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["bat_cha"][-1] += used_ext

        # 2) then local
        if need > 0 and avail_loc > 0:
            used_loc = min(avail_loc, need)
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["bat_cha"][-1] += used_loc
            flow_matrix.loc[region, region] += used_loc

    # 3) fallback Others
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["bat_cha"][-1] += need
        need = 0

    cef_bat.append(cef_sum)
    cost_bat.append(cost_sum)
    return cef_bat, cost_bat, values, resource_usage, flows_by_res, flow_matrix

def cef_es_log(
    idx: int,
    region: str,
    regions: list[str],
    values: pd.DataFrame,
    cef_es: list,
    cost_es: list,
    df_storage_links: pd.DataFrame,
    ratio: float,
    resource_usage: dict,
    flows_by_res: np.ndarray,
    flow_matrix: pd.DataFrame,
    resources: list[str],
    CO2_FACTORS: dict,
    others: dict
) -> tuple:
    """
    Calculate and append the Carbon Emission Factor (CEF) for LDES (OtherStorage) charging at a specific time step,
    updating energy usage and tracking inter-region flows.

    Args:
      idx:                   Current time step index
      region:                Single region name
      regions:               List of all regions
      values:                DataFrame of remaining energy per region_resource at this step
      cef_es:                List accumulating CEF values for LDES charging
      cost_es:               List accumulating costs for LDES charging
      df_storage_links:      DataFrame of LDES charge demand per region
      ratio:                 Charging efficiency ratio
      resource_usage:        Dict recording per-source usage
      flows_by_res:          ndarray (T,U,R,R) inter-region, per-resource inflow volumes
      flow_matrix:           DataFrame (R×R) tracking cumulative source→dest flows
      resources:             List of resource names in order matching flows_by_res
      CO2_FACTORS:           Dict mapping resource→(CO2, cost)
      others:                Dict mapping region→fallback resource name

    Returns:
      (cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix)
    """
    R = len(regions)
    t = idx
    k = regions.index(region)

    # initialize usage for this timestep
    for src in resource_usage:
        resource_usage[src].setdefault("es_cha", []).append(0)

    cef_sum, cost_sum = 0.0, 0.0
    need = df_storage_links.at[t, f"{region}_Other_storage_charger"] * ratio
    if need <= 0:
        cef_es.append(0.0)
        cost_es.append(0.0)
        return cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix

    # resources ordered by CO2 descending for charging
    ordered = list(reversed(resources))

    for src in ordered:
        if need <= 0:
            break
        co2, cst = CO2_FACTORS[src]
        j = resources.index(src)
        col = f"{region}_{src}"

        # compute available ext and loc (without modifying)
        avail_ext = flows_by_res[t, j, :, k].sum()
        total_here = values.at[values.index[t], col]
        avail_loc = total_here - avail_ext

        # 1) external inflow first
        if need > 0 and avail_ext > 0:
            used_ext = min(avail_ext, need)
            prop = flows_by_res[t, j, :, k] / avail_ext
            for i, src_reg in enumerate(regions):
                amt = used_ext * prop[i]
                flows_by_res[t, j, i, k] -= amt
                flow_matrix.loc[src_reg, region] += amt
            values.at[values.index[t], col] -= used_ext
            need    -= used_ext
            cef_sum += used_ext * co2
            cost_sum+= used_ext * cst
            resource_usage[src]["es_cha"][-1] += used_ext

        # 2) then local
        if need > 0 and avail_loc > 0:
            used_loc = min(avail_loc, need)
            values.at[values.index[t], col] -= used_loc
            need    -= used_loc
            cef_sum += used_loc * co2
            cost_sum+= used_loc * cst
            resource_usage[src]["es_cha"][-1] += used_loc
            flow_matrix.loc[region, region] += used_loc

    # 3) fallback Others
    if need > 0:
        co2_o, cst_o = CO2_FACTORS[others[region]]
        cef_sum += need * co2_o
        cost_sum += need * cst_o
        resource_usage['Others']["es_cha"][-1] += need
        need = 0

    cef_es.append(cef_sum)
    cost_es.append(cost_sum)
    return cef_es, cost_es, values, resource_usage, flows_by_res, flow_matrix

def cycle_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, others, gen_bus_carrier, df_storage_links, df_gen_remain, region, regions, CO2_FACTORS, resource_usage,
                   flows_by_res, flow_matrices, flows_by_res_dis, resources):
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
    unit_cost_bat_cycle = []
    unit_cost_es_cycle = []
    emissions_bat_charged_cycle = []
    emissions_es_charged_cycle = []
    emissions_bat_discharged_cycle = []
    emissions_es_discharged_cycle = []
    co2_delta_bat_emissions = []
    co2_delta_es_emissions = []
    cost_delta_bat = []
    cost_delta_es = []
    gen_bus_carrier_0 = gen_bus_carrier
    df_gen_remain_0 = df_gen_remain.copy()

    # Analyze battery charging and discharging cycles
    for cycle_number, times in process_times_bat.items():
        # Calculate total energy used during the cycle
        energy_bat_charge = (df_storage_links[region + '_Battery_charger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        energy_bat_discharge = (
                    -df_storage_links[region + '_Battery_discharger'].iloc[times] * process_ratios_bat[cycle_number]).sum()
        cef_bat_charge = []  # Initialize list for CO2 emissions during charging
        cef_bat_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_bat_charge = []
        cost_bat_discharge = []
        resource_bat = {
            source: {"bat_cha": [], "bat_dis": []} for source in resource_usage.keys()
        }

        for pos, idx in enumerate(times):
            ratio = process_ratios_bat[cycle_number][pos]  # 获取当前时间步对应的比例
            cef_bat_charge, cost_bat_charge, gen_bus_carrier_0, resource_bat, flows_by_res, flow_matrices['bat_cha'] = cef_bat_log(
                idx,
                region,
                regions,
                gen_bus_carrier_0,
                cef_bat_charge,
                cost_bat_charge,
                df_storage_links,
                ratio,
                resource_bat,
                flows_by_res,
                flow_matrices['bat_cha'],
                resources,
                CO2_FACTORS,
                others
            )
            cef_bat_discharge, cost_bat_discharge, df_gen_remain_0, resource_bat, flows_by_res_dis, flow_matrices['bat_dis'] = cef_bat_discharge_log(
                idx,
                region,
                regions,
                df_gen_remain_0,
                cef_bat_discharge,
                cost_bat_discharge,
                df_storage_links,
                ratio,
                resource_bat,
                flows_by_res_dis,
                flow_matrices['bat_dis'],
                resources,
                CO2_FACTORS,
                others
            )

        # Calculate emissions and deltas for the cycle
        emissions_bat_charged = np.sum(cef_bat_charge)
        emissions_bat_discharged = np.sum(cef_bat_discharge)
        cost_bat_charged = np.sum(cost_bat_charge)
        cost_bat_discharged = np.sum(cost_bat_discharge)
        delta_bat_emissions = -emissions_bat_discharged + emissions_bat_charged
        delta_bat_cost = -cost_bat_discharged + cost_bat_charged

        # Append results for batteries
        emissions_bat_charged_cycle.append(emissions_bat_charged)
        emissions_bat_discharged_cycle.append(emissions_bat_discharged)
        co2_bat_charge_cycle.append(delta_bat_emissions / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        unit_cost_bat_cycle.append(delta_bat_cost / energy_bat_discharge if energy_bat_discharge != 0 else 0)
        energy_bat_charge_cycle.append(energy_bat_charge)
        energy_bat_discharged_cycle.append(energy_bat_discharge)
        co2_delta_bat_emissions.append(delta_bat_emissions)
        cost_delta_bat.append(delta_bat_cost)
        for source in resource_usage.keys():
            resource_usage[source]["bat_cha"].append(np.sum(resource_bat[source]["bat_cha"]))
            resource_usage[source]["bat_dis"].append(np.sum(resource_bat[source]["bat_dis"]))

    # Analyze LDES charging and discharging cycles
    for cycle_number, times in process_times_es.items():
        # Calculate total energy used during the cycle
        energy_es_charge = (df_storage_links[region + '_Other_storage_charger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        energy_es_discharge = (
                    -df_storage_links[region + '_Other_storage_discharger'].iloc[times] * process_ratios_es[cycle_number]).sum()
        cef_es_charge = []  # Initialize list for CO2 emissions during charging
        cef_es_discharge = []  # Initialize list for CO2 emissions during discharging
        cost_es_charge = []
        cost_es_discharge = []
        resource_es = {
            source: {"es_cha": [], "es_dis": []} for source in resource_usage.keys()
        }

        for pos, idx in enumerate(times):
            ratio = process_ratios_es[cycle_number][pos]  # 获取当前时间步对应的比例
            cef_es_charge, cost_es_charge,  gen_bus_carrier_0, resource_es, flows_by_res, flow_matrices['es_cha'] = cef_es_log(
                idx,
                region,
                regions,
                gen_bus_carrier_0,
                cef_es_charge,
                cost_es_charge,
                df_storage_links,
                ratio,
                resource_es,
                flows_by_res,
                flow_matrices['es_cha'],
                resources,
                CO2_FACTORS,
                others
            )
            cef_es_discharge, cost_es_discharge, df_gen_remain_0, resource_es, flows_by_res_dis, flow_matrices['es_dis'] = cef_es_discharge_log(
                idx,
                region,
                regions,
                df_gen_remain_0,
                cef_es_discharge,
                cost_es_discharge,
                df_storage_links,
                ratio,
                resource_es,
                flows_by_res_dis,
                flow_matrices['es_dis'],
                resources,
                CO2_FACTORS,
                others
            )

        # Calculate emissions and deltas for the cycle
        emissions_es_charged = np.sum(cef_es_charge)
        emissions_es_discharged = np.sum(cef_es_discharge)
        cost_es_charged = np.sum(cost_es_charge)
        cost_es_discharged = np.sum(cost_es_discharge)
        delta_es_emissions = -emissions_es_discharged + emissions_es_charged
        delta_es_cost = -cost_es_discharged + cost_es_charged

        # Append results for LDES
        emissions_es_charged_cycle.append(emissions_es_charged)
        emissions_es_discharged_cycle.append(emissions_es_discharged)
        co2_es_charge_cycle.append(delta_es_emissions / energy_es_discharge if energy_es_discharge != 0 else 0)
        unit_cost_es_cycle.append(delta_es_cost / energy_es_discharge if energy_es_discharge != 0 else 0)
        energy_es_charge_cycle.append(energy_es_charge)
        energy_es_discharged_cycle.append(energy_es_discharge)
        co2_delta_es_emissions.append(delta_es_emissions)
        cost_delta_es.append(delta_es_cost)
        for source in resource_usage.keys():
            resource_usage[source]["es_cha"].append(np.sum(resource_es[source]["es_cha"]))
            resource_usage[source]["es_dis"].append(np.sum(resource_es[source]["es_dis"]))

    # Return all results as a tuple
    return (unit_cost_bat_cycle, unit_cost_es_cycle, co2_bat_charge_cycle, co2_es_charge_cycle,
            energy_bat_charge_cycle, energy_es_charge_cycle, energy_bat_discharged_cycle, energy_es_discharged_cycle,
            emissions_bat_charged_cycle, emissions_es_charged_cycle, emissions_bat_discharged_cycle, emissions_es_discharged_cycle,
            co2_delta_bat_emissions, co2_delta_es_emissions, cost_delta_bat, cost_delta_es,
            resource_usage)
