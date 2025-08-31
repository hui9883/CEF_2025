import numpy as np
import pandas as pd
import os
import pypsa
import datetime
print(pypsa.__version__)

def build_network_eu_2023(xlsx_path: str, out_nc_path: str):
    # ---- load data (EU-2023) ----
    load = pd.read_excel(xlsx_path, sheet_name='load_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    pv = pd.read_excel(xlsx_path, sheet_name='pv_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    wind = pd.read_excel(xlsx_path, sheet_name='wind_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    wind_offshore = pd.read_excel(xlsx_path, sheet_name='wind_offshore_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    ror = pd.read_excel(xlsx_path, sheet_name='ror_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    inflow_reservoir = pd.read_excel(xlsx_path, sheet_name='inflow_reservoir_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    dispatch_reservoir = pd.read_excel(xlsx_path, sheet_name='dispatch_reservoir_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    chp = pd.read_excel(xlsx_path, sheet_name='chp_timeseries', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()
    chp_bio = pd.read_excel(xlsx_path, sheet_name='chp_bio_timeseries', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

    buses = pd.read_excel(xlsx_path, sheet_name='buses', index_col=0)
    links = pd.read_excel(xlsx_path, sheet_name='links', index_col=0)

    gen_pv = pd.read_excel(xlsx_path, sheet_name='gen_pv', index_col=0)
    gen_wind = pd.read_excel(xlsx_path, sheet_name='gen_wind', index_col=0)
    gen_wind_offshore = pd.read_excel(xlsx_path, sheet_name='gen_wind_offshore', index_col=0)
    gen_gas = pd.read_excel(xlsx_path, sheet_name='gen_gas', index_col=0)
    gen_oil = pd.read_excel(xlsx_path, sheet_name='gen_oil', index_col=0)
    gen_coal = pd.read_excel(xlsx_path, sheet_name='gen_coal', index_col=0)
    gen_nuclear = pd.read_excel(xlsx_path, sheet_name='gen_nuclear', index_col=0)
    gen_biomass = pd.read_excel(xlsx_path, sheet_name='gen_biomass', index_col=0)
    gen_biogas = pd.read_excel(xlsx_path, sheet_name='gen_biogas', index_col=0)
    gen_bio_chp = pd.read_excel(xlsx_path, sheet_name='gen_bio_chp', index_col=0)
    gen_ror = pd.read_excel(xlsx_path, sheet_name='gen_ror', index_col=0)
    gen_other_res = pd.read_excel(xlsx_path, sheet_name='gen_other_res', index_col=0)
    gen_coal_chp = pd.read_excel(xlsx_path, sheet_name='gen_coal_chp', index_col=0)
    gen_oil_chp = pd.read_excel(xlsx_path, sheet_name='gen_oil_chp', index_col=0)

    st_reservoir = pd.read_excel(xlsx_path, sheet_name='st_reservoir', index_col=0)
    st_battery = pd.read_excel(xlsx_path, sheet_name='st_battery', index_col=0)
    st_other = pd.read_excel(xlsx_path, sheet_name='st_other', index_col=0)

    # timeseries for CHP (align columns to generator indices)
    coal_chp_timeseries = pd.DataFrame(data=chp, columns=gen_coal_chp.bus); coal_chp_timeseries.columns = gen_coal_chp.index
    oil_chp_timeseries = pd.DataFrame(data=chp, columns=gen_oil_chp.bus); oil_chp_timeseries.columns = gen_oil_chp.index

    # ---- network build (EU-2023) ----
    network = pypsa.Network()
    network.set_snapshots(load.index)

    network.import_components_from_dataframe(buses, 'Bus')
    network.lines['s_nom_extendable'] = True  # ok even if no Line components

    network.add("Link", links.index, bus0=links['bus0'].tolist(), bus1=links['bus1'].tolist(),
                p_nom=links['p_nom'].tolist(), p_max_pu=links['p_max_pu'].to_list())

    network.add("Load", load.columns, bus=load.columns, p_set=load)

    network.add('Generator', gen_pv['bus'], suffix='_PV', bus=gen_pv['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_pv['p_nom'].to_list(), carrier='PV',
                marginal_cost=gen_pv['marginal_cost'].to_list(), p_max_pu=pv)

    network.add('Generator', gen_wind['bus'], suffix='_Wind', bus=gen_wind['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_wind['p_nom'].to_list(), carrier='Wind',
                marginal_cost=gen_wind['marginal_cost'].to_list(), p_max_pu=wind)

    network.add('Generator', gen_wind_offshore['bus'], suffix='_Wind_offshore',
                bus=gen_wind_offshore['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_wind_offshore['p_nom'].to_list(), carrier='Wind offshore',
                marginal_cost=gen_wind_offshore['marginal_cost'].to_list(), p_max_pu=wind_offshore)

    network.add('Generator', gen_gas.index, bus=gen_gas['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_gas['p_nom'].to_list(), carrier=gen_gas['carrier'].to_list(),
                marginal_cost=gen_gas['marginal_cost'].to_list(), efficiency=gen_gas['efficiency'].to_list())

    network.add('Generator', gen_oil.index, bus=gen_oil['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_oil['p_nom'].to_list(), carrier='Oil',
                marginal_cost=gen_oil['marginal_cost'].to_list(), efficiency=gen_oil['efficiency'].to_list())

    network.add('Generator', gen_coal.index, bus=gen_coal['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_coal['p_nom'].to_list(), carrier=gen_coal['carrier'].to_list(),
                marginal_cost=gen_coal['marginal_cost'].to_list(), efficiency=gen_coal['efficiency'].to_list())

    network.add('Generator', gen_nuclear.index, bus=gen_nuclear['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_nuclear['p_nom'].to_list(), carrier='Nuclear',
                marginal_cost=gen_nuclear['marginal_cost'].to_list(), efficiency=gen_nuclear['efficiency'].to_list(),
                p_max_pu=gen_nuclear['p_max_pu'].to_list(), p_min_pu=gen_nuclear['p_min_pu'].to_list())

    network.add('Generator', gen_biomass.index, bus=gen_biomass['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_biomass['p_nom'].to_list(), carrier=gen_biomass['carrier'].to_list(),
                marginal_cost=gen_biomass['marginal_cost'].to_list(), efficiency=gen_biomass['efficiency'].to_list(),
                p_max_pu=gen_biomass['p_max_pu'].to_list())

    network.add('Generator', gen_biogas['bus'], suffix='_Biogas', bus=gen_biogas['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_biogas['p_nom'].to_list(), carrier='Biogas',
                marginal_cost=gen_biogas['marginal_cost'].to_list(), efficiency=gen_biogas['efficiency'].to_list(),
                p_max_pu=gen_biogas['p_max_pu'].to_list())

    network.add('Generator', gen_ror['bus'], suffix='_ROR', bus=gen_ror['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_ror['p_nom'].to_list(), carrier=gen_ror['carrier'].to_list(),
                marginal_cost=gen_ror['marginal_cost'].to_list(), p_max_pu=ror)

    network.add('Generator', gen_other_res['bus'], suffix='_OtherRES', bus=gen_other_res['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_other_res['p_nom'].to_list(), carrier='Other RES',
                marginal_cost=gen_other_res['marginal_cost'].to_list(), p_max_pu=gen_other_res['p_max_pu'].to_list())

    network.add('Generator', gen_coal_chp.index, bus=gen_coal_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_coal_chp['p_nom'].to_list(), carrier=gen_coal_chp['carrier'].to_list(),
                marginal_cost=gen_coal_chp['marginal_cost'].to_list(), p_max_pu=coal_chp_timeseries,
                p_min_pu=0.9 * coal_chp_timeseries, efficiency=gen_coal_chp['efficiency'].to_list())

    network.add('Generator', gen_oil_chp.index, bus=gen_oil_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_oil_chp['p_nom'].to_list(), carrier=gen_oil_chp['carrier'].to_list(),
                marginal_cost=gen_oil_chp['marginal_cost'].to_list(), p_max_pu=oil_chp_timeseries,
                p_min_pu=0.9 * oil_chp_timeseries, efficiency=gen_oil_chp['efficiency'].to_list())

    network.add('Generator', gen_bio_chp.index, bus=gen_bio_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_bio_chp['p_nom'].to_list(), carrier=gen_bio_chp['carrier'].to_list(),
                marginal_cost=gen_bio_chp['marginal_cost'].to_list(),
                p_max_pu=chp_bio, p_min_pu=chp_bio * 0.9, efficiency=gen_bio_chp['efficiency'].to_list())

    network.add("StorageUnit", st_reservoir.index , bus=st_reservoir['bus'].tolist(),
                carrier=st_reservoir['carrier'].tolist(), p_nom=st_reservoir['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_reservoir['max_hours'].to_list(),
                p_max_pu=dispatch_reservoir, p_min_pu=0,
                efficiency_dispatch=st_reservoir['efficiency_dispatch'].tolist(),
                efficiency_store=st_reservoir['efficiency_store'].tolist(),
                marginal_cost=st_reservoir['marginal_cost'].to_list(), standing_loss=0,
                cyclic_state_of_charge=True,
                state_of_charge_initial=st_reservoir['state_of_charge_initial'].tolist(),
                inflow=inflow_reservoir)

    network.add("StorageUnit", st_battery.index, bus=st_battery['bus'].tolist(),
                carrier=st_battery['carrier'].tolist(), p_nom=st_battery['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_battery['max_hours'].to_list(),
                p_max_pu=st_battery['p_max_pu'].tolist(),
                efficiency_dispatch=st_battery['efficiency_dispatch'].tolist(),
                standing_loss=st_battery['standing_loss'].tolist())

    network.add("StorageUnit", st_other.index, bus=st_other['bus'].tolist(),
                carrier=st_other['carrier'].tolist(), p_nom=st_other['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_other['max_hours'].to_list(),
                p_max_pu=st_other['p_max_pu'].tolist(),
                efficiency_dispatch=st_other['efficiency_dispatch'].tolist(),
                standing_loss=st_other['standing_loss'].tolist())

    # CO2 factors
    dict_carriers = {
        'Lignite': 1.06, 'Hard coal': 0.867, 'CCGT': 0.36, 'SCGT': 0.46, 'Oil': 0.65,
        'Biomass': 0.0, 'Biogas': 0.19656, 'BECCS': 0, 'Wind': 0, 'Wind offshore': 0,
        'PV': 0, 'Hydro': 0, 'Other RES': 0, 'Battery': 0, 'Other storage': 0,
        'Nuclear': 0, 'DSR': 0,
    }

    network.consistency_check()
    network.generators["co2_emissions"] = network.generators["carrier"].map(dict_carriers)
    network.storage_units["co2_emissions"] = network.storage_units["carrier"].map(dict_carriers)

    # export
    os.makedirs(os.path.dirname(out_nc_path), exist_ok=True)
    network.export_to_netcdf(out_nc_path)
    print(f"Saved EU-2023: {out_nc_path}")


def build_network_eu_2030(xlsx_path: str, out_nc_path: str):
    # ---- load data (EU-2030) ----
    load = pd.read_excel(xlsx_path, sheet_name='load_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    pv = pd.read_excel(xlsx_path, sheet_name='pv_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    wind = pd.read_excel(xlsx_path, sheet_name='wind_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    wind_offshore = pd.read_excel(xlsx_path, sheet_name='wind_offshore_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    ror = pd.read_excel(xlsx_path, sheet_name='ror_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    inflow_reservoir = pd.read_excel(xlsx_path, sheet_name='inflow_reservoir_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    dispatch_reservoir = pd.read_excel(xlsx_path, sheet_name='dispatch_reservoir_timeseries', parse_dates=['t'], index_col='t', usecols='A:AB').dropna(axis=1, how='all').dropna()
    chp = pd.read_excel(xlsx_path, sheet_name='chp_timeseries', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()
    chp_bio = pd.read_excel(xlsx_path, sheet_name='chp_bio_timeseries', parse_dates=['t'], index_col='t').dropna(axis=1, how='all').dropna()

    buses = pd.read_excel(xlsx_path, sheet_name='buses', index_col=0)
    links = pd.read_excel(xlsx_path, sheet_name='links', index_col=0)

    gen_pv = pd.read_excel(xlsx_path, sheet_name='gen_pv', index_col=0)
    gen_wind = pd.read_excel(xlsx_path, sheet_name='gen_wind', index_col=0)
    gen_wind_offshore = pd.read_excel(xlsx_path, sheet_name='gen_wind_offshore', index_col=0)
    gen_gas = pd.read_excel(xlsx_path, sheet_name='gen_gas', index_col=0)
    gen_oil = pd.read_excel(xlsx_path, sheet_name='gen_oil', index_col=0)
    gen_coal = pd.read_excel(xlsx_path, sheet_name='gen_coal', index_col=0)
    gen_nuclear = pd.read_excel(xlsx_path, sheet_name='gen_nuclear', index_col=0)
    gen_biomass = pd.read_excel(xlsx_path, sheet_name='gen_biomass', index_col=0)
    gen_biogas = pd.read_excel(xlsx_path, sheet_name='gen_biogas', index_col=0)
    gen_bio_chp = pd.read_excel(xlsx_path, sheet_name='gen_bio_chp', index_col=0)
    gen_ror = pd.read_excel(xlsx_path, sheet_name='gen_ror', index_col=0)
    gen_other_res = pd.read_excel(xlsx_path, sheet_name='gen_other_res', index_col=0)
    gen_dsr = pd.read_excel(xlsx_path, sheet_name='gen_dsr', index_col=0)
    gen_gas_chp = pd.read_excel(xlsx_path, sheet_name='gen_gas_chp', index_col=0)
    gen_coal_chp = pd.read_excel(xlsx_path, sheet_name='gen_coal_chp', index_col=0)
    gen_oil_chp = pd.read_excel(xlsx_path, sheet_name='gen_oil_chp', index_col=0)
    gen_res_chp = pd.read_excel(xlsx_path, sheet_name='gen_res_chp', index_col=0)

    st_reservoir = pd.read_excel(xlsx_path, sheet_name='st_reservoir', index_col=0)
    st_battery = pd.read_excel(xlsx_path, sheet_name='st_battery', index_col=0)
    st_other = pd.read_excel(xlsx_path, sheet_name='st_other', index_col=0)

    # CHP time series (align)
    gas_chp_timeseries = pd.DataFrame(data=chp, columns=gen_gas_chp.bus); gas_chp_timeseries.columns = gen_gas_chp.index
    coal_chp_timeseries = pd.DataFrame(data=chp, columns=gen_coal_chp.bus); coal_chp_timeseries.columns = gen_coal_chp.index
    oil_chp_timeseries = pd.DataFrame(data=chp, columns=gen_oil_chp.bus); oil_chp_timeseries.columns = gen_oil_chp.index
    res_chp_timeseries = pd.DataFrame(data=chp, columns=gen_res_chp.bus); res_chp_timeseries.columns = gen_res_chp.index

    # ---- network build (EU-2030) ----
    network = pypsa.Network()
    network.set_snapshots(load.index)

    network.import_components_from_dataframe(buses, 'Bus')
    network.lines['s_nom_extendable'] = True

    network.add("Link", links.index, bus0=links['bus0'].tolist(), bus1=links['bus1'].tolist(),
                p_nom=links['p_nom'].tolist(), p_max_pu=links['p_max_pu'].to_list())

    network.add("Load", load.columns, bus=load.columns, p_set=load)

    network.add('Generator', gen_pv['bus'], suffix='_PV', bus=gen_pv['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_pv['p_nom'].to_list(), carrier='PV',
                marginal_cost=gen_pv['marginal_cost'].to_list(), p_max_pu=pv)

    network.add('Generator', gen_wind['bus'], suffix='_Wind', bus=gen_wind['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_wind['p_nom'].to_list(), carrier='Wind',
                marginal_cost=gen_wind['marginal_cost'].to_list(), p_max_pu=wind)

    network.add('Generator', gen_wind_offshore['bus'], suffix='_Wind_offshore',
                bus=gen_wind_offshore['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_wind_offshore['p_nom'].to_list(), carrier='Wind offshore',
                marginal_cost=gen_wind_offshore['marginal_cost'].to_list(), p_max_pu=wind_offshore)

    network.add('Generator', gen_gas.index, bus=gen_gas['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_gas['p_nom'].to_list(), carrier=gen_gas['carrier'].to_list(),
                marginal_cost=gen_gas['marginal_cost'].to_list(), efficiency=gen_gas['efficiency'].to_list())

    network.add('Generator', gen_oil.index, bus=gen_oil['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_oil['p_nom'].to_list(), carrier='Oil',
                marginal_cost=gen_oil['marginal_cost'].to_list(), efficiency=gen_oil['efficiency'].to_list())

    network.add('Generator', gen_coal.index, bus=gen_coal['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_coal['p_nom'].to_list(), carrier=gen_coal['carrier'].to_list(),
                marginal_cost=gen_coal['marginal_cost'].to_list(), efficiency=gen_coal['efficiency'].to_list())

    network.add('Generator', gen_nuclear.index, bus=gen_nuclear['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_nuclear['p_nom'].to_list(), carrier='Nuclear',
                marginal_cost=gen_nuclear['marginal_cost'].to_list(), efficiency=gen_nuclear['efficiency'].to_list(),
                p_max_pu=gen_nuclear['p_max_pu'].to_list(), p_min_pu=gen_nuclear['p_min_pu'].to_list())

    network.add('Generator', gen_biomass.index, bus=gen_biomass['bus'].to_list(), p_nom_extendable=False,
        p_nom=gen_biomass['p_nom'].to_list(), carrier=gen_biomass['carrier'].to_list(),
        marginal_cost=gen_biomass['marginal_cost'].to_list(), efficiency=gen_biomass['efficiency'].to_list(),
        p_max_pu=gen_biomass['p_max_pu'].to_list())

    network.add('Generator', gen_biogas['bus'], suffix='_Biogas', bus=gen_biogas['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_biogas['p_nom'].to_list(), carrier='Biogas',
                marginal_cost=gen_biogas['marginal_cost'].to_list(), efficiency=gen_biogas['efficiency'].to_list(),
                p_max_pu=gen_biogas['p_max_pu'].to_list())

    network.add('Generator', gen_ror['bus'], suffix='_ROR', bus=gen_ror['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_ror['p_nom'].to_list(), carrier=gen_ror['carrier'].to_list(),
                marginal_cost=gen_ror['marginal_cost'].to_list(), p_max_pu=ror)

    network.add('Generator', gen_other_res['bus'], suffix='_OtherRES', bus=gen_other_res['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_other_res['p_nom'].to_list(), carrier='Other RES',
                marginal_cost=gen_other_res['marginal_cost'].to_list(), p_max_pu=gen_other_res['p_max_pu'].to_list())

    network.add('Generator', gen_dsr['bus'], suffix='_DSR', bus=gen_dsr['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_dsr['p_nom'].to_list(), carrier='DSR',
                marginal_cost=gen_dsr['marginal_cost'].to_list(), p_max_pu=gen_dsr['p_max_pu'].to_list())

    network.add('Generator', gen_gas_chp.index, bus=gen_gas_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_gas_chp['p_nom'].to_list(), carrier=gen_gas_chp['carrier'].to_list(),
                marginal_cost=gen_gas_chp['marginal_cost'].to_list(), p_max_pu=gas_chp_timeseries,
                p_min_pu=0.9 * gas_chp_timeseries, efficiency=gen_gas_chp['efficiency'].to_list())

    network.add('Generator', gen_coal_chp.index, bus=gen_coal_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_coal_chp['p_nom'].to_list(), carrier=gen_coal_chp['carrier'].to_list(),
                marginal_cost=gen_coal_chp['marginal_cost'].to_list(), p_max_pu=coal_chp_timeseries,
                p_min_pu=0.9 * coal_chp_timeseries, efficiency=gen_coal_chp['efficiency'].to_list())

    network.add('Generator', gen_oil_chp.index, bus=gen_oil_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_oil_chp['p_nom'].to_list(), carrier=gen_oil_chp['carrier'].to_list(),
                marginal_cost=gen_oil_chp['marginal_cost'].to_list(), p_max_pu=oil_chp_timeseries,
                p_min_pu=0.9 * oil_chp_timeseries, efficiency=gen_oil_chp['efficiency'].to_list())

    network.add('Generator', gen_res_chp.index, bus=gen_res_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_res_chp['p_nom'].to_list(), carrier=gen_res_chp['carrier'].to_list(),
                marginal_cost=gen_res_chp['marginal_cost'].to_list(), p_max_pu=res_chp_timeseries,
                p_min_pu=0.9 * res_chp_timeseries, efficiency=gen_res_chp['efficiency'].to_list())

    network.add('Generator', gen_bio_chp.index, bus=gen_bio_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_bio_chp['p_nom'].to_list(), carrier=gen_bio_chp['carrier'].to_list(),
                marginal_cost=gen_bio_chp['marginal_cost'].to_list(), p_max_pu=chp_bio,
                p_min_pu=chp_bio * 0.9, efficiency=gen_bio_chp['efficiency'].to_list())

    network.add("StorageUnit", st_reservoir.index , bus=st_reservoir['bus'].tolist(),
                carrier=st_reservoir['carrier'].tolist(), p_nom=st_reservoir['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_reservoir['max_hours'].to_list(),
                p_max_pu=dispatch_reservoir, p_min_pu=0,
                efficiency_dispatch=st_reservoir['efficiency_dispatch'].tolist(),
                efficiency_store=st_reservoir['efficiency_store'].tolist(),
                marginal_cost=st_reservoir['marginal_cost'].to_list(),
                standing_loss=0, cyclic_state_of_charge=True,
                state_of_charge_initial=st_reservoir['state_of_charge_initial'].tolist(),
                inflow=inflow_reservoir)

    network.add("StorageUnit", st_battery.index, bus=st_battery['bus'].tolist(),
                carrier=st_battery['carrier'].tolist(), p_nom=st_battery['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_battery['max_hours'].to_list(),
                p_max_pu=st_battery['p_max_pu'].tolist(),
                efficiency_dispatch=st_battery['efficiency_dispatch'].tolist(),
                standing_loss=st_battery['standing_loss'].tolist())

    network.add("StorageUnit", st_other.index, bus=st_other['bus'].tolist(),
                carrier=st_other['carrier'].tolist(), p_nom=st_other['p_nom'].tolist(),
                p_nom_extendable=False, max_hours=st_other['max_hours'].to_list(),
                p_max_pu=st_other['p_max_pu'].tolist(),
                efficiency_dispatch=st_other['efficiency_dispatch'].tolist(),
                standing_loss=st_other['standing_loss'].tolist())

    # CO2 factors
    dict_carriers = {
        'Lignite': 1.06, 'Hard coal': 0.867, 'CCGT': 0.36, 'SCGT': 0.46, 'Oil': 0.65,
        'Biomass': 0.0, 'Biogas': 0.19656, 'BECCS': 0, 'Wind': 0, 'Wind offshore': 0,
        'PV': 0, 'Hydro': 0, 'Other RES': 0, 'Battery': 0, 'Other storage': 0,
        'Nuclear': 0, 'DSR': 0,
    }

    network.consistency_check()
    network.generators["co2_emissions"] = network.generators["carrier"].map(dict_carriers)
    network.storage_units["co2_emissions"] = network.storage_units["carrier"].map(dict_carriers)

    # export
    os.makedirs(os.path.dirname(out_nc_path), exist_ok=True)
    network.export_to_netcdf(out_nc_path)
    print(f"Saved EU-2030: {out_nc_path}")


# ---- run both builds, outputs under ./data ----
os.makedirs("./data", exist_ok=True)
build_network_eu_2023("./data/pypsa_eu_2023.xlsx", "./data/network_eu_2023.nc")
build_network_eu_2030("./data/pypsa_eu_2030.xlsx", "./data/network_eu_2030.nc")