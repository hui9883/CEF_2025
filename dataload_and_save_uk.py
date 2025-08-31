import numpy as np
import pandas as pd
import os
import pypsa
import datetime
print(pypsa.__version__)

def build_and_save_network_uk(xlsx_path: str, out_nc_path: str):
    # load data
    load = pd.read_excel(xlsx_path, sheet_name='load', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1, how='all').dropna()
    pv = pd.read_excel(xlsx_path, sheet_name='pv', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1, how='all').dropna()
    wind = pd.read_excel(xlsx_path, sheet_name='wind', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1, how='all').dropna()
    wind_offshore = pd.read_excel(xlsx_path, sheet_name='wind_offshore', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1, how='all').dropna()
    chp = pd.read_excel(xlsx_path, sheet_name='chp', parse_dates=['t'], index_col='t', usecols='A:AD').dropna(axis=1, how='all').dropna()

    buses = pd.read_excel(xlsx_path, sheet_name='buses', index_col=0)
    lines = pd.read_excel(xlsx_path, sheet_name='lines', index_col=0)

    gen_pv = pd.read_excel(xlsx_path, sheet_name='gen_pv', index_col=0)
    gen_wind = pd.read_excel(xlsx_path, sheet_name='gen_wind', index_col=0)
    gen_wind_offshore = pd.read_excel(xlsx_path, sheet_name='gen_wind_offshore', index_col=0)
    gen_gas = pd.read_excel(xlsx_path, sheet_name='gen_gas', index_col=0)
    gen_oil = pd.read_excel(xlsx_path, sheet_name='gen_oil', index_col=0)
    gen_nuclear = pd.read_excel(xlsx_path, sheet_name='gen_nuclear', index_col=0)
    gen_biomass = pd.read_excel(xlsx_path, sheet_name='gen_biomass', index_col=0)
    gen_biogas = pd.read_excel(xlsx_path, sheet_name='gen_biogas', index_col=0)
    gen_hydro = pd.read_excel(xlsx_path, sheet_name='gen_hydro', index_col=0)
    gen_chp = pd.read_excel(xlsx_path, sheet_name='gen_chp', index_col=0)

    st_battery = pd.read_excel(xlsx_path, sheet_name='st_battery', index_col=0)
    st_other = pd.read_excel(xlsx_path, sheet_name='st_other', index_col=0)

    # setting the network
    network = pypsa.Network()
    network.set_snapshots(load.index)

    network.add("Bus", buses.index, **buses.to_dict(orient="list"))
    network.add("Line", name=lines.index, **lines.drop(columns=["name"]).to_dict(orient="list"))
    network.lines['s_nom_extendable'] = False

    network.add("Load", load.columns, bus=load.columns, p_set=load)

    network.add("Generator", gen_pv['bus'], suffix='_PV', bus=gen_pv['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_pv['p_nom'].to_list(), carrier='PV',
                marginal_cost=gen_pv['marginal_cost'].to_list(), capital_cost=150e3, p_max_pu=pv)

    network.add("Generator", gen_wind['bus'], suffix='_Wind', bus=gen_wind['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_wind['p_nom'].to_list(), carrier='Wind',
                marginal_cost=gen_wind['marginal_cost'].to_list(), capital_cost=150e3, p_max_pu=wind)

    network.add("Generator", gen_wind_offshore['bus'], suffix='_Wind_offshore',
                bus=gen_wind_offshore['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_wind_offshore['p_nom'].to_list(), carrier='Wind offshore',
                marginal_cost=gen_wind_offshore['marginal_cost'].to_list(), p_max_pu=wind_offshore)

    network.add("Generator", gen_gas.index, bus=gen_gas['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_gas['p_nom'].to_list(), carrier=gen_gas['carrier'].to_list(),
                marginal_cost=gen_gas['marginal_cost'].to_list(), efficiency=gen_gas['efficiency'].to_list(), p_max_pu=1)

    network.add("Generator", gen_oil.index, bus=gen_oil['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_oil['p_nom'].to_list(), carrier='Oil', marginal_cost=gen_oil['marginal_cost'].to_list(),
                efficiency=gen_oil['efficiency'].to_list(), p_max_pu=1)

    network.add("Generator", gen_nuclear.index, bus=gen_nuclear['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_nuclear['p_nom'].to_list(), carrier='Nuclear',
                marginal_cost=gen_nuclear['marginal_cost'].to_list(),
                efficiency=gen_nuclear['efficiency'].to_list(), p_max_pu=gen_nuclear['p_max_pu'].to_list(),
                p_min_pu=gen_nuclear['p_min_pu'].to_list())

    network.add("Generator", gen_biomass.index, bus=gen_biomass['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_biomass['p_nom'].to_list(), carrier=gen_biomass['carrier'].to_list(),
                marginal_cost=gen_biomass['marginal_cost'].to_list(), efficiency=gen_biomass['efficiency'].to_list(),
                p_max_pu=gen_biomass['p_max_pu'].to_list())

    network.add("Generator", gen_biogas['bus'], suffix='_Biogas', bus=gen_biogas['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_biogas['p_nom'].to_list(), carrier='Biogas',
                marginal_cost=gen_biogas['marginal_cost'].to_list(), efficiency=gen_biogas['efficiency'].to_list(),
                p_max_pu=gen_biogas['p_max_pu'].to_list())

    network.add("Generator", gen_hydro['bus'], suffix='_Hydro', bus=gen_hydro['bus'].to_list(),
                p_nom_extendable=False, p_nom=gen_hydro['p_nom'].to_list(), carrier='Hydro',
                marginal_cost=gen_hydro['marginal_cost'].to_list(), p_max_pu=gen_hydro['p_max_pu'].to_list())

    network.add("Generator", gen_chp.index, bus=gen_chp['bus'].to_list(), p_nom_extendable=False,
                p_nom=gen_chp['p_nom'].to_list(), carrier=gen_chp['carrier'].to_list(),
                marginal_cost=gen_chp['marginal_cost'].to_list(), p_max_pu=chp,
                efficiency=gen_chp['efficiency'].to_list())

    # Storage (battery & other)
    network.add("Bus", st_battery.index, carrier=st_battery['carrier'].to_list())
    network.add("Store", st_battery.index, bus=st_battery.index.to_list(), carrier=st_battery['carrier'].to_list(),
                e_nom=(st_battery['p_nom'] * st_battery['max_hours']).to_list(),
                max_hours=4, e_nom_extendable=False)

    network.add("Link", st_battery.index, suffix='_charger', bus0=st_battery['bus'].to_list(),
                bus1=st_battery.index.to_list(), carrier=st_battery['carrier'].to_list(), efficiency=1, marginal_cost=0,
                p_nom=st_battery['p_nom'].to_list(), p_nom_min=st_battery['p_nom'].to_list(), p_nom_extendable=False)

    network.add("Link", st_battery.index, suffix='_discharger', bus0=st_battery.index.to_list(),
                bus1=st_battery['bus'].to_list(), carrier=st_battery['carrier'].to_list(),
                efficiency=st_battery['efficiency_dispatch'].to_list(), p_nom=st_battery['p_nom'].to_list(),
                p_nom_min=st_battery['p_nom'].to_list(), p_nom_extendable=False)

    network.add("Bus", st_other.index, carrier=st_other['carrier'].to_list())
    network.add("Store", st_other.index, bus=st_other.index.to_list(), carrier=st_other['carrier'].to_list(),
                e_nom=(st_other['p_nom'] * st_other['max_hours']).to_list(), max_hours=4,
                e_nom_extendable=False)

    network.add("Link", st_other.index, suffix='_charger', bus0=st_other['bus'].to_list(),
                bus1=st_other.index.to_list(), carrier=st_other['carrier'].to_list(),
                p_nom_min=st_other['p_nom'].to_list(), efficiency=1, marginal_cost=0,
                p_nom=st_other['p_nom'].to_list(), p_nom_extendable=False)

    network.add("Link", st_other.index, suffix='_discharger', bus0=st_other.index.to_list(),
                bus1=st_other['bus'].to_list(), carrier=st_other['carrier'].to_list(), marginal_cost=0,
                efficiency=st_other['efficiency_dispatch'].to_list(), p_nom=st_other['p_nom'].to_list(),
                p_nom_min=st_other['p_nom'].to_list(), p_nom_extendable=False)

    # Emission factors
    carbon_emission_factors = {
        'Wind': 0,
        'PV': 0,
        'Wind offshore': 0,
        'Hydro': 0,
        'Nuclear': 0,
        'Biogas_CHP': 0.19656,
        'Biogas': 0.19656,
        'CCGT': 0.36,
        'Biomass': 0,
        'Biomass_CHP': 0,
        'SCGT': 0.46,
        'SCGT_CHP': 0.46,
        'Oil': 0.65
    }
    network.generators["co2_emissions"] = network.generators["carrier"].map(carbon_emission_factors)

    # Export
    os.makedirs(os.path.dirname(out_nc_path), exist_ok=True)
    network.export_to_netcdf(out_nc_path)
    print(f"Saved: {out_nc_path}")

# 2023
build_and_save_network_uk("./data/pypsa_uk_2023.xlsx", "./data/network_2023.nc")
# 2030
build_and_save_network_uk("./data/pypsa_uk_2030.xlsx", "./data/network_2030.nc")
