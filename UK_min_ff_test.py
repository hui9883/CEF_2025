import numpy as np
import pandas as pd
import os
import pypsa
import datetime
from fuction.mef_energy_log_new import adjust_gen_by_demand
import re

print(pypsa.__version__)


# define the save dir
save_dir = "./results_min_cost_2023/"  # Case 1, Corresponds to step1 minimisation cost
# save_dir = "./results_min_CO2_2023/"  #Case 2, Corresponds to step2 minimisation emissions
# save_dir = "./results_min_cost_2030/"  #Case 3, Corresponds to step1 minimisation cost
# save_dir = "./results_min_CO2_2030/"  #Case 4, Corresponds to step2 minimisation emissions

if '_2023' in save_dir:
    file = './data/pypsa-uk-prototype_2023.xlsx'
elif '_2030' in save_dir:
    file = './data/pypsa-uk-prototype_2030.xlsx'
else:
    p = None

os.makedirs(save_dir, exist_ok=True)

#load data
load = pd.read_excel(file, sheet_name='load', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1,
                                                                                                      how='all').dropna()
pv = pd.read_excel(file, sheet_name='pv', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1,
                                                                                                  how='all').dropna()
wind = pd.read_excel(file, sheet_name='wind', parse_dates=['t'], index_col='t', usecols='A:M').dropna(axis=1,
                                                                                                      how='all').dropna()
wind_offshore = pd.read_excel(file, sheet_name='wind_offshore', parse_dates=['t'], index_col='t', usecols='A:M').dropna(
    axis=1, how='all').dropna()
chp = pd.read_excel(file, sheet_name='chp', parse_dates=['t'], index_col='t', usecols='A:AD').dropna(axis=1,
                                                                                                     how='all').dropna()
buses = pd.read_excel(file, sheet_name='buses', index_col=0)
lines = pd.read_excel(file, sheet_name='lines', index_col=0)
gen_pv = pd.read_excel(file, sheet_name='gen_pv', index_col=0)
gen_wind = pd.read_excel(file, sheet_name='gen_wind', index_col=0)#
gen_wind_offshore = pd.read_excel(file, sheet_name='gen_wind_offshore', index_col=0)
gen_gas = pd.read_excel(file, sheet_name='gen_gas', index_col=0)
gen_oil = pd.read_excel(file, sheet_name='gen_oil', index_col=0)
gen_nuclear = pd.read_excel(file, sheet_name='gen_nuclear', index_col=0)
gen_biomass = pd.read_excel(file, sheet_name='gen_biomass', index_col=0)
gen_biogas = pd.read_excel(file, sheet_name='gen_biogas', index_col=0)
gen_hydro = pd.read_excel(file, sheet_name='gen_hydro', index_col=0)
gen_chp = pd.read_excel(file, sheet_name='gen_chp', index_col=0)
st_battery_2023 = pd.read_excel(file, sheet_name='st_battery', index_col=0)
st_other_2023 = pd.read_excel(file, sheet_name='st_other', index_col=0)

#setting the network
network = pypsa.Network()
snapshot = load.index
network.set_snapshots(snapshot)
snapshot = np.arange(0, len(snapshot), 1)
network.add("Bus", buses.index, **buses.to_dict(orient="list"))
network.add("Line", name=lines.index, **lines.drop(columns=["name"]).to_dict(orient="list"))
network.lines['s_nom_extendable'] = False
network.add("Load", load.columns, bus=load.columns, p_set=load)
network.add("Generator", gen_pv['bus'], suffix='_PV', bus=gen_pv['bus'].to_list(),
            p_nom_extendable=False, p_nom=gen_pv['p_nom'].to_list(), carrier='PV',
            marginal_cost=gen_pv['marginal_cost'].to_list(), capital_cost=150e3, p_max_pu=pv)  #, capital_cost=150e3
network.add("Generator", gen_wind['bus'], suffix='_Wind', bus=gen_wind['bus'].to_list(),
            p_nom_extendable=False, p_nom=gen_wind['p_nom'].to_list(), carrier='Wind',
            marginal_cost=gen_wind['marginal_cost'].to_list(), capital_cost=150e3, p_max_pu=wind)  #, capital_cost=150e3
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
            marginal_cost=gen_chp['marginal_cost'].to_list(), p_max_pu=chp, #p_min_pu=chp * 0.9,
            efficiency=gen_chp['efficiency'].to_list())
network.add("Bus", st_battery_2023.index, carrier=st_battery_2023['carrier'].to_list())
network.add("Store", st_battery_2023.index, bus=st_battery_2023.index.to_list(), carrier=st_battery_2023['carrier'].to_list(),
            e_nom=(st_battery_2023['p_nom'] * st_battery_2023['max_hours']).to_list(),
            max_hours=4, e_nom_extendable=False) # True capital_cost=31096,
network.add("Link", st_battery_2023.index, suffix='_charger', bus0=st_battery_2023['bus'].to_list(),
            bus1=st_battery_2023.index.to_list(), carrier=st_battery_2023['carrier'].to_list(), efficiency=1, marginal_cost=0,
            p_nom=st_battery_2023['p_nom'].to_list(), p_nom_min=st_battery_2023['p_nom'].to_list(), p_nom_extendable=False) #capital_cost=28534,
network.add("Link", st_battery_2023.index, suffix='_discharger', bus0=st_battery_2023.index.to_list(),
            bus1=st_battery_2023['bus'].to_list(), carrier=st_battery_2023['carrier'].to_list(),
            efficiency=st_battery_2023['efficiency_dispatch'].to_list(), p_nom=st_battery_2023['p_nom'].to_list(),
            p_nom_min=st_battery_2023['p_nom'].to_list(), p_nom_extendable=False)
network.add("Bus", st_other_2023.index, carrier=st_other_2023['carrier'].to_list())
network.add("Store", st_other_2023.index, bus=st_other_2023.index.to_list(), carrier=st_other_2023['carrier'].to_list(),
            e_nom=(st_other_2023['p_nom'] * st_other_2023['max_hours']).to_list(), max_hours=4,
            e_nom_extendable=False) #capital_cost=37720,
network.add("Link", st_other_2023.index, suffix='_charger', bus0=st_other_2023['bus'].to_list(),
            bus1=st_other_2023.index.to_list(), carrier=st_other_2023['carrier'].to_list(), p_nom_min=st_other_2023['p_nom'].to_list(),
            efficiency=1, marginal_cost=0, p_nom=st_other_2023['p_nom'].to_list(), p_nom_extendable=False) #capital_cost=2358/2,
network.add("Link", st_other_2023.index, suffix='_discharger', bus0=st_other_2023.index.to_list(),
            bus1=st_other_2023['bus'].to_list(), carrier=st_other_2023['carrier'].to_list(), marginal_cost=0,
            efficiency=st_other_2023['efficiency_dispatch'].to_list(), p_nom=st_other_2023['p_nom'].to_list(),
            p_nom_min=st_other_2023['p_nom'].to_list(), p_nom_extendable=False) #capital_cost=2358/2,
# Identify components in the PyPSA network
battery_store_ids = [store for store in network.stores.index if 'Battery' in store]
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
# Map carbon emission factors to generators
network.generators["co2_emissions"] = network.generators["carrier"].map(carbon_emission_factors)
# Create optimization model from the PyPSA network
n = network.optimize.create_model()
# Add constraints: Ensure battery storage is zero at 23:00 each day
for store in battery_store_ids:
    for day in pd.date_range("2030-01-01", "2030-12-31", freq="D"):
        today_23pm = day + pd.Timedelta(hours=23)
        n.add_constraints(
            n.variables["Store-e"].loc[today_23pm, store] == 0,#0-20%; 0-50%; <=
            name=f"EndOfDayZeroStorage_{store}_{day.date()}"
        )

end_of_2030 = datetime.datetime(2030, 12, 31, 23)
for store in network.stores.index:
    n.add_constraints(
        n.variables["Store-e"].loc[end_of_2030, store] == 0,
        name=f"FinalZeroStorage_{store}_{end_of_2030.strftime('%Y%m%d_%H')}"
    )
# objective function
if '_min_cost' in save_dir:
    # --- Step 1: Minimize Operational Costs ---
    generator_power = n.variables["Generator-p"]
    marginal_cost = network.generators["marginal_cost"]
    co2_emissions = network.generators["co2_emissions"]
    aligned_marginal_cost = (marginal_cost.reindex(generator_power.coords["Generator"].to_index(), fill_value=0) +
                             0.000001 * co2_emissions.reindex(generator_power.coords["Generator"].to_index(),
                                                              fill_value=0))
    generation_cost = (generator_power * aligned_marginal_cost).sum().sum()
    n.objective = generation_cost
    # Solve the optimization model
    network.optimize.solve_model(solver_name='gurobi', solver_options={'Threads': 10})
    generator_power = network.generators_t.p
    aligned_marginal_cost = network.generators["marginal_cost"].reindex(generator_power.columns)
    generation_cost_cal = (network.generators_t.p * aligned_marginal_cost).sum().sum()
    print("Min Cost Optimization -> Total Operation Cost:", generation_cost_cal)
    min_cost_total_emissions = (network.generators_t.p * network.generators["co2_emissions"]).sum().sum()
    print("Min Cost Optimization -> Total Emissions:", min_cost_total_emissions)
elif '_min_CO2' in save_dir:
    # --- Step 2: Minimize Carbon Emissions ---
    # Define carbon emissions minimization objective function
    generator_power = n.variables["Generator-p"]
    co2_emissions = network.generators["co2_emissions"]
    marginal_cost = network.generators["marginal_cost"]
    aligned_co2_emissions = (co2_emissions.reindex(generator_power.coords["Generator"].to_index()) +
                             (0.000001 * (
                                 marginal_cost.reindex(generator_power.coords["Generator"].to_index(), fill_value=0))))
    emissions_objective = (generator_power * aligned_co2_emissions).sum(["snapshot", "Generator"])
    # Set the objective to minimize carbon emissions
    n.objective = emissions_objective
    # Solve the optimization model for emissions minimization
    network.optimize.solve_model(solver_name='gurobi', solver_options={'Threads': 10})
    # Calculate and print results for emissions minimization
    marginal_cost = network.generators["marginal_cost"]
    aligned_marginal_cost = marginal_cost.reindex(generator_power.coords["Generator"].to_index())
    generation_cost_cal = (network.generators_t.p * aligned_marginal_cost).sum().sum()
    print("Min Emissions Optimization -> Total Operation Cost:", generation_cost_cal)
    min_emissions_total_emissions = (network.generators_t.p * network.generators["co2_emissions"]).sum().sum()
    print("Min Emissions Optimization -> Total Emissions:", min_emissions_total_emissions)
else:
    p = None

# Export network data to a NetCDF file
network.export_to_netcdf(f"{save_dir}network_ff_constrained_time.nc")

# Export power-related results to CSV
network.loads_t.p.to_csv(f"{save_dir}demand_p.csv", header=True)  # Load demand
network.links_t.p0.to_csv(f"{save_dir}links_p0_results.csv", header=True)  # Charge energy
network.links_t.p1.to_csv(f"{save_dir}links_p1_results.csv", header=True)  # Discharge energy
network.stores_t.p.to_csv(f"{save_dir}store_p_results.csv", header=True)  # Storage power
network.buses_t.p.to_csv(f"{save_dir}buses_p_results.csv", header=True)  # Bus power
network.links.p_nom_opt.to_csv(f"{save_dir}links_p_nom_opt.csv", header=True)  # Optimal nominal power for links
network.stores_t.e.to_csv(f"{save_dir}stores_e.csv", header=True)  # Energy in stores
network.stores.e_nom_opt.to_csv(f"{save_dir}stores_e_nom_opt.csv", header=True)  # Optimal nominal energy for stores

# Group storage energy by carrier and save
store_by_carrier = network.stores_t.e.T.groupby(network.stores.carrier).sum().T
store_by_carrier.to_csv(f"{save_dir}store_e_carrier_results.csv", header=True)

# Group generator power by carrier and save
p_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum().T
p_by_carrier.to_csv(f"{save_dir}gen_p_carrier_results.csv", header=True)

# Group storage power by carrier and save
store_by_carrier = network.stores_t.p.T.groupby(network.stores.carrier).sum().T
store_by_carrier.to_csv(f"{save_dir}store_p_carrier_results.csv", header=True)

# # Calculate and save dynamic maximum output for generators
p_max_pu_full = network.generators_t.p_max_pu.reindex(
    columns=network.generators.index
)
missing_generators = network.generators.index.difference(network.generators_t.p_max_pu.columns)

for gen in missing_generators:
    static_val = network.generators.at[gen, "p_max_pu"]  # 取该机组的静态占比
    p_max_pu_full[gen] = static_val

snapshot_max_output = p_max_pu_full.multiply(network.generators["p_nom"], axis=1)

# Calculate remaining output adjusted for efficiency
network.generators_t['max_output'] = snapshot_max_output
network.generators_t['remain_output'] = (network.generators_t['max_output'] - network.generators_t.p)
network.generators_t['remain_output'] *= network.generators['efficiency']

# Group remaining output by bus and carrier
remain_output = network.generators_t['remain_output'].T
remain_output.index = remain_output.index.astype(str)
remain_output['bus'] = remain_output.index.map(network.generators['bus'])
remain_output['carrier'] = remain_output.index.map(network.generators['carrier'])
remain_output['bus_carrier'] = remain_output['bus'] + '_' + remain_output['carrier']
p_by_bus_carrier = remain_output.groupby('bus_carrier').sum().drop(columns=['bus', 'carrier'], errors='ignore')
p_by_bus_carrier = p_by_bus_carrier.where(p_by_bus_carrier >= 0.1, 0).T
p_by_bus_carrier.to_csv(f"{save_dir}p_by_bus_carrier.csv", index=True)

# Group generator remaining power by carrier and save
re_p_by_carrier = network.generators_t['remain_output'].T.groupby(network.generators.carrier).sum().T
re_p_by_carrier = re_p_by_carrier.where(re_p_by_carrier >= 0.1, 0)
re_p_by_carrier.to_csv(f"{save_dir}re_p_carrier_results.csv", header=True)

# Group generator output by bus and carrier
generators_tp = network.generators_t.p.T
generators_tp.index = generators_tp.index.astype(str)
generators_tp['bus'] = generators_tp.index.map(network.generators['bus'])
generators_tp['carrier'] = generators_tp.index.map(network.generators['carrier'])
generators_tp['bus_carrier'] = generators_tp['bus'] + '_' + generators_tp['carrier']
gen_by_bus_carrier = generators_tp.groupby('bus_carrier').sum().drop(columns=['bus', 'carrier'], errors='ignore')
gen_by_bus_carrier = gen_by_bus_carrier.where(gen_by_bus_carrier >= 0.1, 0).T
gen_by_bus_carrier.to_csv(f"{save_dir}gen_by_bus_carrier.csv", index=True)

regions = ['EN_NorthEast','EN_NorthWest','EN_Yorkshire',
           'EN_EastMidlands','EN_WestMidlands',
           'EN_East','EN_London','EN_SouthEast',
           'EN_SouthWest','EN_Wales','Scotland',
           'NorthernIreland']

df_gen_demand = adjust_gen_by_demand(gen_by_bus_carrier,regions, network.loads_t.p)
df_gen_demand.to_csv(f"{save_dir}gen_demand_by_bus_carrier.csv", header=True)

region_pattern = r'(' + '|'.join(regions) + r')_'
df_gen_demand.columns = [re.sub(region_pattern, '', col) for col in df_gen_demand.columns]
p_demand_by_carrier = df_gen_demand.groupby(df_gen_demand.columns, axis=1).sum()
p_demand_by_carrier.to_csv(f"{save_dir}gen_demand_carrier_results.csv", header=True)

# Process network.generators_t.p
generators_t_p = network.generators_t.p.copy()
generators = network.generators.copy()
# Add 'carrier' and 'marginal_cost' rows to generators_t.p
generators_t_p.loc['carrier'] = generators_t_p.columns.map(generators['carrier'])
generators_t_p.loc['marginal_cost'] = generators_t_p.columns.map(generators['marginal_cost'])
# Sort columns by carrier and marginal cost
sorted_columns = (
    generators[['carrier', 'marginal_cost']]
    .sort_values(by=['carrier', 'marginal_cost'], ascending=[True, True])
    .index
)
generators_t_p = generators_t_p[sorted_columns]
# Ensure numerical data types are correct
for row_name in generators_t_p.index:
    if row_name not in ['carrier']:  # Keep 'carrier' row as string
        generators_t_p.loc[row_name] = pd.to_numeric(
            generators_t_p.loc[row_name], errors='coerce'
        )
# Save the result
output_file_p = f"{save_dir}sorted_generators_t_p.csv"
generators_t_p.to_csv(output_file_p)
print("done")
