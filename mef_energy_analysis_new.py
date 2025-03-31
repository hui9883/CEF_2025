import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import os
import pypsa
from fuction.mef_energy_log_new import cycle_analysis, calculate_aef
from fuction.mef_national_energy_log_new import national_cycle_analysis, national_mef_analysis, national_aef_analysis
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fuction.rain_flow_new import cycle_extraction
matplotlib.use('TkAgg')

pre_dic = "./results_min_cost_2023/"  # Case 1, Corresponds to step1 minimisation cost
# pre_dic = "./results_min_CO2_2023/"  #Case 2, Corresponds to step2 minimisation emissions
# pre_dic = "./results_min_cost_2030/"  #Case 3, Corresponds to step1 minimisation cost
# pre_dic = "./results_min_CO2_2030/"  #Case 4, Corresponds to step2 minimisation emissions

save_dir = f"{pre_dic}analysis_output/"
os.makedirs(save_dir, exist_ok=True)
if '_min_cost_2023' in pre_dic:
    p = 'min_cost_2023'
elif '_min_CO2_2023' in pre_dic:
    p = 'min_emission_2023'
elif '_min_cost_2030' in pre_dic:
    p = 'min_cost_2030'
elif '_min_CO2' in pre_dic:
    p = 'min_emission_2030'
else:
    p = None

cmap_t = 'jet'
regions = ['EN_NorthEast','EN_NorthWest','EN_Yorkshire',
           'EN_EastMidlands','EN_WestMidlands',
           'EN_East','EN_London','EN_SouthEast',
           'EN_SouthWest','EN_Wales','Scotland',
           'NorthernIreland']
resource_usage = {
    "PV": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Wind": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Wind offshore": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Hydro": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Nuclear": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Biomass": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Biomass_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Biogas": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Biogas_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "CCGT": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "SCGT_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "SCGT": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Oil": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
    "Others": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}
}
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

# network = pypsa.Network(pre_dic+'network.nc')
network = pypsa.Network(pre_dic+'network_ff_constrained_time.nc')
df = pd.read_csv(pre_dic+'store_e_carrier_results.csv')
df_capacity = pd.read_csv(pre_dic+'stores_e.csv')
df_gen = pd.read_csv(pre_dic+'gen_p_carrier_results.csv')
df_gen_demand = pd.read_csv(pre_dic+'gen_demand_carrier_results.csv')
df_gen_remain = pd.read_csv(pre_dic+'p_by_bus_carrier.csv')
df_gen_remain_carrier = pd.read_csv(pre_dic+'re_p_carrier_results.csv')
df_storage_links = pd.read_csv(pre_dic+'links_p1_results.csv')
df_gen_bus = pd.read_csv(pre_dic+'buses_p_results.csv')
df_gen_bus_carrier_total = pd.read_csv(pre_dic+'gen_demand_by_bus_carrier.csv')
df_gen_bus_carrier_region = pd.read_csv(pre_dic+'gen_by_bus_carrier.csv')
load = pd.read_csv(pre_dic+'demand_p.csv')
df_gen_remain_new = df_gen_remain.copy()
print(network.lines.columns.to_list())
all_carriers = df_gen.columns.to_list()

battery_bus = [s + '_Battery' for s in regions]
ES_bus = [s + '_OtherStorage' for s in regions]
df['soc_batt'] = df['Battery']/df['Battery'].max()
df['soc_ldes'] = df['ES']/df['ES'].max()

battery_charger = [s + '_charger' for s in battery_bus]
battery_discharger = [s + '_discharger' for s in battery_bus]

ES_charger = [s + '_charger' for s in ES_bus]
ES_discharger = [s + '_discharger' for s in ES_bus]

if '_ext' in pre_dic:
    battery_charger_2030 = [s + '_2030_charger' for s in battery_bus]
    battery_discharger_2030 = [s + '_2030_discharger' for s in battery_bus]
    ES_charger_2030 = [s + '_2030_charger' for s in ES_bus]
    ES_discharger_2030 = [s + '_2030_discharger' for s in ES_bus]
    df_storage_links.loc[:, battery_charger] = df_storage_links[battery_charger].to_numpy() + df_storage_links[
        battery_charger_2030].to_numpy()
    df_storage_links.loc[:, battery_discharger] = df_storage_links[battery_discharger].to_numpy() + df_storage_links[
        battery_discharger_2030].to_numpy()
    df_storage_links.loc[:, ES_charger] = df_storage_links[ES_charger].to_numpy() + df_storage_links[
        ES_charger_2030].to_numpy()
    df_storage_links.loc[:, ES_discharger] = df_storage_links[ES_discharger].to_numpy() + df_storage_links[
        ES_discharger_2030].to_numpy()

df_storage_links[battery_charger]=-df_storage_links[battery_charger]
df_storage_links[ES_charger]=-df_storage_links[ES_charger]

# print(df_links_ffmin)
df_storage_links['bus_charger'] = df_storage_links[battery_charger].sum(axis=1)
df_storage_links['bus_discharger'] = df_storage_links[battery_discharger].sum(axis=1)

# print(df_links_ffmin)
df_storage_links['es_bus_charger'] = df_storage_links[ES_charger].sum(axis=1)
df_storage_links['es_bus_discharger'] = df_storage_links[ES_discharger].sum(axis=1)

df_copy = df.copy()
df_copy['snapshot'] = pd.to_datetime(df['snapshot'])
new_snapshot = df_copy['snapshot'].iloc[0] - pd.Timedelta(hours=1)
new_row = pd.DataFrame({col: [0] if col != 'snapshot' else [new_snapshot] for col in df_copy.columns})
df_copy = pd.concat([new_row, df_copy], ignore_index=True)
process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_copy['soc_batt'])
process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_copy['soc_ldes'])


# Create the histogram plot
fig_his, ax_his = plt.subplots()
ax_his.hist(charg_xy, bins=int(len(charg_xy)), color='blue', edgecolor='black', alpha=0.7)  # Histogram with specified style
# Set title and axis labels
ax_his.set_title('Histogram of Battery Cycle Durations')
ax_his.set_xlabel('Cycle Duration [hours]')  # X-axis label
ax_his.set_ylabel('Frequency')               # Y-axis label
# Adding a legend
ax_his.legend(['Battery Cycle Duration'], loc='upper right')  # Optional, may not be necessary for a single histogram
# Save the figure
fig_his.savefig(f"{save_dir}battery_histogram{p}.jpg", format='jpg', dpi=300)

# cef
(cef_bat_t, cef_es_t, unit_ccost_bat_cycle, unit_ccost_es_cycle, unit_dcost_bat_cycle, unit_dcost_es_cycle, unit_cost_bat_cycle, unit_cost_es_cycle,
    carbon_intensity_bat_cycle, carbon_intensity_es_cycle, charge_energy_cost_bat, charge_energy_cost_es, emissions_bat, emissions_es,
 co2_emissions_factor_bat, co2_emissions_factor_es, co2_emissions_bat_cycle, co2_emissions_es_cycle, energy_charge_cycle_bat, energy_charge_cycle_es,
 energy_discharge_cycle_bat, energy_discharge_cycle_es, emissions_charged_bat, emissions_charged_es, emissions_discharged_bat, emissions_discharged_es,
 co2_delta_emissions, co2_delta_emissions_es, resource_usage) = national_cycle_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, df_gen,
                                                                               df_storage_links, df_gen_remain_carrier, resource_usage)

# mef
(mef_bat_t, mef_es_t,carbon_intensity_bat_cycle_mef, carbon_intensity_es_cycle_mef, emissions_bat_mef, emissions_es_mef,
 co2_emissions_factor_bat_mef, co2_emissions_factor_es_mef, co2_emissions_bat_cycle_mef, co2_emissions_es_cycle_mef, energy_charge_cycle_bat_mef, energy_charge_cycle_es_mef,
 energy_discharge_cycle_bat_mef, energy_discharge_cycle_es_mef, emissions_charged_bat_mef, emissions_charged_es_mef, emissions_discharged_bat_mef, emissions_discharged_es_mef,
 co2_delta_emissions_mef, co2_delta_emissions_es_mef) = national_mef_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, df_gen,
                                                                               df_storage_links)

# aef
(aef_bat_t, aef_es_t,carbon_intensity_bat_cycle_aef, carbon_intensity_es_cycle_aef, emissions_bat_aef, emissions_es_aef,
 co2_emissions_factor_bat_aef, co2_emissions_factor_es_aef, co2_emissions_bat_cycle_aef, co2_emissions_es_cycle_aef, energy_charge_cycle_bat_aef, energy_charge_cycle_es_aef,
 energy_discharge_cycle_bat_aef, energy_discharge_cycle_es_aef, emissions_charged_bat_aef, emissions_charged_es_aef, emissions_discharged_bat_aef, emissions_discharged_es_aef,
 co2_delta_emissions_aef, co2_delta_emissions_es_aef) = national_aef_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, df_gen,
                                                                               df_storage_links)

mean_bat = np.mean(co2_emissions_bat_cycle)
mean_es = np.mean(co2_emissions_es_cycle)

mean_bat_mef = np.mean(co2_emissions_bat_cycle_mef)
mean_es_mef = np.mean(co2_emissions_es_cycle_mef)

mean_bat_aef = np.mean(co2_emissions_bat_cycle_aef)
mean_es_aef = np.mean(co2_emissions_es_cycle_aef)


# Create a 1×3 subplot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# Define titles and corresponding y-axis labels
if '_min_cost_2023' in pre_dic:
    titles = [r"$\mathbf{(a)}$" + f" MEF_{p}", r"$\mathbf{(b)}$" + f" AEF_{p}", r"$\mathbf{(c)}$" + f" CEF_{p}"]
elif '_min_cost_2030' in pre_dic:
    titles = [r"$\mathbf{(d)}$" + f" MEF_{p}", r"$\mathbf{(e)}$" + f" AEF_{p}", r"$\mathbf{(f)}$" + f" CEF_{p}"]
else:
    titles = [f"MEF_{p}", f"AEF_{p}", f"CEF_{p}"]
y_labels = ['MEF (tCO2/MWh)', 'AEF (tCO2/MWh)', 'CEF (tCO2/MWh)']

data_sets = [
    (carbon_intensity_bat_cycle_mef, co2_emissions_bat_cycle_mef, carbon_intensity_es_cycle_mef, co2_emissions_es_cycle_mef, mean_bat_mef, mean_es_mef),
    (carbon_intensity_bat_cycle_aef, co2_emissions_bat_cycle_aef, carbon_intensity_es_cycle_aef, co2_emissions_es_cycle_aef, mean_bat_aef, mean_es_aef),
    (carbon_intensity_bat_cycle, co2_emissions_bat_cycle, carbon_intensity_es_cycle, co2_emissions_es_cycle, mean_bat, mean_es)
]

# Plot the three subplots sequentially
for idx, (ax, (title, (bat_x, bat_y, es_x, es_y, mean_bat, mean_es))) in enumerate(zip(axes, zip(titles, data_sets))):
    scatter_bat = ax.scatter(bat_x, bat_y, label='Bat', marker='o', alpha=0.7)
    scatter_es = ax.scatter(es_x, es_y, label='LDES', marker='s', alpha=0.7)

    # Get the colors of the scatter points
    color_bat = scatter_bat.get_facecolors()[0]
    color_es = scatter_es.get_facecolors()[0]

    # Add dashed lines to indicate the average values
    ax.axhline(mean_bat, color=color_bat, linestyle='--', alpha=0.7, label='Avg Bat')
    ax.axhline(mean_es, color=color_es, linestyle='--', alpha=0.7, label='Avg LDES')

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(y_labels[idx], fontsize=12)  # Set different y-labels according to the index
    ax.set_xlabel('Cycle carbon intensity (tCO2/MWh)', fontsize=12)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.invert_xaxis()  # Invert the x-axis
    ax.set_ylim(-0.5, 0.2)

# Adjust layout and save the figure
plt.tight_layout()
fig.savefig(f"{save_dir}emission factors_{p}.jpg", format='jpg', dpi=300)
# plt.show()

unit_ccost_bat_cycle = np.array(unit_ccost_bat_cycle)  # Convert to a NumPy array
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

# Create a scatter plot
fig, ax = plt.subplots(figsize=(5, 4))
# Plot Bat (battery) data points
scatter_bat = ax.scatter(co2_emissions_bat_cycle, unit_ccost_bat_cycle, label='Bat', marker='o', alpha=0.7)
# Plot LDES (long-duration energy storage) data points
scatter_es = ax.scatter(co2_emissions_es_cycle, unit_ccost_es_cycle, label='LDES', marker='s', alpha=0.7)
ax.set_xlim(-0.5, 0.2)
# Invert the x-axis so that it goes from high to low
ax.invert_xaxis()
ax.set_ylim(-2, 85)
# Set the axis labels and title
ax.set_xlabel('CEF (tCO2/MWh)', fontsize=14)
ax.set_ylabel('charging cost factor (£/MWh)', fontsize=14)
ax.set_title(f"charged_cost_{p}", fontsize=14)
# Add legend and grid
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, linestyle='--', alpha=0.7)
# Obtain the maximum and minimum values of the CEF data
cef_values = np.concatenate((co2_emissions_bat_cycle, co2_emissions_es_cycle))
cef_max = np.max(cef_values)
cef_min = np.min(cef_values)
# Add a vertical dashed line: maximum CEF value
ax.axvline(x=cef_max, linestyle='--', color='gray', linewidth=1)
# Add a vertical dashed line: minimum CEF value
ax.axvline(x=cef_min, linestyle='--', color='gray', linewidth=1)
# Add a vertical dashed line: CEF=0
ax.axvline(x=0, linestyle='--', color='gray', linewidth=1)
# Add a horizontal dashed line: CCF=0
ax.axhline(y=0, linestyle='--', color='gray', linewidth=1)
plt.tight_layout()
fig.savefig(f"{save_dir}charging cost vs emission_{p}.jpg", format='jpg', dpi=300)


# Create a scatter plot
fig, ax = plt.subplots(figsize=(5, 4))
# Plot Bat (battery) data points
scatter_bat_d = ax.scatter(co2_emissions_bat_cycle, unit_dcost_bat_cycle, label='Bat', marker='o', alpha=0.7)
# Plot LDES (long-duration energy storage) data points
scatter_es_d = ax.scatter(co2_emissions_es_cycle, unit_dcost_es_cycle, label='LDES', marker='s', alpha=0.7)
ax.set_xlim(-0.5, 0.2)
# Invert the x-axis so that it goes from high to low
ax.invert_xaxis()
ax.set_ylim(-2, 85)
# Set the axis labels and title
ax.set_xlabel('CEF (tCO2/MWh)', fontsize=14)
ax.set_ylabel('discharging cost factor (£/MWh)', fontsize=14)
ax.set_title(f"discharged_cost_{p}", fontsize=14)
# Add legend and grid
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, linestyle='--', alpha=0.7)
# Get the maximum and minimum CEF values (assuming CEF data are stored in co2_emissions_bat_cycle and co2_emissions_es_cycle)
cef_values = np.concatenate((co2_emissions_bat_cycle, co2_emissions_es_cycle))
cef_max = np.max(cef_values)
cef_min = np.min(cef_values)
# Add a vertical dashed line: maximum CEF value
ax.axvline(x=cef_max, linestyle='--', color='gray', linewidth=1)
# Add a vertical dashed line: minimum CEF value
ax.axvline(x=cef_min, linestyle='--', color='gray', linewidth=1)
# Add a vertical dashed line: CEF=0
ax.axvline(x=0, linestyle='--', color='gray', linewidth=1)
# Add a horizontal dashed line: CCF=0
ax.axhline(y=0, linestyle='--', color='gray', linewidth=1)
plt.tight_layout()
fig.savefig(f"{save_dir}discharging cost vs emission_{p}.jpg", format='jpg', dpi=300)
# plt.show()

# Battery quadrant classification
bat_q1 = (unit_cost_bat_cycle > 0) & (co2_emissions_bat_cycle > 0)   # Quadrant 1: cost>0, co2>0
bat_q2 = (unit_cost_bat_cycle > 0) & (co2_emissions_bat_cycle <= 0)  # Quadrant 2: cost>0, co2<=0
bat_q3 = (unit_cost_bat_cycle <= 0) & (co2_emissions_bat_cycle <= 0) # Quadrant 3: cost<=0, co2<=0
bat_q4 = (unit_cost_bat_cycle <= 0) & (co2_emissions_bat_cycle > 0)  # Quadrant 4: cost<=0, co2>0
# Energy Storage quadrant classification
es_q1 = (unit_cost_es_cycle > 0) & (co2_emissions_es_cycle > 0)
es_q2 = (unit_cost_es_cycle > 0) & (co2_emissions_es_cycle <= 0)
es_q3 = (unit_cost_es_cycle <= 0) & (co2_emissions_es_cycle <= 0)
es_q4 = (unit_cost_es_cycle <= 0) & (co2_emissions_es_cycle > 0)
print(f"{p} - Battery data quadrant classification:")
print(f"{p} - Quadrant 1 (cost > 0, co2 > 0): {np.sum(bat_q1)}")
print(f"{p} - Quadrant 2 (cost > 0, co2 <= 0): {np.sum(bat_q2)}")
print(f"{p} - Quadrant 3 (cost <= 0, co2 <= 0): {np.sum(bat_q3)}")
print(f"{p} - Quadrant 4 (cost <= 0, co2 > 0): {np.sum(bat_q4)}")
print(f"\n{p} - Energy Storage data quadrant classification:")
print(f"{p} - Quadrant 1 (cost > 0, co2 > 0): {np.sum(es_q1)}")
print(f"{p} - Quadrant 2 (cost > 0, co2 <= 0): {np.sum(es_q2)}")
print(f"{p} - Quadrant 3 (cost <= 0, co2 <= 0): {np.sum(es_q3)}")
print(f"{p} - Quadrant 4 (cost <= 0, co2 > 0): {np.sum(es_q4)}")
# Calculate the total discharge energy for each quadrant, rounded after dividing by 1e3
# Battery discharge energy
discharge_bat_q1 = np.round(np.sum(energy_discharge_cycle_bat[bat_q1]) / 1e3)
discharge_bat_q2 = np.round(np.sum(energy_discharge_cycle_bat[bat_q2]) / 1e3)
discharge_bat_q3 = np.round(np.sum(energy_discharge_cycle_bat[bat_q3]) / 1e3)
discharge_bat_q4 = np.round(np.sum(energy_discharge_cycle_bat[bat_q4]) / 1e3)
# Energy Storage discharge energy
discharge_es_q1 = np.round(np.sum(energy_discharge_cycle_es[es_q1]) / 1e3)
discharge_es_q2 = np.round(np.sum(energy_discharge_cycle_es[es_q2]) / 1e3)
discharge_es_q3 = np.round(np.sum(energy_discharge_cycle_es[es_q3]) / 1e3)
discharge_es_q4 = np.round(np.sum(energy_discharge_cycle_es[es_q4]) / 1e3)
# Print the output results, including the value of p
print(f"{p} - Battery Discharge Energy Sums (×1e3):")
print(f"{p} - Quadrant 1 (cost > 0, co2 > 0): {discharge_bat_q1}")
print(f"{p} - Quadrant 2 (cost > 0, co2 <= 0): {discharge_bat_q2}")
print(f"{p} - Quadrant 3 (cost <= 0, co2 <= 0): {discharge_bat_q3}")
print(f"{p} - Quadrant 4 (cost <= 0, co2 > 0): {discharge_bat_q4}")
print(f"\n{p} - Energy Storage Discharge Energy Sums (×1e3):")
print(f"{p} - Quadrant 1 (cost > 0, co2 > 0): {discharge_es_q1}")
print(f"{p} - Quadrant 2 (cost > 0, co2 <= 0): {discharge_es_q2}")
print(f"{p} - Quadrant 3 (cost <= 0, co2 <= 0): {discharge_es_q3}")
print(f"{p} - Quadrant 4 (cost <= 0, co2 > 0): {discharge_es_q4}")

# Create scatter plot
if '_min_cost' in pre_dic:
    fig, ax = plt.subplots(figsize=(5, 4))
elif '_min_CO2' in pre_dic:
    fig, ax = plt.subplots(figsize=(5.71, 4))
# Plot Bat (battery) data points
scatter_bat_ccf = ax.scatter(co2_emissions_bat_cycle, unit_cost_bat_cycle, label='Bat', marker='o', alpha=0.7)
# Plot LDES (long-duration energy storage) data points
scatter_es_ccf = ax.scatter(co2_emissions_es_cycle, unit_cost_es_cycle, label='LDES', marker='s', alpha=0.7)
if '_min_cost' in pre_dic:
    ax.set_xlim(-0.5, 0.2)
elif '_min_CO2' in pre_dic:
    ax.set_xlim(-0.6, 0.2)
ax.set_ylim(-60, 65)
# Invert the x-axis so that CEF is sorted from high to low
ax.invert_xaxis()
# Set axis labels and title
ax.set_xlabel('CEF (tCO2/MWh)', fontsize=14)
ax.set_ylabel('CCF (£/MWh)', fontsize=14)
if '_min_cost_2023' in pre_dic:
    ax.set_title(r"$\mathbf{(a)}$" + f" CCF_CEF_{p}", fontsize=14)
elif '_min_CO2_2023' in pre_dic:
    ax.set_title(r"$\mathbf{(b)}$" + f" CCF_CEF_{p}", fontsize=14)
elif '_min_cost_2030' in pre_dic:
    ax.set_title(r"$\mathbf{(c)}$" + f" CCF_CEF_{p}", fontsize=14)
elif '_min_CO2' in pre_dic:
    ax.set_title(r"$\mathbf{(d)}$" + f" CCF_CEF_{p}", fontsize=14)
else:
    ax.set_title(f"CCF_CEF_{p}", fontsize=14)
# Add legend and grid
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.7)
# Get the maximum and minimum CEF values (assuming CEF data are stored in co2_emissions_bat_cycle and co2_emissions_es_cycle)
cef_values = np.concatenate((co2_emissions_bat_cycle, co2_emissions_es_cycle))
cef_max = np.max(cef_values)
cef_min = np.min(cef_values)
# Get the colors of the scatter points
color_bat = scatter_bat_ccf.get_facecolors()[0]
color_es = scatter_es_ccf.get_facecolors()[0]
# Add dashed lines to indicate average values
ax.axhline(np.mean(unit_cost_bat_cycle), color=color_bat, linestyle='--', alpha=0.7)
ax.axhline(np.mean(unit_cost_es_cycle), color=color_es, linestyle='--', alpha=0.7)
# Add vertical dashed line: CEF=0
ax.axvline(x=0, linestyle='--', color='gray', linewidth=1)
# Add horizontal dashed line: CCF=0
ax.axhline(y=0, linestyle='--', color='gray', linewidth=1)
plt.tight_layout()
fig.savefig(f"{save_dir}cost vs emission_{p}.jpg", format='jpg', dpi=300)


# Create scatter plot
if '_min_cost_2023' in pre_dic:
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot Bat (battery) data points
    scatter_bat_ccf = ax.scatter(co2_emissions_bat_cycle, unit_cost_bat_cycle, label='Bat', marker='o', alpha=0.7)
    # Plot LDES (long-duration energy storage) data points
    scatter_es_ccf = ax.scatter(co2_emissions_es_cycle, unit_cost_es_cycle, label='LDES', marker='s', alpha=0.7)
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-50, 40)
    # Invert the x-axis so that it is sorted from high to low
    ax.invert_xaxis()
    # Set axis labels and title
    ax.set_xlabel('CEF (tCO2/MWh)', fontsize=14)
    ax.set_ylabel('CCF (£/MWh)', fontsize=14)
    ax.set_title(f"CCF_CEF_{p}", fontsize=14)
    # Add legend and grid
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    # Get the maximum and minimum CEF values (assuming CEF data are stored in co2_emissions_bat_cycle and co2_emissions_es_cycle)
    cef_values = np.concatenate((co2_emissions_bat_cycle, co2_emissions_es_cycle))
    cef_max = np.max(cef_values)
    cef_min = np.min(cef_values)
    # Get the colors of the scatter points
    color_bat = scatter_bat_ccf.get_facecolors()[0]
    color_es = scatter_es_ccf.get_facecolors()[0]
    # Add vertical dashed line: CEF=0
    ax.axvline(x=0, linestyle='--', color='gray', linewidth=1)
    # Add horizontal dashed line: CCF=0
    ax.axhline(y=0, linestyle='--', color='gray', linewidth=1)
    plt.tight_layout()
    fig.savefig(f"{save_dir}cost vs emission_{p}_widen.jpg", format='jpg', dpi=300)
# plt.show()
#


# --------------------
# process_times_bat is a dictionary where the key x corresponds to a list of indices
indices = process_times_bat[33]
# Use loc to select rows in df_gen and df_storage_links with indices in indices
df_gen_selected = df_gen.loc[indices]
df_gen_remain_selected = df_gen_remain_carrier.loc[indices]
df_storage_links_selected = df_storage_links.loc[indices]
df_gen_grouped = df_gen_selected.drop(columns=['snapshot'])
df_gen_remain_grouped = df_gen_remain_selected.drop(columns=['snapshot'])
# Extract the keys of CO2_FACTORS as the new column order
column_order = list(CO2_FACTORS.keys())
# Reorder the columns of the DataFrame
df_gen_grouped = df_gen_grouped[column_order]
df_gen_remain_grouped = df_gen_remain_grouped[column_order]
# Only keep the 'bus_charger' and 'bus_discharger' columns
df_storage_links_selected = df_storage_links_selected[['bus_charger', 'bus_discharger']]
# Create the figure
fig, (ax_top, ax_bottom) = plt.subplots(
    2, 1, sharex=True, figsize=(10,5),
    gridspec_kw={'height_ratios': [3,1]}
)
bar_positions = np.arange(len(df_gen_grouped.index))
# --------------------
# Upper subplot (ax_top)
# --------------------
# Extract CO2 emission factors (first value)
co2_values = {key: value[0] for key, value in CO2_FACTORS.items()}
# Separate energy sources with zero CO2 emission and non-zero CO2 emission
zero_co2 = [k for k, v in co2_values.items() if v == 0]
nonzero_co2 = {k: v for k, v in co2_values.items() if v > 0}
# Green (low carbon) gradient & gray (high carbon) gradient
green_cmap = mcolors.LinearSegmentedColormap.from_list("GreenScale", ["darkgreen", "lightgreen"])
gray_cmap = mcolors.LinearSegmentedColormap.from_list("GrayScale", ["lightgray", "black"])
# Assign colors to the green part in order of indices
green_norm_values = np.linspace(0, 1, len(zero_co2))
colors_map = {carrier: green_cmap(norm) for carrier, norm in zip(zero_co2, green_norm_values)}
# Normalize gray for energy sources with carbon
if nonzero_co2:
    norm_gray = mcolors.Normalize(vmin=min(nonzero_co2.values()), vmax=max(nonzero_co2.values()))
    for carrier, co2 in nonzero_co2.items():
        colors_map[carrier] = gray_cmap(norm_gray(co2))
# Upper bar chart: positive part
bottom = np.zeros(len(bar_positions))
for carrier in df_gen_grouped.columns:
    ax_top.bar(
        bar_positions,
        df_gen_grouped[carrier].values / 1e3,
        bottom=bottom,
        width=0.8,
        color=colors_map[carrier],
        label=carrier
    )
    bottom += df_gen_grouped[carrier].values / 1e3
# Upper bar chart: negative part
bottom_neg = np.zeros(len(bar_positions))
for carrier in df_gen_remain_grouped.columns:
    ax_top.bar(
        bar_positions,
        -df_gen_remain_grouped[carrier].values / 1e3,
        bottom=bottom_neg,
        width=0.8,
        color=colors_map[carrier]
    )
    bottom_neg -= df_gen_remain_grouped[carrier].values / 1e3
# Grid & axis limits
ax_top.grid(axis='y', linestyle='--', alpha=0.7)
ax_top.set_ylim(-43, 50)  # y-axis range
# Second y-axis (ax2_top) for MEF / AEF / CEF
ax2_top = ax_top.twinx()
ax2_top.plot(bar_positions, np.array(mef_bat_t[33]) / 1e3, marker='s', linestyle='--',
             color="crimson", linewidth=1, markersize=2, label="MEF")
ax2_top.plot(bar_positions, np.array(aef_bat_t[33]) / 1e3, marker='D', linestyle='-.',
             color="darkorange", linewidth=1, markersize=2, label="AEF")
ax2_top.plot(bar_positions, np.array(cef_bat_t[33]) / 1e3, marker='o', linestyle='-',
             color="deepskyblue", linewidth=1, markersize=2, label="CEF")
ax2_top.set_ylim(-4.3, 5)  # y-axis range
# Set labels and title
ax_top.set_ylabel("Generation (GW)")
ax2_top.set_ylabel("Storage emissions (tCO2)")
ax_top.set_title("Emissions calculation comparison in a same cycle")
ax_top.tick_params(axis='x', bottom=False)  # Remove x-axis tick marks
ax_top.spines["bottom"].set_visible(False)
# --------------------
# Lower subplot (ax_bottom)
# --------------------
# Plot battery charging and discharging data
ax_bottom.bar(
    bar_positions,
    df_storage_links_selected['bus_charger'].values/1e3,
    width=0.8,
    color="gold",
    label="Bat_cha"
)
ax_bottom.bar(
    bar_positions,
    df_storage_links_selected['bus_discharger'].values/1e3,
    width=0.8,
    color="olive",
    label="Bat_dis"
)
# Lower axis limits
ax_bottom.set_ylim(-12.5, 11)
# # Remove top border
# ax_bottom.spines["top"].set_visible(False) # Remove x-axis label
ax_bottom.grid(axis='y', linestyle='--', alpha=0.7)
ax_bottom.set_ylabel("Battery power (GW)")
ax_bottom.spines["top"].set_visible(False)
# You can also place x-axis tick labels on the lower subplot:
# ax_bottom.set_xlabel("Time index (hours)")
# ax_bottom.tick_params(labelbottom=True)
# ax_top.tick_params(labelbottom=False)
# --------------------
# Share x-axis tick labels
# --------------------
ax_bottom.set_xticks(bar_positions)
ax_bottom.set_xticklabels(df_gen_grouped.index.astype(str), rotation=45)
ax_bottom.set_xlabel("Time index (hours)", labelpad=5)  # Slightly raise the x-axis label
# ax_bottom.xaxis.set_label_coords(0.5, 0)  # Manually adjust position (center in x-direction, upward in y-direction)
# --------------------
# Merge legends
# --------------------
lines_top, labels_top = ax_top.get_legend_handles_labels()
lines_top2, labels_top2 = ax2_top.get_legend_handles_labels()
lines_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
# Place them together on the upper subplot
ax2_top.legend(
    lines_top + lines_top2 + lines_bottom,
    labels_top + labels_top2 + labels_bottom,
    loc="center left",
    bbox_to_anchor=(1.08, 0.3),
    fontsize=10,
    frameon=True
)
# Layout & save
plt.tight_layout()
fig.subplots_adjust(hspace=0.02)  # Adjust spacing between subplots
fig.savefig(f"{save_dir}cycle_factor_new_{p}.jpg", format='jpg', dpi=300, bbox_inches="tight")
# plt.show()

# Extract data
resources = list(resource_usage.keys())
bat_cha = [np.sum(resource_usage[res]["bat_cha"]) for res in resources]
bat_dis = [-np.sum(resource_usage[res]["bat_dis"]) for res in resources]
es_cha = [np.sum(resource_usage[res]["es_cha"]) for res in resources]
es_dis = [-np.sum(resource_usage[res]["es_dis"]) for res in resources]
x = np.arange(len(resources))  # Resource index positions
width = 0.4  # Width of each group of bars
# Compute global y-axis limits
all_data = bat_cha + bat_dis + es_cha + es_dis
y_min = min(all_data)
y_max = max(all_data)
y_limit = (y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1)  # Leave a 10% margin
# Define colors and transparency
bat_charge_color = 'blue'
bat_discharge_color = 'orange'
es_charge_color = 'green'
es_discharge_color = 'red'
# Create the plot
fig, ax = plt.subplots(figsize=(8, 4))
# Plot Battery data
ax.bar(x - width/2, bat_cha, width, label='Battery Charge', color=bat_charge_color, alpha=0.8)
ax.bar(x - width/2, bat_dis, width, label='Battery Discharge', color=bat_discharge_color, alpha=0.8)
# Plot Energy Storage (LDES) data
ax.bar(x + width/2, es_cha, width, label='LDES Charge', color=es_charge_color, alpha=0.8)
ax.bar(x + width/2, es_dis, width, label='LDES Discharge', color=es_discharge_color, alpha=0.8)
# Add helper line
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
# Set axes and title
ax.set_xticks(x)
ax.set_xticklabels(resources, rotation=45, ha="right")
ax.set_ylim(y_limit)  # Set uniform y-axis range
ax.set_ylabel('Energy (MWh)', fontsize=14)
ax.set_title(f"National Battery and LDES Usage by Resource_{p}", fontsize=14)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
# Adjust layout and save the figure
fig.tight_layout()
fig.savefig(f"{save_dir}bat_es_cha_dis_{p}.jpg", format='jpg', dpi=300)
# plt.show()

# For bat_cha, rows represent each time step/record and columns represent different resources
df_bat_cha = pd.DataFrame(
    {res: resource_usage[res]["bat_cha"] for res in resources}
)
# Similarly, for bat_dis
df_bat_dis = pd.DataFrame(
    {res: resource_usage[res]["bat_dis"] for res in resources}
)
# For es_cha
df_es_cha = pd.DataFrame(
    {res: resource_usage[res]["es_cha"] for res in resources}
)
# For es_dis
df_es_dis = pd.DataFrame(
    {res: resource_usage[res]["es_dis"] for res in resources}
)
output_path = f"{save_dir}resource_usage_summary.xlsx"
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_bat_cha.to_excel(writer, sheet_name="bat_cha", index=False)
    df_bat_dis.to_excel(writer, sheet_name="bat_dis", index=False)
    df_es_cha.to_excel(writer, sheet_name="es_cha", index=False)
    df_es_dis.to_excel(writer, sheet_name="es_dis", index=False)

print('national results done')


# # regional results
for region_i in regions:
    region_gen_p = network.generators_t.p.T.groupby(network.generators.bus).get_group(region_i)
    region_gen_p_carrier = region_gen_p.groupby(network.generators.carrier).sum().T

    # region_demand = (network.loads_t.p.groupby(network.generators.bus, axis=1).get_group(region_i))
    region_gen_p_carrier['demand'] = network.loads_t.p[region_i]

    # print(region_gen_p)
    file_name = region_i +'_carrier.csv'
    region_gen_p_carrier.to_csv(pre_dic+file_name,header=True)

battery_soc=[]
ES_soc=[]
aef_oth_regs = calculate_aef(df_gen_demand, df_gen_bus_carrier_total)
fig_bus_batt, ax_co2_bus = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={'hspace': 0.5})
df_store = (network.stores_t.e)
stores = ['Battery', 'OtherStorage']
storage_batt = []
storage_ldes = []
for i in regions:
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), gridspec_kw={'hspace': 0.5})
    for j in stores:
        store_carrier = i + '_' + j
        df_store.plot(y=store_carrier, ax=ax[0])
        ax[0].set_title('Energy storage SOC in MWh')
        ax[0].set_ylabel('SOC [MWh]')

        if j == 'Battery':
            storage_batt.append(df_store[store_carrier].max())
        else:
            storage_ldes.append(df_store[store_carrier].max())
    battery_i = i + '_Battery'
    ES_i = i + '_OtherStorage'
    if '_ext' in pre_dic:

        battery_i_2030 = i + '_Battery_2030'
        df_capacity[battery_i + '_soc'] = (df_capacity[battery_i_2030] + df_capacity[battery_i]) / (
                    df_capacity[battery_i_2030].max() + df_capacity[battery_i].max())

        ES_i_2030 = i + '_OtherStorage_2030'
        df_capacity[ES_i + '_soc'] = (df_capacity[ES_i_2030] + df_capacity[ES_i]) / (
                    df_capacity[ES_i_2030].max() + df_capacity[ES_i].max())
    else:
        df_capacity[battery_i + '_soc'] = df_capacity[battery_i]/df_capacity[battery_i].max()
        df_capacity[ES_i + '_soc'] = df_capacity[ES_i] / df_capacity[ES_i].max()

    battery_soc.append(battery_i + '_soc')
    ES_soc.append(ES_i + '_soc')

    file_name = i + '_carrier.csv'
    df_gen_bus_carrier = pd.read_csv(pre_dic+file_name)

    df_gen_bus_carrier = df_gen_bus_carrier.reindex(columns=all_carriers, fill_value=0)
    df_gen_bus_carrier_update = df_gen_bus_carrier.copy()
    resource_usage_i = {
        "PV": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Wind": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Wind offshore": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Hydro": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Nuclear": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Biomass": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Biomass_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Biogas": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Biogas_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "CCGT": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "SCGT_CHP": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "SCGT": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Oil": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []},
        "Others": {"bat_cha": [], "es_cha": [], "bat_dis": [], "es_dis": []}
    }
    print(i)

    df_capacity_copy = df_capacity.copy()
    df_capacity_copy['snapshot'] = pd.to_datetime(df['snapshot'])
    new_snapshot = df_capacity_copy['snapshot'].iloc[0] - pd.Timedelta(hours=1)
    new_row = pd.DataFrame({col: [0] if col != 'snapshot' else [new_snapshot] for col in df_capacity_copy.columns})
    df_capacity_copy = pd.concat([new_row, df_capacity_copy], ignore_index=True)
    process_times_bat, process_ratios_bat, charg_xy = cycle_extraction(df_capacity_copy[battery_i + '_soc'])
    process_times_es, process_ratios_es, charg_xy_es = cycle_extraction(df_capacity_copy[ES_i + '_soc'])

    (emissions_bat, emissions_es, co2_emissions_factor_bat, co2_emissions_factor_es, co2_emissions_bat_cycle, co2_emissions_es_cycle,
     energy_charge_cycle_bat, energy_charge_cycle_es,
     energy_discharge_cycle_bat, energy_discharge_cycle_es, emissions_charged_bat, emissions_charged_es,
     emissions_discharged_bat, emissions_discharged_es,
     co2_delta_emissions, co2_delta_emissions_es, resource_usage_i) = cycle_analysis(process_times_bat, process_ratios_bat, process_times_es, process_ratios_es, aef_oth_regs, df_gen_bus_carrier_region,
                                                                   df_storage_links, df_gen_remain_new,  i, resource_usage_i)

    # Create a figure and axis
    fig_factor, ax_factor = plt.subplots()
    ax_factor.scatter(charg_xy, co2_emissions_bat_cycle, label='bat', marker='o', alpha=0.7)
    ax_factor.scatter(charg_xy_es, co2_emissions_es_cycle, label='ldes', marker='s', alpha=0.7)
    ax_factor.set_xscale('log')
    ax_factor.set_title('emissions per cycle', fontsize=12)
    ax_factor.set_ylabel('emissions [ton/MWh]', fontsize=12)
    ax_factor.set_xlabel('Storage Duration [hour]', fontsize=12)
    ax_factor.legend(fontsize=10)
    ax_factor.grid(True, linestyle='--', alpha=0.7)
    fig_factor.savefig(f"{save_dir}{i}_emissions_duration_bat{p}.jpg", format='jpg', dpi=300)

    fig_factor1, ax_factor1 = plt.subplots()
    ax_factor1.scatter(energy_discharge_cycle_bat, co2_emissions_bat_cycle, label='bat', marker='o', alpha=0.7)
    ax_factor1.scatter(energy_discharge_cycle_es, co2_emissions_es_cycle, label='ldes', marker='s', alpha=0.7)
    ax_factor1.set_xscale('log')
    ax_factor1.set_title('CO2 Emissions Factor vs Energy Discharged Per Cycle', fontsize=12)
    ax_factor1.set_ylabel('CO2 Emissions Factor (tons/MWh)', fontsize=12)
    ax_factor1.set_xlabel('Energy Discharged Per Cycle (MWh)', fontsize=12)
    ax_factor1.legend(fontsize=10)
    ax_factor1.grid(True, linestyle='--', alpha=0.7)
    fig_factor1.savefig(f"{save_dir}{i}_energy_discharged_{p}.jpg", format='jpg', dpi=300)

    ax_co2_bus[0].scatter(charg_xy, co2_delta_emissions, label=i)
    scatter_4 = ax[2].scatter(emissions_charged_bat, emissions_discharged_bat, c=charg_xy, cmap=cmap_t, label='battery', marker='o')

    ax_co2_bus[1].scatter(charg_xy_es, co2_delta_emissions_es, marker='s', label=i)
    scatter_4_es = ax[2].scatter(emissions_charged_es, emissions_discharged_es,c=charg_xy_es, cmap=cmap_t, marker='s', label='ldes')

    ax[2].legend()
    ax[2].set_xlabel('Charge emissions [tCO2]', fontsize=12)
    ax[2].set_ylabel('Discharge emissions [tCO2]', fontsize=12)
    # first colorbar
    divider_4 = make_axes_locatable(ax[2])
    cax_4 = divider_4.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(scatter_4, cax=cax_4, label='Battery Storage Duration [hour]')
    # sec colorbar
    cax_4_es = divider_4.append_axes("right", size="5%", pad=0.7)
    fig.colorbar(scatter_4_es, cax=cax_4_es, label='LDES Storage Duration [hour]')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlim(1e0, 1e6)
    ax[2].set_ylim(1e0, 1e6)
    ax[2].plot([0, 1e6], [0, 1e6], '--')

    ax_co2_bus[0].set_title('all region results - Batt')
    ax_co2_bus[0].legend()
    ax_co2_bus[0].set_title('battery')
    ax_co2_bus[0].set_xlabel('Storage Duration [hour]')
    ax_co2_bus[0].set_ylabel('CO2 Emissions [ton/MWh]')

    ax_co2_bus[1].set_title('all region results - LDES')
    ax_co2_bus[1].set_xlabel('Storage Duration [hour]')
    ax_co2_bus[1].set_ylabel('CO2 Emissions [ton/MWh]')

    # Extract data
    resources = list(resource_usage_i.keys())
    bat_cha = [np.sum(resource_usage_i[res]["bat_cha"]) for res in resources]
    bat_dis = [-np.sum(resource_usage_i[res]["bat_dis"]) for res in resources]
    es_cha = [np.sum(resource_usage_i[res]["es_cha"]) for res in resources]
    es_dis = [-np.sum(resource_usage_i[res]["es_dis"]) for res in resources]
    x = np.arange(len(resources))  # Resource index positions
    width = 0.2  # Width of each bar group
    # Compute global y-axis limits
    all_data = bat_cha + bat_dis + es_cha + es_dis
    y_min = min(all_data)
    y_max = max(all_data)
    y_limit = (y_min - abs(y_min) * 0.1, y_max + abs(y_max) * 0.1)  # Leave a 10% margin
    # Define colors and transparency
    bat_charge_color = 'blue'
    bat_discharge_color = 'orange'
    es_charge_color = 'green'
    es_discharge_color = 'red'
    # Plot Battery data
    ax[1].bar(x - width / 2, bat_cha, width, label='Battery Charge', color=bat_charge_color, alpha=0.8)
    ax[1].bar(x - width / 2, bat_dis, width, label='Battery Discharge', color=bat_discharge_color, alpha=0.8)
    # Plot Energy Storage (LDES) data
    ax[1].bar(x + width / 2, es_cha, width, label='LDES Charge', color=es_charge_color, alpha=0.8)
    ax[1].bar(x + width / 2, es_dis, width, label='LDES Discharge', color=es_discharge_color, alpha=0.8)
    # Add helper line
    ax[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    # Set axes and title
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(resources, rotation=45, ha="right")
    ax[1].set_ylim(y_limit)  # Set uniform y-axis range
    ax[1].set_ylabel('Energy (MWh)')
    ax[1].set_title(f"Battery and LDES Usage by Resource_{i}_{p}")
    ax[1].legend(title="Operations")
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    # Plotting and saving within the loop
    fig.savefig(f"{save_dir}{i}_energy_storage_emission_{p}.jpg", format='jpg', dpi=300)

fig_bus_batt.savefig(f"{save_dir}regional_battery_co2_{p}.jpg", format='jpg', dpi=300)
# plt.show()
print('---------------------------------')
print('regional results done')