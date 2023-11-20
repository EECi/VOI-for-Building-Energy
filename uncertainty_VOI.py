import matplotlib.pyplot as plt
import numpy as np

##%  consider the uncertainty of performance improvement
# Define the parameters for the function
beta_a = 0.05
beta_b = 2.5
gamma = 1.4

# Define the range of maintenance activities (N_m)
N_m = np.linspace(0, 12, 100)  # Assuming a maximum of 20 maintenance activities per year

# Define the function beta
beta = (beta_a * N_m**gamma) / (beta_b + N_m**gamma)

# Define the uncertainty, which increases with the number of maintenance activities
# We'll use a simple linear model for uncertainty
beta_uncertainty = 0.002 * N_m  # 0.2% uncertainty per maintenance activity

# Upper and lower bounds of beta considering uncertainty
upper_bound_beta = beta + beta_uncertainty
lower_bound_beta = beta - beta_uncertainty

# Plotting
plt.figure(figsize=(10, 6))
plt.fill_between(N_m, lower_bound_beta, upper_bound_beta, color='lightgreen', alpha=0.4, label="Uncertainty Range")
plt.plot(N_m, beta, label="Function β(N_m)", color='green')
plt.title("Performance improvement vs. Number of Maintenance Activities (N_m) with Uncertainty")
plt.xlabel("Number of Maintenance Activities (N_m) per Year")
plt.ylabel("β")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/Performance improvement distribution.png', dpi=300)
plt.show()


##% Define the uncertainty of heating load measurements
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# Load the Excel file
file_path = 'Inputs/2020-01-01 to 2020-12-31heat_meter.xlsx'
heat_meter_data = pd.read_excel(file_path)

# Specific heat capacity of water
cp_water = 4.18  # in kJ/kg·K

# Assumed temperature difference
delta_T = 10  # in degrees Celsius

# Convert the heating load (Q) from kW to kJ/s for calculation
# 1 kW = 1 kJ/s
Q_kJ_s = heat_meter_data['C020 Building Gas']

# Calculating the flow rate (m) in kg/s
# m = Q / (cp * delta_T)
flow_rate_kg_s = Q_kJ_s / (cp_water * delta_T)


# Assumed errors in sensors
temp_sensor_error = 0.5  # degrees Celsius for temperature sensor    ref:
flow_meter_error_percent = 2  # percentage error for flow meter      ref:

# Number of simulations for Monte Carlo
num_simulations = 1000

# Arrays to store simulated values
simulated_heating_loads = np.zeros((num_simulations, len(Q_kJ_s)))

# Simulating flow rate and heating load with assumed sensor errors
for i in range(num_simulations):
    print(f'the number of simulation is {i}')
    for j in range(len(Q_kJ_s)):
        # Simulating error in temperature measurement (±0.5 degrees)
        temp_error = np.random.uniform(-temp_sensor_error, temp_sensor_error)
        simulated_delta_T = delta_T + temp_error

        # Simulating error in flow rate measurement (±2% of the reading)
        flow_rate_error = np.random.uniform(-flow_meter_error_percent, flow_meter_error_percent) / 100
        simulated_flow_rate = flow_rate_kg_s[j] * (1 + flow_rate_error)

        # Calculating heating load with simulated values
        simulated_heating_load = simulated_flow_rate * cp_water * simulated_delta_T
        simulated_heating_loads[i, j] = simulated_heating_load
#
# Calculate cumulative heating load for each simulation
cumulative_heating_loads = np.sum(simulated_heating_loads, axis=1)


# Fit a normal distribution to the cumulative heating loads
mu, std = norm.fit(cumulative_heating_loads)

# Plotting the distribution of the cumulative heating loads
plt.figure(figsize=(10, 6))
plt.hist(cumulative_heating_loads, bins=50, density=True, alpha=0.6, color='g')

# Plot the PDF of the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = f"Fit results: mu = {mu:.2f},  std = {std:.2f}"
plt.title(title)
plt.xlabel('Cumulative Heating Load (kW)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('plots/Cumulative Heating Load.png', dpi=300)
plt.show()

##% define the distribution of electicity price
#ref: https://www.gov.uk/government/statistical-data-sets/annual-domestic-energy-price-statistics

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = 'Inputs/past electricity_price.xlsx'

# Re-loading the Excel file to select the requested columns for the year 2022
electricity_price_data = pd.read_excel(file_path)

# Filtering the data for the year 2022
electricity_price_2022 = electricity_price_data[electricity_price_data['Year'] == 2022]

# Selecting the requested columns
credit_prices_2022 = electricity_price_2022["Credit: Average variable unit price (£/kWh)[Note 2]"]
debit_prices_2022 = electricity_price_2022["Direct debit: Average variable unit price (£/kWh)[Note 2]"]
prepayment_prices_2022 = electricity_price_2022["Prepayment: Average variable unit price (£/kWh)[Note 2]"]

# Combining the prices into a single array
all_prices_2022 = np.concatenate([credit_prices_2022, debit_prices_2022, prepayment_prices_2022])

# Fit a normal distribution to the combined electricity prices
mu, std = norm.fit(all_prices_2022)

# Plotting the distribution of the electricity prices
plt.figure(figsize=(10, 6))
plt.hist(all_prices_2022, bins=30, density=True, alpha=0.6, color='c')

# Plot the PDF of the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = f"2022 Electricity Prices Distribution: mu = {mu:.4f}, std = {std:.4f}"
plt.title(title)
plt.xlabel('Electricity Price (£/kWh)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('plots/2022 Electricity Prices.png', dpi=300)
plt.show()


