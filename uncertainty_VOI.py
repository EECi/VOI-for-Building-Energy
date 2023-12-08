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
beta_base_values  = (beta_a * N_m**gamma) / (beta_b + N_m**gamma)

# Define a fixed percentage for the uncertainty bounds (e.g., ±10%)
uncertainty_percentage = 0.1

# Calculate the upper and lower bounds
beta_upper_bound = beta_base_values * (1 + uncertainty_percentage)
beta_lower_bound = beta_base_values * (1 - uncertainty_percentage)

# Plotting
plt.figure(figsize=(10, 6))
plt.fill_between(N_m, beta_lower_bound, beta_upper_bound , color='lightgreen', alpha=0.4, label="Uncertainty Range")
plt.plot(N_m, beta_base_values, label="Function β($N_m$)", color='green')
plt.title("Performance improvement vs. Number of Maintenance Activities ($N_m$) with Uncertainty")
plt.xlabel("Number of Maintenance Activities ($N_m$) per Year")
plt.ylabel("β")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/Performance_Improvement_Distribution.png', dpi=300)
plt.show()

##% define the distribution of electicity price
#ref: https://www.gov.uk/government/statistical-data-sets/annual-domestic-energy-price-statistics

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = 'Inputs/table_223.xlsx'

# Reading the data
electricity_price_data = pd.read_excel(file_path, sheet_name='2.2.3',skiprows=11)
electricity_price_data.columns
#
# # Re-loading the Excel file to select the requested columns for the year 2022
# electricity_price_data = pd.read_excel(file_path)

# Filtering the data for the year 2022
electricity_price_2022 = electricity_price_data[electricity_price_data['Year'] == 2022]

electricity_price_2022 = electricity_price_2022[electricity_price_2022['Region'] != 'Northern Ireland']
# plt.bar(electricity_price_2022['Region'], electricity_price_2022["Credit: Unit cost (Pence per kWh)"])
# plt.show()
# Selecting the requested columns
credit_prices_2022 = electricity_price_2022["Credit: Unit cost (Pence per kWh)"]
debit_prices_2022 = electricity_price_2022["Direct debit: Unit cost (Pence per kWh)"]
prepayment_prices_2022 = electricity_price_2022["Prepayment: Unit cost (Pence per kWh)"]

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
# Adding a note at the bottom of the figure
plt.figtext(0.02, 0.01, "Note: The Electricity Prices of Northern Ireland were excluded.",
            wrap=True, horizontalalignment='left', fontsize=10)

plt.tight_layout()
plt.savefig('plots/2022_Electricity_Prices.png', dpi=300)
plt.show()

##% heating enegry use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.stats import norm
# continuous
AllData = pd.read_csv('Inputs/A011_1389_OLDSCHOOL.csv')
gas_df = AllData[AllData['type.1'] == 'Gas']

locations = gas_df.location.unique()

print(locations)

loc_2 = gas_df.number.unique()

print(loc_2)

# Split by location

#location = locations[1]

gas_df_location = gas_df[gas_df['location'] == locations[0]]

# Extract data
gas_data = gas_df_location.loc[:, 'date':'24:00']
gas_data = gas_data.drop('totalunits', axis=1)

gas_data.rename(columns={'00:30': '00:00:00', '01:00': '00:30:00',
                           '01:30': '01:00:00', '02:00': '01:30:00',
                           '02:30': '02:00:00', '03:00': '02:30:00',
                           '03:30': '03:00:00', '04:00': '03:30:00',
                           '04:30': '04:00:00', '05:00': '04:30:00',
                           '05:30': '05:00:00', '06:00': '05:30:00',
                           '06:30': '06:00:00', '07:00': '06:30:00',
                           '07:30': '07:00:00', '08:00': '07:30:00',
                           '08:30': '08:00:00', '09:00': '08:30:00',
                           '09:30': '09:00:00', '10:00': '09:30:00',
                           '10:30': '10:00:00', '11:00': '10:30:00',
                           '11:30': '11:00:00', '12:00': '11:30:00',
                           '12:30': '12:00:00', '13:00': '12:30:00',
                           '13:30': '13:00:00', '14:00': '13:30:00',
                           '14:30': '14:00:00', '15:00': '14:30:00',
                           '15:30': '15:00:00', '16:00': '15:30:00',
                           '16:30': '16:00:00', '17:00': '16:30:00',
                           '17:30': '17:00:00', '18:00': '17:30:00',
                           '18:30': '18:00:00', '19:00': '18:30:00',
                           '19:30': '19:00:00', '20:00': '19:30:00',
                           '20:30': '20:00:00', '21:00': '20:30:00',
                           '21:30': '21:00:00', '22:00': '21:30:00',
                           '22:30': '22:00:00', '23:00': '22:30:00',
                           '23:30': '23:00:00', '24:00': '23:30:00'}, inplace=True)
# Re-arrange timestamp
gas_melt = pd.melt(gas_data, id_vars = ['date'])

s = pd.to_datetime(gas_melt['date'], format='%Y-%m-%d %H:%M:%S').dt.date

t = pd.to_datetime(gas_melt['variable']).dt.time

nd = np.size(gas_melt, axis=0)

dtout = np.zeros(shape=(nd,), dtype='datetime64[ns]')

for jj in range(nd):
    dtout[jj] = pd.Timestamp.combine(s[jj],t[jj])

d = {'Time': dtout, 'Gas': gas_melt.value}

gas_location = pd.DataFrame(data=d)
TimeHistory = gas_location.sort_values(by='Time')

tt = TimeHistory.Time
consumption = TimeHistory.Gas

plt.plot(tt,consumption, color='black')

plt.title(locations[0])

plt.ylim([-250,250])

Year_sum=TimeHistory
# extract year
Year_sum['Year'] = Year_sum['Time'].dt.year

# calcuate sum
yearly_sum = Year_sum.groupby('Year')['Gas'].sum()

# save
yearly_sum.to_csv('Inputs/yearly_sum_'+locations[0]+'_.csv')

filtered_yearly_sum=yearly_sum.loc[2012:2021]
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(filtered_yearly_sum.index, filtered_yearly_sum.values, 'o', label='Data',color='g')
plt.xlabel('Year')
plt.ylabel('Annual Heating Load (kWh)')
plt.title('Annual Heating Load from 2012-2021')
plt.tight_layout()
plt.savefig('plots/Historical_Heating_Load.png', dpi=300)
plt.show()
filtered_yearly_sum=filtered_yearly_sum[filtered_yearly_sum.index!=2020]
# Fit the data to a normal distribution
mu, std = stats.norm.fit(filtered_yearly_sum)

# Plotting the distribution of the electricity prices
plt.figure(figsize=(10, 6))
plt.hist(filtered_yearly_sum, density=True, alpha=0.6, color='g')

# Plot the PDF of the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)

title = f"2012-2021 Heating Load Distribution: mu = {mu:.4f}, std = {std:.4f}"
plt.title(title)
plt.xlabel('Annual Heating Load (kWh)')
plt.ylabel('Density')
plt.figtext(0.02, 0.01, "Note: Heating Load in 2020 was excluded due to Covid impact.",
            wrap=True, horizontalalignment='left', fontsize=10)
plt.tight_layout()
plt.savefig('plots/Cumulative_Heating_Load.png', dpi=300)
plt.show()



