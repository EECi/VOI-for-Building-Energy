## Overview for determining ASHP maintenance uncertain parameter
This  branch focuses on developing probabilistic models to estimate heating loads, beta parameter variables, and electricity prices. It leverages uncertainty analysis and simulation techniques. Detailed implementation can be found in `uncertainty_VOI.py`.

### 1. Heating Load Uncertainty
Data on the yearly heating energy consumption of an educational building (A011 old school) in Cambridge was gathered for the period 2012 to 2021. A normal distribution model was applied to characterize the pattern of this heating energy usage.

**Figure 1: Historical Annual Heating Load from 2012 to 2021**
![Annual Heating Load](/plots/Historical_Heating_Load.png)

**Figure 2: Fitted Distribution of Annual Heating Load**
![Annual Heating Load](/plots/Cumulative_Heating_Load.png)

### 2. Beta Parameter
The relationship between the annual number of maintenance activities (N<sub>m</sub>) and the beta parameter (`β`) is given by:

β = (β<sub>a</sub> * N<sub>m</sub><sup>γ</sup>) / (β<sub>b</sub> + N<sub>m</sub><sup>γ</sup>) (1+ϵ)

where β<sub>a</sub> = 0.05, β<sub>b</sub>  = 2.5, γ = 1.4, N<sub>m</sub> is the annual number of maintenance activities, and `ϵ` follows log-normal distribution Lognormal (0,0.1<sup>2</sup>), representing ±10% uncertainty in this case. 

**Figure 3: Performance Improvement Distribution**
![Performance Improvement Distribution](/plots/Performance_improvement_distribution.png)

### 3. Electricity Prices
Electricity price data was downloaded from [Gov.uk](https://www.gov.uk/government/statistical-data-sets/annual-domestic-energy-price-statisticshttps://www.gov.uk/government/statistical-data-sets/annual-domestic-energy-price-statistics), using the 2022 unit electricity costs for different payment methods (i.e.,credit,direct debit and prepayment) across different UK regions (i.e.,East Midlands, Eastern, London, Merseyside & North Wales, North East, North Scotland, North West, Northern Ireland, South East, South Scotland, South Wales, South West, Southern, West Midlands, Yorkshire) to fit the distribution of electricity prices.

**Figure 4: 2022 Electricity Prices**
![2022 Electricity Prices](/plots/2022_Electricity_Prices.png)

