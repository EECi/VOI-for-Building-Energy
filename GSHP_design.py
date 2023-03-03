"""Perform EVPI calculation for ground source heat pump supply system design example."""

import numpy as np
import pandas as pd

import os
import json
from utils import cached

from evpi import compute_EVPI
from models import EnergyPile


if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    borehole_lengths = np.linspace(140,180,9)

    # 2. Define sampling function for uncertain parameter(s)
    def ks_sampler(n_samples, mu=2, sigma=0.12):
        """Sample values from ground thermal conductivity prior and
        discretise for computational efficiency (enable utility caching)."""
        cont_samples = np.random.normal(mu,sigma,size=n_samples)
        discr_points = np.arange(0,4,1e-1)
        return np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in cont_samples])

    # 3. Define system dynamics (intermediate computations)
    # ========================================================================
    # read in heating load data and manipulate (reduce sampling rate & extend to create 50 years of data)
    load_data = pd.read_csv(os.path.join('data','LondonGHELoad_1kW_5yrs_w12h.csv'),header=None,names=['Hours','Load_W'])
    load_data['Load_W']  = load_data['Load_W']*-1 # sign correction
    reduction_factor = 10 # factor to reduce no. of data rows by using chunk means over rows
    load_data = load_data.groupby(np.arange(len(load_data)) // reduction_factor).mean()
    extension_factor = 10 # factor to extend length of reduced dataframe out by, i.e. increase effective duration by factor
    load_data = pd.concat([load_data]*extension_factor, ignore_index=True)
    time_step = load_data['Hours'][1] - load_data['Hours'][0] # hours
    load_data['Hours'] = (load_data['Hours'][0] + time_step*np.arange(len(load_data))).astype(int)

    def create_BH(bh_length,load_data,ks):
        """Create borehole object to perform simulation."""

        # define Pile or Borehole propreties and defining thermal design parameters
        # H = 100               # BHE length [m] - varied here
        rho = 1950.6            # Density [kg/m^3]
        Cp = 956.23             # Specific heat capacity [J/(kg K)]
        Cpvs = rho*Cp           # soil specific heat * soil density
        rb = 0.15               # BHE radius [m]
        T0 = 12.5               # Farfield temperature [degC] 
        # ks = 1.6925           # soil thermal conductivity [W/mK] - varied here (called lambda in model description)
        Rb = 0.1                # BHE thermal resistance [(mK)/W]
        qrate = 20/60           # Fluid flow rate in L/s

        # define pipe properties, radius, themral conductivity
        pipe_r_inner = 0.0269/2
        pipe_r_outter = 0.0023+0.0269/2
        pipe_k = 0.4
        no_of_pipe_legs = 4

        BH = EnergyPile(bh_length,T0,rb,ks,Cpvs,Rb,qrate)
        BH.thermal_design(load_data) # add thermal load data
        BH.pipe_properties(no_of_pipe_legs,pipe_r_inner,pipe_r_outter,pipe_k) # set pipe properties
        BH.Rb_split() # calculate borehole resistance etc.
        BH.FLSM_Gfunc('Lamarche') # calculate G-function for this method

        return BH

    def compute_bh_energy_usage(T, ground_load_used, time_step):
        """Compute energy consumed by GSHP system (kWh)."""
        COP_dist = 4.0279 + 0.1319*T # specific for heating, down to -3 degC
        energy_grnd = abs(ground_load_used*time_step)/1000
        energy_bld = energy_grnd/(1-1/COP_dist)
        energy_consumed = energy_bld/COP_dist
        return energy_consumed.sum()

    def compute_aux_energy_usage(load_data, time_step):
        """Compute energy consumed by auxiliary heating system (kWh)."""
        aux_cop = 1
        energy_usage = abs(load_data*time_step)/1000/aux_cop
        return energy_usage.sum()

    def compute_system_energy_usage(bh_length, ks, load_data, num_boreholes):
        """Compute energy usage by GSHP and auxiliary heating system
        for given borehole length and ground conductivity (ks)."""
        # define building load properties
        max_factor = 4.7484 # represents building with 13.24 kW average, 25.18 kW peak

        time_step = load_data['Hours'][1] - load_data['Hours'][0] # hours

        BH = create_BH(bh_length,load_data, ks)
        f,_,_ = BH.loadfactor_estimate(35,5,0.5,0.01,'FLSM_Lamarche',mode='heating') # find max load based on temperature limits

        if (f >= max_factor): # GSHP can provide heating demand
            f = max_factor
            aux_energy_usage = 0
        else: # auxiliary heating system required
            f_aux = max_factor - f
            aux_energy_usage = compute_aux_energy_usage(load_data['Load_W'].to_numpy()*f_aux*num_boreholes,time_step)

        ground_load_used_per_bh = load_data['Load_W'].to_numpy()*f # compute actual ground load consumed by each borehole
        used_load_df = pd.DataFrame({'Hours':load_data['Hours'],'Load_W':ground_load_used_per_bh})
        true_BH = create_BH(bh_length, used_load_df, ks)
        temps = true_BH.Fluid_Temp('FLSM_Lamarche') # compute fluid temps over true borehole operation
        bh_energy_usage = compute_bh_energy_usage(temps, ground_load_used_per_bh*num_boreholes, time_step)

        return bh_energy_usage, aux_energy_usage
    # ========================================================================

    # 4. Define utility function
    @cached
    def utility(bh_length, ks, load_df=load_data):

        # set up cost parameters
        num_boreholes = 9 # number of boreholes supplying building
        bh_cost_per_m = 70 # borehole capital costs per meter length, £/m
        elec_unit_cost = 0.34 # £/kWh

        bh_energy_usage, aux_energy_usage = compute_system_energy_usage(bh_length, ks, load_df, num_boreholes) # kWh for both

        return -1*((bh_energy_usage + aux_energy_usage)*elec_unit_cost + bh_length*bh_cost_per_m*num_boreholes)

    # load cached utility evaluations if available
    cache_path = os.path.join('data','caches','GSHP_utility_cache_redf-%s.json'%reduction_factor)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            utility.__wrapped__.cache = json.load(file)

    # 5. Perform EVPI computation
    EVPI, Eu_prior, Eu_preposterior, astar_prior = compute_EVPI(borehole_lengths, ks_sampler, utility, n_samples=int(1e5))

    # save utility evaluation cache
    with open(cache_path, 'w') as file:
        json.dump({key:utility.__wrapped__.cache[key] for key in sorted(utility.__wrapped__.cache.keys())}, file, indent=4)

    print("EVPI: ", np.round(EVPI,3))
    print("Expected prior utility: ", np.round(Eu_prior,3))
    print("Expected pre-posterior utility: ", np.round(Eu_preposterior,3))
    print("Prior action decision: ", astar_prior)