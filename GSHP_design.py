"""Perform EVPI calculation for ground source heat pump supply system design example."""

import numpy as np
import pandas as pd

import os
import json
from utils import cached

from voi import compute_EVPI
import pygfunction as gt
from scipy.constants import pi

if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    borehole_lengths = np.linspace(140,200,13)

    # 2. Define sampling function for uncertain parameter(s)
    def ks_sampler(n_samples, mu=2, sigma=0.12):
        """Sample values from ground thermal conductivity prior and
        discretise for computational efficiency (enable utility caching)."""
        cont_samples = np.random.normal(mu,sigma,size=n_samples)
        discr_points = np.arange(0,4,1e-2)
        return np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in cont_samples])

    # 3. Define system dynamics (intermediate computations)
    # ========================================================================
    # read in heating load data and manipulate (reduce sampling rate & extend to create 50 years of data)
    load_data = pd.read_csv(os.path.join('data','Simplified_London_new.csv'),header=None,names=['Hours','Load_W'])
    load_data['Load_W']  = load_data['Load_W']*-1 # sign correction
    reduction_factor = 10 # factor to reduce no. of data rows by using chunk means over rows
    load_data['Load_W'] = load_data['Load_W'].rolling(reduction_factor).mean()
    load_data = load_data.iloc[::reduction_factor,:]
    load_data.iloc[0] = 0
    extension_factor = 50 # factor to extend length of reduced dataframe out by, i.e. increase effective duration by factor
    load_data = pd.concat([load_data]*extension_factor, ignore_index=True)
    time_step = load_data['Hours'][1] - load_data['Hours'][0] # hours
    load_data['Hours'] = (load_data['Hours'][0] + time_step*np.arange(len(load_data))).astype(int)

    
    def create_BH_pyg(BHE_L,ks):
        """Create borehole object to perform simulation."""
        # Defining Pile or Borehole propreties and defining thermal design parameters
        D = 1.0             # Borehole buried depth (m)
        # H = 100               # BHE length [m] - varied here
        rb = 0.15               # BHE radius [m]   
        # ks = 1.6925           # soil thermal conductivity [W/mK] - varied here
        m_flow = 20/60       # Total fluid mass flow rate [kg/s]
        k_g = 1.0           # Grout thermal conductivity [W/m.K]
        
        # pipe properties, radius, position, thermal conductivity
        pipe_r_inner = 0.0269/2
        pipe_r_outter = 0.0023+0.0269/2
        D_s = 0.1          # Shank spacing [m]
        pipe_k = 0.4        # pipe thermal conductivity [W/mK]
        epsilon = 1.0e-6    # Pipe roughness [m]
        pos_double = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]
        
        # Fluid properties - The fluid is propylene-glycol (20 %) at 20 degC
        m_flow = 20/60       # Total fluid mass flow rate [kg/s]
        fluid = gt.media.Fluid('MPG', 20.)
        cp_f = fluid.cp     # Fluid specific isobaric heat capacity [J/kg.K]
        den_f = fluid.rho   # Fluid density [kg/m3]
        visc_f = fluid.mu   # Fluid dynamic viscosity [kg/m.s]
        k_f = fluid.k       # Fluid thermal conductivity [W/m.K]
        
        borehole = gt.boreholes.Borehole(BHE_L, D, rb, x=0., y=0.)
        boreField = [borehole]
        
        # Pipe thermal resistance
        R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            pipe_r_inner, pipe_r_outter, pipe_k)
        # Fluid to inner pipe wall thermal resistance (Single U-tube and double
        # U-tube in series)
        h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow, pipe_r_inner, visc_f, den_f, k_f, cp_f, epsilon)
        R_f_ser = 1.0/(h_f*2*pi*pipe_r_inner)
        
        # U-tubes class - double U-tube connected in series
        UTubes = gt.pipes.MultipleUTube(pos_double, pipe_r_inner, pipe_r_outter,
                                                 borehole, ks, k_g, R_f_ser + R_p,
                                                 nPipes=2, config='series')
        
        return boreField, UTubes, m_flow, cp_f

    def simulate_BH_pyg(BHE_L,ks,df_load_data,boreField,UTubes,m_flow,cp_f):
        
        # Define ground properties
        rho = 1950.6            # Density [kg/m^3]
        Cp = 956.23             # Specific heat capacity [J/(kg K)]
        Cpvs = rho*Cp           # soil specific heat * soil density
        alpha = ks/Cpvs         # soil thermal diffussivity
        T0 = 12.5               # Farfield temperature [degC] 
    
        # Simulation parameters - time
        time = df_load_data["Hours"].values #dt * np.arange(1, Nt+1)
        dt = (time[-1]-time[-2])*3600.                  # Time step (s)
        tmax = time[-1]* 3600. #1.*8760. * 3600.     # Maximum time (s)
        Nt = len(time) #int(np.ceil(tmax/dt))  # Number of time steps
        
        # Loading
        Q = df_load_data["Load_W"].values
        # Load aggregation scheme
        LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)
    
        # G-function calculations
        # g-Function calculation options
        options = {'nSegments': 8,
                   'disp': False}
        # Get time values needed for g-function evaluation
        time_req = LoadAgg.get_times_for_simulation()
        # Calculate g-function
        gFunc = gt.gfunction.gFunction(
            boreField, alpha, time=time_req, options=options)
        # Initialize load aggregation scheme
        LoadAgg.initialize(gFunc.gFunc/(2*pi*ks))
        
        T_b = np.zeros(Nt)
        T_f_in_double_ser = np.zeros(Nt)
        T_f_out_double_ser = np.zeros(Nt)
        
        for i, (t, Q_b_i) in enumerate(zip(time, Q)):
            # Increment time step by (1)
            LoadAgg.next_time_step(t)
    
            # Apply current load
            LoadAgg.set_current_load(Q_b_i/BHE_L)
    
            # Evaluate borehole wall temperature
            deltaT_b = LoadAgg.temporal_superposition()
            T_b[i] = T0 - deltaT_b
    
            # Evaluate inlet fluid temperature
            T_f_in_double_ser[i] = UTubes.get_inlet_temperature(
                    Q[i], T_b[i], m_flow, cp_f)
    
            # Evaluate outlet fluid temperature
            T_f_out_double_ser[i] = UTubes.get_outlet_temperature(
                    T_f_in_double_ser[i],  T_b[i], m_flow, cp_f)
        Tf = np.array((T_f_in_double_ser,T_f_out_double_ser)).T
        
        return Tf
    
    def find_factor_BH(Tmax, Tmin, Tincr, Tprec, BHE_L,ks,df_load_data, boreField, UTubes, m_flow, cp_f):
        
        def get_factor_linear(factor_low, factor_high, T_low,T_high,Tdesired):
            if (T_low==T_high):
                return float('inf')
            
            m = (factor_high - factor_low)/(T_high-T_low)
            c = factor_low-m*T_low
            return c + m*Tdesired
        
        def get_initial_estimate(Tmax, Tmin, df_load_data):
               
                # get better estimates for factor
                factor_1 = 0.1
                df_load_data_copy = df_load_data.copy()
                df_load_data_copy["Load_W"] = df_load_data_copy["Load_W"]*factor_1
                Tf = simulate_BH_pyg(BHE_L, ks, df_load_data_copy, boreField, UTubes, m_flow, cp_f)
                maxT_1 = Tf.max()
                minT_1 = Tf.min()
    
                factor_2 = 10
                df_load_data_copy = df_load_data.copy()
                df_load_data_copy["Load_W"] = df_load_data_copy["Load_W"]*factor_2
                Tf = simulate_BH_pyg(BHE_L, ks, df_load_data_copy, boreField, UTubes, m_flow, cp_f)
                maxT_2 = Tf.max()
                minT_2 = Tf.min()
                
                factor_max = get_factor_linear(factor_1, factor_2, maxT_1, maxT_2, Tmax)
                factor_min = get_factor_linear(factor_1, factor_2, minT_1, minT_2, Tmin)
    
                return min(factor_max,factor_min) 
            
        factor = get_initial_estimate(Tmax, Tmin, df_load_data)
        df_load_data_copy = df_load_data.copy()
        df_load_data_copy["Load_W"] = df_load_data_copy["Load_W"]*factor
        Tf = simulate_BH_pyg(BHE_L, ks, df_load_data_copy, boreField, UTubes, m_flow, cp_f)
        Tmax_calc = Tf.max()
        Tmin_calc = Tf.min()
        
        while (abs(Tmax_calc-Tmax)>Tprec and abs(Tmin_calc-Tmin)>Tprec):
            if (Tmax_calc>Tmax or Tmin_calc<Tmin):
                factor = factor-Tincr
            else:
                factor = factor+Tincr
            df_load_data_copy = df_load_data.copy()
            df_load_data_copy["Load_W"] = df_load_data_copy["Load_W"]*factor
            Tf = simulate_BH_pyg(BHE_L, ks, df_load_data_copy, boreField, UTubes, m_flow, cp_f)
            Tmax_calc = Tf.max()
            Tmin_calc = Tf.min()
        return factor, Tmax_calc, Tmin_calc
        
    
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
        max_factor = 61.09925996/num_boreholes # represents building with 13.24 kW average, 25.18 kW peak

        time_step = load_data['Hours'][1] - load_data['Hours'][0] # hours

        boreField, UTubes, m_flow, cp_f = create_BH_pyg(bh_length,ks)
        f,_,_= find_factor_BH(35, 5, 0.5, 0.1, bh_length, ks, load_data, boreField, UTubes, m_flow, cp_f)

        if (f >= max_factor): # GSHP can provide heating demand
            f = max_factor
            aux_energy_usage = 0
        else: # auxiliary heating system required
            f_aux = max_factor - f
            aux_energy_usage = compute_aux_energy_usage(load_data['Load_W'].to_numpy()*f_aux*num_boreholes,time_step)

        ground_load_used_per_bh = load_data['Load_W'].to_numpy()*f # compute actual ground load consumed by each borehole
        used_load_df = pd.DataFrame({'Hours':load_data['Hours'],'Load_W':ground_load_used_per_bh})
        temps = simulate_BH_pyg(bh_length, ks, used_load_df, boreField, UTubes, m_flow, cp_f).mean(axis=1) # compute fluid temps over true borehole operation
        bh_energy_usage = compute_bh_energy_usage(temps, ground_load_used_per_bh*num_boreholes, time_step)

        return bh_energy_usage, aux_energy_usage
    # ========================================================================

    # 4. Define utility function
    @cached
    def utility(bh_length, ks, load_df=load_data):

        # set up cost parameters
        num_boreholes = 12 # number of boreholes supplying building
        bh_cost_per_m = 70 # borehole capital costs per meter length, £/m
        elec_unit_cost = 0.34 # £/kWh

        bh_energy_usage, aux_energy_usage = compute_system_energy_usage(bh_length, ks, load_df, num_boreholes) # kWh for both

        return -1*((bh_energy_usage + aux_energy_usage)*elec_unit_cost + bh_length*bh_cost_per_m*num_boreholes)

    # load cached utility evaluations if available
    cache_path = os.path.join('data','caches','GSHP_utility_new_cache_redf-%s.json'%reduction_factor)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            utility.__wrapped__.cache = json.load(file)

    # 5. Perform EVPI computation
    results = compute_EVPI(borehole_lengths, ks_sampler, utility, n_samples=int(1e6))

    # save utility evaluation cache
    with open(cache_path, 'w') as file:
        json.dump({key:utility.__wrapped__.cache[key] for key in sorted(utility.__wrapped__.cache.keys())}, file, indent=4)

    print("EVPI: ", np.round(results[0],3))
    print("Expected prior utility: ", np.round(results[1],3))
    print("Expected pre-posterior utility: ", np.round(results[2],3))
    print("Prior action decision: ", results[3])
    print("Pre-posterior action decision counts: ", results[4])
