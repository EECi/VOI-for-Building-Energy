"""Functions defining GSHP system model."""

import numpy as np
import pygfunction as gt

from scipy.constants import pi


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