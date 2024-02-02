"""Perform EVPI calculation for ground source heat pump supply system design example."""

import numpy as np
import pandas as pd

import os
import csv
import json
from utils import cached

import logging
from cmdstanpy import CmdStanModel

from models import create_BH_pyg, simulate_BH_pyg, find_factor_BH, compute_bh_energy_usage, compute_aux_energy_usage
from voi import compute_EVPI, compute_EVII

if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    borehole_lengths = np.linspace(110,190,17)

    # 2. Define sampling functions for uncertain parameter(s)
    prior_mu = 1.94
    prior_sigma = 0.31
    discr_points = np.arange(0,4,1e-2) # discretisation points for ground thermal conductivity

    # Turn off cmdstan info logging
    stan_log = False
    if not stan_log:
        logger = logging.getLogger('cmdstanpy')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.WARNING)

    def prior_ks_sampler(n_samples, mu=prior_mu, sigma=prior_sigma, discr_points=discr_points):
        """Sample values from ground thermal conductivity prior distribution.
        Discretise to enable utility caching for computational efficiency."""
        cont_samples = np.random.normal(mu,sigma,size=n_samples)
        return np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in cont_samples])

    def prior_theta_z_sampler(n_samples, error, mu=prior_mu, sigma=prior_sigma, discr_points=discr_points, thin_factor=10):
        """Sample theta and z values from prior model of thermal conductivity and measurements."""
        data = {'mu': mu,'sigma': sigma,'error': error}
        prior_stan_model_path = os.path.join('stan_models','GSHP_prior.stan')
        prior_stan_model = CmdStanModel(stan_file=prior_stan_model_path)
        thin_factor = 10
        prior_fit = prior_stan_model.sample(
            data=data,
            inits={'theta': mu, 'z':mu}, # prevent invalid initial values
            iter_warmup=n_samples, iter_sampling=n_samples*thin_factor, chains=1,
            show_progress=False
        )

        thetas = prior_fit.stan_variable('theta')
        reduced_thetas = np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in thetas[::thin_factor]])
        zs = prior_fit.stan_variable('z')
        reduced_zs = np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in zs[::thin_factor]])

        return reduced_thetas, reduced_zs

    def posterior_theta_sampler(measurement, n_samples, error, mu=prior_mu, sigma=prior_sigma, discr_points=discr_points, thin_factor=10):
        """Sample theta values from posterior model of thermal conductivity given measurement, z."""
        data = {'mu': mu,'sigma': sigma,'error': error, 'z': measurement}
        post_stan_model_path = os.path.join('stan_models','GSHP_posterior.stan')
        post_stan_model = CmdStanModel(stan_file=post_stan_model_path)
        post_fit = post_stan_model.sample(
            data=data,
            inits={'theta': measurement},
            iter_warmup=n_samples, iter_sampling=n_samples*thin_factor, chains=1,
            show_progress=False
        )

        candidate_thetas = post_fit.stan_variable('theta')
        reduced_candidate_thetas = np.array([discr_points[np.argmin(np.abs(discr_points-s))] for s in candidate_thetas[::thin_factor]])

        return reduced_candidate_thetas

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
    def utility(bh_length, k, load_df=load_data):

        # set up cost parameters
        num_boreholes = 12 # number of boreholes supplying building
        bh_cost_per_m = 70 # borehole capital costs per meter length, £/m
        elec_unit_cost = 0.326 # £/kWh

        bh_energy_usage, aux_energy_usage = compute_system_energy_usage(bh_length, k, load_df, num_boreholes) # kWh for both

        return -1*((bh_energy_usage + aux_energy_usage)*elec_unit_cost + bh_length*bh_cost_per_m*num_boreholes)

    # load cached utility evaluations if available
    cache_path = os.path.join('data','caches','GSHP_utility_cache_redf-%s.json'%reduction_factor)
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            utility.__wrapped__.cache = json.load(file)


    # 5. Perform VOI calculations
    # ========================================================================
    print("\nPerforming EVPI calculation...")

    EVPI_results = compute_EVPI(
        borehole_lengths,
        prior_ks_sampler,
        utility,
        n_samples=int(1e6)
    )

    print("EVPI: ", np.round(EVPI_results[0],3))
    print("Expected prior utility: ", np.round(EVPI_results[1],3))
    print("Expected pre-posterior utility: ", np.round(EVPI_results[2],3))
    print("Prior action decision: ", EVPI_results[3])
    print("Pre-posterior action decision counts: ", EVPI_results[4])

    # save utility evaluation cache
    with open(cache_path, 'w') as file:
        json.dump({key:utility.__wrapped__.cache[key] for key in sorted(utility.__wrapped__.cache.keys())}, file, indent=4)


    print("\nPerforming EVII calculations...")

    results_file = os.path.join('results','GSHP_EVII_results.csv')
    columns = ['error-sigma','EVII','expected_prior_utility','expected_preposterior_utility','n_prior_samples','n_measurement_samples']
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    n_prior_samples = int(1e5)
    n_measurement_samples = int(1e5)

    test_errors = [0.125,0.085,0.05,0.025]
    for error in test_errors:
        # construct sampling functions for given error
        prior_theta_z_sampler_with_error = lambda n_samples: prior_theta_z_sampler(n_samples, error)
        posterior_theta_sampler_with_error = lambda measurement, n_samples: posterior_theta_sampler(measurement, n_samples, error)

        EVII_results = compute_EVII(
            borehole_lengths,
            prior_theta_z_sampler_with_error,
            posterior_theta_sampler_with_error,
            utility,
            n_prior_samples=n_prior_samples,
            n_measurement_samples=n_measurement_samples
        )

        print("\nMesurement error: %s%%"%np.round(error*100,1))
        print("EVII: ", np.round(EVII_results[0],3))
        print("Expected prior utility: ", np.round(EVII_results[1],3))
        print("Expected pre-posterior utility: ", np.round(EVII_results[2],3))
        print("Pre-posterior std error: ", np.round(EVII_results[5],3))

        # save results
        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([error, EVII_results[0], EVII_results[1], EVII_results[2], n_prior_samples, n_measurement_samples, EVII_results[5]])
    # ========================================================================

    # save utility evaluation cache
    with open(cache_path, 'w') as file:
        json.dump({key:utility.__wrapped__.cache[key] for key in sorted(utility.__wrapped__.cache.keys())}, file, indent=4)
