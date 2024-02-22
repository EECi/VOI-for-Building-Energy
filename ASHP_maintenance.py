"""Perform EVPI calculation for air source heatpump maintenance scheduling example."""

import os
import csv

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import numba

from voi import fast_EVPI, fast_EVII


if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    maintenance_freqs = np.arange(13) # number of maintenance operations per year

    # 2. Define sampling function for uncertain parameter(s)
    @numba.njit(fastmath=True)
    def discritise_samples(samples, discr_points,bins):
        return discr_points[(np.digitize(samples,bins)-1)]

    def prior_theta_sampler(n_samples):
        """Sample vectors of uncertain parameter values from prior distributions."""

        # set up parameters
        # =================
        # sample discretisation used to allow caching for compuational efficiency
        discr_delta = 1e-2

        # alpha - truncated Gaussian [0,1)
        alpha_mu = 1e-2
        alpha_sigma = 0.25
        alpha_discr_points = np.arange(0,1-discr_delta,discr_delta)
        alpha_bins = np.array([-1*np.inf, *[np.mean(alpha_discr_points[i:i+2]) for i in range(len(alpha_discr_points)-1)], np.inf])
        # epsilon - log Normal
        epsilon_mu = 0
        epsilon_sigma = 0.1
        epsilon_discr_points = np.arange(-1,1,discr_delta)
        epsilon_bins = np.array([-1*np.inf, *[np.mean(epsilon_discr_points[i:i+2]) for i in range(len(epsilon_discr_points)-1)], np.inf])
        # SPF' - Gaussian
        spf_dash_mu = 2.9
        spf_dash_sigma = 0.167
        spf_dash_discr_points = np.arange(2,4,discr_delta)
        spf_dash_bins = np.array([-1*np.inf, *[np.mean(spf_dash_discr_points[i:i+2]) for i in range(len(spf_dash_discr_points)-1)], np.inf])
        # elec unit cost (£/kWh) - Gaussian
        elec_unit_cost_mu = 0.326
        elec_unit_cost_sigma = 0.016
        # annual load - Gaussian
        annual_load_mu = 12.56e6
        annual_load_sigma = 1.358e6

        theta_matrix = np.vstack([
            discritise_samples(stats.truncnorm(-1*alpha_mu/alpha_sigma,(1-alpha_mu)/alpha_sigma,loc=alpha_mu,scale=alpha_sigma).rvs(n_samples),alpha_discr_points,alpha_bins), # alpha
            discritise_samples(stats.norm(loc=epsilon_mu,scale=epsilon_sigma).rvs(n_samples),epsilon_discr_points,epsilon_bins), # epsilon
            discritise_samples(stats.norm(loc=spf_dash_mu,scale=spf_dash_sigma).rvs(n_samples),spf_dash_discr_points,spf_dash_bins), # spf_dash
            stats.norm(loc=elec_unit_cost_mu,scale=elec_unit_cost_sigma).rvs(n_samples), # elec_unit_cost
            stats.norm(loc=annual_load_mu,scale=annual_load_sigma).rvs(n_samples), # annual_load
        ])

        return theta_matrix.T

    def prior_theta_and_partial_perfect_z_sampler(n_samples, perfect_info_params=None):
        """Sample thetas and zs for partial perfect information for specified parameters."""

        thetas = prior_theta_sampler(n_samples)

        parameters = ['alpha', 'epsilon', 'spf_dash', 'elec_unit_cost', 'annual_load']
        if perfect_info_params is None:
            perfect_info_params = {param: False for param in parameters}
        measure_params = np.array(list(perfect_info_params.values()),dtype=bool)

        zs = np.where(measure_params==True, thetas, np.nan)

        return thetas, zs

    def partial_perfect_info_theta_sampler(measurement, n_samples):
        """Sample thetas for case of partial perfect information with specified measurement."""

        thetas = prior_theta_sampler(n_samples)
        thetas_with_perfect_info = np.where(np.isnan(measurement), thetas, measurement)

        return thetas_with_perfect_info


    # 3. Define system dynamics (intermediate computations)
    @numba.njit(fastmath=True)
    def compute_beta(maint_freq, epsilon):
        beta_a = 0.05
        beta_b = 2.5
        gamma = 1.4
        return ((beta_a*maint_freq**gamma)/(beta_b+maint_freq**gamma))*(1+epsilon)

    @numba.njit(fastmath=True)
    def compute_spf(alpha, beta, spf_dash):
        """Compute seasonal performance factor."""
        return spf_dash*(1-alpha)*(1+beta)

    # 4. Define utility function
    @numba.njit(fastmath=True)
    def utility(maint_freq, theta):

        # unpack theta
        alpha = theta[0]
        epsilon = theta[1]
        spf_dash = theta[2]
        elec_unit_cost = theta[3] # £/kWh
        annual_load = theta[4] # kWh/year

        # set up cost parameters
        maint_unit_cost = 552.5*30 # £ per maintainence operation on 10 ASHPs (originally 4 for 1.75GWh/year)

        # compute spf
        beta = compute_beta(maint_freq, epsilon)
        spf = compute_spf(alpha, beta, spf_dash)

        # compute cost contributions
        maintenance_cost = maint_unit_cost*maint_freq # £/year
        electricity_cost = annual_load*elec_unit_cost/spf # £/year

        return -1*(maintenance_cost+electricity_cost) # utility [+£/year]



    # 5. Perform VOI calculations
    # ========================================================================
    print("\nPerforming EVPI calculation...")

    n_samples = int(1e7)

    results = fast_EVPI(
        maintenance_freqs,
        prior_theta_sampler,
        utility,
        n_samples=n_samples,
        report_prepost_freqs=True,
        return_utility_samples=True
    )

    print("EVPI: ", np.round(results[0],3))
    print("Expected prior utility: ", np.round(results[1],3))
    print("Expected pre-posterior utility: ", np.round(results[2],3))
    print("Prior action decision: ", results[3])
    print("Pre-posterior action decision counts: ", results[4])

    # Plot results
    fig, ax = plt.subplots()
    colors = ['grey' if i != results[3] else 'k' for i in list(results[4].keys())]
    sns.barplot(x=list(results[4].keys()),y=np.array(list(results[4].values()))/n_samples, palette=colors, ax=ax)
    plt.xlabel("Maintenance frequency")
    plt.ylabel("Propotion of true optimal actions")
    plt.show()

    clip_lower = -1e7
    clip_upper = -5e5
    fig, ax = plt.subplots()
    sns.kdeplot(np.clip(results[-2],clip_lower,clip_upper)/1e6, ax=ax, c='k')
    plt.vlines(results[1]/1e6,0,1.3,colors='k',linestyles='dashed')
    plt.text(results[1]/1e6+0.1, 0.4, "Expected prior utility", rotation=90, verticalalignment='center', horizontalalignment='left')
    plt.xlim(-5,-0.8)
    plt.ylim(0,1.3)
    plt.xlabel("Utility (£m/year)")
    plt.ylabel("Density")
    plt.show()