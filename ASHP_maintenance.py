"""Perform EVPI calculation for air source heatpump maintenance scheduling example."""

import numpy as np
import scipy.stats as stats
from functools import cache

from evpi import compute_EVPI


if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    maintenance_freqs = np.arange(1,13) # number of maintenance operations per year

    # 2. Define sampling function for uncertain parameter(s)
    def alpha_sampler(n_samples, mu=1e-2, sigma=0.25):
        #return np.random.normal(mu,sigma,size=n_samples)
        return stats.truncnorm(-1*mu/sigma,np.inf,loc=mu,scale=sigma).rvs(n_samples)

    # 3. Define system dynamics (intermediate computations)
    @cache
    def compute_beta(maint_freq):
        beta_a = 0.05
        beta_b = 2.5
        gamma = 1.4
        return (beta_a*maint_freq**gamma)/(beta_b+maint_freq**gamma)
    
    def compute_spf(alpha, beta):
        """Compute seasonal performance factor."""
        spf_dash = 3 # initial heat pump efficiency
        return spf_dash*(1-alpha)*(1+beta)

    # 4. Define utility function
    def utility(maint_freq, alpha):

        # set up cost parameters
        elec_unit_cost = 0.51 # £/kWh
        annual_load = 1750000 # kWh/year
        maint_unit_cost = 2210 # £ per maintainence operation on all 4 ASHPs

        # compute spf
        beta = compute_beta(maint_freq)
        spf = compute_spf(alpha, beta)

        # compute cost contributions
        maintenance_cost = maint_unit_cost*maint_freq # £/year
        electricity_cost = annual_load*elec_unit_cost/spf # £/year

        return -1*(maintenance_cost+electricity_cost) # £/year

    # 5. Perform EVPI computation
    EVPI, Eu_prior, Eu_preposterior, astar_prior = compute_EVPI(maintenance_freqs, alpha_sampler, utility, n_samples=int(1e6))

    print("EVPI: ", np.round(EVPI,3))
    print("Expected prior utility: ", np.round(Eu_prior,3))
    print("Expected pre-posterior utility: ", np.round(Eu_preposterior,3))
    print("Prior action decision: ", astar_prior)