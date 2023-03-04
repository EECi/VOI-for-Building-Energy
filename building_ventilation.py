"""Perform EVPI calculation for building ventilation scheduling example."""

import numpy as np
from functools import cache

from evpi import compute_EVPI


if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    ventilation_rates = [1,3,5,10] # air changes per hour

    # 2. Define sampling function for uncertain parameter(s)
    def occupancy_sampler(n_samples, n_staff=55):
        return np.random.randint(0,n_staff,size=n_samples)

    # 3. Define system dynamics (intermediate computations)
    @cache
    def R_function(vent_rate, occ_value):
        """Compute probability of individual infection."""

        vent_fresh = vent_rate/3600
        kappa = 0.39/3600
        lam = 0.636/3600
        loss_rate = vent_fresh + kappa + lam
        InfectionRate = 0.0218
        NumInfected = InfectionRate*occ_value
        riskConst = 410
        inhRate = 0.521/1000
        N_r = 0.484
        V = 2000
        t_max = 8*3600
        nstep = 3201
        deltat = t_max/(nstep-1)

        r = np.zeros((nstep))
        c = np.zeros((nstep))
        ninh=np.zeros((nstep))

        for ii in range(nstep-1):
            c[ii+1] = NumInfected*N_r/V/loss_rate + (c[ii] - (NumInfected*N_r/V/loss_rate))*np.exp(-1*loss_rate*deltat)
            ninh[ii+1] = ninh[ii] + inhRate*deltat*c[ii+1]
            r[ii+1] = 1-np.exp(-1*ninh[ii+1]/riskConst)

        return r[-1]

    # 4. Define utility function
    @cache
    def utility(vent_rate, occ_value):

        # set up cost parameters
        sick_day_cost = 128 # £/day
        sick_days_per_infection = 3
        venting_costs = {1:6.08, 3:18.25, 5:30.42, 10:60.83} # £/day - pre-computed

        # compute individual infection probability
        infection_prob = R_function(vent_rate, occ_value)

        # compute expected cost over binomial distribution of number of infections on the given day (occupancy, infeciton prob.)
        num_infections_samples = np.random.binomial(occ_value,infection_prob,size=10000) # sample of number of infections on day
        #cost_samples = [sick_days_per_infection*sick_day_cost*infections + venting_costs[vent_rate] for infections in num_infections_samples]
        cost_samples = sick_days_per_infection*sick_day_cost*num_infections_samples + venting_costs[vent_rate]

        return -1*np.mean(cost_samples) # utility [+£/day]

    # 5. Perform EVPI computation
    results = compute_EVPI(ventilation_rates, occupancy_sampler, utility, n_samples=int(1e5))

    print("EVPI: ", np.round(results[0],3))
    print("Expected prior utility: ", np.round(results[1],3))
    print("Expected pre-posterior utility: ", np.round(results[2],3))
    print("Prior action decision: ", results[3])
    print("Pre-posterior action decision counts: ", results[4])