"""Perform EVPI calculation for building ventilation scheduling example."""

from calendar import c
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from functools import cache
from voi import fast_EVPI


if __name__ == '__main__':

    np.random.seed(0) # seed sampling

    # 1. Define action space
    ventilation_rates = [1,3,6,12,20] # air changes per hour

    # 2. Define sampling function for uncertain parameter(s)
    def prior_theta_sampler(n_samples, n_staff=100):
        # building occupancy - Uni. random int in range [0,n_staff]
        return np.random.randint(0,n_staff,size=n_samples)

    # 3. Define system dynamics (intermediate computations)
    @cache
    def R_function(vent_rate, occupancy, volume, N_r, InfectionRate):
        """Compute probability of individual infection."""

        vent_fresh = vent_rate/3600
        kappa = 0.39/3600 # gravitational settling
        lam = 0.636/3600 # viral decay
        loss_rate = vent_fresh + kappa + lam
        NumInfected = InfectionRate*occupancy
        riskConst = 410 # constant calculated for Covid-19
        inhRate = 0.521/1000 # inhalation rate average sedentary person

        office_hours = 8 # number of hours spent in office per day
        t_max = 3600*office_hours
        deltat = 10 # seconds
        n_steps = t_max//deltat

        r = np.zeros((n_steps))
        c = np.zeros((n_steps))
        ninh=np.zeros((n_steps))

        for ii in range(n_steps-1):
            c[ii+1] = NumInfected*N_r/volume/loss_rate + (c[ii] - (NumInfected*N_r/volume/loss_rate))*np.exp(-1*loss_rate*deltat)
            ninh[ii+1] = ninh[ii] + inhRate*deltat*c[ii+1]
            r[ii+1] = 1-np.exp(-1*ninh[ii+1]/riskConst)

        return r[-1]

    # 4. Define utility function
    def utility(vent_rate, occupancy, floor_area, elec_cost, N_r, InfectionRate):

        # set up cost parameters
        sick_day_cost = 128 # £/day
        sick_days_per_infection = 3
        ceiling_height = 2.4 # m
        volume = ceiling_height*floor_area # room volume, m^3

        # compute venting costs
        vent_hours = 10 # hours per day ventilation is run
        hourly_vented_volume = vent_rate*volume*1000 # litres
        venting_load = 1.9 # W/l/s
        venting_efficiency = 0.6
        vent_power = (hourly_vented_volume/3600)*venting_load/venting_efficiency # required power to run ventilation, W
        vent_energy = vent_power*vent_hours/1000 # kWh/day
        venting_cost = vent_energy*elec_cost # £/day

        # compute individual infection probability
        infection_prob = R_function(vent_rate, occupancy, volume, N_r, InfectionRate)

        # compute expected cost (cost of expected number of infections on the given day + venting cost)
        # treat infections as Binomial r.v. - expectation is n*p
        expected_infections = occupancy*infection_prob
        total_cost = sick_days_per_infection*sick_day_cost*expected_infections + venting_cost

        return -1*total_cost # utility [+£/day]


    # 5. Perform EVPI computations
    # ========================================================================

    def calculate_EVPI(floor_area_per_person, elec_cost, N_r, InfectionRate, n_samples=int(1e6)):

        n_staff = 100 # number of staff in building
        # NOTE, all costs *should*? scale linearly with this value
        # well, sort of, there is some discrepency introduced by the discretisation
        # as there are only integer occupancy values sampled

        floor_area = floor_area_per_person*n_staff

        def scenario_theta_sampler(n_samples):
            return prior_theta_sampler(n_samples, n_staff)

        @np.vectorize
        def scenario_utility(ventilation_rates, theta):
            return utility(ventilation_rates, theta, floor_area, elec_cost, N_r, InfectionRate)

        results = fast_EVPI(
            ventilation_rates,
            scenario_theta_sampler,
            scenario_utility,
            n_samples,
            report_prepost_freqs=True,
            return_utility_samples=True
        )

        return results


    base_N_r = 0.484 # generation rate calculated using aerosol cut-off of 10 microns and viral load 10^9 copies per ml
    base_InfectionRate = 0.0218 # prevalence of infection in population
    base_floor_area_per_person = 10 # m^2/person
    base_elec_cost = 0.326 # £/kWh

    # Calculate EVPI for base case
    base_results = calculate_EVPI(base_floor_area_per_person, base_elec_cost, base_N_r, base_InfectionRate)

    print("(Base) EVPI: ", np.round(base_results[0],3))
    print("(Base) Expected prior utility: ", np.round(base_results[1],3))
    print("(Base) Expected pre-posterior utility: ", np.round(base_results[2],3))
    print("(Base) Prior action decision: ", base_results[3])
    print("(Base) Pre-posterior action decision counts: ", base_results[4])

    # Plot prior utility distributions for each action
    n_staff=100
    thetas = prior_theta_sampler(n_samples=int(1e6), n_staff=n_staff)
    @np.vectorize
    def scenario_utility(ventilation_rates, theta):
        return utility(ventilation_rates, theta, base_floor_area_per_person*n_staff, base_elec_cost, base_N_r, base_InfectionRate)
    a_utils = [scenario_utility(a,thetas.T) for a in ventilation_rates]
    fig, ax = plt.subplots()
    for i,a in enumerate(ventilation_rates):
        sns.kdeplot(a_utils[i], ax=ax, label=a, cut=0)
        a_ax = ax.lines[-1]
        plt.vlines(a_ax.get_xdata()[-1],0,a_ax.get_ydata()[-1],color=a_ax.get_c(),linestyle="--")
    plt.xlim(-600,0)
    # plt.ylim(0,1.3)
    plt.xlabel("Utility (£/day)")
    plt.ylabel("Density")
    plt.legend(title="Ventilation rate", ncols=5)
    plt.savefig(os.path.join('plots',"builing_vent_prior_u_dists_by_action.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

    # # Calculate EVPI for varying floor area per person
    # print("\nFloor area per person:")
    # fapp_values = [5,10,15,20,25] # m^2/person
    # fapp_results = []
    # for fapp in fapp_values:
    #     fapp_results.append(calculate_EVPI(fapp, base_elec_cost, base_N_r, base_InfectionRate))

    # # Calculate EVPI for varying electricity cost
    # print("\nElectricity cost:")
    # elec_cost_values = [0.26,0.28,0.30,0.326,0.35,0.40]
    # elec_cost_results = []
    # for elec_cost in elec_cost_values:
    #     elec_cost_results.append(calculate_EVPI(base_floor_area_per_person, elec_cost, base_N_r, base_InfectionRate))

    # # Calculate EVPI for varying N_r
    # print("\nN_r:")
    # N_r_values = [0.048,0.484,4.843]
    # N_r_results = []
    # for N_r in N_r_values:
    #     N_r_results.append(calculate_EVPI(base_floor_area_per_person, base_elec_cost, N_r, base_InfectionRate))

    # Calculate EVPI for varying InfectionRate
    print("\nInfectionRate:")
    InfectionRate_values = [0.005,0.01,0.02,0.03,0.04,0.05]
    InfectionRate_results = []
    for InfectionRate in InfectionRate_values:
        InfectionRate_results.append(calculate_EVPI(base_floor_area_per_person, base_elec_cost, base_N_r, InfectionRate))

    # # Print results
    # print("\nVarying floor area per person:")
    # print(fapp_values)
    # print([r[0] for r in fapp_results])
    # print([r[3] for r in fapp_results])
    # print("\nVarying electricity cost:")
    # print(elec_cost_values)
    # print([r[0] for r in elec_cost_results])
    # print([r[3] for r in elec_cost_results])
    # print("\nVarying N_r:")
    # print(N_r_values)
    # print([r[0] for r in N_r_results])
    # print([r[3] for r in N_r_results])
    print("\nVarying InfectionRate:")
    print(InfectionRate_values)
    print([r[0] for r in InfectionRate_results])
    print([r[3] for r in InfectionRate_results])

    # Plot change in prior utility distributions as infrection rate varies
    fig, ax = plt.subplots()
    for i,IR in enumerate(InfectionRate_values):
        sns.kdeplot(InfectionRate_results[i][-2], ax=ax, label=IR, cut=0)
        a_ax = ax.lines[-1]
        plt.vlines(a_ax.get_xdata()[-1],0,a_ax.get_ydata()[-1],color=a_ax.get_c(),linestyle="--")
    plt.xlim(-450,0)
    # plt.ylim(0,1.3)
    plt.xlabel("Utility (£/day)")
    plt.ylabel("Density")
    plt.legend(title="Infection rate", ncols=3)
    plt.savefig(os.path.join('plots',"builing_vent_prior_u_dists_with_InfRate.pdf"), format="pdf", bbox_inches="tight")
    plt.show()