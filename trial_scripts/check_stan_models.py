"""Test pystan."""

# NOTE
# Need to use cmdstanpy installed via conda-forge
# see https://mc-stan.org/cmdstanpy/installation.html#conda-install-cmdstanpy-cmdstan-c-toolchain

import os
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cmdstanpy import CmdStanModel

# define parameters of probabilistic model
data = {
    'mu': 10,
    'sigma': 1,
    'error': 0.15
}

# sample (theta,z) values from posterior theta|z
prior_stan_model_path = os.path.join('stan_models','GSHP_prior.stan')
prior_stan_model = CmdStanModel(stan_file=prior_stan_model_path)

prior_fit = prior_stan_model.sample(
    data=data,
    inits={'theta': 10, 'z':10}, # prevent invalid initial values
    iter_warmup=5000, iter_sampling=100000, chains=1,
    show_progress=True
)

thetas = prior_fit.stan_variable('theta')
zs = prior_fit.stan_variable('z')

thinner = 100 # thin samples for plotting
plt.scatter(thetas[::thinner], zs[::thinner], alpha=0.1)
plt.xlabel('theta')
plt.ylabel('z')
plt.show()

# sample thetas from conditional theta|z(k)
measurements = [5,8,10,12,15]

fig,ax = plt.subplots()

for z in measurements:
    post_data = copy.deepcopy(data)
    post_data['z'] = z
    post_stan_model_path = os.path.join('stan_models','GSHP_posterior.stan')
    post_stan_model = CmdStanModel(stan_file=post_stan_model_path)

    post_fit = post_stan_model.sample(
        data=post_data,
        inits={'theta': z},
        iter_warmup=5000, iter_sampling=100000, chains=1,
        show_progress=True
        )

    hyp_thetas = post_fit.stan_variable('theta')

    pd.DataFrame(hyp_thetas).plot(kind='density', ax=ax)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, [f'z={z}' for z in measurements])
ax.set_xlabel('Candidate theta value')
plt.show()