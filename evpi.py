"""Perform EVPI calculation for general 1-stage stochastic decision problem."""

import numpy as np
from tqdm import tqdm


def compute_EVPI(action_space, sampling_function, utility_function, n_samples=int(1e5)):

    # 1. Sample from prior distribution of uncertain parameter(s)
    samples = sampling_function(n_samples)

    # 2. Perform Prior analysis
    prior_E_utilities = [np.mean([utility_function(a,s) for s in tqdm(samples)]) for a in action_space]
    Eu_prior = np.max(prior_E_utilities)
    astar_prior = action_space[np.argmax(prior_E_utilities)]

    # 3. Perform Pre-Posterior analysis
    pre_posterior_utility_samples = [np.max([utility_function(a,s) for a in action_space]) for s in tqdm(samples)]
    Eu_preposterior = np.mean(pre_posterior_utility_samples)
    astar_freq_prepost = {action_space[val]:count for (val,count) in zip(*np.unique([np.argmax([utility_function(a,s) for a in action_space]) for s in tqdm(samples)], return_counts=True))}

    # 4. Compute EVPI
    EVPI = Eu_preposterior - Eu_prior

    return EVPI, Eu_prior, Eu_preposterior, astar_prior, astar_freq_prepost