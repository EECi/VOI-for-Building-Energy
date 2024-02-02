"""Perform VoI calculations for general 1-stage Bayesian decision problem."""

import os
from multiprocess import Pool
from functools import partial
from utils import cached
import numpy as np
from tqdm import tqdm


def compute_EVPI(action_space, sampling_function, utility_function, n_samples=int(1e6)):
    """Compute Expected Value of Perfect Information (EVPI) for a generic 1-stage Bayesian
    decision problem, defined by: action space, prior probabilitic distribution over
    uncertain parameter(s) values (sampling function), and utility function.

    Args:
        action_space (list): list of feasible actions for decision problem
        sampling_function (fn): function to sample from prior distribution of
            uncertain parameter(s). Must take integer argument n_samples as only
            input and return a numpy array of samples which are then passed to the
            utility function.
        utility_function (fn): function to compute utility of action for defined
            decision problem when the true value of the uncertain parameter(s) is
            known. Must take two arguments: action and true value of uncertain
            parameter(s) and return a scalar value.
        n_samples (int, optional): number of samples to use when computing
            MC estimates of expectations. Defaults to int(1e6).

    Returns:
        EVPI (float): Expected Value of Perfect Information for defined decision
            problem.
        Eu_prior (float): Expected utility of optimal prior decision.
        Eu_preposterior (float): Expected utility of pre-posterior decisions.
        astar_prior (?): Optimal prior decision.
        astar_freq_prepost (dict): Frequency of optimal pre-posterior decisions.
    """

    # 1. Sample from prior distribution of uncertain parameter(s)
    thetas = sampling_function(n_samples)

    # 2. Perform Prior analysis
    prior_E_utilities = [np.mean([utility_function(a,s) for s in tqdm(thetas)]) for a in action_space]
    Eu_prior = np.max(prior_E_utilities)
    astar_prior = action_space[np.argmax(prior_E_utilities)]

    # 3. Perform Pre-Posterior analysis
    posterior_utilities_samples = [[utility_function(a,s) for a in action_space] for s in tqdm(thetas)]
    pre_posterior_utility_samples = [np.max(l) for l in posterior_utilities_samples]
    Eu_preposterior = np.mean(pre_posterior_utility_samples)
    astar_freq_prepost = {action_space[val]:count for (val,count) in zip(*np.unique([np.argmax(l) for l in posterior_utilities_samples], return_counts=True))}
    prepost_std_error = np.std(pre_posterior_utility_samples)/n_samples

    # 4. Compute EVPI
    EVPI = Eu_preposterior - Eu_prior

    return EVPI, Eu_prior, Eu_preposterior, astar_prior, astar_freq_prepost, prepost_std_error


def compute_EVII(action_space, prior_sampling_function, measurement_sampling_function, utility_function, n_prior_samples=int(1e6), n_measurement_samples=int(1e3)):
    """Compute Expected Value of Imperfect Information (EVII) for a generic 1-stage Bayesian
    decision problem, defined by: action space, prior probabilitic distribution over
    uncertain parameter(s) values and obtained measurement values (prior sampling function),
    posterior probabilistic model of uncertain parameter(s) given measurement value (measurement
    sampling function), and utility function.

    NOTE: Partial perfect information calculations are equivalent to EVII calculations where the
    measurement sampling function is the prior probabilistic model for non-measured uncertain
    parmeters, and a deterministic function (delta spike at the prior sample value) for the
    measured uncertain parameters.

    Args:
        action_space (list): list of feasible actions for decision problem
        prior_sampling_function (fn): function to sample from prior distribution of
            uncertain parameter(s) as well as corresponding distribution over obtained
            measurement values. Must take integer argument n_samples as only input and
            return two numpy arrays of uncertain parameter value and measurement value
            samples.
        measurement_sampling_function (fn): sampling function definining probablistic
            model of hypothesised uncertain parameter values given obtained measurement
            value. Must take two arguments: measurement value and integer argument of
            number of samples to draw from posterior distribution of uncertain parameter.
        utility_function (fn): function to compute utility of action for defined
            decision problem when the true value of the uncertain parameter(s) is
            known. Must take two arguments: action and true value of uncertain
            parameter(s) and return a scalar value.
        n_samples (int, optional): number of samples to use when computing
            MC estimates of expectations. Defaults to int(1e6).

    Returns:
        EVPI (float): Expected Value of Perfect Information for defined decision
            problem.
        Eu_prior (float): Expected utility of optimal prior decision.
        Eu_preposterior (float): Expected utility of pre-posterior decisions.
        astar_prior (?): Optimal prior decision.
        astar_freq_prepost (dict): Frequency of optimal pre-posterior decisions.
    """

    # 1. Sample from prior distribution of uncertain parameter(s) and obtained measurements
    thetas, zs = prior_sampling_function(n_prior_samples)

    # 2. Perform Prior analysis
    prior_E_utilities = [np.mean([utility_function(a,s) for s in tqdm(thetas)]) for a in action_space]
    Eu_prior = np.max(prior_E_utilities)
    astar_prior = action_space[np.argmax(prior_E_utilities)]

    # 3. Perform Pre-Posterior analysis
    posterior_expected_utilities = lambda z: [np.mean([utility_function(a,s) for s in measurement_sampling_function(z,n_prior_samples)]) for a in action_space]
    posterior_expected_utilities_samples = [posterior_expected_utilities(z) for z in tqdm(zs[::n_prior_samples//n_measurement_samples])]
    pre_posterior_utility_samples = [np.max(l) for l in posterior_expected_utilities_samples]
    Eu_preposterior = np.mean(pre_posterior_utility_samples)
    astar_freq_prepost = {action_space[val]:count for (val,count) in zip(*np.unique([np.argmax(l) for l in posterior_expected_utilities_samples], return_counts=True))}
    prepost_std_error = np.std(pre_posterior_utility_samples)/n_measurement_samples

    # 4. Compute EVII
    EVII = Eu_preposterior - Eu_prior

    return EVII, Eu_prior, Eu_preposterior, astar_prior, astar_freq_prepost, prepost_std_error



def fast_EVPI(action_space, sampling_function, utility_function, n_samples=int(1e6), report_prepost_freqs=False):
    """Compute EVII using tricks for computational efficiency.

    NOTE: sampling functions must return samples as np.arrays with dimensions (n,d)
    where n in the number of samples and d is the dimension of the uncertain parameters.
    """

    # 1. Sample from prior distribution of uncertain parameter(s)
    thetas = sampling_function(n_samples)

    # 2. Perform Prior analysis
    prior_E_utilities = [np.mean(utility_function(a,thetas.T)) for a in tqdm(action_space)]
    Eu_prior = np.max(prior_E_utilities)
    astar_prior = action_space[np.argmax(prior_E_utilities)]

    # 3. Perform Pre-Posterior analysis
    posterior_utilities_samples = np.array([utility_function(a,thetas.T) for a in tqdm(action_space)]).T
    pre_posterior_utility_samples = np.max(posterior_utilities_samples,axis=1)
    Eu_preposterior = np.mean(pre_posterior_utility_samples)
    astar_freq_prepost = {action_space[val]:count for (val,count) in zip(*np.unique([np.argmax(l) for l in posterior_utilities_samples], return_counts=True))} if report_prepost_freqs else None
    prepost_std_error = np.std(pre_posterior_utility_samples)/n_samples

    # 4. Compute EVPI
    EVPI = Eu_preposterior - Eu_prior

    return EVPI, Eu_prior, Eu_preposterior, astar_prior, astar_freq_prepost, prepost_std_error


def fast_EVII(action_space, prior_sampling_function, measurement_sampling_function, utility_function, n_prior_samples=int(1e6), n_measurement_samples=int(1e3), report_prepost_freqs=False):
    """Compute EVII using tricks for computational efficiency.
    
    NOTE:
    - sampling functions must return samples as np.arrays with dimensions (n,d)
    where n in the number of samples and d is the dimension of the uncertain parameters.
    - any cached functions must have the cache filled prior to calling this function for
    thread safety reasons.
    """

    # 1. Sample from prior distribution of uncertain parameter(s) and obtained measurements
    thetas, zs = prior_sampling_function(n_prior_samples)

    # 2. Perform Prior analysis
    prior_E_utilities = [np.mean(utility_function(a,thetas.T)) for a in tqdm(action_space)]
    Eu_prior = np.max(prior_E_utilities)
    astar_prior = action_space[np.argmax(prior_E_utilities)]

    # 3. Perform Pre-Posterior analysis
    def compute_posterior_expected_utilities(z):
        posterior_samples = measurement_sampling_function(z,n_prior_samples)
        # reuse posterior samples for computing expected utilities for all actions
        return [np.mean(utility_function(a,posterior_samples.T)) for a in action_space]

    thinned_zs = [z for z in zs[::n_prior_samples//n_measurement_samples]]
    with Pool(processes=min(os.cpu_count(),16)) as pool:
        posterior_expected_utilities_samples = list(tqdm(pool.imap(compute_posterior_expected_utilities, thinned_zs, chunksize=100), total=len(thinned_zs)))
        pool.close()
        pool.join()
    pre_posterior_utility_samples = np.max(posterior_expected_utilities_samples,axis=1)
    Eu_preposterior = np.mean(pre_posterior_utility_samples)
    astar_freq_prepost = {action_space[val]:count for (val,count) in zip(*np.unique([np.argmax(l) for l in posterior_expected_utilities_samples], return_counts=True))}  if report_prepost_freqs else None
    prepost_std_error = np.std(pre_posterior_utility_samples)/n_measurement_samples

    # 4. Compute EVII
    EVII = Eu_preposterior - Eu_prior

    return EVII, Eu_prior, Eu_preposterior, astar_prior, astar_freq_prepost, prepost_std_error