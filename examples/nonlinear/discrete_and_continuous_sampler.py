"""
File: examples/nonlinear/discrete_and_continuous_sampler.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: Script showing how to use the Sampler class to evaluate complex
             likelihoods like those arising from TruncatedBasisHyperModel
             objects.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, DiscreteUniformDistribution,\
    InfiniteUniformDistribution, ExponentialDistribution, DistributionSet,\
    AdjacencyJumpingDistribution, GaussianJumpingDistribution,\
    JumpingDistributionSet
from pylinex import Basis, TruncatedBasisHyperModel, GaussianLoglikelihood,\
    Sampler, NLFitter, BurnRule

seed = 1
np.random.seed(seed)
fontsize = 24

x_values = np.linspace(-1, 1, 100)
powers = np.arange(2)
basis = Basis(x_values[np.newaxis,:] ** powers[:,np.newaxis])
model = TruncatedBasisHyperModel(basis, min_terms=1, max_terms=2)

true_a0 = 100
true_a1 = 10
noise_level = 100
parameters = np.array([true_a0, true_a1, 2])
noiseless_data = model(parameters)

error = np.ones_like(noiseless_data) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise

loglikelihood = GaussianLoglikelihood(data, error, model)

prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(ExponentialDistribution(1), 'nterms')
prior_distribution_set.add_distribution(InfiniteUniformDistribution(ndim=2),\
    ['a{:d}'.format(which) for which in range(2)])

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(DiscreteUniformDistribution(1, 2),\
    'nterms')
guess_distribution_set.add_distribution(\
    UniformDistribution(true_a0 - 10, true_a0 + 10), 'a0')
guess_distribution_set.add_distribution(\
    UniformDistribution(true_a1 - 10, true_a1 + 10), 'a1')

jumping_distribution_set = JumpingDistributionSet()
nterms_jumping_distribution =\
    AdjacencyJumpingDistribution(jumping_probability=0.2, minimum=1, maximum=2)
jumping_distribution_set.add_distribution(nterms_jumping_distribution,\
    'nterms')
coefficient_jumping_covariance = np.identity(2)
coefficient_jumping_distribution =\
     GaussianJumpingDistribution(coefficient_jumping_covariance)
jumping_distribution_set.add_distribution(coefficient_jumping_distribution,\
    ['a{:d}'.format(which) for which in range(2)])

file_name =\
    'TESTINGDISCRETEANDCONTINUOUSPARAMETERSAMPLINGDELETEIFYOUSEETHIS.hdf5'
num_walkers = 20
steps_per_checkpoint = 500
num_checkpoints = 20
verbose = True
restart_mode = 'continue'
desired_acceptance_fraction = 0.25

try:
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
        restart_mode=restart_mode,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(num_checkpoints, silence_error=True)
    sampler.close()
    fitter = NLFitter(file_name)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    fitter.plot_acceptance_fraction(log_scale=False, ax=ax, show=False)
    fitter.plot_chain(parameters='a.*', show=False,\
        reference_value_mean=parameters[:2])
    fitter.close()
    burn_rule = BurnRule(min_checkpoints=10, desired_fraction=0.5)
    fitter = NLFitter(file_name, burn_rule)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    fitter.plot_rescaling_factors(ax=ax, show=False)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    number = 1000
    probabilities = [0.68, 0.95]
    alphas = [0.4, 0.2]
    parameter_regex = '.*'
    signals = fitter.plot_reconstruction_confidence_intervals(number,\
        probabilities, parameter_regex, model, true_curve=noiseless_data,\
        x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    ax.scatter(x_values, data, color='k', label='data')
    ax.set_xlabel('x', size=fontsize)
    ax.set_ylabel('y', size=fontsize)
    ax.legend(fontsize=fontsize)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.show()

