"""
File: examples/nonlinear/sampler.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing a simple use case of the Sampler class to sample
             a GaussianLoglikelihood object.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution, DistributionSet,\
    GaussianJumpingDistribution, JumpingDistributionSet
from pylinex import GaussianModel, TanhModel, GaussianLoglikelihood, BurnRule,\
    Sampler, NLFitter
from dare.util.ares_expression_model import make_ares_signal_model

seed = 1234567890

nwalkers = 30
nthreads = 3 # to test multithreading, set to > 1
file_name = 'TESTING_SAMPLER_CLASS.hdf5'
num_channels = 100
num_iterations = 100
quarter_ncheckpoints = 25
steps_per_checkpoint = 100
x_values = np.linspace(99, 101, num_channels)
error = np.ones_like(x_values) * 0.1
model = GaussianModel(x_values)

true = [1, 100, 0.1]
input_curve = model(true)
input_noise = np.random.normal(0, 1, size=num_channels) * error
input_data = input_curve + input_noise

loglikelihood = GaussianLoglikelihood(input_data, error, model)

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(\
    UniformDistribution(0.8, 1.2), 'gaussian_A')
guess_distribution_set.add_distribution(\
    UniformDistribution(99.8, 100.2), 'gaussian_mu')
guess_distribution_set.add_distribution(\
    UniformDistribution(-2, 0), 'gaussian_sigma', 'log10')

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'gaussian_A')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'gaussian_mu')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-2),\
    'gaussian_sigma', 'log10')

#nwalkers = 2 * loglikelihood.num_parameters
np.random.seed(seed)

try:
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode=None)
    sampler.run_checkpoints(quarter_ncheckpoints)
    sampler.close()
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='continue')
    sampler.run_checkpoints(quarter_ncheckpoints)
    sampler.close()
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='reinitialize')
    sampler.run_checkpoints(quarter_ncheckpoints)
    sampler.close()
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='continue')
    sampler.run_checkpoints(quarter_ncheckpoints)
    sampler.close()
    fitter = NLFitter(file_name)
    fitter.plot_acceptance_fraction(log_scale=True, ax=None, show=False)
    fitter.close()
    burn_rule = BurnRule(min_checkpoints=10, desired_fraction=0.5)
    fitter = NLFitter(file_name, burn_rule)
    fitter.plot_rescaling_factors(ax=None, show=False)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    number = 1000
    probabilities = [0.68, 0.95]
    alphas = [0.4, 0.2]
    parameter_regex = '.*'
    signals = fitter.plot_reconstruction_confidence_intervals(number,\
        probabilities, parameter_regex, model, true_curve=input_data,\
        x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    fitter.close()
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)
    pl.show()

