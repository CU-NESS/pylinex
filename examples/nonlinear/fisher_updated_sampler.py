"""
File: examples/nonlinear/sampler.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing a simple use case of the Sampler class to sample
             a GaussianLoglikelihood object using the 'fisher_update'
             restart_mode.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution, DistributionSet,\
    GaussianJumpingDistribution, JumpingDistributionSet
from pylinex import GaussianModel, GaussianLoglikelihood, BurnRule, Sampler,\
    NLFitter

seed = 1234567890
np.random.seed(seed)

desired_acceptance_fraction = 0.25
nwalkers = 20
nthreads = 1 # to test multithreading, set to > 1
file_name = 'TESTING_SAMPLER_CLASS.hdf5'
num_channels = 1000
num_iterations = 100
half_ncheckpoints = 100
steps_per_checkpoint = 10
noise_level = 0.01
x_values = np.linspace(99, 101, num_channels)
error = np.ones_like(x_values) * noise_level
model = GaussianModel(x_values)

true_amplitude = 1
true_center = 100
true_scale = 0.1
true = [true_amplitude, true_center, true_scale]
input_curve = model(true)
input_noise = np.random.normal(0, 1, size=num_channels) * error
input_data = input_curve + input_noise

loglikelihood = GaussianLoglikelihood(input_data, error, model)

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(\
    UniformDistribution(1.1, 1.2), 'amplitude')
guess_distribution_set.add_distribution(\
    UniformDistribution(99.9, 100.1), 'center')
guess_distribution_set.add_distribution(\
    UniformDistribution(-1.25, -0.75), 'scale', 'log10')

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(0.1),\
    'amplitude')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(0.1),\
    'center')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-2),\
    'scale', 'log10')

#nwalkers = 2 * loglikelihood.num_parameters

try:
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode=None)
    sampler.run_checkpoints(half_ncheckpoints, silence_error=True)
    sampler.close()
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, nthreads=nthreads,\
        steps_per_checkpoint=steps_per_checkpoint,\
        restart_mode='fisher_update',\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(half_ncheckpoints, silence_error=True)
    sampler.close()
    fitter = NLFitter(file_name)
    fitter.plot_acceptance_fraction_four_types()
    fitter.plot_lnprobability_both_types()
    reference_value_mean = np.array([true_amplitude, true_center, true_scale])
    reference_value_covariance = (model, error)
    fitter.plot_chain(show=False, reference_value_mean=reference_value_mean,\
        reference_value_covariance=reference_value_covariance, figsize=(8,8))
    fig = fitter.triangle_plot(parameters='.*', plot_type='contourf',\
        reference_value_mean=reference_value_mean,\
        reference_value_covariance=reference_value_covariance, figsize=(8, 8),\
        contour_confidence_levels=[0.40, 0.95], fontsize=16,\
        kwargs_2D={'reference_alpha': 0.3}, show=False)
    fig.subplots_adjust(left=0.2)
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
        probabilities, parameter_regex, model, true_curve=input_curve,\
        x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    ax.scatter(x_values, input_data, color='b')
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    number = 100
    ax = fitter.plot_reconstructions(number,\
        parameters=parameter_regex, model=model, true_curve=input_curve,\
        x_values=x_values, ax=ax, alpha=0.05, color='r', show=False)
    ax.scatter(x_values, input_data, color='k', label='data')
    fitter.close()
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.show()

