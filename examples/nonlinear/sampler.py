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
    GaussianJumpingDistribution, JumpingDistributionSet,\
    TruncatedGaussianDistribution
from pylinex import GaussianModel, GaussianLoglikelihood, BurnRule, Sampler,\
    NLFitter

remove_file = True

seed = 1234567890
np.random.seed(seed)

desired_acceptance_fraction = 0.25
num_walkers = 30
num_threads = 1 # to test multithreading, set to > 1
file_name = 'TESTING_SAMPLER_CLASS.hdf5'
num_channels = 100
quarter_ncheckpoints = 25
steps_per_checkpoint = 10
x_values = np.linspace(99, 101, num_channels)
error = np.ones_like(x_values) * 0.1
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
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'amplitude')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'center')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-2),\
    'scale', 'log10')

prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(GaussianDistribution(1, 100),\
    'amplitude')
prior_distribution_set.add_distribution(GaussianDistribution(100, 100),\
    'center')
prior_distribution_set.add_distribution(\
    TruncatedGaussianDistribution(0.1, 100, low=0), 'scale')

#num_walkers = 2 * loglikelihood.num_parameters

try:
    if os.path.exists(file_name):
        sampler = Sampler(file_name, num_walkers, loglikelihood,\
            jumping_distribution_set=None, guess_distribution_set=None,\
            prior_distribution_set=prior_distribution_set,\
            num_threads=num_threads,\
            steps_per_checkpoint=steps_per_checkpoint,\
            restart_mode='continue')
    else:
        sampler = Sampler(file_name, num_walkers, loglikelihood,\
            jumping_distribution_set=jumping_distribution_set,\
            guess_distribution_set=guess_distribution_set,\
            prior_distribution_set=prior_distribution_set,\
            num_threads=num_threads,\
            steps_per_checkpoint=steps_per_checkpoint, restart_mode=None)
    sampler.run_checkpoints(quarter_ncheckpoints, silence_error=True)
    sampler.close()
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=prior_distribution_set,\
        num_threads=num_threads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='update',\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(quarter_ncheckpoints, silence_error=True)
    sampler.close()
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=prior_distribution_set,\
        num_threads=num_threads, steps_per_checkpoint=steps_per_checkpoint,\
        restart_mode='reinitialize',\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(quarter_ncheckpoints, silence_error=True)
    sampler.close()
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=prior_distribution_set,\
        num_threads=num_threads,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='continue')
    sampler.run_checkpoints(quarter_ncheckpoints, silence_error=True)
    sampler.close()
    fitter = NLFitter(file_name)
    fitter.plot_acceptance_fraction_four_types()
    fitter.plot_lnprobability_both_types(which='posterior')
    fitter.plot_lnprobability_both_types(which='likelihood')
    fitter.plot_lnprobability_both_types(which='prior')
    reference_value_mean = np.array([true_amplitude, true_center, true_scale])
    reference_value_covariance = (model, error)
    fitter.plot_chain(show=False, reference_value_mean=reference_value_mean,\
        reference_value_covariance=reference_value_covariance)
    fig = fitter.triangle_plot(parameters='.*', plot_type='contourf',\
        reference_value_mean=reference_value_mean,\
        reference_value_covariance=reference_value_covariance,\
        figsize=(12, 12), contour_confidence_levels=[0.40, 0.95],\
        kwargs_2D={'reference_alpha': 0.5}, show=False)
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
    ax = fitter.plot_reconstruction_confidence_intervals(number,\
        probabilities, parameters=parameter_regex, model=model,\
        true_curve=input_data, x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax = fitter.plot_reconstruction_confidence_intervals(number,\
        probabilities, parameters=parameter_regex, model=model,\
        true_curve=input_curve, subtract_truth=True, x_values=x_values, ax=ax,\
        alphas=alphas, show=False,\
        title='68% and 95% confidence intervals with truth subtracted')
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    number = 100
    ax = fitter.plot_reconstructions(number,\
        parameters=parameter_regex, model=model, true_curve=input_data,\
        x_values=x_values, ax=ax, alpha=0.05, color='r', show=False)
    fitter.close()
except:
    if remove_file:
        os.remove(file_name)
    raise
else:
    if remove_file:
        os.remove(file_name)

pl.show()

