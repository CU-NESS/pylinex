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
from pylinex import GaussianModel, GaussianLoglikelihood, BurnRule, Sampler,\
    NLFitter

remove_file = True

seed = 1234567890
np.random.seed(seed)

nwalkers = 30
nthreads = 1 # to test multithreading, set to > 1
file_name = 'TESTING_SAMPLER_CLASS.hdf5'
num_channels = 100
num_iterations = 100
quarter_ncheckpoints = 10
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

#nwalkers = 2 * loglikelihood.num_parameters

try:
    if os.path.exists(file_name):
        sampler = Sampler(file_name, nwalkers, loglikelihood,\
            jumping_distribution_set=None, guess_distribution_set=None,\
            prior_distribution_set=None, nthreads=nthreads,\
            steps_per_checkpoint=steps_per_checkpoint, restart_mode='continue')
    else:
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
        steps_per_checkpoint=steps_per_checkpoint, restart_mode='update')
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
    fitter.plot_chain(show=False, amplitude=true_amplitude,\
        center=true_center, scale=true_scale)
    fig = fitter.triangle_plot(parameters='.*', plot_type='contourf',\
        reference_value_mean=np.array([true_amplitude, true_center,\
        true_scale]), reference_value_covariance=(model, error),\
        figsize=(12, 12), contour_confidence_levels=[0.40, 0.95], show=False)
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
        probabilities, parameter_regex, model, true_curve=input_data,\
        x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    fitter.close()
except:
    if remove_file:
        os.remove(file_name)
    raise
else:
    if remove_file:
        os.remove(file_name)

pl.show()

