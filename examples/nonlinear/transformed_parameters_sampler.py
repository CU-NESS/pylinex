"""
File: examples/nonlinear/transformed_parameters_sampler.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: File showing the use of the Sampler class when exploring
             parameters in transformed spaces.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, DistributionSet,\
    GaussianJumpingDistribution, JumpingDistributionSet
from pylinex import GaussianModel, SlicedModel, GaussianLoglikelihood,\
    Sampler, NLFitter, BurnRule

fontsize = 24

seed = 1
np.random.seed(seed)

num_channels = 1000
noise_level = 1e1
x_values = np.linspace(-1, 1, num_channels)

model = SlicedModel(GaussianModel(x_values), center=0)
parameters = np.array([100, 0.1])
noiseless_data = model(parameters)
error = np.ones_like(noiseless_data) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise

loglikelihood = GaussianLoglikelihood(data, error, model)

prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(UniformDistribution(-3, 3),\
    'amplitude', 'log10')
prior_distribution_set.add_distribution(UniformDistribution(-3, 0), 'scale',\
    'log10')

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(UniformDistribution(2.5, 1.5),\
    'amplitude', 'log10')
guess_distribution_set.add_distribution(UniformDistribution(-1.5, -0.5),\
    'scale', 'log10')

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'amplitude', 'log10')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'scale', 'log10')

file_name = 'TESTINGSAMPLERDELETEIFYOUSEETHIS.hdf5'
nwalkers = 50
steps_per_checkpoint = 100
num_checkpoints = 20
verbose = True

try:
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose)
    sampler.run_checkpoints(num_checkpoints)
    sampler.close()
    fitter = NLFitter(file_name)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    fitter.plot_acceptance_fraction(log_scale=False, ax=ax, show=False)
    fitter.plot_chain(parameters='.*', ax=ax, show=False,\
        amplitude=parameters[0], scale=parameters[1])
    fitter.close()
    burn_rule = BurnRule(min_checkpoints=1, desired_fraction=0.1)
    fitter = NLFitter(file_name, burn_rule)
    fitter.triangle_plot(parameters='.*', figsize=(12, 12), show=False,\
        fontsize=28, nbins=100, plot_type='contour',\
        reference_value_mean=parameters,\
        reference_value_covariance=(model, error), apply_transforms=True)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    fitter.plot_rescaling_factors(ax=ax, show=False)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    number = 1000
    probabilities = [0.68]
    alphas = [0.2]
    parameter_regex = '.*'
    signals = fitter.plot_reconstruction_confidence_intervals(number,\
        probabilities, parameter_regex, model, true_curve=noiseless_data,\
        x_values=x_values, ax=ax, alphas=alphas,\
        title='68% confidence interval', show=False)
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

