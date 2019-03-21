"""
File: examples/nonlinear/poisson_sampler.py
Author: Keith Tauscher
Date: 26 Feb 2018

Description: Example showing how to use a PoissonLoglikelihood with the
             Sampler class.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution,\
    JumpingDistributionSet, GaussianJumpingDistribution
from pylinex import PoissonLoglikelihood, GaussianModel, Sampler, NLFitter,\
    BurnRule

file_name = 'TEST_DELETE_THIS.hdf5'
seed = 0
num_iterations = 100
steps_per_checkpoint = 100
ncheckpoints = 50
nwalkers = 50

x_values = np.linspace(-1, 1, 21)
model = GaussianModel(x_values)

true_pars = (100., 0., 1.)
(true_amplitude, true_mean, true_width) = true_pars
mean_values = model(np.array(true_pars))
data = np.array([np.random.poisson(mean_value) for mean_value in mean_values])

loglikelihood = PoissonLoglikelihood(data, model)

prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(\
    UniformDistribution(0, 100), 'amplitude')
prior_distribution_set.add_distribution(\
    UniformDistribution(-5, 5), 'center')
prior_distribution_set.add_distribution(\
    UniformDistribution(0, 10), 'scale')

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(\
    UniformDistribution(90, 110), 'amplitude')
guess_distribution_set.add_distribution(\
    UniformDistribution(-0.1, 0.1), 'center')
guess_distribution_set.add_distribution(\
    UniformDistribution(0.9, 1.1), 'scale')

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1),\
    'amplitude')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-4),\
    'center')
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(1e-2),\
    'scale', 'log10')

np.random.seed(seed)

try:
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=None,\
        steps_per_checkpoint=steps_per_checkpoint, restart_mode=None)
    sampler.run_checkpoints(ncheckpoints)
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
        probabilities, parameters=parameter_regex, model=model,\
        true_curve=data, x_values=x_values, ax=ax, alphas=alphas,\
        title='68% and 95% confidence intervals', show=False)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    number = 100
    ax = fitter.plot_reconstructions(number, parameters=parameter_regex,\
        model=model, true_curve=data, x_values=x_values, ax=ax, alpha=0.05,\
        color='r', show=False)
    ax.scatter(x_values, data, color='k', label='data')
    fitter.close()
except:
    try:
        os.remove(file_name)
    except:
        pass
    raise
else:
    os.remove(file_name)
    pl.show()

