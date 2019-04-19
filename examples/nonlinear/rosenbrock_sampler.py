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
from pylinex import GaussianModel, RosenbrockLoglikelihood, BurnRule, Sampler,\
    NLFitter

remove_file = True

seed = 1234567890
np.random.seed(seed)

nwalkers = 500
nthreads = 1 # to test multithreading, set to > 1
file_name = 'TESTING_ROSENBROCK_SAMPLER.hdf5'
ncheckpoints = 20
steps_per_checkpoint = 100
nbins = 200

xmin = 1.
scale_ratio = 100.
overall_scale = 0.5

true_a0 = xmin
true_a1 = (xmin ** 2)
true = np.array([true_a0, true_a1])

loglikelihood = RosenbrockLoglikelihood(xmin=xmin, scale_ratio=scale_ratio,\
    overall_scale=overall_scale)

initial_covariance = np.array([[0.05, 0.], [0., 0.05]])

guess_distribution_set = DistributionSet(\
    [(GaussianDistribution(true, initial_covariance), ['a0', 'a1'], None)])

jumping_distribution_set = JumpingDistributionSet(\
    [(GaussianJumpingDistribution(initial_covariance), ['a0', 'a1'], None)])

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
sampler.run_checkpoints(ncheckpoints, silence_error=True)
sampler.close()
fitter = NLFitter(file_name)
fitter.plot_acceptance_fraction(log_scale=True, ax=None, show=False)
fitter.plot_chain(show=False, a0=true_a0, a1=true_a1)
fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
fig = fitter.plot_bivariate_histogram('a0', 'a1', ax=ax,\
    matplotlib_function='contourf', reference_value_mean=true,\
    reference_value_covariance=\
    loglikelihood.parameter_covariance_fisher_formalism(true),\
    contour_confidence_levels=[0.95], reference_alpha=0.4, bins=nbins,\
    xlabel='$x$', ylabel='$y$', show=False,\
    title='Samples from Rosenbrock function-inspired distribution')
fitter.close()
burn_rule = BurnRule(min_checkpoints=10, desired_fraction=0.5)
fitter = NLFitter(file_name, burn_rule)
fitter.plot_rescaling_factors(ax=None, show=False)
fitter.close()

pl.show()
os.remove(file_name)

