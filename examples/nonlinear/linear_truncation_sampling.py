"""
File: examples/nonlinear/linear_truncation_sampling.py
Author: Keith Tauscher
Date: 17 Dec 2018

Description: Example script showing simple case of sampling a
             LinearTruncationLoglikelihood object.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import DiscreteUniformDistribution, DistributionSet,\
    GridHopJumpingDistribution, JumpingDistributionSet
from pylinex import FourierBasis, BasisSum, Fitter,\
    LinearTruncationLoglikelihood, Sampler, BurnRule, NLFitter

seed = 0
np.random.seed(seed)

fontsize = 24
band_enlargement = 10

num_channels = 1000
noise_level = 1
true_amplitudes = [100, 3]
x_values = np.linspace(-np.pi, np.pi, num_channels)
names = ['even', 'odd']
ndim = len(names)
jumping_probability = 0.5
nterms_maxima = np.array([40, 40])
nterms_minima = np.array([1, 1])
error = np.ones(num_channels) * noise_level
base_basis = FourierBasis(num_channels, 50, error=error)
bases = [base_basis.even_subbasis, base_basis.odd_subbasis]
nterms_maxima = np.array([basis.num_basis_vectors for basis in bases])
basis_sum = BasisSum(names, bases)
true_curve = (true_amplitudes[0] * np.sin(x_values) *\
    (1 - ((x_values / np.pi) ** 2))) +\
    (true_amplitudes[1] * ((x_values / np.pi) ** 2))
noise = np.random.normal(0, 1, size=error.shape) * error
data = noise + true_curve
information_criterion = 'deviance_information_criterion'

loglikelihood = LinearTruncationLoglikelihood(basis_sum, data, error,\
    information_criterion=information_criterion)

prior_distribution_set = DistributionSet()
for (parameter, nterms) in zip(loglikelihood.parameters, nterms_maxima):
    prior_distribution_set.add_distribution(\
        DiscreteUniformDistribution(1, nterms), parameter)

guess_distribution_set = prior_distribution_set.copy()

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(\
    GridHopJumpingDistribution(ndim=ndim,\
    jumping_probability=jumping_probability, minima=nterms_minima,\
    maxima=nterms_maxima), loglikelihood.parameters)

file_name = 'TEST_LINEAR_TRUNCATION_SAMPLER.hdf5'
nwalkers = 64
steps_per_checkpoint = 100
num_checkpoints = 10
verbose = True
restart_mode = None
desired_acceptance_fraction = 0.25

try:
    sampler = Sampler(file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
        restart_mode=restart_mode,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(num_checkpoints, silence_error=True)
    sampler.close()
    burn_rule = BurnRule(min_checkpoints=1, desired_fraction=1)
    fitter = NLFitter(file_name, burn_rule=burn_rule)
    fitter.triangle_plot(parameters=loglikelihood.parameters,\
        figsize=(12, 12), fontsize=28, nbins=np.mean(nterms_maxima),\
        plot_type='histogram')
    fitter.plot_chain(parameters='.*', figsize=(12, 12), show=False)
    truncated_basis_sum = loglikelihood.truncated_basis_sum(\
        fitter.maximum_probability_parameters.astype(int))
    maxprob_linfit = Fitter(truncated_basis_sum, data, error)
    (band_center, band_width) =\
        (maxprob_linfit.channel_mean, maxprob_linfit.channel_error)
    fig = pl.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.scatter(x_values, data, color='k')
    ax.plot(x_values, band_center, color='r')
    ax.fill_between(x_values, band_center - (band_enlargement * band_width),\
        band_center + (band_enlargement * band_width), color='r', alpha=0.5)
    ax.set_xlabel('$x$', size=fontsize)
    ax.set_ylabel('$y$', size=fontsize)
    ax.set_title('Chosen truncated fitter ({:d}x enlarged)'.format(\
        band_enlargement), size=fontsize)
    ax.tick_params(labelsize=fontsize, width=5, length=15, which='major')
    ax.tick_params(labelsize=fontsize, width=3, length=9, which='minor')
    print("mcmc_chosen={}".format({parameter: value\
        for (parameter, value) in zip(loglikelihood.parameters,\
        fitter.maximum_probability_parameters.astype(int))}))
except:
    if os.path.exists(file_name):
        os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.show()


