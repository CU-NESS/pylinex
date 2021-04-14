"""
File: examples/fitter/maa_fitter_chi_squared.py
Author: Keith Tauscher
Date: 14 Apr 2021

Description: Example showing how the reduced_chi_squared and
             reduced_chi_squared_expected_distribution properties of the
             MAAFitter class behave.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from pylinex import RepeatExpander, Basis, BasisSum, MAAFitter

seed = 0
fontsize = 24
num_x_values = 10000
num_curves = 10000
num_channels_per = 100
num_repeats = 2
num_channels = num_repeats * num_channels_per
num_basis_vectors = 25
arbitrary_constant = 343.013

np.random.seed(seed)

error = np.ones(num_channels)
basis = Basis(np.identity(num_channels)[:num_basis_vectors,:])
basis_sum = BasisSum(['sole'], [basis])

prior_mean = np.random.normal(loc=0, scale=100, size=(num_basis_vectors))
prior_variance = 1e-4
prior_variances = np.ones(num_basis_vectors) * prior_variance
prior_covariance = np.diag(prior_variances)
prior = GaussianDistribution(prior_mean, prior_covariance)

noiseless_data =\
    np.array([basis(prior.draw()) for icurve in range(num_curves)])

data = noiseless_data + (error[np.newaxis,:] *\
    np.random.standard_normal(size=(num_curves, num_channels)))

expander = RepeatExpander(num_repeats)

fitter_without_priors = MAAFitter(expander, basis_sum, data, error=error)
fitter_with_priors = MAAFitter(expander, basis_sum, data, error=error,\
    sole_prior=prior)

fig = pl.figure(figsize=(18,9))
ax = fig.add_subplot(121)

ax.hist(fitter_without_priors.reduced_chi_squared, histtype='bar', color='C0',\
    label='No priors', bins=100, density=True, alpha=0.5)
ax.hist(fitter_with_priors.reduced_chi_squared, histtype='bar', color='C1',\
    label='With priors', bins=100, density=True, alpha=0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_values = np.linspace(xlim[0], xlim[1], num_x_values)

fitter_without_priors.reduced_chi_squared_expected_distribution.plot(x_values,\
    color='C0', ax=ax, fontsize=fontsize)
fitter_with_priors.reduced_chi_squared_expected_distribution.plot(x_values,\
    color='C1', ax=ax, fontsize=fontsize)

expected_mean_with_priors =\
    fitter_with_priors.reduced_chi_squared_expected_mean
ax.plot([1] * 2, ylim, color='C0', linestyle='--')
ax.plot([np.mean(fitter_without_priors.reduced_chi_squared)] * 2, ylim,\
    color='C0', linestyle=':')
ax.plot([expected_mean_with_priors] * 2, ylim, color='C1', linestyle='--')
ax.plot([np.mean(fitter_with_priors.reduced_chi_squared)] * 2, ylim,\
    color='C1', linestyle=':')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel('$\chi^2$', size=fontsize)
ax.set_ylabel('PDF', size=fontsize)
ax.set_title('Full data $\chi^2$', size=fontsize)
ax.legend(fontsize=fontsize)




#fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(122)

ax.hist(fitter_without_priors.desired_reduced_chi_squared(0), histtype='bar',\
    color='C0', label='No priors', bins=100, density=True, alpha=0.5)
ax.hist(fitter_with_priors.desired_reduced_chi_squared(0), histtype='bar',\
    color='C1', label='With priors', bins=100, density=True, alpha=0.5)

xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_values = np.linspace(xlim[0], xlim[1], num_x_values)

fitter_without_priors.desired_reduced_chi_squared_expected_distribution.plot(\
    x_values, color='C0', ax=ax, fontsize=fontsize)
fitter_with_priors.desired_reduced_chi_squared_expected_distribution.plot(\
    x_values, color='C1', ax=ax, fontsize=fontsize)

expected_mean_with_priors =\
    fitter_with_priors.desired_reduced_chi_squared_expected_mean
ax.plot([1] * 2, ylim, color='C0', linestyle='--')
ax.plot([np.mean(fitter_without_priors.desired_reduced_chi_squared(0))] * 2,\
    ylim, color='C0', linestyle=':')
ax.plot([expected_mean_with_priors] * 2, ylim, color='C1', linestyle='--')
ax.plot([np.mean(fitter_with_priors.desired_reduced_chi_squared(0))] * 2,\
    ylim, color='C1', linestyle=':')

ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_xlabel('$\chi^2$', size=fontsize)
ax.set_ylabel('PDF', size=fontsize)
ax.set_title('Desired component $\chi^2$', size=fontsize)
ax.legend(fontsize=fontsize)

fig.subplots_adjust(top=0.945, bottom=0.11, left=0.065, right=0.985,\
    wspace=0.23)

pl.show()

