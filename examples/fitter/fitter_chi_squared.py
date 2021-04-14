"""
File: examples/fitter/fitter_chi_squared.py
Author: Keith Tauscher
Date: 10 Apr 2021

Description: Example showing how the reduced_chi_squared and
             reduced_chi_squared_expected_distribution properties of the Fitter
             class behave.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from pylinex import Basis, BasisSum, Fitter

seed = 0
fontsize = 24
num_x_values = 10000
num_curves = 10000
num_channels = 100
num_basis_vectors = 25
arbitrary_constant = 343.013

np.random.seed(seed)

error = np.ones(num_channels)
basis = Basis(np.identity(num_channels)[:num_basis_vectors,:])
basis_sum = BasisSum(['sole'], [basis])

prior_mean = np.random.normal(loc=0, scale=100, size=(num_basis_vectors))
prior_variance = 1
prior_variances = np.ones(num_basis_vectors) * prior_variance
prior_covariance = np.diag(prior_variances)
prior = GaussianDistribution(prior_mean, prior_covariance)

noiseless_data =\
    np.array([basis(prior.draw()) for icurve in range(num_curves)])

data = noiseless_data + (error[np.newaxis,:] *\
    np.random.standard_normal(size=(num_curves, num_channels)))

fitter_without_priors = Fitter(basis_sum, data, error=error)
fitter_with_priors = Fitter(basis_sum, data, error=error, sole_prior=prior)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)

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
ax.legend(fontsize=fontsize)

pl.show()

