"""
File: examples/loglikelihood/conditional_fit_gaussian_loglikelihood.py
Author: Keith Tauscher
Date: 6 Jun 2019

Description: Example showing a very simple use-case of the
             ConditionalFitGaussianLoglikelihood class.
"""
from __future__ import division
import os
import numpy as np
from distpy import GaussianDistribution
from pylinex import LegendreBasis, BasisModel, GaussianModel, SumModel,\
    Fitter, SingleConditionalFitModel, GaussianLoglikelihood,\
    ConditionalFitGaussianLoglikelihood, load_loglikelihood_from_hdf5_file

half_num_channels = 500
num_channels = (2 * half_num_channels) + 1
num_basis_vectors = 10
basis = LegendreBasis(num_channels, num_basis_vectors - 1)
basis_model = BasisModel(basis)
gaussian_model = GaussianModel(np.linspace(-1, 1, num_channels))
sum_model = SumModel(['gaussian', 'basis'], [gaussian_model, basis_model])
basis_parameters = np.random.normal(0, 10, size=num_basis_vectors)
gaussian_parameters = np.array([-5, 0.1, 0.25])
noiseless_data =\
    sum_model(np.concatenate([gaussian_parameters, basis_parameters]))
noise_level = 1
error = np.ones(num_channels) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise
prior_mean = np.zeros(num_basis_vectors)
prior_covariance = np.ones(num_basis_vectors)
prior = GaussianDistribution(prior_mean, prior_covariance)
priors = {'sole_prior': prior}
conditional_fit_model = SingleConditionalFitModel(sum_model, data, error,\
    ['basis'], prior=prior)
conditional_fit_gaussian_loglikelihood =\
    ConditionalFitGaussianLoglikelihood(conditional_fit_model)
full_loglikelihood = GaussianLoglikelihood(data, error, sum_model)
assert(conditional_fit_gaussian_loglikelihood.parameters ==\
    ['gaussian_{!s}'.format(par) for par in ['amplitude', 'center', 'scale']])
assert(conditional_fit_gaussian_loglikelihood.full_loglikelihood ==\
    full_loglikelihood)
conditional_fit_evaluated_loglikelihood =\
    conditional_fit_gaussian_loglikelihood(gaussian_parameters)
fitter =\
    Fitter(basis, data - gaussian_model(gaussian_parameters), error, **priors)
fitter_mean = fitter.parameter_mean
fitter_covariance = fitter.parameter_covariance
full_evaluated_conditional_posterior =\
    full_loglikelihood(np.concatenate([gaussian_parameters, fitter_mean])) +\
    prior.log_value(fitter_mean) +\
    (np.linalg.slogdet(fitter_covariance)[1] / 2)
assert(np.isclose(conditional_fit_evaluated_loglikelihood,\
    full_evaluated_conditional_posterior, rtol=0, atol=1e-12))

file_name = 'TESTINGCONDITIONALFITGAUSSIANLOGLIKELIHOOD.hdf5'
conditional_fit_gaussian_loglikelihood.save(file_name)
try:
    assert(ConditionalFitGaussianLoglikelihood.load(file_name) ==\
        conditional_fit_gaussian_loglikelihood)
    assert(load_loglikelihood_from_hdf5_file(file_name) ==\
        conditional_fit_gaussian_loglikelihood)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

