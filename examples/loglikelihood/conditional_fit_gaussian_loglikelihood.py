"""
File: examples/loglikelihood/conditional_fit_gaussian_loglikelihood.py
Author: Keith Tauscher
Date: 6 Jun 2019

Description: Example showing a very simple use-case of the
             ConditionalFitGaussianLoglikelihood class.
"""
import os
import numpy as np
from pylinex import LegendreBasis, BasisModel, Fitter, ConditionalFitModel,\
    GaussianLoglikelihood, ConditionalFitGaussianLoglikelihood,\
    load_loglikelihood_from_hdf5_file

half_num_channels = 50
num_channels = (2 * half_num_channels) + 1
num_basis_vectors = 5
basis = LegendreBasis(num_channels, num_basis_vectors - 1)
basis_model = BasisModel(basis)
basis_parameters = np.random.normal(0, 10, size=num_basis_vectors)
noiseless_data = basis_model(basis_parameters)
noise_level = 1
error = np.ones(num_channels) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise
conditional_fit_model = ConditionalFitModel(basis_model, data, error, [])
conditional_fit_gaussian_loglikelihood =\
    ConditionalFitGaussianLoglikelihood(conditional_fit_model)
full_loglikelihood = GaussianLoglikelihood(data, error, basis_model)
assert(conditional_fit_gaussian_loglikelihood.parameters == [])
assert(conditional_fit_gaussian_loglikelihood.full_loglikelihood ==\
    full_loglikelihood)
conditional_fit_evaluated_loglikelihood =\
    conditional_fit_gaussian_loglikelihood(np.array([]))
full_evaluated_loglikelihood =\
    full_loglikelihood(Fitter(basis, data, error).parameter_mean)
assert(np.isclose(conditional_fit_evaluated_loglikelihood,\
    full_evaluated_loglikelihood, rtol=0, atol=1e-12))

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

