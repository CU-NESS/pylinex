"""
File: examples/nonlinear/least_square_fit.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the LeastSquareFitter class to fit a
             model.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution,\
    KroneckerDeltaDistribution, DistributionSet
from pylinex import GaussianModel, GaussianLoglikelihood, LeastSquareFitter,\
    autocorrelation

file_name = 'TESTING_LEAST_SQUARE_FITTER_DELETE_THIS.hdf5'

seed = 123456789
np.random.seed(seed)

num_channels = 1000
num_iterations = 100
x_values = np.linspace(-1, 1, num_channels)
noise_level = 1e-2
error = np.ones_like(x_values) * noise_level
model = GaussianModel(x_values)

true = [1, 0, 0.1]
input_curve = model(true)
input_noise = np.random.normal(0, 1, size=num_channels) * error
input_data = input_curve + input_noise

loglikelihood = GaussianLoglikelihood(input_data, error, model)

prior_set = DistributionSet()
prior_set.add_distribution(UniformDistribution(-2, 2), 'amplitude')
prior_set.add_distribution(UniformDistribution(-1, 1), 'center')
prior_set.add_distribution(UniformDistribution(0, 1), 'scale')

exactly_correct_prior_set = DistributionSet()
exactly_correct_prior_set.add_distribution(KroneckerDeltaDistribution(1),\
    'amplitude')
exactly_correct_prior_set.add_distribution(KroneckerDeltaDistribution(0),\
    'center')
exactly_correct_prior_set.add_distribution(KroneckerDeltaDistribution(0.1),\
    'scale')

bounds = {'amplitude': (-10, 10), 'center': (-1, 1), 'scale': (0, 1)}

try:
    least_square_fitter = LeastSquareFitter(loglikelihood=loglikelihood,\
        prior_set=prior_set, file_name=file_name, **bounds)
    assert(least_square_fitter.num_iterations == 0)
    least_square_fitter.run(num_iterations // 2)
    assert(least_square_fitter.num_iterations == num_iterations // 2)
    least_square_fitter = LeastSquareFitter(file_name=file_name)
    assert(least_square_fitter.num_iterations == num_iterations // 2)
    least_square_fitter.run(num_iterations // 2)
    assert(least_square_fitter.num_iterations == (num_iterations // 2) * 2)
    least_square_fitter = LeastSquareFitter(loglikelihood=loglikelihood,\
        prior_set=exactly_correct_prior_set, **bounds)
    least_square_fitter.run(iterations=num_iterations,\
        cutoff_loglikelihood=((-1) * num_channels))
    assert(least_square_fitter.num_iterations == 1)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

argmin = least_square_fitter.argmin
print('true={}'.format(true))
print('found={}'.format(argmin))
print('true-found={}'.format(true - argmin))
print('reduced_chi_squared({0:d})={1:.4g}'.format(\
    least_square_fitter.degrees_of_freedom,\
    least_square_fitter.reduced_chi_squared_statistic))

solution = model(argmin)
residual = (input_data - solution) / error
(correlation, correlation_noise_level) = autocorrelation(residual)
length = len(correlation)
indices = np.arange(length)

fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(indices[1:], correlation[1:], color='k',\
    label='Observed points')
ax.fill_between(indices[1:], -correlation_noise_level[1:],\
    correlation_noise_level[1:], color='r',\
    alpha=0.2, linewidth=3, label='$1\sigma$ error')
ax.fill_between(indices[1:], -2 * correlation_noise_level[1:],\
    2 * correlation_noise_level[1:], color='r',\
    alpha=0.1, linewidth=3, label='$3\sigma$ error')
ax.legend()
ax.set_xlim((indices[0], indices[-1]))
ax.set_ylim((1e-4, 1e0))

pl.figure()
pl.plot(x_values, input_data, color='k', linewidth=2)
pl.plot(x_values, solution, color='r', linewidth=2)
pl.xlim((x_values[0], x_values[-1]))
pl.show()

