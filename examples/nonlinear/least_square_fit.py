"""
File: examples/nonlinear/least_square_fit.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the LeastSquareFitter class to fit a
             model.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution, DistributionSet
from pylinex import GaussianModel, TanhModel, GaussianLoglikelihood,\
    LeastSquareFitter, autocorrelation

seed = 1234
np.random.seed(seed)

num_channels = 1000
num_iterations = 100
x_values = np.linspace(-1, 1, num_channels)
error = np.ones_like(x_values) * 1e0
model = GaussianModel(x_values)

true = [1, 0, 0.1]
input_curve = model(true)
input_noise = np.random.normal(0, 1, size=num_channels) * error
input_data = input_curve + input_noise

loglikelihood = GaussianLoglikelihood(input_data, error, model)

prior_set = DistributionSet()
prior_set.add_distribution(UniformDistribution(-10, 10), 'amplitude')
prior_set.add_distribution(UniformDistribution(-1, 1), 'center')
prior_set.add_distribution(UniformDistribution(0, 1), 'standard_deviation')

least_square_fitter = LeastSquareFitter(loglikelihood, prior_set)
least_square_fitter.run(num_iterations)

argmin = least_square_fitter.argmin
print('true={}'.format(true))
print('found={}'.format(argmin))
print('true-found={}'.format(true - argmin))
print('reduced_chi_squared({0:d})={1:.4g}'.format(\
    least_square_fitter.degrees_of_freedom,\
    least_square_fitter.reduced_chi_squared_statistic))

solution = model(argmin)
residual = input_data - solution
correlation = autocorrelation(residual)
length = len(correlation)
indices = np.arange(length)
expected_correlation_mean = (indices == 0).astype(float)
expected_correlation_standard_deviation_smallest =\
    np.sqrt((1. / (length - indices)) + (2. / length))
expected_correlation_standard_deviation_largest = 1 / np.sqrt(length - indices)

fig = pl.figure()
ax = fig.add_subplot(111)
ax.scatter(indices[1:], np.abs(correlation[1:]), color='k',\
    label='Observed points')
ax.fill_between(indices[1:],\
    expected_correlation_standard_deviation_smallest[1:],\
    expected_correlation_standard_deviation_largest[1:], color='r',\
    alpha=0.5, linewidth=3, label='possible $1\sigma$ error')
ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim((indices[0], indices[-1]))
ax.set_ylim((1e-4, 1e0))

pl.figure()
pl.plot(x_values, input_data, color='k', linewidth=2)
pl.plot(x_values, solution, color='r', linewidth=2)
pl.xlim((x_values[0], x_values[-1]))
pl.show()

