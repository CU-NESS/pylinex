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
    LeastSquareFitter

seed = 12345
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
prior_set.add_distribution(UniformDistribution(-10, 10), 'gaussian_A')
prior_set.add_distribution(UniformDistribution(-1, 1), 'gaussian_mu')
prior_set.add_distribution(UniformDistribution(0, 1), 'gaussian_sigma')

least_square_fitter = LeastSquareFitter(loglikelihood, prior_set)
least_square_fitter.run(num_iterations)

argmin = least_square_fitter.argmin
print('true={}'.format(true))
print('found={}'.format(argmin))
print('true-found={}'.format(true - argmin))

solution = model(argmin)

pl.plot(x_values, input_data, color='k', linewidth=2)
pl.plot(x_values, solution, color='r', linewidth=2)
pl.xlim((x_values[0], x_values[-1]))
pl.show()

