"""
File: examples/loglikelihood/fisher_formalism.py
Author: Keith Tauscher
Date: 25 Nov 2018

Description: Example showing a simple use of the Fisher matrix approximation.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import ExponentialDistribution, InfiniteUniformDistribution,\
    DistributionSet
from pylinex import GaussianModel, GaussianLoglikelihood, GammaLoglikelihood

num_points = int(1e2)
noise_level = 0.25
x_values = np.linspace(-1, 1, num_points)
error = np.ones(num_points) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error

model = GaussianModel(x_values)
true_parameters = np.array([1, 0, 0.5])
mean = model(true_parameters)
data = mean + noise

loglikelihood = GaussianLoglikelihood(data, error, model)

maximum_likelihood_parameters = true_parameters
transform_list = None
max_standard_deviations = np.inf
prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(InfiniteUniformDistribution(),\
    'amplitude')
prior_distribution_set.add_distribution(InfiniteUniformDistribution(),\
    'center')
prior_distribution_set.add_distribution(ExponentialDistribution(1),\
    'scale')
prior_distribution_list =\
    prior_distribution_set.distribution_list(model.parameters)

fisher_information_kwargs = {}

distribution_set = loglikelihood.parameter_distribution_fisher_formalism(\
    maximum_likelihood_parameters, transform_list=transform_list,\
    max_standard_deviations=max_standard_deviations,\
    prior_to_impose_in_transformed_space=prior_distribution_list)

ndraw = int(1e6)
parameters = model.parameters
in_transformed_space = True
figsize = (12, 12)
kwargs_1D = {'color': 'C0'}
kwargs_2D = {'colors': 'C0'}
fontsize = 12
fig = None
show = False
nbins = 100
plot_type = 'contour'
reference_value_mean = true_parameters
reference_value_covariance = None
contour_confidence_levels = 0.95
parameter_renamer = (lambda x: x)

fig = distribution_set.triangle_plot(ndraw, parameters=parameters,\
    in_transformed_space=in_transformed_space, figsize=figsize, fig=fig,\
    show=show, kwargs_1D=kwargs_1D, kwargs_2D=kwargs_2D, fontsize=fontsize,\
    nbins=nbins, plot_type=plot_type,\
    reference_value_mean=reference_value_mean,\
    reference_value_covariance=reference_value_covariance,\
    contour_confidence_levels=contour_confidence_levels,\
    parameter_renamer=parameter_renamer)

curve_sample = model.curve_sample(distribution_set, 100)
fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.plot(x_values, curve_sample.T, color='k', linestyle='-', alpha=0.1)
ax.scatter(x_values, data, color='r')

pl.show()

