"""
File: examples/nonlinear/gaussian_sampler.py
Author: Keith Tauscher
Date: 24 Nov 2018

Description: Example showing use of the GaussianLoglikelihood class in an MCMC
             sampling context and comparison with the Fisher matrix formalism.
"""
from __future__ import division
import os, time, glob
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
from distpy import GaussianDistribution, TruncatedGaussianDistribution,\
    DistributionSet
from pylinex import ConstantModel, GaussianLoglikelihood, LeastSquareFitCluster

seed = 1
np.random.seed(seed)

linewidth = 3
fontsize = 20

true_value = 1
num_points = 1000
noise_level = 1

nwalkers = 100
steps_per_checkpoint = 100
num_checkpoints = 100

error = np.ones(num_points) * noise_level
model = ConstantModel(num_points)
true_parameters = np.ones(1) * true_value
true_variance = (noise_level ** 2) / num_points
mean = model(true_parameters)
data = true_value + (np.random.normal(0, 1, size=error.shape) * error)
loglikelihood = GaussianLoglikelihood(data, error, model)

prefix = 'DELETETHESE'
restart_mode = None
guess_distribution_set = DistributionSet([(TruncatedGaussianDistribution(\
    true_value, true_variance / 1000, low=0), 'a', None)])
num_fits = 1000
least_square_fit_cluster = LeastSquareFitCluster(loglikelihood,\
    guess_distribution_set, prefix, num_fits)
t0 = time.time()
least_square_fit_cluster.run(iterations=1000)
t1 = time.time()
np.random.seed(seed)
least_square_fit_cluster = LeastSquareFitCluster.load_from_first_file(\
    '{!s}.hdf5'.format(prefix), num_fits)
least_square_fit_cluster.run()
t2 = time.time()
gaussian_distribution =\
    least_square_fit_cluster.approximate_gaussian_distribution

x_values = true_value +\
    (np.linspace(-1, 1, 1000) * (4 * np.sqrt(true_variance)))
x_values = x_values[x_values > 0]

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
pdf_values = np.exp([gaussian_distribution.log_value(x_value)\
    for x_value in x_values])
pdf_values = pdf_values / np.max(pdf_values)
ax.plot(x_values, pdf_values, color='b', linewidth=linewidth, linestyle='--',\
    label='LeastSquareFitCluster')

true_distribution = GaussianDistribution(true_value, true_variance)
pdf_values = np.exp([true_distribution.log_value(x_value)\
    for x_value in x_values])
pdf_values = pdf_values / np.max(pdf_values)
ax.plot(x_values, pdf_values, color='r', linewidth=linewidth, linestyle='--',\
    label='No noise scatter')


transform_list = None
max_standard_deviations = np.inf
prior_distribution = None
larger_differences = 1e-5
smaller_differences = 1e-6
fisher_distribution_set =\
    loglikelihood.parameter_distribution_fisher_formalism(\
    gaussian_distribution.mean.A[0], transform_list=transform_list,\
    max_standard_deviations=max_standard_deviations,\
    prior_to_impose_in_transformed_space=prior_distribution,\
    larger_differences=larger_differences,\
    smaller_differences=smaller_differences)
fisher_distribution = fisher_distribution_set._data[0][0]
pdf_values = np.exp([fisher_distribution.log_value(x_value)\
    for x_value in x_values])
pdf_values = pdf_values / np.max(pdf_values)
ax.plot(x_values, pdf_values, color='g', linewidth=linewidth, linestyle='--',\
    label='Fisher matrix')

ax.legend(fontsize=fontsize, frameon=False)
ax.set_xlim((x_values[0], x_values[-1]))
ax.set_ylim((-0.025, 1.025))
ax.xaxis.set_major_locator(MultipleLocator(5e-2))
ax.xaxis.set_minor_locator(MultipleLocator(1e-2))
ax.yaxis.set_major_locator(MultipleLocator(2e-1))
ax.yaxis.set_minor_locator(MultipleLocator(5e-2))
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

first_duration = t1 - t0
second_duration = t2 - t1
print("First call of run took {:.6g} s.".format(first_duration))
print("Second call of run took {:.6g} s.".format(second_duration))

least_square_fit_cluster.triangle_plot(fontsize=12, nbins=num_fits//25,\
    plot_reference_gaussian=True, contour_confidence_levels=0.95)

os.remove('{!s}.hdf5'.format(prefix))
os.remove('{!s}.summary.hdf5'.format(prefix))
pl.show()

