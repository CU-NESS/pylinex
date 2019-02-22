"""
File: examples/basis/training_set_plot.py
Author: Keith Tauscher
Date: 22 Feb 2019

Description: Example script showing how to use the plot_training_set_with_modes
             function to plot a three-panel figure summarizing a training set.
"""
from __future__ import division
import numpy as np
from distpy import DistributionSet, GaussianDistribution
from pylinex import plot_training_set_with_modes, GaussianModel

seed = 0
np.random.seed(seed)

num_modes = 5
x_values = np.linspace(40, 120, 81)
integration_time = 100
error = 83 * np.power(x_values / 50, -2.5) / np.sqrt(integration_time)
xlabel = '$\\nu$ [MHz]'
extra_ylabel_string = ' [mK]'
title = 'Gaussian training set'
fontsize = 24
curve_slice = slice(0, None, 2)
alpha = 1

model = GaussianModel(x_values)
distribution_set = DistributionSet()
distribution_set.add_distribution(GaussianDistribution(-100, 50 ** 2),\
    'amplitude')
distribution_set.add_distribution(GaussianDistribution(80, 20 ** 2), 'center')
distribution_set.add_distribution(GaussianDistribution(10, 2 ** 2), 'scale')
ndraw = 10
return_parameters = False
training_set = model.curve_sample(distribution_set, ndraw,\
    return_parameters=return_parameters)

plot_training_set_with_modes(training_set, num_modes, x_values=x_values,\
    curve_slice=curve_slice, alpha=alpha, fontsize=fontsize, xlabel=xlabel,\
    extra_ylabel_string=extra_ylabel_string, title=title, show=True)

