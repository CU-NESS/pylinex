"""
File: examples/basis/trained_basis_with_and_without_translation.py
Author: Keith Tauscher
Date: 14 Dec 2020

Description: Script showing that using the mean_translation parameter of the
             TrainedBasis class changes the basis vectors when the mean of the
             training set is nonzero.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from pylinex import TrainedBasis, GaussianModel

fontsize = 24
seed = 0
np.random.seed(seed)

ntrain = 10000
nvar = 10
xs = np.linspace(-1, 1, 100)
stdvs = (1 / (1 + np.arange(nvar))) / 100
training_set_mean = GaussianModel(xs)(np.array([10, 0.25, 0.5]))
mean_vector = np.zeros(nvar)
covariance_matrix = np.diag(np.power(stdvs, 2))
coefficient_distribution = GaussianDistribution(mean_vector, covariance_matrix)
training_set_coefficients = coefficient_distribution.draw(ntrain)
training_set = np.array([(np.polyval(coeff[-1::-1], xs) + training_set_mean)\
    for coeff in training_set_coefficients])

num_basis_vectors_to_plot = 5
error = np.ones_like(xs) * 0.01

#training_set = training_set +\
#    (np.random.normal(0, 1, size=training_set.shape) * error[np.newaxis,:])

basis_without_translation = TrainedBasis(training_set, len(training_set),\
    error=error, mean_translation=False)
basis_with_translation = TrainedBasis(training_set, len(training_set),\
    error=error, mean_translation=True)

ax = basis_without_translation[:5].plot(x_values=xs, linestyle='-')
ax.plot(xs, basis_with_translation.translation, color='k', linestyle='-.')
basis_with_translation[:5].plot(ax=ax, x_values=xs, linestyle='--', show=False)

ax = basis_without_translation.plot_RMS_spectrum(color='k',\
    label='without mean subtracted', plot_reference_lines=False)
basis_with_translation.plot_RMS_spectrum(color='r',\
    label='with mean subtracted', plot_reference_lines=False, ax=ax)
ax.legend(fontsize=fontsize)

pl.show()

