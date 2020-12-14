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
from distpy import GaussianDistribution
from pylinex import TrainedBasis

seed = 0
np.random.seed(seed)

ntrain = 1000
nvar = 5
xs = np.linspace(-1, 1, 100)
stdvs = 1 / (1 + np.arange(nvar))
mean_vector = np.random.normal(2, 1, size=nvar)
covariance_matrix = np.diag(np.power(stdvs, 2))
coefficient_distribution = GaussianDistribution(mean_vector, covariance_matrix)
training_set_coefficients = coefficient_distribution.draw(ntrain)
training_set = np.array([np.polyval(coeff[-1::-1], xs)\
    for coeff in training_set_coefficients])

num_basis_vectors = 5
error = np.ones_like(xs)

basis_without_translation = TrainedBasis(training_set, num_basis_vectors,\
    error=error, mean_translation=False)
basis_with_translation = TrainedBasis(training_set, num_basis_vectors,\
    error=error, mean_translation=True)

ax = basis_without_translation.plot(x_values=xs, linestyle='-')
basis_with_translation.plot(ax=ax, x_values=xs, linestyle='--', show=True)
