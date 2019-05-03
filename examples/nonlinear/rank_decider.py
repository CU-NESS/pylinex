"""
File: examples/nonlinear/rank_decider.py
Author: Keith Tauscher
Date: 24 Apr 2019

Description: Example script showing how to use the RankDecider class, which
             decides how many terms to use in a multi-component data fit.
"""
import numpy as np
from distpy import Expression
from pylinex import PolynomialBasis, FourierBasis, BasisSet, FixedModel,\
    GaussianModel, RankDecider

half_num_channels = 100
num_channels = (2 * half_num_channels) + 1
xs = np.linspace(-1, 1, num_channels)
(polynomial_amplitude, fourier_amplitude) = (34, 86)
polynomial_basis = PolynomialBasis(xs, 10)
polynomial_part = polynomial_amplitude * (xs ** 2) * (1 - xs) * (1 + xs)
fourier_basis = FourierBasis(num_channels, 10)
fourier_part = fourier_amplitude * np.sin(np.pi * xs)
gaussian_model = GaussianModel(xs)
true_gaussian_parameters = np.array([10, 0.1, 0.1])
gaussian_part = gaussian_model(true_gaussian_parameters)
total = polynomial_part + fourier_part + gaussian_part
error = np.ones_like(total)
data = total + (error * np.random.normal(0, 1, size=error.shape))
expression = Expression('({0}+{1}+{2})*{3}')
parameter_penalty = np.log(num_channels)
non_basis_models =\
    {'gaussian': gaussian_model, 'constant': FixedModel(np.ones(num_channels))}

names = ['polynomial', 'fourier', 'gaussian', 'constant']
basis_set =\
    BasisSet(['polynomial', 'fourier'], [polynomial_basis, fourier_basis])
initial_nterms = {'polynomial': 10, 'fourier': 10}
true_parameters = {'gaussian': true_gaussian_parameters}
true_curves = {'polynomial': (polynomial_part, error),\
    'fourier': (fourier_part, error)}
return_trail = False
can_backtrack = False
bounds = {'gaussian_scale': (0, None)}

rank_decider = RankDecider(names, basis_set, data, error, expression,\
    parameter_penalty=parameter_penalty, **non_basis_models)
optimal_nterms = rank_decider.minimize_information_criterion(initial_nterms,\
    true_parameters, true_curves, return_trail=return_trail,\
    can_backtrack=can_backtrack, **bounds)
expected_optimal_nterms = {'polynomial': 5, 'fourier': 2}

print("expected_optimal_nterms={}".format(expected_optimal_nterms))
print("optimal_nterms={}".format(optimal_nterms))

