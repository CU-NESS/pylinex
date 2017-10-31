"""
File: examples/fitter/fitter.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example of how to use Fitter class to perform least square fit
             with single basis. Even though in this example, there is only one
             component basis, the fitter class can handle an arbitrary number
             of component basis sets to separate.
"""
import numpy as np
import numpy.random as rand
from pylinex import PolynomialBasis, BasisSum, Fitter

xs = np.linspace(-1, 1, 100)

noise_level = 1e-1
noiseless_data = np.polyval([4, 3, 2, 1, 0, 0, 9], xs)
data = noiseless_data + (rand.normal(0, 1, xs.shape) * noise_level)

name = 'polynomial'
basis = PolynomialBasis(xs, 15)
basis_sum = BasisSum([name], [basis])
fitter = Fitter(basis_sum, data)


fitter.plot_subbasis_fit(nsigma=1, name=name, true_curve=noiseless_data,\
    x_values=xs, colors='r', show=True)

