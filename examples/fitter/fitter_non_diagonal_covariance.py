"""
File: examples/fitter/fitter.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example of how to use Fitter class to perform least square fit
             with single basis. Even though in this example, there is only one
             component basis, the fitter class can handle an arbitrary number
             of component basis sets to separate.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from distpy import SparseSquareBlockDiagonalMatrix, SparseGaussianDistribution
from pylinex import PolynomialBasis, BasisSum, Fitter

correlation = 0.5
quarter_number = 25
xs = np.linspace(-1, 1, 4 * quarter_number)

noise_level = 1e-1
block =\
    (np.ones((4,4)) * correlation) + (np.identity(4) * (1 - correlation)) *\
    (noise_level ** 2)
error = SparseSquareBlockDiagonalMatrix([block] * (len(xs) // 4))
noise_distribution = SparseGaussianDistribution(np.zeros(len(xs)), error)
noiseless_data = np.polyval([4, 3, 2, 1, 0, 0, 9], xs)
noise = noise_distribution.draw()
data = noiseless_data + noise

name = 'polynomial'
basis = PolynomialBasis(xs, 15)
basis_sum = BasisSum([name], [basis])
fitter = Fitter(basis_sum, data, error)


fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.scatter(xs, data, color='k')
fitter.plot_subbasis_fit(nsigma=1, name=name, true_curve=None,\
    x_values=xs, colors='r', ax=ax, show=False)
pl.show()

