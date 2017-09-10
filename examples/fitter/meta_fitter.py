"""
File: examples/fitter/meta_fitter.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use MetaFitter class to perform many least
             square fits and autonomously choose the "best" basis based on a
             given statistic (BPIC in this case). The MetaFitter class can
             handle any number of dimensions and can calculate any number of
             quantities at each grid square.
"""
import numpy as np
import numpy.random as rand
from pylinex import AttributeQuantity, PolynomialBasis, BasisSet, MetaFitter

xs = np.linspace(-1, 1, 100)

noise_level = 1e-1
error = np.ones_like(xs)
noiseless_data = np.polyval([4, 3, 2, 1, 0, 0, 9], xs)
data = noiseless_data + (rand.normal(0, 1, xs.shape) * error)

name = 'polynomial'
basis = PolynomialBasis(xs, 15)
basis_set = BasisSet([name], [basis])

quantity = AttributeQuantity('BPIC')

dimension = {'polynomial': np.arange(1, 16)}

meta_fitter = MetaFitter(basis_set, data, error, quantity, dimension)
fitter = meta_fitter.minimize_quantity('BPIC')
fitter.plot_subbasis_fit(nsigma=1, name=name, true_curve=noiseless_data,\
    x_values=xs, colors='r', show=True)

