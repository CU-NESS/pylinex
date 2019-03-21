"""
File: examples/fitter/extractor.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example of how to use Extractor class to go from training set,
             data, and error to find subbasis fits. In this example, only one
             component basis is used, but in general, the Extractor class can
             deal with an arbitrary number of subbasis to separate from each
             other.
"""
from __future__ import division
import os
import numpy as np
import numpy.random as rand
from distpy import GaussianDistribution
from pylinex import AttributeQuantity, CompiledQuantity, Extractor

ntrain = 1000
nvar = 20
xs = np.linspace(-1, 1, 100)
mean_vector = np.zeros(nvar)
stdvs = 1 / (1 + np.arange(nvar))
covariance_matrix = np.diag(np.power(stdvs, 2))
coefficient_distribution = GaussianDistribution(mean_vector, covariance_matrix)
training_set_coefficients = coefficient_distribution.draw(ntrain)
training_set = np.array([np.polyval(coeff[-1::-1], xs)\
                                       for coeff in training_set_coefficients])
quantity = CompiledQuantity('compiled', AttributeQuantity('BPIC'))

error = np.ones_like(xs) * 1e-3
data = rand.normal(0, 1, error.shape) * error
names = ['signal']
training_sets = [training_set]
dimensions = [{'signal': np.arange(1, 21)}]

extractor = Extractor(data, error, names, training_sets, dimensions,\
    compiled_quantity=quantity, quantity_to_minimize='BPIC',\
    expanders=None, num_curves_to_score=0, use_priors_in_fit=False,\
    verbose=True)

extractor.fitter.plot_subbasis_fit(name='signal', show=True)

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
try:
    extractor.save(hdf5_file_name)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

