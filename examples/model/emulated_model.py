"""
File: examples/model/emulated_model.py
Author: Keith Tauscher
Date: 8 Jul 2019

Description: Example showing how to initialize and use the EmulatedModel class.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import TruncatedGaussianDistribution, GaussianDistribution,\
    DistributionSet
from pylinex import GaussianModel, EmulatedModel, load_model_from_hdf5_file

try:
    import emupy
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
except:
    raise ImportError("The EmulatedModel class (and the emulated_model.py " +\
        "example script) cannot be used if emupy and/or sklearn are not " +\
        "installed.")

hdf5_file = 'TESTINGEMULATEDMODEL.hdf5'

np.random.seed(0)

nchannels = 501
channels = np.linspace(-1, 1, nchannels)

train_ndraw = 100
test_ndraw = 10

nbasis_vectors = 20

true_model = GaussianModel(channels)

amplitude_mean = -100
amplitude_stdv = 10
center_mean = 0
center_stdv = 0.1
sigma_mean = 0.25
sigma_stdv = 0.1

prior_set = DistributionSet()
prior_set.add_distribution(\
    GaussianDistribution(amplitude_mean, amplitude_stdv ** 2), 'amplitude')
prior_set.add_distribution(\
    GaussianDistribution(center_mean, center_stdv ** 2), 'center')
prior_set.add_distribution(\
    TruncatedGaussianDistribution(sigma_mean, sigma_stdv ** 2, low=0), 'scale')

draw1 = prior_set.draw(train_ndraw)
inputs1 = np.stack([draw1[par] for par in true_model.parameters], axis=1)
true_outputs1 = np.array([true_model(inp) for inp in inputs1])

draw2 = prior_set.draw(test_ndraw)
inputs2 = np.stack([draw2[par] for par in true_model.parameters], axis=1)
true_outputs2 = np.array([true_model(inp) for inp in inputs2])

amplitude_scale = 10
center_scale = 0.1
sigma_scale = 0.1

kernel = Matern(np.array([amplitude_scale, center_scale, sigma_scale]), nu=2.5)
emulated_model = EmulatedModel(true_model.parameters, inputs1, true_outputs1,\
    error=None, num_modes=nbasis_vectors, kernel=kernel, verbose=False)

emulated_outputs2 = np.array([emulated_model(inp) for inp in inputs2])
emulated_model.save(hdf5_file)
emulated_model_again = load_model_from_hdf5_file(hdf5_file)
emulated_outputs2_again =\
    np.array([emulated_model_again(inp) for inp in inputs2])

assert(np.all(emulated_outputs2 == emulated_outputs2_again))

emulation_errors = true_outputs2 - emulated_outputs2
pl.plot(channels, (emulated_outputs2 - true_outputs2).T, color='r',\
    linewidth=1, alpha=1)
pl.xlim((channels[0], channels[-1]))

os.remove(hdf5_file)

pl.show()

