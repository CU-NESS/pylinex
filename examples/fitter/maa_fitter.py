"""
File: examples/fitter/maa_fitter.py
Author: Keith Tauscher
Date: 19 Jul 2020

Description: Example script showing how to use the MAAFitter class.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import ChiSquaredDistribution
from pylinex import RepeatExpander, MultipleExpander, Basis, LegendreBasis,\
    MAAFitter, GaussianModel

seed = 0
np.random.seed(seed)
fontsize = 24

num_fits = 10000
stride = num_fits // 10

num_curves = 2
curve_length = 101
(start, end) = (50, 100)
channels = np.linspace(start, end, curve_length)
num_channels = num_curves * curve_length

epsilon = 1
multiplying_factors = np.linspace(1, 1 + epsilon, num_curves)

desired_expander = RepeatExpander(num_curves)
undesired_expander = MultipleExpander(multiplying_factors)
num_basis_vectors = 10
undesired_basis = LegendreBasis(curve_length, num_basis_vectors - 1,\
    expander=undesired_expander)

error = np.ones(num_channels) * np.sqrt(num_curves)

signals = GaussianModel(channels)(np.array([-10, 80, 5]))
signals = signals[np.newaxis,:] * np.ones((num_fits, 1))
nuisances =\
    np.dot(np.random.randn(num_fits, num_basis_vectors), undesired_basis.basis)
noises = np.random.randn(num_fits, num_channels) * error[np.newaxis,:]

data = [(desired_expander(signal) + undesired_expander(nuisance) + noise)\
    for (signal, nuisance, noise) in zip(signals, nuisances, noises)]

priors = {}

fitter =\
    MAAFitter(desired_expander, undesired_basis, data, error=error, **priors)

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
fitter.save(hdf5_file_name)
try:
    loaded_fitter = MAAFitter.load(hdf5_file_name)
    assert np.allclose(loaded_fitter.desired_reduced_chi_squared(signals),\
        fitter.desired_reduced_chi_squared(signals), rtol=0, atol=1e-6)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

fitter.plot_reduced_chi_squared_histogram(xlabel='$\chi^2$', ylabel='PDF',\
    title='PDF of full data $\chi^2$', show=False, bins=100, alpha=0.5,\
    color='C0')
fitter.plot_desired_reduced_chi_squared_histogram(signals, xlabel='$\chi^2$',\
    ylabel='PDF', title='PDF of desired component $\chi^2$', show=False,\
    bins=100, alpha=0.5, color='C0')
which = 10
fitter.plot_desired_mean(nsigmas=[1,2], alphas=[0.5,0.2], which=which,\
    desired_component=signals[which], channels=channels, xlabel='x',\
    ylabel='y', title='Desired component mean vs. channel')
which = slice(None, None, stride)
fitter.plot_desired_bias(signals[which], which=which, channels=channels,\
    xlabel='x', ylabel='bias', title='Desired component bias vs. channel',\
    fontsize=fontsize, color='C0', plot_desired_error=True,\
    plot_desired_noise_level=True, alpha=0.1)
fitter.plot_desired_error(plot_desired_noise_level=True, channels=channels,\
    xlabel='$x$', ylabel='$1\sigma$ uncertainty',\
    title='Uncertainty on desired component', color='k')
fitter.plot_desired_correlation(title='MAA channel correlation matrix',\
    axis_label='$x$', colorbar_label='$\\rho$',\
    channel_lim=(channels[0], channels[-1]))

pl.show()

