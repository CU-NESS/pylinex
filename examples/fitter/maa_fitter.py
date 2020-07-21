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
    MAAFitter

fontsize = 24

num_fits = 10000
stride = num_fits // 100

num_curves = 2
curve_length = 101
num_channels = num_curves * curve_length

epsilon = 1
multiplying_factors = np.linspace(1, 1 + epsilon, num_curves)

desired_expander = RepeatExpander(num_curves)
undesired_expander = MultipleExpander(multiplying_factors)
num_basis_vectors = 10
undesired_basis = LegendreBasis(curve_length, num_basis_vectors - 1,\
    expander=undesired_expander)

error = np.ones(num_channels) * np.sqrt(num_curves)

signals = np.random.randn(num_fits, curve_length)
nuisances =\
    np.dot(np.random.randn(num_fits, num_basis_vectors), undesired_basis.basis)
noises = np.random.randn(num_fits, num_channels) * error[np.newaxis,:]

data = [(desired_expander(signal) + undesired_expander(nuisance) + noise)\
    for (signal, nuisance, noise) in zip(signals, nuisances, noises)]

priors = {}

fitter =\
    MAAFitter(desired_expander, undesired_basis, data, error=error, **priors)

signal_chi_squared = fitter.desired_reduced_chi_squared(signals)

hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
fitter.save(hdf5_file_name)
try:
    loaded_fitter = MAAFitter.load(hdf5_file_name)
    assert np.allclose(loaded_fitter.desired_reduced_chi_squared(signals),\
        signal_chi_squared, rtol=0, atol=1e-6)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.hist(signal_chi_squared, density=True, bins=100, alpha=0.5, color='C0')
xlim = ax.get_xlim()
fitter.desired_reduced_chi_squared_expected_distribution.plot(\
    np.linspace(xlim[0], xlim[1], 1001)[1:], ax=ax, color='k',\
    fontsize=fontsize, xlabel='$\chi^2$', ylabel='PDF',\
    title='PDF of desired component $\chi^2$')
ax.set_xlim((xlim[0], xlim[1]))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
channels = np.arange(curve_length)
ax.plot(channels, (fitter.desired_mean - signals)[0], color='C0', alpha=0.1,\
    label='biases')
ax.plot(channels, (fitter.desired_mean - signals)[stride::stride].T,\
    color='C0', alpha=0.1)
ax.plot(channels, fitter.desired_noise_level, color='C1',\
    label='$1\sigma$ noise level')
ax.plot(channels, -fitter.desired_noise_level, color='C1')
ax.plot(channels, fitter.desired_error, color='C2', label='$1\sigma$ error')
ax.plot(channels, -fitter.desired_error, color='C2')
ax.legend(fontsize=fontsize, frameon=False)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
ax.set_xlabel('Channel #', size=fontsize)
ax.set_ylabel('Bias', size=fontsize)
ax.set_title('Desired component bias vs. channel', size=fontsize)
ax.set_xlim((channels[0], channels[-1]))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
image = ax.imshow(fitter.desired_correlation, vmin=-1, vmax=1, cmap='coolwarm')
pl.colorbar(image)

pl.show()

