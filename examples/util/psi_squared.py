"""
File: examples/util/psi_squared.py
Author: Keith Tauscher
Date: 12 Sep 2018

Description: File showing example of how to use the psi_squared function.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from pylinex import psi_squared

seed = 0
np.random.seed(seed)
(num_curves, num_channels) = (1000, 10000)

error = np.linspace(1, 10, num_channels)
curves = np.random.normal(0, 1, size=(num_curves, num_channels)) *\
    error[np.newaxis,:]
psi_squareds = [psi_squared(curve, normalize_by_chi_squared=False,\
    return_null_hypothesis_error=True, error=error,\
    minimum_correlation_spacing=1) for curve in curves]
psi_squared_error = psi_squareds[0][1]
psi_squareds =\
    np.array([psi_squared_tuple[0] for psi_squared_tuple in psi_squareds])

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.hist(psi_squareds)
ylim = ax.get_ylim()
ax.plot([1] * 2, ylim, color='k', linestyle='-')
ax.plot([1 - psi_squared_error] * 2, ylim, color='k', linestyle='--')
ax.plot([1 + psi_squared_error] * 2, ylim, color='k', linestyle='--')
pl.show()

