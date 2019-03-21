"""
File: examples/util/rectangular_binner.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: Example script showing the use of the RectangularBinner class.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from pylinex import RectangularBinner

fontsize = 24

num_old_x_values = 1000
num_new_x_values = 20

wavelength = 0.4

old_x_values = np.linspace(-1, 1, num_old_x_values)[1:-1]
old_error = np.ones_like(old_x_values)
old_y_values =\
    np.sin(2 * np.pi * old_x_values / wavelength) * np.sinh(old_x_values)
new_x_bin_edges = np.linspace(-1, 1, num_new_x_values + 1)
weights = np.ones_like(old_y_values)

binner = RectangularBinner(old_x_values, new_x_bin_edges)
new_x_values = binner.binned_x_values
(new_y_values, new_weights) = binner.bin(old_y_values, weights=weights,\
    return_weights=True)
new_error = binner.bin_error(old_error, weights=weights, return_weights=False)
print("new_error={}".format(new_error))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.plot(old_x_values, old_y_values, label='unbinned')
ax.plot(new_x_values, new_y_values, label='binned')
ax.legend(fontsize=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

