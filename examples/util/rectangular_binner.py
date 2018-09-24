"""
File: examples/util/rectangular_binner.py
Author: Keith Tauscher
Date: 22 Sep 2018

Description: Example script showing the use of the RectangularBinner class.
"""
import numpy as np
import matplotlib.pyplot as pl
from pylinex import rect_bin

fontsize = 24

num_old_x_values = 1000
num_new_x_values = 20

wavelength = 0.4

old_x_values = np.linspace(-1, 1, num_old_x_values)[1:-1]
old_y_values =\
    np.sin(2 * np.pi * old_x_values / wavelength) * np.sinh(old_x_values)
new_x_bin_edges = np.linspace(-1, 1, num_new_x_values + 1)
weights = np.ones_like(old_y_values)

(new_x_values, new_y_values, new_weights) = rect_bin(new_x_bin_edges,\
    old_x_values, old_y_values, weights=weights, return_weights=True)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.plot(old_x_values, old_y_values, label='unbinned')
ax.plot(new_x_values, new_y_values, label='binned')
ax.legend(fontsize=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()
