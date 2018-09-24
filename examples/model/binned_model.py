"""
File: examples/model/binned_model.py
Author: Keith Tauscher
Date: 21 Sep 2018

Description: Script showing correct usage and syntax of the BinnedModel class.
"""
import os
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as pl
from pylinex import RectangularBinner, BinnedModel, GaussianModel,\
    load_model_from_hdf5_file

fontsize = 24

original_num_channels = 1000
final_num_channels = 10

old_x_values = np.linspace(-1, 1, original_num_channels + 2)[1:-1]
submodel = GaussianModel(old_x_values)
new_bin_edges = np.linspace(-1, 1, final_num_channels + 1)
binner = RectangularBinner(old_x_values, new_bin_edges)
new_x_values = binner.binned_x_values
model = BinnedModel(submodel, binner, weights=np.ones_like(old_x_values))

hdf5_file_name = 'TESTINGBINNEDMODELCLASSDELETEIFYOUSEETHIS.hdf5'
try:
    model.save(hdf5_file_name)
    assert(model == load_model_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

parameters = np.array([-1, 0.34, 0.12])

fig = pl.figure(figsize=(16,12))
ax = fig.add_subplot(111)
ax.plot(old_x_values, submodel(parameters), label='raw')
ax.plot(new_x_values, model(parameters), label='binned')
ax.legend(fontsize=fontsize)
ax.xaxis.set_major_locator(MultipleLocator(0.25))
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
ax.set_xlim((-1, 1))
ax.set_xlabel('$x$', size=fontsize)
ax.set_ylabel('$y$', size=fontsize)
ax.set_title('Effect of binning', size=fontsize)

pl.show()

