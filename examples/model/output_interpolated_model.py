"""
File: examples/model/output_interpolated_model.py
Author: Keith Tauscher
Date: 30 Jul 2019

Description: Shows a usage of the OutputInterpolatedModel class.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from pylinex import FixedModel, OutputInterpolatedModel,\
    load_model_from_hdf5_file

file_name = 'TESTINGOUTPUTINTERPOLATEDMODELDELETETHIS.hdf5'

(amplitude, phase) = (5.1468, 0.29)
half_small_num_channels = 50
half_large_num_channels = 1000
small_num_channels = 1 + (2 * half_small_num_channels)
large_num_channels = 1 + (2 * half_large_num_channels)
fixed_slice = np.array([index for index in range(small_num_channels) if\
    (((index < 20) or (index > 30)) and ((index < 70) or (index > 80)) and\
    ((index < 45) or (index > 55)))])
fixed_xs = np.linspace(-1, 1, small_num_channels)
interpolation_xs = np.linspace(-1, 1, large_num_channels)
fixed_ys = amplitude * np.sin((fixed_xs * np.pi) + phase)

fixed_xs = fixed_xs[fixed_slice]
fixed_ys = fixed_ys[fixed_slice]

fixed_model = FixedModel(fixed_ys)
interpolated_model =\
    OutputInterpolatedModel(fixed_model, fixed_xs, interpolation_xs, order=1)
interpolated_by_model = interpolated_model(np.array([]))
interpolated_by_numpy = np.interp(interpolation_xs, fixed_xs, fixed_ys)

try:
    interpolated_model.save(file_name)
    assert(interpolated_model == load_model_from_hdf5_file(file_name))
except:
    if os.path.exists(file_name):
        os.remove(file_name)
    raise
else:
    os.remove(file_name)

assert(np.allclose(\
    interpolated_by_model, interpolated_by_numpy, atol=1e-12, rtol=0))

pl.plot(interpolation_xs, interpolated_by_model, color='C0',\
    label='after_interpolation')
pl.scatter(fixed_xs, fixed_ys, color='k', label='before interpolation')
pl.xlim((-1, 1))
pl.ylim((-1.1 * amplitude, 1.1 * amplitude))
pl.legend()

pl.show()


