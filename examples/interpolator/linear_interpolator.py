"""
File: examples/interpolator/linear_interpolator.py
Author: Keith Tauscher
Date: 19 Apr 2019

Description: Example showing how to initialize and use a LinearInterpolator
             object.
"""
from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as pl
from pylinex import LinearInterpolator

fontsize = 'xx-large'
seed = 0
ninputs = 1000
npoints = 100000
ndim = 7
inputs = np.random.uniform(0.5, 1, size=(ninputs, ndim))
outputs = np.prod(inputs, axis=1)

interpolator = LinearInterpolator(inputs, outputs, scale_to_cube=True)

np.random.seed(seed)


points_to_interpolate =\
    np.random.uniform(0.65, 0.85, size=(npoints, ndim))
absolute_errors = np.ndarray(npoints)
relative_errors = np.ndarray(npoints)
times = np.ndarray(npoints)
for ipoint in range(npoints):
    point_to_interpolate = points_to_interpolate[ipoint]
    actual_value = np.prod(point_to_interpolate)
    actual_gradient = actual_value / point_to_interpolate
    start_time = time.time()
    (interpolated_value, interpolated_gradient, interpolated_hessian) =\
        interpolator.value_gradient_and_hessian(point_to_interpolate)
    times[ipoint] = time.time() - start_time
    absolute_error = interpolated_value - actual_value
    absolute_errors[ipoint] = absolute_error
    relative_errors[ipoint] = (absolute_error / actual_value)

print('')
print('First time: {} s'.format(times[0]))
print('Average of all other times: {} s'.format(np.mean(times[1:])))
print('Total of all times: {} s'.format(np.sum(times)))

fig = pl.figure()
ax = fig.add_subplot(111)
ax.hist(absolute_errors, bins=100)
ax.set_title('Absolute error (expected to be skewed left)', size=fontsize)
fig = pl.figure()
ax = fig.add_subplot(111)
bins = np.concatenate(([-1e7], np.linspace(-1, 1, 201), [1e7]))
ax.hist(relative_errors, bins=bins)
ax.set_title('Fractional error', size=fontsize)
ax.set_xlim((-1.1, 1.1))

pl.show()

