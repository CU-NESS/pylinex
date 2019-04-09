"""
File: examples/model/direct_sum_model.py
Author: Keith Tauscher
Date: 1 Jul 2018

Description: File containing an example script showing various uses of the
             DirectSumModel class.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from pylinex import PadExpander, ConstantModel, ExpandedModel, DirectSumModel,\
    load_model_from_hdf5_file

seed = 0
fontsize = 24

smaller_num_channels = 100
num_models = 10
larger_num_channels = num_models * smaller_num_channels
channels = np.arange(larger_num_channels)
noise_level = 1


np.random.seed(seed)
names = [chr(ord('A') + index) for index in range(num_models)]
expanders = [PadExpander('{:d}*'.format(index),\
    '{:d}*'.format(num_models - index - 1)) for index in range(num_models)]
models = [ExpandedModel(ConstantModel(smaller_num_channels), expander)\
    for expander in expanders]
direct_sum_model = DirectSumModel(names, models)

hdf5_file_name = 'TESTING_DIRECT_SUM_MODEL_CLASS_DELETE_THIS.hdf5'
direct_sum_model.save(hdf5_file_name)
try:
    assert(direct_sum_model == load_model_from_hdf5_file(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

true_parameters = ((np.arange(num_models) + 1) ** 2)
true_curve = np.concatenate([(parameter * np.ones(smaller_num_channels))\
    for parameter in true_parameters], axis=0)
error = np.ones_like(true_curve) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error

data = true_curve + noise

(mean, covariance) = direct_sum_model.quick_fit(data, error=error)
variances = np.sqrt(np.diag(covariance))
expected_variances =\
    ((noise_level / np.sqrt(smaller_num_channels)) * np.ones(num_models))
assert(np.all(np.abs(mean - true_parameters) < np.abs(5 * expected_variances)))
fit_curve = direct_sum_model(mean)

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.plot(channels, data, label='data')
ax.plot(channels, fit_curve, label='fit')
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('y', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
ax.legend(fontsize=fontsize, loc='upper left')

pl.show()

