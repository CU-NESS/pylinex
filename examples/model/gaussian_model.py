"""
File: examples/model/gaussian_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the GaussianModel class, including how
             to save and load them to and from hdf5 files.
"""
from __future__ import division
import os
import numpy as np
from pylinex import GaussianModel, load_model_from_hdf5_file

num_channels = 100
x_values = np.linspace(-1, 1, num_channels)
model = GaussianModel(x_values)

value = model([1, 0, 1])
gradient = model.gradient([1, 0, 1])
hessian = model.hessian([1, 0, 1])
assert np.allclose(np.exp((x_values ** 2) / (-2.)), value)
assert np.allclose(gradient[:,0], value)
assert np.allclose(gradient[:,1], value * x_values)
assert np.allclose(gradient[:,2], value * (x_values ** 2))
for index1 in range(3):
    for index2 in range(index1):
        assert np.all(hessian[:,index1,index2] == hessian[:,index2,index1])
assert np.all(hessian[:,0,0] == 0)
assert np.allclose(hessian[:,0,1], gradient[:,1])
assert np.allclose(hessian[:,0,2], gradient[:,2])

file_name = 'test_TESTING_GAUSSIANMODEL_CLASS.hdf5'
model.save(file_name)
model = load_model_from_hdf5_file(file_name)
try:
    assert np.allclose(np.exp((x_values ** 2) / (-2.)), model([1, 0, 1]))
except:
    os.remove(file_name)
    raise

os.remove(file_name)

