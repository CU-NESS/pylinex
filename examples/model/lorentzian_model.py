"""
File: examples/model/lorentzian_model.py
Author: Keith Tauscher
Date: 7 Jul 2018

Description: Example showing how to use the LorentzianModel class, including
             how to save and load them to and from hdf5 files.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from pylinex import LorentzianModel, load_model_from_hdf5_file

num_channels = 1000
x_values = np.linspace(-10, 10, num_channels)
model = LorentzianModel(x_values)

value = model([1, 0, 1])
gradient = model.gradient([1, 0, 1])
hessian = model.hessian([1, 0, 1])

base_value = model.base_function(x_values)
base_derivative = model.base_function_derivative(x_values)
#base_second_derivative = model.base_function_second_derivative(x_values)

expected_value = model.base_function(x_values)
expected_gradient = np.stack([base_value,\
    ((-1) * base_derivative), (((-1) * base_derivative) * x_values)], axis=1)
#expected_hessian = model.base_function_second_derivative(x_values)

assert(np.allclose(value, expected_value))
assert(np.allclose(gradient, expected_gradient))

file_name = 'test_TESTING_LORENTZIANMODEL_CLASS.hdf5'
model.save(file_name)
model = load_model_from_hdf5_file(file_name)
try:
    assert np.allclose(expected_value, model([1, 0, 1]))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.plot(x_values, value)
pl.show()

