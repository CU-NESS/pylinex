"""
File: examples/model/basis_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the BasisMode class, including saving
             and loading it to and from hdf5 files.
"""
import os
import numpy as np
from pylinex import PolynomialBasis, BasisModel, load_model_from_hdf5_file

num_channels = 100
x_values = np.linspace(-1, 1, num_channels)
basis = PolynomialBasis(x_values, 3)
model = BasisModel(basis)

assert np.allclose(x_values ** 2 + x_values, model([0, 1, 1]))
assert np.allclose(np.stack([x_values ** power for power in range(3)],\
    axis=1), model.gradient([0, 0, 0]))
assert np.allclose(np.zeros((num_channels, 3, 3)), model.hessian([0, 0, 0]))

file_name = 'test_TESTING_BASISMODEL_CLASS.hdf5'
model.save(file_name)
try:
    model = load_model_from_hdf5_file(file_name)
    assert np.allclose(x_values ** 2 + x_values, model([0, 1, 1]))
    model = BasisModel.load(file_name)
    assert np.allclose(x_values ** 2 + x_values, model([0, 1, 1]))
except:
    os.remove(file_name)
    raise

os.remove(file_name)

