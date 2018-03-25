"""
File: examples/model/constant_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the ConstantModel class, including
             saving and loading it to and from hdf5 files.
"""
import os
import numpy as np
from pylinex import ConstantModel, load_model_from_hdf5_file

constant_model = ConstantModel(50)

modeled = constant_model(100.)
assert (modeled.shape == (50,))
assert np.all(modeled == 100.)

gradient = constant_model.gradient(100.)
assert (gradient.shape == (50, 1))
assert np.all(gradient == 1.)

hessian = constant_model.hessian(100.)
assert (hessian.shape == (50, 1, 1))
assert np.all(hessian == 0.)

file_name = 'test_TESTING_CONSTANTMODEL_CLASS.hdf5'
constant_model.save(file_name)

try:
    constant_model = load_model_from_hdf5_file(file_name)
    modeled = constant_model(0.)
    assert (modeled.shape == (50,))
    assert np.all(modeled == 0.)
    constant_model = ConstantModel.load(file_name)
    modeled = constant_model(0.)
    assert (modeled.shape == (50,))
    assert np.all(modeled == 0.)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

