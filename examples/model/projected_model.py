"""
File: examples/model/projected_model.py
Author: Keith Tauscher
Date: 13 Jul 2018

Description: 
"""
import os
import numpy as np
from pylinex import Basis, BasisModel, ProjectedModel,\
    load_model_from_hdf5_file

num_inputs = 50
num_powers = 5

x_values = np.linspace(-1, 1, 101)
powers = np.arange(num_powers)
basis = Basis(x_values[np.newaxis,:] ** powers[:,np.newaxis])
model = ProjectedModel(BasisModel(basis), basis)

file_name = 'TESTING_PROJECTEDMODEL_CLASS_DELETETHISIFYOUSEEIT.hdf5'
try:
    model.save(file_name)
    assert(model == load_model_from_hdf5_file(file_name))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

inputs = np.random.normal(size=(num_inputs, num_powers))
for inp in inputs:
    assert(np.allclose(inp, model(inp)))

