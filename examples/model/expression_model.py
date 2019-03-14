"""
File: examples/model/expression_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to use the ExpressionModel class, including
             how to save and load them to and from hdf5 files.
"""
import os
import numpy as np
from distpy import Expression
from pylinex import Basis, BasisModel, ExpressionModel,\
    load_model_from_hdf5_file

num_channels = 2
basis = Basis(np.identity(2))
reference_model = BasisModel(basis)

import_strings = ['import numpy as np']

expression = Expression('np.array([{0}, {1}])', import_strings=import_strings)
zero = np.zeros(num_channels)
one = np.ones(num_channels)
gradient_expressions = [Expression('np.array([1, 0])', num_arguments=2,\
    import_strings=import_strings), Expression('np.array([0, 1])',\
    num_arguments=2, import_strings=import_strings)]
hessian_expressions =\
    [[Expression('zero', num_arguments=2, kwargs={'zero': zero})] * 2] * 2
parameters = ['x', 'y']
expression_model = ExpressionModel(expression, parameters,\
    gradient_expressions=gradient_expressions,\
    hessian_expressions=hessian_expressions)

random_pars = np.random.randn(2)
assert np.allclose(reference_model(random_pars), expression_model(random_pars))
assert np.allclose(reference_model.gradient(random_pars),\
    expression_model.gradient(random_pars))
assert np.allclose(reference_model.hessian(random_pars),\
    expression_model.hessian(random_pars))

file_name = 'test_TESTING_EXPRESSIONMODEL_CLASS.hdf5'
expression_model.save(file_name)
expression_model = load_model_from_hdf5_file(file_name)
try:
    assert np.allclose(reference_model(random_pars),\
        expression_model(random_pars))
    assert np.allclose(reference_model.gradient(random_pars),\
        expression_model.gradient(random_pars))
    assert np.allclose(reference_model.hessian(random_pars),\
        expression_model.hessian(random_pars))
except:
    os.remove(file_name)
    raise

os.remove(file_name)

