"""
File: examples/model/composite_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to initialize, use, save, and load the
             CompositeModel class.
"""
import os
import numpy as np
from distpy import Expression
from pylinex import PolynomialBasis, BasisModel, CompositeModel,\
    load_model_from_hdf5_file

num_channels = 11
x_values = np.linspace(-1, 1, num_channels)
basis = PolynomialBasis(x_values, 2)
reference_model = BasisModel(basis)

expression = Expression('{0}+{1}')
zero = np.zeros(num_channels)
one = np.ones(num_channels)
gradient_expressions = [Expression('one', num_arguments=2,\
    kwargs={'one': one}) for power in range(2)]
hessian_expressions =\
    [[Expression('zero', num_arguments=2, kwargs={'zero': zero})] * 2] * 2
names = ['a', 'b']
models = [BasisModel(basis[0:1]), BasisModel(basis[1:2])]
composite_model = CompositeModel(expression, names, models,\
    gradient_expressions=gradient_expressions,\
    hessian_expressions=hessian_expressions)

random_pars = np.random.randn(2)
assert np.allclose(reference_model(random_pars), composite_model(random_pars))
assert np.allclose(reference_model.gradient(random_pars),\
    composite_model.gradient(random_pars))
assert np.allclose(reference_model.hessian(random_pars),\
    composite_model.hessian(random_pars))

file_name = 'test_TESTING_COMPOSITEMODEL_CLASS.hdf5'
composite_model.save(file_name)
composite_model = load_model_from_hdf5_file(file_name)
try:
    assert np.allclose(reference_model(random_pars),\
        composite_model(random_pars))
    assert np.allclose(reference_model.gradient(random_pars),\
        composite_model.gradient(random_pars))
    assert np.allclose(reference_model.hessian(random_pars),\
        composite_model.hessian(random_pars))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

