"""
File: examples/util/expression.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing multiple ways of initializing and using the
             Expression class.
"""
import os
import numpy as np
from pylinex import Expression

simple_sum = Expression('{0}+{1}')
assert (simple_sum(1, 1) == 2)
assert np.all(simple_sum(np.ones(10) * 2.5, 1) == np.ones(10) * 3.5)

numpy_sum = Expression('np.sum({0})', import_strings=['import numpy as np'])
assert np.isclose(numpy_sum(np.ones(10) * 2.5), 25)

numpy_sum_with_extra_argument = Expression('np.sum({0})', num_arguments=2,\
    import_strings=['import numpy as np'], kwargs={})
assert np.isclose(numpy_sum_with_extra_argument(np.ones(10) * 2.5,\
    'should not be used'), 25)

expression = Expression('np.exp(a * {0})', num_arguments=2,\
    import_strings=['import numpy as np'], kwargs={'a': np.linspace(-2, 2, 5)})
file_name = 'test_TESTING_EXPRESSION_CLASS.hdf5'
expression.save(file_name)
try:
    expression = Expression.load(file_name)
    assert np.allclose(expression(1, 'b'),\
        [1 / (np.e ** 2), 1 / np.e, 1, np.e, np.e ** 2])
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

