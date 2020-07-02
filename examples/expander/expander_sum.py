"""
File: examples/expander/expander_sum.py
Author: Keith Tauscher
Date: 1 Jul 2020

Description: Example of how to create and use a ExpanderSum object, which
             represents the sum of multiple Expander objects.
"""
from __future__ import division
import os
import numpy as np
import numpy.random as rand
from pylinex import NullExpander, ExpanderSum, load_expander_from_hdf5_file

expander = ExpanderSum(NullExpander(), NullExpander())
error = rand.rand(100) ** 2

arrays = [np.arange(100), rand.rand(100), np.linspace(-np.pi, np.pi, 1000)]

for array in arrays:
    assert np.all(2 * array == expander(array))

assert\
    np.allclose(error / 2, expander.contract_error(error), atol=1e-10, rtol=0)

file_name = 'test_expander_sum_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)
