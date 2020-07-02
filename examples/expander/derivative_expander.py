"""
File: examples/expander/null_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example of how to create and use a NullExpander object, which does
             nothing to its inputs.
"""
from __future__ import division
import os
import numpy as np
import numpy.random as rand
from pylinex import DerivativeExpander, load_expander_from_hdf5_file

channel_width = 2.64

expander = DerivativeExpander(channel_width)

arrays = [np.arange(100), rand.rand(100), np.linspace(-np.pi, np.pi, 1000)]

for array in arrays:
    expanded_array = (array[1:] - array[:-1]) / channel_width
    expanded_array = np.concatenate([[expanded_array[0]],\
        (expanded_array[1:] + expanded_array[:-1]) / 2, [expanded_array[-1]]])
    assert np.all(expanded_array == expander(array))


file_name = 'test_derivative_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)
