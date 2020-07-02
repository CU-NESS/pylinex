"""
File: examples/expander/null_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example of how to create and use a NullExpander object, which does
             nothing to its inputs.
"""
import os
import numpy as np
import numpy.random as rand
from pylinex import MultipleExpander, load_expander_from_hdf5_file

multiples = rand.uniform(0, 1, size=10)
expander = MultipleExpander(multiples)
error = np.ones(100 * len(multiples))

arrays = [np.arange(100), rand.rand(100), np.linspace(-np.pi, np.pi, 1000)]

for array in arrays:
    expanded_array =\
        np.concatenate([(multiple * array) for multiple in multiples])
    assert np.all(expanded_array == expander(array))

contracted_error = np.ones(100) / np.sqrt(np.sum(np.power(multiples, 2)))
assert np.all(contracted_error == expander.contract_error(error))

file_name = 'test_null_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)
