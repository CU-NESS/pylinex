"""
File: examples/expander/repeat_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use RepeatExpander class.
"""
import os
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as pl
from pylinex import RepeatExpander, load_expander_from_hdf5_file

arrays = [np.linspace(-1, 5, 1000), rand.rand(3203)]

error = rand.rand(300)
tiled_error = np.tile(error, (3,))

expander = RepeatExpander(3)

for array in arrays:
    expanded_array = expander(array)
    third_length = len(expanded_array) // 3
    assert np.all(expanded_array[:third_length] == array)
    assert np.all(expanded_array[third_length:-third_length] == array)
    assert np.all(expanded_array[-third_length:] == array)
assert np.allclose(expander.contract_error(tiled_error), error / np.sqrt(3),\
    rtol=0, atol=1e-9)

file_name = 'test_repeat_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

