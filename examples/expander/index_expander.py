"""
File: examples/expander/index_expander.py
Author: Keith Tauscher
Date: 27 Mar 2019

Description: Example script showing how to use the index expander and testing
             some of its outputs.
"""
import numpy as np
from pylinex import IndexExpander

expanded_shape = (3, 5)
axis = 1
indices = [0, 1, 4]
modulating_factors = [1, 2, 3]
pad_value = 0

expander = IndexExpander(expanded_shape, axis, indices,\
    modulating_factors=modulating_factors, pad_value=pad_value)

input_array = np.array([[1, 1, 1, 10, 10, 10, 100, 100, 100],\
    [2, 2, 2, 20, 20, 20, 200, 200, 200]])
expected_output_array =\
    np.array([[1, 2, 0, 0, 3, 10, 20, 0, 0, 30, 100, 200, 0, 0, 300],\
    [2, 4, 0, 0, 6, 20, 40, 0, 0, 60, 200, 400, 0, 0, 600]])
output_array = expander(input_array)

assert(np.all(output_array == expected_output_array))

