"""
File: examples/expander/axis_expander.py
Author: Keith Tauscher
Date: 4 Feb 2019

Description: Example showing the usage of the AxisExpander class.
"""
import numpy as np
from pylinex import AxisExpander

old_shape = (2, 2)
new_axis_position = 1
new_axis_length = 4
index = 1
pad_value = 0

expander = AxisExpander(old_shape, new_axis_position, new_axis_length, index,\
    pad_value=pad_value)

input_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
expected_array = np.array([[0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0],\
    [0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0]])
output_array = expander(input_array)

assert(np.all(output_array == expected_array))

