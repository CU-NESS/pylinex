"""
File: examples/expander/composite_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing usage of the CompositeExpander class which
             concatenates usage of multiple individual Expander objects. In
             this case, data is padded with zeros and then repeated.
"""
import os
import numpy as np
from pylinex import PadExpander, RepeatExpander, CompositeExpander,\
    load_expander_from_hdf5_file

expanders = [PadExpander('1+', '2+'), RepeatExpander(2)]
composite_expander = CompositeExpander(*expanders)

array = np.arange(1, 4)
expanded_array = composite_expander(array)
expected_expanded_array = np.array([0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0])
assert np.all(expanded_array == expected_expanded_array)


file_name = 'test_composite_expander_TEMP.hdf5'
composite_expander.save(file_name)
try:
    assert composite_expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

