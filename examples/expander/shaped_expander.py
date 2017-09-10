import os
import numpy as np
from pylinex import NullExpander, ShapedExpander, load_expander_from_hdf5_file

expander = ShapedExpander(NullExpander(), (2, 6), (3, 4))


input_array = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
expected_output = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

assert np.all(expander(input_array) == expected_output)

file_name = 'test_shaped_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

