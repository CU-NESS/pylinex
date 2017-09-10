import os
import numpy as np
from pylinex import PadExpander, load_expander_from_hdf5_file

array = np.arange(100)
error = np.tile(np.linspace(1, 2, 20), (5,))

expander = PadExpander(1, 3)
expanded_array = expander(array)
assert np.all(expanded_array[:1] == 0)
assert np.all(expanded_array[1:-3] == array)
assert np.all(expanded_array[-3:] == 0)
assert np.all(expander.contract_error(error) == error[1:-3])

expander = PadExpander('2+', '3*')
expanded_array = expander(array)
assert np.all(expanded_array[:2] == 0)
assert np.all(expanded_array[2:-300] == array)
assert np.all(expanded_array[-300:] == 0)
assert np.all(expander.contract_error(\
    np.concatenate([[0, 0], error, [0] * 300])) == error)

expander = PadExpander('3*', '1*', pad_value=1)
expanded_array = expander(array)
assert np.all(expanded_array[:300] == 1)
assert np.all(expanded_array[300:-100] == array)
assert np.all(expanded_array[-100:] == 1)
assert np.all(expander.contract_error(error) == error[:20])

file_name = 'test_pad_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

