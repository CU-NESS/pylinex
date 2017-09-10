import os
import numpy as np
import numpy.random as rand
from pylinex import NullExpander, load_expander_from_hdf5_file

expander = NullExpander()
error = rand.rand(100) ** 2

arrays = [np.arange(100), rand.rand(100), np.linspace(-np.pi, np.pi, 1000)]

for array in arrays:
    assert np.all(array == expander(array))

assert np.all(error == expander.contract_error(error))

file_name = 'test_null_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)
