"""
File: examples/expander/modulation_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the ModulationExpander class.
"""
import os
import numpy as np
from pylinex import ModulationExpander, load_expander_from_hdf5_file

array = np.linspace(40, 50, 100)
error = np.ones(100)

modulators = [np.ones(100), np.linspace(-1, 1, 100)]
xs = np.linspace(0, 1, 100)
expectations = [array, (20 * (xs ** 2)) + (70 * xs) - 40]
expected_contracted_errors = [error, error / np.abs(np.linspace(-1, 1, 100))]

expander = ModulationExpander(modulators[0])
assert np.all(expander(array) == expectations[0])
assert np.all(expander.contract_error(error) == expected_contracted_errors[0])

expander = ModulationExpander(modulators[1])
assert np.allclose(expander(array), expectations[1], rtol=0, atol=1e-9)
assert np.allclose(expander.contract_error(error),\
    expected_contracted_errors[1], rtol=0, atol=1e-9)

expander = ModulationExpander(modulators)
assert np.allclose(expander(array), np.concatenate(expectations), rtol=0,\
    atol=1e-9)

file_name = 'test_modulation_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)
