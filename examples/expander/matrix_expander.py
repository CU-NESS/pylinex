"""
File: examples/expander/matrix_expander.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use a MatrixExpander object to perform
             Fourier transforms of data.
"""
from __future__ import division
import os
import numpy as np
import numpy.random as rand
from pylinex import MatrixExpander, load_expander_from_hdf5_file

big_int = 100
ints = np.arange(big_int)
matrix = (1 / np.sqrt(big_int)) *\
    np.exp(-1j * ints[:,np.newaxis] * ints[np.newaxis,:] * 2 * np.pi / big_int)
expander = MatrixExpander(matrix)

error = np.ones(big_int)
data = rand.rand(big_int)
fft_data = np.fft.fft(data, norm='ortho')

assert np.allclose(expander(data), fft_data, rtol=0, atol=1e-9)
assert np.allclose(expander.contract_error(error), error, rtol=0, atol=1e-9)
assert np.allclose(expander(expander(data).conj()), data, rtol=0, atol=1e-9)

file_name = 'test_matrix_expander_TEMP.hdf5'
expander.save(file_name)
try:
    assert expander == load_expander_from_hdf5_file(file_name)
except:
    os.remove(file_name)
    raise
os.remove(file_name)

