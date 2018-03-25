"""
File: examples/basis/fourier_basis.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to create a simple Fourier basis. Since no
             error is given to the FourierBasis initializer, the basis
             functions should be simple sines and cosines (custom errors mix
             the different sines and cosines).
"""
import os
import numpy as np
from pylinex import Basis, FourierBasis

x_values = np.linspace(-1, 1, 100)
basis = FourierBasis(100, 5)
hdf5_file_name = 'TEST_DELETE_THIS.hdf5'
basis.save(hdf5_file_name)
try:
    loaded_basis = Basis.load(hdf5_file_name)
    assert np.all(loaded_basis.basis == basis.basis)
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)
basis.plot(x_values=x_values, show=True)
