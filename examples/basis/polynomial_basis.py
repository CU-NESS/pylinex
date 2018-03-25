"""
File: examples/basis/polynomial_basis.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to create a simple Polynomial basis.
"""
import os
import numpy as np
from pylinex import Basis, PolynomialBasis

x_values = np.linspace(-1, 1, 100)
basis = PolynomialBasis(x_values, 10)
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

