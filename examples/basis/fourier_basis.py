"""
File: examples/basis/fourier_basis.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to create a simple Fourier basis. Since no
             error is given to the FourierBasis initializer, the basis
             functions should be simple sines and cosines (custom errors mix
             the different sines and cosines).
"""
import numpy as np
from pylinex import FourierBasis

x_values = np.linspace(-1, 1, 100)
basis = FourierBasis(100, 5)
basis.plot(x_values=x_values, show=True)
