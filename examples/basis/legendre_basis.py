"""
File: examples/basis/legendre_basis.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to create a simple Legendre basis. Since no
             error is given to the FourierBasis initializer, the basis
             functions should be simple Legendre polynomials (custom errors mix
             the different Legendre polynomials).
"""
import numpy as np
from pylinex import LegendreBasis

x_values = np.linspace(-1, 1, 100)
basis = LegendreBasis(100, 5)
basis.plot(x_values=x_values, show=True)

