"""
File: examples/basis/polynomial_basis.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to create a simple Polynomial basis.
"""
import numpy as np
from pylinex import PolynomialBasis

x_values = np.linspace(-1, 1, 100)
basis = PolynomialBasis(x_values, 10)
basis.plot(x_values=x_values, show=True)

