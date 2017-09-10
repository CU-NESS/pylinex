"""
File: pylinex/basis/PolynomialBasis.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing basis set whose basis vectors
             are integer powers of a given set of x values.
"""
import numpy as np
from .Basis import Basis

class PolynomialBasis(Basis):
    """
    Class representing basis set whose basis vectors are integer powers of a
    given set of x values.
    """
    def __init__(self, xs, num_basis_vectors, expander=None):
        """
        Initializes a new PolynomialBasis with the given parameters.
        
        xs: x values at which to evaluate polynomials
        num_basis_vectors: number of basis vectors to include in this Basis
        expander: Expander object to use to expand basis vectors to full space
        """
        basis_vectors =\
            np.array([(xs ** i) for i in range(num_basis_vectors)])
        Basis.__init__(self, basis_vectors, expander=expander)

