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
    def __init__(self, xs, num_basis_vectors, expander=None, translation=None):
        """
        Initializes a new PolynomialBasis with the given parameters.
        
        xs: x values at which to evaluate polynomials
        num_basis_vectors: number of basis vectors to include in this Basis
        expander: Expander object to use to expand basis vectors to full space
        translation: if None, no constant additive is included in the results
                              of calling this Basis.
                     otherwise: should be a 1D numpy.ndarray of the same length
                                as xs. It is the (unexpanded) result of calling
                                the basis on a zero-vector of parameters
        """
        basis = np.array([(xs ** i) for i in range(num_basis_vectors)])
        Basis.__init__(self, basis, expander=expander, translation=translation)

