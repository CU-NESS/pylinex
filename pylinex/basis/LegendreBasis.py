"""
File: pylinex/basis/LegendreBasis.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a set of Legendre polynomial-
             like basis vectors.
"""
from .GramSchmidtBasis import orthonormal_polynomial_basis, GramSchmidtBasis

class LegendreBasis(GramSchmidtBasis):
    """
    Class representing a set of generalized Legendre basis vectors (orthogonal
    basis vectors made of polynomials).
    """
    def __init__(self, npoints, max_degree, error=None, expander=None):
        """
        Initializes a new LegendreBasis.
        
        npoints: the number of data points in the basis vectors
        max_degree: the number of nodes in the highest frequency mode
        error: the error with which to define 'orthogonal'. Default: None
        expander: if None, no expansion is applied
                  otherwise, expander must be an Expander object.
        """
        self.basis =\
            orthonormal_polynomial_basis(npoints, max_degree, error=error)
        self.expander = expander

