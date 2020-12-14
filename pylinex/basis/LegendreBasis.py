"""
File: pylinex/basis/LegendreBasis.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a set of Legendre polynomial-
             like basis vectors.
"""
from .Basis import Basis
from .GramSchmidtBasis import orthonormal_polynomial_basis, GramSchmidtBasis

class LegendreBasis(GramSchmidtBasis):
    """
    Class representing a set of generalized Legendre basis vectors (orthogonal
    basis vectors made of polynomials).
    """
    def __init__(self, npoints, max_degree, error=None, expander=None,\
        translation=None):
        """
        Initializes a new LegendreBasis.
        
        npoints: the number of data points in the basis vectors
        max_degree: the number of nodes in the highest frequency mode
        error: the error with which to define 'orthogonal'. Default: None
        expander: if None, no expansion is applied
                  otherwise, expander must be an Expander object.
        translation: if None, no constant additive is included in the results
                              of calling this Basis.
                     otherwise: should be a 1D numpy.ndarray of length npoints.
                                It is the (unexpanded) result of calling the
                                basis on a zero-vector of parameters
        """
        basis = orthonormal_polynomial_basis(npoints, max_degree, error=error)
        Basis.__init__(self, basis, expander=expander, translation=translation)

