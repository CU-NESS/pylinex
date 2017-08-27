"""
File: extractpy/basis/GramSchmidtBasis.py
Author: Keith Tauscher
Date: 26 Aug 2017 2017

Description: File containing subclass of Basis which implements a set of basis
             vectors created through a Gram-Schmidt orthogonalization
             procedure.
"""
import numpy as np
from ..expander import NullExpander
from .Basis import Basis

def orthonormal_basis(seed_vectors, error=None):
    """
    Creates an orthonormal basis from the given seed vectors using a
    Gram-Schmidt orthogonalization procedure.
    
    seed_vectors: vectors with which to generate new modes using the
                  Gram-Schmidt orthogonalization procedure
    error: 1D numpy.ndarray containing error values with which to define
           'orthogonal'
    
    returns: a set of basis vectors. a numpy.ndarray of the same shape as the
             seed vectors
    """
    if error is None:
        error = np.ones_like(seed_vectors[0])
    inverse_error = 1 / error
    covariance = np.diag(error ** 2)
    inverse_covariance = np.diag(inverse_error ** 2)
    correction_matrix = inverse_covariance.copy()
    def dot_product(u, v):
        return np.dot(u * inverse_error, v * inverse_error)
    basis = []
    for degree in xrange(len(seed_vectors)):
        pbv = seed_vectors[degree]
        cpbv = np.dot(pbv, correction_matrix)
        basis.append(np.dot(covariance, cpbv) / np.sqrt(np.dot(pbv, cpbv)))
        CinvF = np.reshape(np.dot(inverse_covariance, basis[-1]), (-1, 1))
        correction_matrix -= np.dot(CinvF, CinvF.T)
    return np.array(basis)

def orthonormal_polynomial_basis(num_points, max_degree, error=None):
    """
    Creates a new Legendre polynomial-like basis with the given number of
    points and basis vectors.
    
    num_points: number of points to include in the basis vectors
    max_degree: one less than the number of basis vectors desired
    error: 1D numpy.ndarray with which to define 'orthogonal'
    
    returns: 2D numpy.ndarray of shape (md+1,np) where md is the max_degree
             given and np is the num_points given
    """
    xs = np.linspace(-1, 1, num_points)
    seed_vectors = [(xs ** degree) for degree in xrange(max_degree + 1)]
    return orthonormal_basis(seed_vectors, error=error)

def orthonormal_harmonic_basis(num_points, max_degree, error=None):
    """
    Creates a new Fourier series-like basis with the given number of
    points and basis vectors.
    
    num_points: number of points to include in the basis vectors
    max_degree: determines number of basis vectors returned
    error: 1D numpy.ndarray with which to define 'orthogonal'
    
    returns: 2D numpy.ndarray of shape (2*md+1,np) where md is the max_degree
             given and np is the num_points given
    """
    xs = np.linspace(-np.pi, np.pi, num_points)
    num_basis = (2 * max_degree) + 1
    def vector(index):
        is_sin = bool(index % 2)
        if is_sin:
            return np.sin(((index + 1) / 2) * xs)
        else:
            return np.cos(((index + 1) / 2) * xs)
    seed_vectors = np.array([vector(i) for i in xrange(num_basis)])
    return orthonormal_basis(seed_vectors, error=error)

class GramSchmidtBasis(Basis):
    """
    Class representing a Basis object whose vectors are calculated through a
    Gram-Schmidt orthogonalization procedure.
    """
    def __init__(self, seed_vectors, error=None, expander=None):
        """
        Initializes the new GramSchmidtBasis object.
        
        seed_vectors: the vectors with which to generate further modes through
                      the Gram-Schmidt orthogonalization procedure
        error: the error with which to define 'orthogonal'
        expander: if None, no expansion is applied
                  otherwise, expander must be an Expander object.
        """
        self.basis = orthonormal_basis(seed_vectors, error=error)
        self.expander = expander
    
    @property
    def even_subbasis(self):
        """
        Property storing the subbasis of this one which contains only the even
        functions (as long as the even-index seed vectors given to the
        initializer were even). Returns Basis object.
        """
        return self[::2]
    
    @property
    def odd_subbasis(self):
        """
        Property storing the subbasis of this one which contains only the odd
        functions (as long as the odd-index seed vectors given to the
        initializer were odd). Returns Basis object.
        """
        return self[1::2]

