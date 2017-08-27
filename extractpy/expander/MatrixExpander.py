"""
File: extractpy/expander/MatrixExpander.py
Author: Keith Tauscher
Date: 26 Aug 2017

Description: File containing a class representing an Expander which expands
             data using a matrix (typically one with more rows than columns).
             Most of the Expander subclasses in this module can be represented
             by this class. The drawback is that this class typically uses much
             more memory than other implementations.
"""
import numpy as np
import numpy.linalg as la
from .Expander import Expander

class MatrixExpander(Expander):
    """
    Class representing an Expander which expands data using a matrix (typically
    one with more rows than columns). Most of the Expander subclasses in this
    module can be represented by this class. The drawback is that this class
    typically uses much more memory than other implementations.
    """
    def __init__(self, matrix):
        """
        Initializes this MatrixExpander with the given matrix.
        
        matrix: a numpy.ndarray of at least 2 dimensions. The last dimension
                represents the size of the input space. Any extra dimensions
                will be effectively flattened into one. They are left for
                organizational purposes.
        """
        self.matrix = matrix
    
    @property
    def matrix(self):
        """
        Property storing the matrix at the heart of this MatrixExpander. It is
        a numpy.ndarray of at least 2 dimensions. The last dimension represents
        the size of the input space. Any extra dimensions will be effectively
        flattened into one. They are left for organizational purposes.
        """
        if not hasattr(self, '_matrix'):
            raise AttributeError("matrix was referenced before it was set.")
        return self._matrix
    
    @matrix.setter
    def matrix(self, value):
        """
        Setter for the matrix at the heart of this MatrixExpander.
        
        value: a numpy.ndarray of at least 2 dimensions. The last dimension
               represents the size of the input space. Any extra dimensions
               will be effectively flattened into one. They are left for
               organizational purposes.
        """
        value = np.array(value)
        if value.ndim > 1:
            self._matrix = value
        else:
            raise TypeError("matrix was set to an array of less than 2 " +\
                            "dimensions.")
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return np.dot(self.matrix, vector).flatten()
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        flattened_matrix = np.reshape(self.matrix, (-1, self.matrix.shape[-1]))
        flattened_matrix = flattened_matrix / error[:,np.newaxis]
        inverse_covariance =\
            np.dot(np.conj(flattened_matrix).T, flattened_matrix)
        covariance = la.inv(inverse_covariance)
        variances = np.real(np.diag(covariance))
        return np.sqrt(variances)
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == self.matrix.shape[-1]) and\
            (expanded_space_size == np.prod(self.matrix.shape[:-1])))
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'MatrixExpander'
        group.create_dataset('matrix', data=self.matrix)
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, MatrixExpander):
            if self.matrix.shape == other.matrix.shape:
                return\
                    np.allclose(self.matrix, other.matrix, rtol=0, atol=1e-9)
            else:
                return False
        else:
            return False

