"""
File: pylinex/expander/MatrixExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an Expander which expands
             data using a matrix (typically one with more rows than columns).
             Most of the Expander subclasses in this module can be represented
             by this class. The drawback is that this class typically uses much
             more memory than other implementations.
"""
import numpy as np
import numpy.linalg as la
from ..util import create_hdf5_dataset
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
        if value.ndim > 1:
            self._matrix = np.reshape(value, (-1, value.shape[-1]))
        else:
            raise TypeError("matrix was set to an array of less than 2 " +\
                            "dimensions.")
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        if original_space_size != self.matrix.shape[-1]:
            raise ValueError("Given original_space_size is not the number " +\
                "of columns of the matrix this MatrixExpander is based upon.")
        return self.matrix
    
    def expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        if original_space_size != self.matrix.shape[-1]:
            raise ValueError("Given original_space_size is not the number " +\
                "of columns of the matrix this MatrixExpander is based upon.")
        return self.matrix
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied MatrixExpander
        """
        return MatrixExpander(self.matrix.copy())
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return np.dot(self.matrix, vector.T).T
    
    def covariance_from_error(self, error):
        """
        Gets covariance in smaller space from error in larger space.
        """
        flattened_matrix = np.reshape(self.matrix, (-1, self.matrix.shape[-1]))
        flattened_matrix = flattened_matrix / error[:,np.newaxis]
        inverse_covariance =\
            np.dot(np.conj(flattened_matrix).T, flattened_matrix)
        return la.inv(inverse_covariance)
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        return np.sqrt(np.real(np.diag(self.covariance_from_error(error))))
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        covariance = self.covariance_from_error(error)
        double_weighted_data = data / np.power(error, 2)
        return np.dot(covariance, np.dot(self.matrix.T, double_weighted_data))
    
    @property
    def input_size(self):
        """
        Property storing the length of the expected inputs.
        """
        if not hasattr(self, '_input_size'):
            self._input_size = self.matrix.shape[-1]
        return self._input_size
    
    @property
    def output_size(self):
        """
        Property storing the length of the expected outputs.
        """
        if not hasattr(self, '_output_size'):
            self._output_size = np.prod(self.matrix.shape[:-1])
        return self._output_size
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == self.input_size) and\
            (expanded_space_size == self.output_size))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        if expanded_space_size == self.output_size:
            return self.input_size
        else:
            raise ValueError(("expanded_space_size ({0}) was not equal to " +\
                "expected output size ({1}).").format(expanded_space_size,\
                self.output_size))
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        if original_space_size == self.input_size:
            return self.output_size
        else:
            raise ValueError(("original_space_size ({0}) was not equal to " +\
                "expected input size ({1}).").format(original_space_size,\
                self.input_size))
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        if original_space_size == self.input_size:
            return np.sum(np.abs(self.matrix), axis=-1).flatten().nonzero()[0]
        else:
            raise ValueError(("original_space_size ({0}) was not equal to " +\
                "expected input size ({1}).").format(original_space_size,\
                self.input_size))
    
    def fill_hdf5_group(self, group, matrix_link=None):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        matrix_link: if None, matrix is saved directly
                     otherwise, this must be an extant h5py.Dataset,
                     h5py.Group, or HDF5Link (from pylinex.util) object
        """
        group.attrs['class'] = 'MatrixExpander'
        create_hdf5_dataset(group, 'matrix', data=self.matrix,\
            link=matrix_link)
    
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

