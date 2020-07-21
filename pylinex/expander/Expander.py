"""
File: pylinex/expander/Expander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing an object which expands vectors
             from one space to another.
"""
import numpy as np
import numpy.linalg as la
from ..util import Savable

cannot_instantiate_expander_error =\
    NotImplementedError("Cannot directly instantiate Expander class.")

class Expander(Savable):
    """
    Class representing an object which expands vectors from one space to
    another.
    """
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander. This default method should be
        re-implemented by subclasses if possible because it is not very
        time-efficient.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        indices = np.arange(original_space_size)
        expansion_matrix = []
        for index in range(original_space_size):
            unit_vector = (indices == index).astype(int)
            expansion_matrix.append(self.apply(unit_vector))
        return np.array(expansion_matrix).T
    
    @property
    def cached_original_space_size(self):
        """
        Property storing the original space size corresponding to the cached
        matrix.
        """
        if not hasattr(self, '_cached_original_space_size'):
            self._cached_original_space_size = None
        return self._cached_original_space_size
    
    @property
    def cached_expansion_matrix(self):
        """
        Property storing a cached expansion matrix. This allows the matrix
        method to work quickly without recomputing the matrix each time is it
        called.
        """
        if not hasattr(self, '_cached_expansion_matrix'):
            self._cached_expansion_matrix = None
        return self._cached_expansion_matrix
    
    def expansion_matrix(self, original_space_size):
        """
        Gets expansion matrix of this expander.
        
        original_space_size: the size of the space before expansion
        
        returns: 2D matrix of shape (expanded_space_size, original_space_size)
        """
        if original_space_size != self.cached_original_space_size:
            expansion_matrix = self.make_expansion_matrix(original_space_size)
            self._cached_original_space_size = original_space_size
            self._cached_expansion_matrix = expansion_matrix
        return self.cached_expansion_matrix
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander. Must be implemented
        separately by each subclass of Expander.
        
        returns: copied Expander of same class as self
        """
        raise cannot_instantiate_expander_error
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        This method can be implemented by subclasses for more efficient
        application. If it is not, the matrix method will be used.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        original_space_size = vector.shape[-1]
        expansion_matrix = self.expansion_matrix(original_space_size)
        return np.dot(expansion_matrix, vector)
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        This method can be implemented by subclasses for more efficient
        application. If it is not, the matrix method will be used.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        expanded_space_size = len(error)
        original_space_size = self.original_space_size(expanded_space_size)
        expansion_matrix = self.expansion_matrix(original_space_size)
        expansion_matrix_over_error = expansion_matrix / error[:,np.newaxis]
        contracted_covariance = la.inv(np.dot(expansion_matrix_over_error.T,\
            expansion_matrix_over_error))
        return np.sqrt(np.diag(contracted_covariance))
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error. This method can be
        implemented by subclasses for more efficient application. If it is not,
        the matrix method will be used.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        if len(data) != len(error):
            raise ValueError("data and error did not have the same size.")
        expanded_space_size = len(data)
        original_space_size = self.original_space_size(expanded_space_size)
        expansion_matrix = self.expansion_matrix(original_space_size)
        expansion_matrix_over_error = expansion_matrix / error[:,np.newaxis]
        data_over_error = data / error
        contracted_covariance = la.inv(np.dot(expansion_matrix_over_error.T,\
            expansion_matrix_over_error))
        return np.dot(contracted_covariance,\
            np.dot(expansion_matrix_over_error.T, data_over_error))
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        raise cannot_instantiate_expander_error
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        raise cannot_instantiate_expander_error
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        raise cannot_instantiate_expander_error
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        raise cannot_instantiate_expander_error
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        raise cannot_instantiate_expander_error
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        raise cannot_instantiate_expander_error
    
    def __call__(self, vector):
        """
        Causes calls of this object to be equivalent to calls to the apply
        function of this Expander.
        
        vector: 1D vector from original space
        """
        return self.apply(vector)
    
    def __ne__(self, other):
        """
        Ensures that checks for inequality are consistent with checks for
        equality.
        
        other: object with which to check for inequality
        
        returns: False if this object and the other are identical,
                 True otherwise 
        """
        return (not self.__eq__(other))

