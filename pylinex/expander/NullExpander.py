"""
File: pylinex/expander/NullExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an Expander which doesn't
             actually do anything to its inputs. It simply returns them back.
"""
import numpy as np
from .Expander import Expander

class NullExpander(Expander):
    """
    Class representing an Expander which doesn't actually do anything to its
    inputs. It simply returns them back.
    """
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        return np.identity(original_space_size)
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: another NullExpander
        """
        return NullExpander()
    
    def overlap(self, vectors, error=None):
        """
        Computes Psi^T C^{-1} y for one or more vectors y and for a diagonal C
        defined by the given error.
        
        vectors: either a 1D array of length expanded_space_size or a 2D array
                 of shape (nvectors, expanded_space_size)
        error: the standard deviations of the independent noise defining the
               dot product
        
        returns: if vectors is 1D, result is a 1D array of length
                                   original_space_size
                 else, result is a 2D array of shape
                       (nvectors, original_space_size)
        """
        if type(error) is type(None):
            return vectors
        elif vectors.ndim == 1:
            return vectors / error
        else:
            return vectors / error[np.newaxis,:]
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return vector
    
    def contracted_covariance(self, error):
        """
        Finds the covariance matrix associated with contracted noise.
        
        error: 1D vector from expanded space
        
        returns: 2D array of shape (original_space_size, original_space_size)
        """
        return np.diag(error ** 2)
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        return error
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return (original_space_size == expanded_space_size)
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        return expanded_space_size
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        return original_space_size
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return np.arange(original_space_size)
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        return data
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'NullExpander'
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other. Returns True iff
        other is another NullExpander.
        """
        return isinstance(other, NullExpander)

