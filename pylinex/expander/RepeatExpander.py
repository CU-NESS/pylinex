"""
File: pylinex/expander/RepeatExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an Expander which expands
             vectors by repeating them multiple times.
"""
import numpy as np
from ..util import int_types
from .Expander import Expander

class RepeatExpander(Expander):
    """
    Class representing an Expander which expands vectors by repeating them
    multiple times.
    """
    def __init__(self, nrepeats):
        """
        Initializes a new RepeatExpander with the given number of repeats
        
        nrepeats: positive integer number of times input is repeated
        """
        self.nrepeats = nrepeats
    
    @property
    def nrepeats(self):
        """
        Property storing the positive integer number of times this Expander
        should repeat its input.
        """
        if not hasattr(self, '_nrepeats'):
            raise AttributeError("nrepeats was referenced before it was set.")
        return self._nrepeats
    
    @nrepeats.setter
    def nrepeats(self, value):
        """
        Setter for the number of times this Expander should repeat its input.
        
        value: positive integer number of times this Expander should repeat its
               input
        """
        if type(value) in int_types:
            self._nrepeats = value
        else:
            raise TypeError("nrepeats was set to a non-integer.")
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied RepeatExpander
        """
        return RepeatExpander(self.nrepeats)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space, must be numpy.ndarray so as not
                                               to waste time casting it here
        
        returns: 1D vector from expanded space
        """
        return np.tile(vector, ((1,) * (vector.ndim - 1)) + (self.nrepeats,))
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        error_length = len(error)
        reshaped_error = np.reshape(error, (self.nrepeats, -1))
        return 1 / np.sqrt(np.sum(np.power(reshaped_error, -2), axis=0))
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        new_shape = (nrepeats, -1)
        reshaped_data = np.reshape(data, new_shape)
        reshaped_error = np.reshape(error, new_shape)
        squared_final_error = 1 / np.sum(np.power(reshaped_error, -2), axis=0)
        square_weighted_final_data =\
            np.sum(reshaped_data / np.power(reshaped_error, 2), axis=0)
        return squared_final_error * square_weighted_final_data
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size * self.nrepeats) == expanded_space_size)
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        if (expanded_space_size % self.nrepeats) == 0:
            return (expanded_space_size // self.nrepeats)
        else:
            raise ValueError("Given expanded_space_size was not compatible " +\
                "as it was not a factor of the number of repeats of this " +\
                "expander.")
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        return original_space_size * self.nrepeats
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return np.arange(self.expanded_space_size(original_space_size))
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'RepeatExpander'
        group.attrs['nrepeats'] = self.nrepeats
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, RepeatExpander):
            return (self.nrepeats == other.nrepeats)
        else:
            return False

