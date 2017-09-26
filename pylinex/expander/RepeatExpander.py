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
            if value > 0:
                self._nrepeats = value
            else:
                raise ValueError("nrepeats must be positive.")
        else:
            raise TypeError("nrepeats was set to a non-integer.")
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        vector = np.array(vector)
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
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size * self.nrepeats) == expanded_space_size)
    
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

