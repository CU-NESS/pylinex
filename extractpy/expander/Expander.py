"""
File: extractpy/expander/Expander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing an object which expands vectors
             from one space to another.
"""
from ..util import Savable

class Expander(Savable):
    """
    Class representing an object which expands vectors from one space to
    another.
    """
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        raise NotImplementedError("Cannot directly instantiate Expander " +\
                                  "class.")
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        raise NotImplementedError("Cannot directly instantiate Expander " +\
                                  "class.")
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        raise NotImplementedError("Cannot directly instantiate Expander " +\
                                  "class.")
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        raise NotImplementedError("Cannot directly instantiate Expander " +\
                                  "class.")
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        raise NotImplementedError("Cannot directly instantiate Expander " +\
                                  "class.")
    
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

