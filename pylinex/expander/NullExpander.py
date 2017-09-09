"""
File: pylinex/expander/NullExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an Expander which doesn't
             actually do anything to its inputs. It simply returns them back.
"""
from .Expander import Expander

class NullExpander(Expander):
    """
    Class representing an Expander which doesn't actually do anything to its
    inputs. It simply returns them back.
    """
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return vector
    
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

