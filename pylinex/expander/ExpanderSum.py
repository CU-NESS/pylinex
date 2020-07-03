"""
File: pylinex/expander/ExpanderSum.py
Author: Keith Tauscher
Date: 9 Jun 2020

Description: File containing a class representing the sum of multiple
             Expanders.
"""
import numpy as np
from ..util import sequence_types
from .Expander import Expander

class ExpanderSum(Expander):
    """
    Class representing the sum of multiple Expanders. invert and contract_error
    methods are intentionally left off so that the default ones defined in the
    Expander class are used. These default methods use the full expansion
    matrices in memory.
    """
    def __init__(self, *expanders):
        """
        Initializes a new ExpanderSum.
        
        expanders: Expander objects to add together.
        """
        self.expanders = expanders
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        return np.sum([expander.make_expansion_matrix(original_space_size)\
            for expander in self.expanders], axis=0)
    
    @property
    def expanders(self):
        """
        Property storing a sequence of Expander objects that are to be added
        together.
        
        returns: list of Expander objects
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders was referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Setter for the set of Expander objects that are to be added together.
        
        value: sequence of Expander objects
        """
        if type(value) in sequence_types:
            if len(value) > 0:
                if all([isinstance(element, Expander) for element in value]):
                    self._expanders = [element for element in value]
                else:
                    raise TypeError("At least one element of the expanders " +\
                        "sequence was not an Expander object.")
            else:
                raise ValueError("expanders sequence was empty, so this " +\
                    "object is degenerate.")
        else:
            raise TypeError("expanders was set to a non-sequence.")
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: another ExpanderSum
        """
        return ExpanderSum([expander.copy() for expander in self.expanders])
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return\
            np.sum([expander(vector) for expander in self.expanders], axis=0)
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return all([expander.is_compatible(original_space_size,\
            expanded_space_size) for expander in self.expanders])
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        original_space_sizes = [expander.original_space_size(\
            expanded_space_size) for expander in self.expanders]
        if all([(original_space_size == original_space_sizes[0])\
            for original_space_size in original_space_sizes]):
            return original_space_sizes[0]
        else:
            raise ValueError("The given expanded space size cannot be " +\
                "consistently created by all of the expanders in this " +\
                "Expander (i.e. different expanders would require " +\
                "different original space sizes to produce this expanded " +\
                "space size).")
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        expanded_space_sizes = [expander.expanded_space_size(\
            original_space_size) for expander in self.expanders]
        if all([(expanded_space_size == expanded_space_sizes[0])\
            for expanded_space_size in expanded_space_sizes]):
            return expanded_space_sizes[0]
        else:
            raise ValueError("The given original space size cannot be " +\
                "consistently created by all of the expanders in this " +\
                "Expander (i.e. different expanders would generate " +\
                "different expanded space sizes from this original space " +\
                "size).")
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        channels_affected = [expander.channels_affected(original_space_size)\
            for expander in self.expanders]
        return np.unique(np.concatenate(channels_affected))
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'ExpanderSum'
        for (iexpander, expander) in enumerate(self.expanders):
            subgroup = group.create_group('expander_{}'.format(iexpander))
            expander.fill_hdf5_group(subgroup)
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other. Returns True iff
        other is another ExpanderSum with the same expanders.
        """
        if not isinstance(other, ExpanderSum):
            return False
        if len(self.expanders) != len(other.expanders):
            return False
        others_accounted_for = []
        for sexpander in self.expanders:
            accounted_for = False
            for (ioexpander, oexpander) in enumerate(other.expanders):
                if ioexpander in others_accounted_for:
                    continue
                elif oexpander == sexpander:
                    accounted_for = True
                    others_accounted_for.append(ioexpander)
                    break
            if not accounted_for:
                return False
        return True
                

