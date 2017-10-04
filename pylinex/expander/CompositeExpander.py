"""
File: pylinex/expander/CompositeExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an Expander which expands by
             successively applying a sequence of individual Expander objects.
"""
from ..util import sequence_types
from .Expander import Expander

class CompositeExpander(Expander):
    """
    Class representing an Expander which expands by successively applying a
    sequence of individual Expander objects.
    """
    def __init__(self, *expanders):
        """
        Initializes this CompositeExpander with the given sequence of Expander
        objects.
        
        expanders: sequence of 1 or more Expander objects which are mutually
                   incompatible.
        """
        self.expanders = expanders
    
    @property
    def expanders(self):
        """
        Property storing the sequence of Expander objects underlying this
        CompositeExpander.
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders was referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Setter for the sequence of Expander objects underlying this
        CompositeExpander.
        
        value: must be a sequence where every element is an Expander object
        """
        if type(value) in sequence_types:
            if all([isinstance(element, Expander) for element in value]):
                self._expanders = value
            else:
                raise TypeError("Not all elements of expanders sequence " +\
                                "were Expander objects.")
        else:
            raise TypeError("expanders was set to a non-sequence.")
    
    @property
    def num_expanders(self):
        """
        Property storing the length of the sequence of Expander objects
        underlying this CompositeExpander.
        """
        if not hasattr(self, '_num_expanders'):
            self._num_expanders = len(self.expanders)
        return self._num_expanders
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        current = vector
        for expander in self.expanders:
            current = expander(current)
        return current
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        current = error
        for expander in self.expanders[-1::-1]:
            current = expander.contract_error(current)
        return current
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        current_data = data
        current_error = error
        for expander in self.expanders[-1::-1]:
            current_data = expander.invert(current_data, current_error)
            current_error = expander.contract_error(current_error)
        return current_data
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return (expanded_space_size ==\
            self.expanded_space_size(original_space_size))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        current_size = expanded_space_size
        for expander in self.expanders[-1::-1]:
            current_size = expander.original_space_size(current_size)
        return current_size
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        current_size = original_space_size
        for expander in self.expanders:
            current_size = expander.expanded_space_size(current_size)
        return current_size
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        current_size = original_space_size
        for expander in self.expanders[:-1]:
            current_size = expander.expanded_space_size(current_size)
        return self.expanders[-1].channels_affected(current_size) # TODO reconsider
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'CompositeExpander'
        for (iexpander, expander) in enumerate(self.expanders):
            subgroup = group.create_group('expander_{}'.format(iexpander))
            expander.fill_hdf5_group(subgroup)
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, CompositeExpander):
            if self.num_expanders == other.num_expanders:
                for iexpander in range(self.num_expanders):
                    this_expander = self.expanders[iexpander]
                    other_expander = other.expanders[iexpander]
                    if this_expander != other_expander:
                        return False
                return True
            else:
                return False
        else:
            return False

