"""
File: pylinex/expander/PadExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing an Expander which expands the
             data by padding it with zeros (or any other value).
"""
import numpy as np
from ..util import int_types, numerical_types
from .Expander import Expander
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
class PadExpander(Expander):
    """
    Class representing an Expander which expands the data by padding it with
    zeros (or any other value).
    """
    def __init__(self, pads_before, pads_after, pad_value=0):
        """
        Initializes a new PadExpander.
        
        pads_before, pads_after: strings of the form N+S where N is a positive
                                 integer string and S is either '+' or '*'.
                                 If S is '*', then the number in N is taken to
                                 be the factor by which inputs should be
                                 expanded. If S is '+', then the number in N is
                                 taken to be the number of pads to put in the
                                 given position regardless of the input size.
        pad_value: numerical value to place in the pad positions
        """
        self.pads_before = pads_before
        self.pads_after = pads_after
        self.pad_value = pad_value
    
    def check_and_reprocess_pad_number(self, pad_number):
        """
        Checks the given pad_number string to see if it has the right format
        and reprocesses it.
        
        pad_number: string of the form N+S where N is a positive integer string
                    and S is either '+' or '*'. If S is '*', then the number in
                    N is taken to be the factor by which inputs should be
                    expanded. If S is '+', then the number in N is taken to be
                    the number of pads to put in the given position regardless
                    of the input size.
        
        returns: tuple of the form (number, is_multiplicative) representing
                 the number and symbol represented in the string.
        """
        if type(pad_number) in int_types:
            if pad_number >= 0:
                return (pad_number, False)
            else:
                raise ValueError("pad_number cannot be negative.")
        elif isinstance(pad_number, basestring):
            if pad_number[-1] in ['+', '*']:
                try:
                    int_part = int(pad_number[:-1])
                except:
                    raise ValueError("pad_number should be of the form X+Y " +\
                                     "where X is a string form of an " +\
                                     "integer and Y is either '+' or '*'.")
                else:
                    if int_part >= 0:
                        return (int_part, pad_number[-1] == '*')
                    else:
                        raise ValueError("integer part of pad number " +\
                                         "string must be non-negative.")
            else:
                raise ValueError("If pad_number is a string, it must end " +\
                                 "in '+' or '*'.")
        else:
            raise TypeError("pad_number was neither a string nor an integer.")
    
    @property
    def pads_before(self):
        """
        Property storing the number associated with the pads before the input,
        regardless of whether or not it is to be taken as multiplicative.
        """
        if not hasattr(self, '_pads_before'):
            raise AttributeError("pads_before was referenced before it was " +\
                                 "set.")
        return self._pads_before
    
    @property
    def pads_before_multiplicative(self):
        """
        Property storing whether or not the pads_before property is to be taken
        as multiplicative.
        """
        if not hasattr(self, '_pads_before_multiplicative'):
            raise AttributeError("pads_before_multiplicative was " +\
                                 "referenced before it was set.")
        return self._pads_before_multiplicative
    
    @pads_before.setter
    def pads_before(self, value):
        """
        Setter for the pads_before string.
        
        value: string of the form N+S where N is a positive integer string and
               S is either '+' or '*'. If S is '*', then the number in N is
               taken to be the factor by which inputs should be expanded. If S
               is '+', then the number in N is taken to be the number of pads
               to put in the given position regardless of the input size.
        """
        (self._pads_before, self._pads_before_multiplicative) =\
            self.check_and_reprocess_pad_number(value)
    
    @property
    def pads_after(self):
        """
        Property storing the number associated with the pads after the input,
        regardless of whether or not it is to be taken as multiplicative.
        """
        if not hasattr(self, '_pads_after'):
            raise AttributeError("pads_after was referenced before it was " +\
                                 "set.")
        return self._pads_after
    
    @property
    def pads_after_multiplicative(self):
        """
        Property storing whether or not the pads_after property is to be taken
        as multiplicative.
        """
        if not hasattr(self, '_pads_after_multiplicative'):
            raise AttributeError("pads_after_multiplicative was referenced " +\
                                 "before it was set.")
        return self._pads_after_multiplicative
    
    @pads_after.setter
    def pads_after(self, value):
        """
        Setter for the pads_after string.
        
        value: string of the form N+S where N is a positive integer string and
               S is either '+' or '*'. If S is '*', then the number in N is
               taken to be the factor by which inputs should be expanded. If S
               is '+', then the number in N is taken to be the number of pads
               to put in the given position regardless of the input size.
        """
        (self._pads_after, self._pads_after_multiplicative) =\
            self.check_and_reprocess_pad_number(value)
    
    @property
    def pad_value(self):
        """
        Property storing the value which will pad either side of the input.
        """
        if not hasattr(self, '_pad_value'):
            raise AttributeError("pad_value was referenced before it was set.")
        return self._pad_value
    
    @pad_value.setter
    def pad_value(self, value):
        """
        Setter for the value with which to fill bad positions
        
        value: single number with which to fill pad positions
        """
        if type(value) in numerical_types:
            self._pad_value = value
        else:
            raise TypeError("pad_value was set to a non-number.")
    
    def get_pad_size(self, original_space_size, pad_number, is_multiplicative):
        """
        Gets the pad size associated with the given input size as well as a
        number of pads and whether that number should be taken as
        multiplicative.
        
        original_space_size: the size of the input vector
        pad_number: number with which to find number of pad values
        is_multiplicative: bool determining whether pad_number is to be taken
                           as multiplicative
        
        returns: the actual number of pad values to put in the given position
        """
        if is_multiplicative:
            return pad_number * original_space_size
        else:
            return pad_number
    
    def get_pad_sizes_from_original_space_length(self, original_space_size):
        """
        Gets the sizes of the pads both before and after the input vector from
        the size of the input vector.
        
        original_space_size: length of input vector
        
        returns: tuple of form (number_of_pads_before, number_of_pads_after)
        """
        size_before = self.get_pad_size(original_space_size, self.pads_before,\
            self.pads_before_multiplicative)
        size_after = self.get_pad_size(original_space_size, self.pads_after,\
            self.pads_after_multiplicative)
        return (size_before, size_after)
    
    def get_pad_sizes_from_expanded_space_length(self, expanded_space_size):
        """
        Gets the sizes of the pads both before and after the input vector from
        the size of the output vector.
        
        expanded_space_size: length of output vector
        
        returns: tuple of form (number_of_pads_before, number_of_pads_after)
        """
        if self.pads_before_multiplicative and self.pads_after_multiplicative:
            original_space_size =\
                expanded_space_size // (1 + self.pads_before + self.pads_after)
            return (self.pads_before * original_space_size,\
                self.pads_after * original_space_size)
        elif self.pads_before_multiplicative:
            original_space_size = (expanded_space_size - self.pads_after) //\
                (1 + self.pads_before)
            return (self.pads_before * original_space_size, self.pads_after)
        elif self.pads_after_multiplicative:
            original_space_size = (expanded_space_size - self.pads_before) //\
                (1 + self.pads_after)
            return (self.pads_before, self.pads_after * original_space_size)
        else:
            return (self.pads_before, self.pads_after)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space by
        padding the vector with expander.pad_value.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        vector_length = vector.shape[-1]
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_original_space_length(vector_length)
        pad_before_shape = vector.shape[:-1] + (pad_size_before,)
        pad_after_shape = vector.shape[:-1] + (pad_size_after,)
        pad_array_before = np.ones(pad_before_shape) * self.pad_value
        pad_array_after = np.ones(pad_after_shape) * self.pad_value
        return np.concatenate([pad_array_before, vector, pad_array_after],\
            axis=-1)
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space
        simply by slicing.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        error_length = len(error)
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_expanded_space_length(error_length)
        return error[pad_size_before:error_length-pad_size_after]
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        num_channels = len(data)
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_expanded_space_length(num_channels)
        return data[pad_size_before:num_channels-pad_size_after]
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_original_space_length(original_space_size)
        expected_expanded_space_size =\
            (pad_size_before + original_space_size + pad_size_after)
        return (expected_expanded_space_size == expanded_space_size)
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_expanded_space_length(expanded_space_size)
        return (expanded_space_size - (pad_size_before + pad_size_after))
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_original_space_length(original_space_size)
        return (pad_size_before + original_space_size + pad_size_after)
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        (pad_size_before, pad_size_after) =\
            self.get_pad_sizes_from_original_space_length(original_space_size)
        return np.arange(original_space_size) + pad_size_before
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'PadExpander'
        if self.pads_before_multiplicative:
            group.attrs['pads_before'] = ('{}*'.format(self.pads_before))
        else:
            group.attrs['pads_before'] = ('{}+'.format(self.pads_before))
        if self.pads_after_multiplicative:
            group.attrs['pads_after'] = ('{}*'.format(self.pads_after))
        else:
            group.attrs['pads_after'] = ('{}+'.format(self.pads_after))
        group.attrs['pad_value'] = self.pad_value
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, PadExpander):
            if self.pads_before != other.pads_before:
                return False
            if self.pads_before_multiplicative !=\
                other.pads_before_multiplicative:
                return False
            if self.pads_after != other.pads_after:
                return False
            if self.pads_after_multiplicative !=\
                other.pads_after_multiplicative:
                return False
            return True
        else:
            return False

