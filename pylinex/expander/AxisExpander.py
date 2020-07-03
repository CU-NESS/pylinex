"""
File: pylinex/expander/AxisExpander.py
Author: Keith Tauscher
Date: 4 Feb 2019

Description: File containing class representing an Expander which expands the
             data by filling it with zeros (or any other value) in such a way
             that mimics an added axis.
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types
from .Expander import Expander

class AxisExpander(Expander):
    """
    Class representing an Expander which expands the data by filling it with
    zeros (or any other value) in such a way that mimics an added axis.
    """
    def __init__(self, old_shape, new_axis_position, new_axis_length, index,\
        pad_value=0):
        """
        Initializes a new AxisExpander.
        
        old_shape: tuple of ints whose product is equal to the unexpanded size
                   of the inputs to this expander
        new_axis_position: integer corresponding to the final position of the
                           axis added
        new_axis_length: the reciprocal of the fraction of the final array
                         which is filled with the unexpanded data given
        index: the index of the new axis where the given data is placed
        pad_value: numerical value to place in the pad positions
        """
        self.old_shape = old_shape
        self.new_axis_position = new_axis_position
        self.new_axis_length = new_axis_length
        self.index = index
        self.pad_value = pad_value
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        if pad_value != 0:
            raise ValueError("If pad_value is not zero, then Expander " +\
                "cannot be represented by an expansion matrix.")
        if original_space_size != np.prod(self.old_shape):
            raise ValueError("original_space_size is not compatible with " +\
                "the old_shape given at initialization.")
        old_indices = np.arange(original_space_size)
        expanded_old_indices = self.apply(old_indices + 1) - 1
        new_indices = np.unique(expanded_old_indices, return_index=True)[1][1:]
        expansion_matrix =\
            np.zeros((np.prod(self.new_shape), np.prod(self.old_shape)))
        expansion_matrix[new_indices,old_indices] = 1
        return expansion_matrix
    
    @property
    def old_shape(self):
        """
        Property storing the shape whose product 
        """
        if not hasattr(self, '_old_shape'):
            raise AttributeError("old_shape was referenced before it was set.")
        return self._old_shape
    
    @old_shape.setter
    def old_shape(self, value):
        """
        Setter for the old_shape property.
        
        value: tuple of ints
        """
        if type(value) in sequence_types:
            if len(value) <= 31:
                if all([type(element) in int_types for element in value]):
                    self._old_shape =\
                        tuple([int(element) for element in value])
                else:
                    raise TypeError("Not all elements of old_shape were " +\
                        "integers.")
            else:
                raise ValueError("old_shape was longer than 31. So, " +\
                    "new_shape will not be able to source a numpy array.")
        else:
            raise TypeError("old_shape was set to a non-sequence.")
    
    @property
    def new_axis_position(self):
        """
        Property storing the integer corresponding to the final position of the
        added axis.
        """
        if not hasattr(self, '_new_axis_position'):
            raise AttributeError("new_axis_position was referenced before " +\
                "it was set.")
        return self._new_axis_position
    
    @new_axis_position.setter
    def new_axis_position(self, value):
        """
        Setter for the new_axis_position property.
        
        value: non-negative integer less than or equal to the length of the
               old_shape property
        """
        if type(value) in int_types:
            self._new_axis_position = (value % (len(self.old_shape) + 1))
        else:
            raise TypeError("new_axis_position was set to a non-integer.")
    
    @property
    def new_axis_length(self):
        """
        Property storing the length of the new axis.
        """
        if not hasattr(self, '_new_axis_length'):
            raise AttributeError("new_axis_length was referenced before it " +\
                "was set.")
        return self._new_axis_length
    
    @new_axis_length.setter
    def new_axis_length(self, value):
        """
        Setter for the length of the added axis.
        
        value: integer greater than 1 (it could be 1, but that would make this
               expander an inefficient conceptual copy of the NullExpander)
        """
        if type(value) in int_types:
            if value > 0:
                self._new_axis_length = value
            else:
                raise ValueError("new_axis_length was set to a " +\
                    "non-positive integer.")
        else:
            raise TypeError("new_axis_length was set to a non-integer.")
    
    @property
    def new_shape(self):
        """
        Property storing the expanded (but not flattened) array.
        """
        if not hasattr(self, '_new_shape'):
            self._new_shape = self.old_shape[:self.new_axis_position] +\
                (self.new_axis_length,) +\
                self.old_shape[self.new_axis_position:]
        return self._new_shape
    
    @property
    def index(self):
        """
        Property storing the integer index of the unexpanded array in the
        expanded array.
        """
        if not hasattr(self, '_index'):
            raise AttributeError("index was referenced before it was set.")
        return self._index
    
    @index.setter
    def index(self, value):
        """
        Setter for the index of the unexpanded array in the expanded array.
        """
        if type(value) in int_types:
            if value >= 0:
                if value < self.new_axis_length:
                    self._index = value
                else:
                    raise ValueError("index was set to an integer")
            else:
                raise ValueError("index was set to a negative integer.")
        else:
            raise TypeError("index was set to a non-integer.")
    
    @property
    def expansion_slice(self):
        """
        Property storing the slice into the expanded array where the unexpanded 
        """
        if not hasattr(self, '_expansion_slice'):
            self._expansion_slice = ((slice(None),) *\
                self.new_axis_position) + (self.index,) + ((slice(None),) *\
                (len(self.old_shape) - self.new_axis_position))
        return self._expansion_slice
    
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
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied AxisExpander
        """
        return AxisExpander(self.old_shape, self.new_axis_position,\
            self.new_axis_length, self.index, pad_value=self.pad_value)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space by
        padding the vector with expander.pad_value.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        pre_vector_shape = vector.shape[:-1]
        array = (np.ones(pre_vector_shape + self.new_shape) * self.pad_value)
        expansion_slice =\
            ((slice(None),) * len(pre_vector_shape)) + self.expansion_slice
        array[expansion_slice] = np.reshape(vector,\
            pre_vector_shape + self.old_shape)
        return np.reshape(array, pre_vector_shape + (-1,))
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space
        simply by slicing.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        return\
            np.reshape(error, self.new_shape)[self.expansion_slice].flatten()
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        return np.reshape(data, self.new_shape)[self.expansion_slice].flatten()
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == np.prod(self.old_shape)) and\
            (expanded_space_size == np.prod(self.new_shape)))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        return np.prod(self.old_shape)
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        return np.prod(self.new_shape)
    
    @property
    def affected_channels(self):
        """
        Property storing the indices of the channels affected by this expander.
        """
        if not hasattr(self, '_affected_channels'):
            self._affected_channels = np.zeros(self.new_shape, dtype=bool)
            self._affected_channels[self.expansion_slice] = True
            self._affected_channels = self._affected_channels.flatten()
            self._affected_channels = np.where(self._affected_channels)[0]
        return self._affected_channels
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return self.affected_channels
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'AxisExpander'
        group.attrs['old_shape'] = self.old_shape
        group.attrs['new_axis_position'] = self.new_axis_position
        group.attrs['new_axis_length'] = self.new_axis_length
        group.attrs['index'] = self.index
        group.attrs['pad_value'] = self.pad_value
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if not isinstance(other, AxisExpander):
            return False
        if self.old_shape != other.old_shape:
            return False
        if self.new_axis_position != other.new_axis_position:
            return False
        if self.new_axis_length != other.new_axis_length:
            return False
        if self.index != other.index:
            return False
        return (self.pad_value == other.pad_value)

