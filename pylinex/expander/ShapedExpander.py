"""
File: pylinex/expander/ShapedExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing an Expander which has arbitrarily shaped inputs
             and outputs.
"""
import numpy as np
from ..util import int_types
from .Expander import Expander

class ShapedExpander(Expander):
    """
    Expander subclass which has arbitrarily shaped inputs and outputs, whereas
    other Expander subclasses use flattened inputs and outputs.
    """
    def __init__(self, expander, input_shape, output_shape):
        """
        Initializes a new ShapedExpander with the given Expander object
        underlying it and the input and output shapes.
        
        expander: Expander object which expands the flattened inputs and
                  outputs
        input_shape, output_shape: tuples of ints whose products are compatible
                                   with given Expander object
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.expander = expander
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied ShapedExpander
        """
        expander = self.expander.copy()
        input_shape = tuple([dimension for dimension in self.input_shape])
        output_shape = tuple([dimension for dimension in self.output_shape])
        return ShapedExpander(expander, input_shape, output_shape)
    
    @property
    def input_shape(self):
        """
        Property storing tuple of ints describing shape of expected inputs.
        """
        if not hasattr(self, '_input_shape'):
            raise AttributeError("input_shape referenced before it was set.")
        return self._input_shape
    
    @input_shape.setter
    def input_shape(self, value):
        """
        Sets the input shape of this ShapedExpander.
        
        value: tuple of ints
        """
        value = np.array(value)
        if value.dtype in int_types:
            self._input_shape = tuple(value)
        else:
            raise TypeError("input_shape was set to a sequence of non-ints.")
    
    @property
    def input_size(self):
        """
        Property storing the product of all elements of the input shape.
        """
        if not hasattr(self, '_input_size'):
            self._input_size = np.prod(self.input_shape)
        return self._input_size
    
    @property
    def output_shape(self):
        """
        Property storing tuple of ints describing shape of expected outputs.
        """
        if not hasattr(self, '_output_shape'):
            raise AttributeError("output_shape referenced before it was set.")
        return self._output_shape
    
    @output_shape.setter
    def output_shape(self, value):
        """
        Sets the output shape of this ShapedExpander.
        
        value: tuple of ints
        """
        value = np.array(value)
        if value.dtype in int_types:
            self._output_shape = tuple(value)
        else:
            raise TypeError("output_shape was set to a sequence of non-ints.")
    
    @property
    def output_size(self):
        """
        Property storing the product of all elements of the output shape.
        """
        if not hasattr(self, '_output_size'):
            self._output_size = np.prod(self.output_shape)
        return self._output_size
    
    @property
    def expander(self):
        """
        Property storing the Expander object which expands flattened inputs and
        outputs.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Setter for the Expander object which expands flattened inputs and
        outputs.
        
        value: Expander object which expands the flattened inputs and outputs
        """
        if isinstance(value, Expander):
            if value.is_compatible(self.input_size, self.output_size):
                self._expander = value
            else:
                raise ValueError("expander not compatible with input and " +\
                                 "output sizes.")
        else:
            raise TypeError("expander was set to a non-Expander object.")
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space (or 2D array where axis just
                described is second one)
        
        returns: 1D vector from expanded space
        """
        return np.reshape(self.expander(vector), self.output_shape)
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        return np.reshape(self.expander.contract_error(error.flatten()),\
            self.input_shape)
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        reshaped_error = error.flatten()
        reshaped_data =\
            np.reshape(data, data.shape[:-len(self.input_shape)] + (-1,))
        return np.reshape(self.expander.invert(reshaped_data, reshaped_error),\
            self.input_shape)
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == self.input_size) and\
            (expanded_space_size == self.output_size))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        if expanded_space_size == self.output_size:
            return self.input_size
        else:
            raise ValueError(("expanded_space_size ({0}) was not equal to " +\
                "expected output size ({1}).").format(expanded_space_size,\
                self.output_size))
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        if original_space_size == self.input_size:
            return self.output_size
        else:
            raise ValueError(("original_space_size ({0}) was not equal to " +\
                "expected input size ({1}).").format(original_space_size,\
                self.input_size))
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return self.expander.channels_affected(original_space_size)
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'ShapedExpander'
        group.attrs['input_shape'] = self.input_shape
        group.attrs['output_shape'] = self.output_shape
        self.expander.fill_hdf5_group(group.create_group('expander'))
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, ShapedExpander):
            input_shapes_equal = (self.input_shape == other.input_shape)
            output_shapes_equal = (self.output_shape == other.output_shape)
            expanders_equal = (self.expander == other.expander)
            return\
                input_shapes_equal and output_shapes_equal and expanders_equal
        else:
            return False

