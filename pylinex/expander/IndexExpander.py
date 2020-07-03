"""
File: pylinex/expander/IndexExpander.py
Author: Keith Tauscher
Date: 27 Mar 2019

Description: File containing class representing an Expander that interprets
             input as flattened version of an ND shape and expands it to the
             given output shape before reflattening.
"""
import numpy as np
from ..util import int_types, numerical_types, sequence_types,\
    create_hdf5_dataset
from .Expander import Expander

class IndexExpander(Expander):
    """
    Class representing an Expander that interprets input as flattened version
    of an ND shape and expands it to the given output shape before
    reflattening. make_expansion_matrix method is intentionally left off so
    that the default one defined in the Expander class is used. This default
    method computes the full expansion matrix iteratively.
    """
    def __init__(self, expanded_shape, axis, indices, modulating_factors=1,\
        pad_value=0):
        """
        Initializes a new IndexExpander.
        
        expanded_shape: the un-flattened shape of output vectors
        axis: integer axis index in expanded_shape which has been expanded here
        indices: sequence of indices corresponding to a dimension of length
                 expanded_shape[axis] onto which input values will be mapped in
                 the output of this expander
        modulating_factors: extra factor to apply to inputs while expanding.
                            This should be either a single number or a 1D array
                            of the same length as indices. Should be nonzero
        pad_value: numerical value to place in the pad positions
        """
        self.expanded_shape = expanded_shape
        self.axis = axis
        self.indices = indices
        self.modulating_factors = modulating_factors
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
        return Expander.make_expansion_matrix(self, original_space_size)
    
    @property
    def expanded_shape(self):
        """
        Property storing the shape whose product 
        """
        if not hasattr(self, '_expanded_shape'):
            raise AttributeError("expanded_shape was referenced before it " +\
                "was set.")
        return self._expanded_shape
    
    @expanded_shape.setter
    def expanded_shape(self, value):
        """
        Setter for the expanded_shape property.
        
        value: tuple of ints
        """
        if type(value) in sequence_types:
            if len(value) <= 31:
                if all([type(element) in int_types for element in value]):
                    self._expanded_shape =\
                        tuple([int(element) for element in value])
                else:
                    raise TypeError("Not all elements of expanded_shape " +\
                        "were integers.")
            else:
                raise ValueError("expanded_shape was longer than 31. So, " +\
                    "new_shape will not be able to source a numpy array.")
        else:
            raise TypeError("expanded_shape was set to a non-sequence.")
    
    @property
    def expanded_size(self):
        """
        Property storing the size of arrays expected as outputs of this
        expander.
        """
        if not hasattr(self, '_expanded_size'):
            self._expanded_size = np.prod(self.expanded_shape)
        return self._expanded_size
    
    @property
    def ndim(self):
        """
        Property storing the integer number of dimensions that the input/output
        consist of (when ignoring flattening before and after processing by
        this expander)
        """
        if not hasattr(self, '_ndim'):
            self._ndim = len(self.expanded_shape)
        return self._ndim
    
    @property
    def axis(self):
        """
        Property storing the integer index of the axis which is expanded here
        """
        if not hasattr(self, '_axis'):
            raise AttributeError("axis was referenced before it was set.")
        return self._axis
    
    @axis.setter
    def axis(self, value):
        """
        Setter for the axis which is expanded by this object
        
        value: integer which will be modded by self.ndim before setting the
               axis property, allowing for indexing Xth to last axis as -X
        """
        if type(value) in int_types:
            self._axis = (value % self.ndim)
        else:
            raise TypeError("axis was set to a non-int.")
    
    @property
    def indices(self):
        """
        Property storing the indices onto which inputs will be expanded into
        the axis (specified by property 'axis') of expanded_shape.
        """
        if not hasattr(self, '_indices'):
            raise AttributeError("indices was referenced before it was set.")
        return self._indices
    
    @indices.setter
    def indices(self, value):
        """
        Setter of the indices of expanded array into which to place input
        
        value: sequence of length corresponding to the value of
               input_shape[axis], which is usually smaller than
               expanded_shape[axis]. For every integer i from 0 (inclusive) to
               len(value) (exclusive), value[i] should be the index into
               expanded_shape[axis] where input[...,i,...] should be placed,
               where input is an input array to this expander
        """
        if type(value) in int_types:
            raise ValueError("Since indices was set to an int, it would be " +\
                "better to use the AxisExpander class than the " +\
                "IndexExpander class.")
        if type(value) in sequence_types:
            if all([(type(element) in int_types) for element in value]):
                value = np.mod(value, self.expanded_shape[self.axis])
                if len(value) == len(np.unique(value)):
                    self._indices = value
                else:
                    raise ValueError("At least one index of indices " +\
                        "appeared twice or more.")
            else:
                raise TypeError("Not all elements of the indices sequence " +\
                    "were integers.")
        else:
            raise TypeError("indices was set to a non-sequence.")
    
    @property
    def input_shape(self):
        """
        Property storing the shape as to interpret the input arrays of this
        expander.
        """
        if not hasattr(self, '_input_shape'):
            shape = [element for element in self.expanded_shape]
            shape[self.axis] = len(self.indices)
            self._input_shape = tuple(shape)
        return self._input_shape
    
    @property
    def input_size(self):
        """
        Property storing the size of arrays expected as inputs to this
        expander.
        """
        if not hasattr(self, '_input_size'):
            self._input_size = np.prod(self.input_shape)
        return self._input_size
    
    @property
    def output_slice(self):
        """
        Property storing the slice of the (un-flattened) output into which the
        (un-flattened) input is placed.
        """
        if not hasattr(self, '_output_slice'):
            self._output_slice =\
                (((slice(None),) * self.axis) + (self.indices,) +\
                ((slice(None),) * (self.ndim - self.axis - 1)))
        return self._output_slice
    
    @property
    def output_channels_affected(self):
        """
        Property storing the channels of the (re-flattened) output that are
        affected by the input.
        """
        if not hasattr(self, '_output_channels_affected'):
            is_affected = np.zeros(self.expanded_shape, dtype=bool)
            is_affected[self.output_slice] = True
            self._output_channels_affected = np.flatnonzero(is_affected)
        return self._output_channels_affected
    
    @property
    def modulating_factors(self):
        """
        Property storing the factors by which inputs are multiplied, indexed in
        the same way as the input is along the axis specified by the axis
        property.
        """
        if not hasattr(self, '_modulating_factors'):
            raise AttributeError("modulating_factors was referenced before " +\
                "it was set.")
        return self._modulating_factors
    
    @modulating_factors.setter
    def modulating_factors(self, value):
        """
        Setter for the factors by which inputs are multiplied in giving outputs
        
        value: if a single number, inputs are multiplied by this value
               if a sequence of numbers, determines how inputs are multiplied
                                         along the axis given by the axis
                                         property. Should be indexed as the
                                         input.
               No matter, the type, 0 should never be included
        """
        if type(value) in numerical_types:
            self._modulating_factors = np.ones(self.indices.shape) * value
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.indices.shape:
                if np.any(value == 0):
                    raise ValueError("At least one of modulating_factors " +\
                        "was zero.")
                else:
                    self._modulating_factors = value
            else:
                raise ValueError("The sequence modulating_factors given " +\
                    "was not of the same shape as the indices given.")
        else:
            raise TypeError("modulating_factors was set to neither a " +\
                "single number or a sequence of length len(self.indices).")
    
    @property
    def shaped_modulating_factors(self):
        """
        Property storing the modulating_factors array with extra dimensions of
        length 1 corresponding to non-expanded dimensions of the (un-flattened)
        input/output of this expander.
        """
        if not hasattr(self, '_shaped_modulating_factors'):
            view = (((np.newaxis,) * self.axis) + (slice(None),) +\
                ((np.newaxis,) * (self.ndim - self.axis - 1)))
            self._shaped_modulating_factors = self.modulating_factors[view]
        return self._shaped_modulating_factors
    
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
        
        returns: copied IndexExpander
        """
        expanded_shape = tuple([element for element in self.expanded_shape])
        axis = self.axis
        indices = self.indices.copy()
        modulating_factors = self.modulating_factors.copy()
        pad_value = self.pad_value
        return IndexExpander(expanded_shape, axis, indices,\
            modulating_factors=modulating_factors, pad_value=pad_value)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space by
        padding the vector with expander.pad_value.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        pre_shape = vector.shape[:-1]
        reshaped_input = np.reshape(vector, pre_shape + self.input_shape)
        output = np.ones(pre_shape + self.expanded_shape) * self.pad_value
        output[((slice(None),) * len(pre_shape)) + self.output_slice] =\
            (reshaped_input * self.shaped_modulating_factors)
        return np.reshape(output, pre_shape + (-1,))
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space
        simply by slicing.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        reshaped_error = np.reshape(error, self.expanded_shape)
        sliced_reshaped_error = reshaped_error[self.output_slice]
        input_space_error =\
            sliced_reshaped_error / self.shaped_modulating_factors
        return input_space_error.flatten()
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        reshaped_data = np.reshape(data, self.expanded_shape)
        sliced_reshaped_data = reshaped_data[self.output_slice]
        input_space_data =\
            sliced_reshaped_data / self.shaped_modulating_factors
        return input_space_data.flatten()
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == self.input_size) and\
            (expanded_space_size == self.expanded_size))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        return self.input_size
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        return self.expanded_size
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return self.output_channels_affected
    
    def fill_hdf5_group(self, group, indices_link=None,\
        modulating_factors_link=None):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        indices_link: if None, indices are saved directly
                      otherwise, this should be an extant h5py.Group,
                                 h5py.Dataset, or HDF5Link (from pylinex.util)
                                 object
        modulating_factors_link: if None, modulating_factors are saved directly
                                 otherwise, this should be an extant
                                 h5py.Group, h5py.Dataset, or HDF5Link (from
                                 pylinex.util) object
        """
        group.attrs['class'] = 'IndexExpander'
        group.attrs['expanded_shape'] = self.expanded_shape
        group.attrs['axis'] = self.axis
        create_hdf5_dataset(group, 'indices', data=self.indices,\
            link=indices_link)
        create_hdf5_dataset(group, 'modulating_factors',\
            data=self.modulating_factors, link=modulating_factors_link)
        group.attrs['pad_value'] = self.pad_value
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if not isinstance(other, IndexExpander):
            return False
        if self.expanded_shape != other.expanded_shape:
            return False
        if self.axis != other.axis:
            return False
        if np.any(self.indices != other.indices):
            return False
        if np.any(self.modulating_factors != other.modulating_factors):
            return False
        if self.pad_value != other.pad_value:
            return False
        return True

