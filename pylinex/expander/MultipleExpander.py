"""
File: pylinex/expander/MultipleExpander.py
Author: Keith Tauscher
Date: 22 Jun 2020

Description: File containing class representing an Expander which expands its
             inputs by multiplying it by one or a set of multiplying factors.
"""
from __future__ import division
import numpy as np
from ..util import create_hdf5_dataset, sequence_types, real_numerical_types
from .Expander import Expander

class MultipleExpander(Expander):
    """
    Class representing an Expander which expands its inputs by multiplying it
    by one or a set of multiplying factors.
    """
    def __init__(self, multiplying_factors):
        """
        Initialized a new ModulationExpander with the new modulating factors.
        
        multiplying_factors: numpy.ndarray whose length is the ratio of the
                             expanded space size to the original space size.
        """
        self.multiplying_factors = multiplying_factors
    
    @property
    def multiplying_factors(self):
        """
        Property storing the factors by which inputs should be modulated.
        """
        if not hasattr(self, '_multiplying_factors'):
            raise AttributeError("multiplying_factors was referenced " +\
                                 "before it was set.")
        return self._multiplying_factors
    
    @multiplying_factors.setter
    def multiplying_factors(self, value):
        """
        Setter for the multiplying_factors array.
        
        value: numpy.ndarray whose length is the ratio of the expanded space
               size to the original space size.
        """
        if type(value) in real_numerical_types:
            self._multiplying_factors = np.array([value])
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._multiplying_factors = value
            else:
                raise ValueError("multiplying_factors was set to an array " +\
                    "that was not 1D.")
        else:
            raise TypeError("multiplying_factors was neither a real number " +\
                "or an array.")
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        square_components = []
        for factor in self.multiplying_factors:
            square_components.append(np.identity(original_space_size) * factor)
        return np.concatenate(square_components, axis=0)
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied MultipleExpander
        """
        return MultipleExpander(self.multiplying_factors.copy())
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        return (self.multiplying_factors[:,np.newaxis] *\
            vector[np.newaxis,:]).flatten()
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        try:
            reshaped_error = np.reshape(error, (self.num_factors, -1))
        except ValueError:
            raise ValueError("Length of error not a multiple of " +\
                "len(multiplying_factors).")
        weighted_inverse_error =\
            self.multiplying_factors[:,np.newaxis] / reshaped_error
        return np.power(np.sum((weighted_inverse_error) ** 2, axis=0), -0.5)
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        if len(data) != len(error):
            raise ValueError("data and error do not have the same length.")
        try:
            reshaped_data = np.reshape(data, (self.num_factors, -1))
            reshaped_error = np.reshape(error, (self.num_factors, -1))
        except ValueError:
            raise ValueError("Length of data and/or error not a multiple " +\
                "of len(multiplying_factors).")
        weighted_factors =\
            self.multiplying_factors[:,np.newaxis] / reshaped_error
        weighted_data = reshaped_data / reshaped_error
        return np.sum(weighted_data * weighted_factors, axis=0) /\
            np.sum(weighted_factors ** 2, axis=0)
    
    @property
    def num_factors(self):
        """
        Property storing the number of factors.
        """
        if not hasattr(self, '_num_factors'):
            self._num_factors = len(self.multiplying_factors)
        return self._num_factors
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return\
            (expanded_space_size == (original_space_size * self.num_factors))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        if (expanded_space_size % self.num_factors) == 0:
            return expanded_space_size // self.num_factors
        else:
            raise ValueError(("expanded_space_size ({0:d}) was not a " +\
                "multiple of the number of multiplying factors " +\
                "({1:d}).").format(expanded_space_size, self.num_factors))
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        return (original_space_size * self.num_factors)
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        channels_affected = []
        for (ifactor, factor) in enumerate(self.multiplying_factors):
            if factor != 0:
                channels_affected.append(\
                    np.arange(ifactor * original_space_size,\
                    (ifactor + 1) * original_space_size))
        return np.concatenate(channels_affected)
    
    def fill_hdf5_group(self, group, multiplying_factors_link=None):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        multiplying_factors_link: if None, multiplying_factors are saved
                                  directly otherwise, this should be an extant
                                 h5py.Group, h5py.Dataset, or HDF5Link (from
                                 pylinex.util) object
        """
        group.attrs['class'] = 'MultipleExpander'
        create_hdf5_dataset(group, 'multiplying_factors',\
            data=self.multiplying_factors, link=multiplying_factors_link)
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, ModulationExpander):
            if self.multiplying_factors.shape ==\
                other.multiplying_factors.shape:
                return np.allclose(self.multiplying_factors,\
                    other.multiplying_factors, atol=1e-8, rtol=0)
            else:
                return False
        else:
            return False

