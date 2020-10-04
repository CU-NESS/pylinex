"""
File: pylinex/expander/ModulationExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing an Expander which expands its
             inputs by multiplying it by a set of modulating factors.
"""
import numpy as np
from ..util import create_hdf5_dataset, sequence_types
from .Expander import Expander

class ModulationExpander(Expander):
    """
    Class representing an Expander which expands its inputs by multiplying it
    by a set of modulating factors.
    """
    def __init__(self, modulating_factors):
        """
        Initialized a new ModulationExpander with the new modulating factors.
        
        modulating_factors: numpy.ndarray whose last axis represents the space
                            of the input. Any additional axes represent
                            multiple parts of the output which should be
                            modulated differently from the input.
        """
        self.modulating_factors = modulating_factors
    
    @property
    def modulating_factors(self):
        """
        Property storing the factors by which inputs should be modulated.
        """
        if not hasattr(self, '_modulating_factors'):
            raise AttributeError("modulating_factors was referenced before " +\
                                 "it was set.")
        return self._modulating_factors
    
    @modulating_factors.setter
    def modulating_factors(self, value):
        """
        Setter for the modulating_factors array.
        
        value: numpy.ndarray whose last axis represents the space of the input.
               Any additional axes represent multiple parts of the output which
               should be modulated differently from the input.
        """
        if type(value) in sequence_types:
            self._modulating_factors = np.array(value)
        else:
            raise TypeError("modulating_factors should be an array of at " +\
                "least one dimension whose last axis represents the space " +\
                "of the input.")
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        if original_space_size != self.input_size:
            raise ValueError("original_space_size was not the same size as " +\
                "input size.")
        modulating_factors =\
            np.reshape(self.modulating_factors, (-1, self.input_size))
        return np.concatenate([np.diag(modulating_factor)\
            for modulating_factor in modulating_factors], axis=0)
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: copied ModulationExpander
        """
        return ModulationExpander(self.modulating_factors.copy())
    
    @property
    def ndim(self):
        """
        Property storing the total dimension of the modulating factors array.
        """
        if not hasattr(self, '_ndim'):
            self._ndim = self.modulating_factors.ndim
        return self._ndim
    
    @property
    def expansion_slice(self):
        """
        Property storing the slice by which inputs should be expanded before
        being modulated by modulating_factors.
        """
        if not hasattr(self, '_expansion_slice'):
            self._expansion_slice =\
                ((np.newaxis,) * (self.ndim - 1)) + (slice(None),)
        return self._expansion_slice
    
    def contracted_covariance(self, error):
        """
        Finds the covariance matrix associated with contracted noise.
        
        error: 1D vector from expanded space
        
        returns: 2D array of shape (original_space_size, original_space_size)
        """
        return np.diag(self.contract_error(error) ** 2)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        extra_dims = vector.ndim - 1
        extra_dim_slices = ((slice(None),) * extra_dims)
        extra_dim_newaxes = ((np.newaxis,) * extra_dims)
        modulating_factors_expansion_slice = (extra_dim_newaxes +\
            ((slice(None),) * self.modulating_factors.ndim))
        full_expansion_slice = extra_dim_slices + self.expansion_slice
        return np.reshape(\
            self.modulating_factors[modulating_factors_expansion_slice] *\
            vector[full_expansion_slice], vector.shape[:-1] + (-1,))
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        
        error: 1D vector from expanded space
        
        returns: 1D vector from original space
        """
        reshaped_error = np.reshape(error, self.modulating_factors.shape)
        weighted_modulating_factors = self.modulating_factors / reshaped_error
        new_shape = (-1, self.modulating_factors.shape[-1])
        weighted_modulating_factors =\
            np.reshape(weighted_modulating_factors, new_shape)
        inverse_squared_errors = np.sum(np.conj(weighted_modulating_factors) *\
            weighted_modulating_factors, axis=0)
        return 1 / np.sqrt(inverse_squared_errors)
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error.
        
        data: data vector from which to imply an original space cause
        error: Gaussian noise level in data
        
        returns: most likely original-space curve to cause given data
        """
        if data.ndim != 1:
            raise ValueError("This function only supports 1D data.")
        reshaped_data = np.reshape(data, self.modulating_factors.shape)
        reshaped_error = np.reshape(error, self.modulating_factors.shape)
        weighted_modulating_factors = self.modulating_factors / reshaped_error
        weighted_data = reshaped_data / reshaped_error
        new_shape = (-1, weighted_modulating_factors.shape[-1])
        weighted_data = np.reshape(weighted_data, new_shape)
        weighted_modulating_factors =\
            np.reshape(weighted_modulating_factors, new_shape)
        return np.sum(weighted_modulating_factors * weighted_data, axis=0) /\
            np.sum(np.power(weighted_modulating_factors, 2), axis=0)
            
    
    @property
    def input_size(self):
        """
        Property storing the length of the expected inputs.
        """
        if not hasattr(self, '_input_size'):
            self._input_size = self.modulating_factors.shape[-1]
        return self._input_size
    
    @property
    def output_size(self):
        """
        Property storing the length of the expected outputs.
        """
        if not hasattr(self, '_output_size'):
            self._output_size = self.modulating_factors.size
        return self._output_size
    
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
        if original_space_size == self.input_size:
            return self.modulating_factors.flatten().nonzero()[0]
        else:
            raise ValueError(("original_space_size ({0}) was not equal to " +\
                "expected input size ({1}).").format(original_space_size,\
                self.input_size))
    
    def fill_hdf5_group(self, group, modulating_factors_link=None):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        modulating_factors_link: if None, modulating_factors are saved directly
                                 otherwise, this should be an extant
                                 h5py.Group, h5py.Dataset, or HDF5Link (from
                                 pylinex.util) object
        """
        group.attrs['class'] = 'ModulationExpander'
        create_hdf5_dataset(group, 'modulating_factors',\
            data=self.modulating_factors, link=modulating_factors_link)
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other.
        
        other: object with which to check for equality
        
        returns: True if this object and other are identical,
                 False otherwise
        """
        if isinstance(other, ModulationExpander):
            if self.modulating_factors.shape == other.modulating_factors.shape:
                return np.allclose(self.modulating_factors,\
                    other.modulating_factors, atol=1e-8, rtol=0)
            else:
                return False
        else:
            return False

