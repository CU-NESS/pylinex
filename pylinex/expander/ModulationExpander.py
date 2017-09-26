"""
File: pylinex/expander/ModulationExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing an Expander which expands its
             inputs by multiplying it by a set of modulating factors.
"""
import numpy as np
import numpy.linalg as la
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
        value = np.array(value)
        if value.ndim >= 1:
            self._modulating_factors = value
        else:
            raise ValueError("modulating_factors must have nonzero shape.")
    
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
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        vector = np.array(vector)
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
        return np.power(inverse_squared_errors, -0.5)
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        return ((original_space_size == self.modulating_factors.shape[-1]) and\
           (expanded_space_size == self.modulating_factors.size))
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'ModulationExpander'
        group.create_dataset('modulating_factors',\
            data=self.modulating_factors)
    
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

