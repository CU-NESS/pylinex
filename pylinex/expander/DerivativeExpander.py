"""
File: pylinex/expander/DerivativeExpander.py
Author: Keith Tauscher
Date: 12 Jun 2020

Description: File containing a class representing an Expander which performs a
             finite-difference derivative approximation on its inputs.
"""
import numpy as np
from ..util import bool_types, real_numerical_types, sequence_types
from .Expander import Expander

class DerivativeExpander(Expander):
    """
    Class representing an Expander which performs a finite-difference
    derivative approximation on its inputs.
    """
    def __init__(self, differences, interpolate=True):
        """
        Initializes a new DerivativeExpander by assuming the given differences
        indicate space between channel x-values.
        
        differences: either a single real number or an array of real numbers
                     with one fewer element the number of expected channels in
                     the array to differentiate
        """
        self.differences = differences
        self.interpolate = interpolate
    
    @property
    def differences(self):
        """
        Property storing the differences in x-value between the channels of
        array to be differentiated. This can either be a single real number or
        an array with one fewer element than the array to be differentiated.
        """
        if not hasattr(self, '_differences'):
            raise AttributeError("differences was referenced before it was " +\
                "set.")
        return self._differences
    
    @differences.setter
    def differences(self, value):
        """
        Setter for the differences to use when taking the finite-difference
        approximation to the derivative.
        
        value: either a single real number or an array of real numbers with one
               fewer element the number of expected channels in the array to
               differentiate
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._differences = 1. * value
            else:
                raise ValueError("differences was set to a non-positive " +\
                    "number.")
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                if np.all(value > 0):
                    self._differences = 1. * value
                else:
                    raise ValueError("Not all differences were positive.")
            else:
                raise ValueError("differences was set to a sequence with " +\
                    "multiple dimensions.")
        else:
            raise TypeError("differences was set to neither a single " +\
                "number nor a 1D sequence of numbers.")
    
    @property
    def single_number_differences(self):
        """
        Property storing a boolean describing whether differences is a single
        number or not.
        """
        if not hasattr(self, '_single_number_differences'):
            self._single_number_differences =\
                (type(self.differences) in real_numerical_types)
        return self._single_number_differences
    
    @property
    def interpolate(self):
        """
        Property storing boolean determining whether or not derivative is
        interpolated to same space as input or whether it is left in the space
        of midpoints between the original points.
        """
        if not hasattr(self, '_interpolate'):
            raise AttributeError("interpolate was referenced before it was " +\
                "set.")
        return self._interpolate
    
    @interpolate.setter
    def interpolate(self, value):
        """
        Setter for the interpolate property, which determines whether
        derivative is interpolated or not.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._interpolate = value
        else:
            raise TypeError("interpolate was set to a non-bool.")
    
    def make_expansion_matrix(self, original_space_size):
        """
        Computes the matrix of this expander.
        
        original_space_size: size of unexpanded space
        
        returns: expansion matrix of this expander
        """
        self.check_original_space_size(original_space_size)
        if self.interpolate:
            expansion_matrix =\
                np.zeros((original_space_size, original_space_size))
            if self.single_number_differences:
                expansion_matrix[0,0] = -2
                expansion_matrix[0,1] = 2
                if original_space_size > 2:
                    two_less_indices = np.arange(original_space_size - 2)
                    expansion_matrix[two_less_indices+1,two_less_indices] = -1
                    expansion_matrix[two_less_indices+1,two_less_indices+2] = 1
                expansion_matrix[-1,-2] = -2
                expansion_matrix[-1,-1] = 2
                expansion_matrix = expansion_matrix / (2 * self.differences)
            else:
                expansion_matrix[0,0] = -1 / self.differences[0]
                expansion_matrix[0,1] = 1 / self.differences[0]
                if original_space_size > 2:
                    difference_ratios =\
                        self.differences[:-1] / self.differences[1:]
                    squared_difference_ratios = difference_ratios ** 2
                    one_plus_difference_ratios = difference_ratios + 1
                    denominators =\
                        self.differences[:-1] * one_plus_difference_ratios
                    two_less_indices = np.arange(original_space_size - 2)
                    expansion_matrix[two_less_indices+1,two_less_indices+2] =\
                        squared_difference_ratios / denominators
                    expansion_matrix[two_less_indices+1,two_less_indices+1] =\
                        (1 - squared_difference_ratios) / denominators
                    expansion_matrix[two_less_indices+1,two_less_indices] =\
                        (-1) / denominators
                expansion_matrix[-1,-2] = -1 / self.differences[-1]
                expansion_matrix[-1,-1] = 1 / self.differences[-1]
        else:
            expansion_matrix =\
                np.zeros((original_space_size - 1, original_space_size))
            difference_indices = np.arange(original_space_size - 1)
            expansion_matrix[difference_indices,difference_indices] =\
                -1 / self.differences
            expansion_matrix[difference_indices,difference_indices+1] =\
                1 / self.differences
        return expansion_matrix
    
    def copy(self):
        """
        Finds and returns a deep copy of this expander.
        
        returns: another DerivativeExpander with copied differences
        """
        if self.single_number_differences:
            return DerivativeExpander(self.differences,\
                interpolate=self.interpolate)
        else:
            return DerivativeExpander(self.differences.copy(),\
                interpolate=self.interpolate)
    
    def apply(self, vector):
        """
        Expands vector from smaller original space to larger expanded space.
        
        vector: 1D vector from original space
        
        returns: 1D vector from expanded space
        """
        self.check_original_space_size(len(vector))
        midpoint_derivatives = (vector[1:] - vector[:-1]) / self.differences
        if self.interpolate:
            low_fraction = self.differences[1:] /\
                (self.differences[1:] + self.differences[:-1])
            interpolated_derivatives =\
                (low_fraction * midpoint_derivatives[:-1]) +\
                ((1 - low_fraction) * midpoint_derivatives[1:])
            return np.concatenate([[midpoint_derivatives[0]],\
                interpolated_derivatives, [midpoint_derivatives[-1]]])
        else:
            return midpoint_derivatives
    
    def is_compatible(self, original_space_size, expanded_space_size):
        """
        Checks whether this Expander is compatible with the given sizes of the
        original expanded spaces.
        
        original_space_size: size of (typically smaller) original space
        expanded_space_size: size of (typically larger) expanded space
        
        returns: True iff the given sizes are compatible with this Expander
        """
        try:
            self.check_original_space_size(original_space_size)
        except:
            return False
        else:
            return (original_space_size ==\
                expanded_space_size + (0 if self.interpolate else 1))
    
    def original_space_size(self, expanded_space_size):
        """
        Finds the input space size from the output space size.
        
        expanded_space_size: positive integer compatible with this Expander
        
        returns: input space size
        """
        original_space_size =\
            expanded_space_size + (0 if self.interpolate else 1)
        self.check_original_space_size(original_space_size)
        return original_space_size
    
    def check_original_space_size(self, original_space_size):
        """
        Function which checks whether this Expander is compatible with the
        given input size and throws an error if it is not.
        
        original_space_size: size to assume for input array
        """
        if (not self.single_number_differences) and\
            (original_space_size != len(self.differences) + 1):
            raise ValueError("The given space size was not compatible with " +\
                "the number of channel differences provided.")
    
    def expanded_space_size(self, original_space_size):
        """
        Finds the output space size from the input space size.
        
        original_space_size: positive integer compatible with this Expander
        
        returns: output space size
        """
        self.check_original_space_size(original_space_size)
        return original_space_size - (0 if self.interpolate else 1)
    
    def channels_affected(self, original_space_size):
        """
        Finds the indices of the data channels affected by data of the given
        size given to this Expander object.
        
        original_space_size: positive integer to assume as input size
        
        returns: 1D numpy.ndarray of indices of data channels possibly affected
                 by data expanded by this Expander object 
        """
        return np.arange(self.expanded_space_size(original_space_size))
    
    def contract_error(self, error):
        """
        Contracts error from full expanded space to smaller original space.
        This cannot be done for this expander for any error because the
        derivative is not full rank, so an error is thrown.
        """
        raise NotImplementedError("The DerivativeExpander cannot contract " +\
            "any error vector because the expansion matrix is not full rank.")
    
    def invert(self, data, error):
        """
        (Pseudo-)Inverts this expander in order to infer an original-space
        curve from the given expanded-space data and error. This cannot be done
        for this expander for any data and error because the derivative is not
        full rank, so an error is thrown.
        """
        raise NotImplementedError("The DerivativeExpander cannot be " +\
            "pseudo-inverted because the expansion matrix is not full rank.")
    
    def fill_hdf5_group(self, group):
        """
        Saves data about this in the given hdf5 file group.
        
        group: hdf5 file group to which to write
        """
        group.attrs['class'] = 'DerivativeExpander'
        group.attrs['differences'] = self.differences
        group.attrs['interpolate'] = self.interpolate
    
    def __eq__(self, other):
        """
        Checks for equality between this Expander and other. Returns True iff
        other is another DerivativeExpander with the same differences and
        interpolate properties.
        """
        if not isinstance(other, DerivativeExpander):
            return False
        if self.interpolate != other.interpolate:
            return False
        if self.single_number_differences != other.single_number_differences:
            return False
        if self.single_number_differences:
            return self.differences == other.differences
        elif len(self.differences) == len(other.differences):
            return np.all(self.differences == other.differences)
        else:
            return False

