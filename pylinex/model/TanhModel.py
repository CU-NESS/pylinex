"""
File: pylinex/model/TanhModel.py
Author: Keith Tauscher
Date: 7 Jul 2018

Description: File containing a class representing a model which is a simple
             tanh, with parameters ['amplitude', 'center', 'scale'].
"""
from __future__ import division
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value, bool_types
from .ShiftedRescaledModel import ShiftedRescaledModel

class TanhModel(ShiftedRescaledModel):
    """
    Class representing a model which is a simple tanh, with parameters
    ['amplitude', 'center', 'scale'].
    """
    def __init__(self, x_values, zero_minimum=False):
        """
        Initializes this TanhModel by providing the x_values at which it should
        be evaluated to produce the output.
        
        x_values: 1D numpy.ndarray of values at which to evaluate the tanh
        zero_minimum: if True (default: False), the minimum value of this model
                      is 0 because the model is taken to be (1+tanh(x))/2
                      instead of simply tanh(x)
        """
        self.x_values = x_values
        self.zero_minimum = zero_minimum
    
    @property
    def zero_minimum(self):
        """
        Property storing a boolean indicating whether the minimum value of this
        model should be 0 (True) or -1 (False).
        """
        if not hasattr(self, '_zero_minimum'):
            raise AttributeError("zero_minimum was referenced before it " +\
                "was set.")
        return self._zero_minimum
    
    @zero_minimum.setter
    def zero_minimum(self, value):
        """
        Setter for the boolean indicating whether the minimum value of this
        model is 0 (True) or -1 (False).
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._zero_minimum = bool(value)
        else:
            raise TypeError("zero_minimum was set to a non-bool.")
    
    def base_function(self, scaled_x_values):
        """
        This function calculates the base (unscaled) tanh at the given
        pre-scaled x_values.
        
        scaled_x_values: array of pre-scaled x_values
        """
        if self.zero_minimum:
            return (1 + np.tanh(scaled_x_values)) / 2
        else:
            return np.tanh(scaled_x_values)
    
    def base_function_derivative(self, scaled_x_values):
        """
        Calculates the derivative of the base (unscaled) tanh.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return np.power(np.cosh(scaled_x_values), -2) /\
            (2 if self.zero_minimum else 1)
    
    def base_function_second_derivative(self, scaled_x_values):
        """
        Calculates the second derivative of the base (unscaled) tanh.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return (-1.) * (1 if self.zero_minimum else 2) *\
            np.tanh(scaled_x_values) * np.power(np.cosh(scaled_x_values), -2)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'TanhModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
        group.attrs['zero_minimum'] = self.zero_minimum
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this TanhModel.
        
        other: object to check for equality
        
        returns: False unless other is a TanhModel with the same x_values
        """
        return isinstance(other, TanhModel) and\
            (self.zero_minimum == other.zero_minimum) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        return TanhModel(get_hdf5_value(group['x_values']),\
            zero_minimum=group.attrs['zero_minimum'])
    
