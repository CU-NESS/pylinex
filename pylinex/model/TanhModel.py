"""
File: pylinex/model/TanhModel.py
Author: Keith Tauscher
Date: 7 Jul 2018

Description: File containing a class representing a model which is a simple
             tanh, with parameters ['amplitude', 'center', 'scale'].
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .ShiftedRescaledModel import ShiftedRescaledModel

class TanhModel(ShiftedRescaledModel):
    """
    Class representing a model which is a simple tanh, with parameters
    ['amplitude', 'center', 'scale'].
    """
    def __init__(self, x_values):
        """
        Initializes this TanhModel by providing the x_values at which it should
        be evaluated to produce the output.
        
        x_values: 1D numpy.ndarray of values at which to evaluate the tanh
        """
        self.x_values = x_values
    
    def base_function(self, scaled_x_values):
        """
        This function calculates the base (unscaled) tanh at the given
        pre-scaled x_values.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return np.tanh(scaled_x_values)
    
    def base_function_derivative(self, scaled_x_values):
        """
        Calculates the derivative of the base (unscaled) tanh.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return np.power(np.cosh(scaled_x_values), -2)
    
    def base_function_second_derivative(self, scaled_x_values):
        """
        Calculates the second derivative of the base (unscaled) tanh.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return (-2.) * np.tanh(scaled_x_values) *\
            np.power(np.cosh(scaled_x_values), -2)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'TanhModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this TanhModel.
        
        other: object to check for equality
        
        returns: False unless other is a TanhModel with the same x_values
        """
        return isinstance(other, TanhModel) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        return TanhModel(get_hdf5_value(group['x_values']))
    
