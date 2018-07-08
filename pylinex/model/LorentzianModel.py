"""
File: pylinex/model/LorentzianModel.py
Author: Keith Tauscher
Date: 7 Jul 2018

Description: File containing a class representing a model which is a simple
             Lorentzian, with parameters ['amplitude', 'center', 'scale'].
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .ShiftedRescaledModel import ShiftedRescaledModel

class LorentzianModel(ShiftedRescaledModel):
    """
    Class representing a model which is a simple Lorentzian, with parameters
    ['amplitude', 'center', 'scale'].
    """
    def __init__(self, x_values):
        """
        Initializes this LorentzianModel by providing the x_values at which it
        should be evaluated to produce the output.
        
        x_values: 1D numpy.ndarray of values at which to evaluate Lorentzian
        """
        self.x_values = x_values
    
    def base_function(self, scaled_x_values):
        """
        This function calculates the base (standard) Lorentzian at the given
        pre-scaled x_values.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return (1 / (1 + (scaled_x_values ** 2)))
    
    def base_function_derivative(self, scaled_x_values):
        """
        Calculates the derivative of the base (standard) Lorentzian.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return (((-2) * scaled_x_values) / ((1 + (scaled_x_values ** 2)) ** 2))
    
    def base_function_second_derivative(self, scaled_x_values):
        """
        Calculates the second derivative of the base (standard) Lorentzian.
        
        scaled_x_values: array of pre-scaled x_values
        """
        return (((6 * (scaled_x_values ** 2)) - 2) /\
            ((1 + (scaled_x_values ** 2)) ** 3))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'LorentzianModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this LorentzianModel.
        
        other: object to check for equality
        
        returns: False unless other is a LorentzianModel with the same x_values
        """
        return isinstance(other, LorentzianModel) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        return LorentzianModel(get_hdf5_value(group['x_values']))

