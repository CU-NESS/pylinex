"""
File: pylinex/model/ConstantModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a simple Model which is a constant across
             channels. That constant is the sole parameter of the model.
"""
import numpy as np
from ..util import Savable, int_types
from .Model import Model

class ConstantModel(Model):
    """
    A simple Model which is a constant across channels. That constant is the
    sole parameter of the model.
    """
    def __init__(self, num_channels):
        """
        Initializes this ConstantModel by describing the number of channels it
        will output.
        
        num_channels: the integer number of channels to produce in the output
        """
        self.num_channels = num_channels
    
    @property
    def num_channels(self):
        """
        Property storing the integer number of channels to produce in the
        output.
        """
        if not hasattr(self, '_num_channels'):
            raise AttributeError("num_channels referenced before it was set.")
        return self._num_channels
        
    @num_channels.setter
    def num_channels(self, value):
        """
        Setter for the integer number of channels to produce in the output.
        
        value: must be a positive integer
        """
        if (type(value) in int_types) and (value > 0):
            self._num_channels = value
        else:
            raise TypeError("num_channels was set to a non-int.")
    
    @property
    def parameters(self):
        """
        The parameters list of this model is simply ['a'].
        """
        return ['a']
    
    def __call__(self, pars):
        """
        Evaluates this model at the given parameter values.
        
        pars: either a length-1 sequence of parameter values or a single
              numerical parameter value
        
        returns: the output of this model. This is the single parameter's value
                 concatenated num_channels times
        """
        try:
            return np.ones(self.num_channels) * pars[0]
        except:
            return np.ones(self.num_channels) * pars
    
    @property
    def gradient_computable(self):
        """
        Property storing that the gradient of this model is computable.
        """
        return True
    
    def gradient(self, pars):
        """
        Function which calculates the gradient of this model which is simply 1.
        
        pars: unused array of parameters
        
        returns: 1s in a numpy.ndarray of shape (num_channels, 1)
        """
        return np.ones((self.num_channels, 1))
    
    @property
    def hessian_computable(self):
        """
        Property storing that the hessian of this model is computable.
        """
        return True
    
    def hessian(self, pars):
        """
        Function which calculates the hessian of this model which is simply 0.
        
        pars: unused array of parameters
        
        returns: 0s in a numpy.ndarray of shape (num_channels, 1, 1)
        """
        return np.zeros((self.num_channels, 1, 1))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ConstantModel'
        group.attrs['num_channels'] = self.num_channels
    
    def __eq__(self, other):
        """
        Checks if other is the equivalent to this model.
        
        other: object to check for equality
        
        returns: False unless other is a ConstantModel with the same number of
                 channels as this model.
        """
        return isinstance(other, ConstantModel) and\
            (self.num_channels == other.num_channels)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ConstantModel from the given group. The load_from_hdf5_group of
        a given subclass model should always be called.
        
        group: the hdf5 file group from which to load the ConstantModel
        
        returns: a ConstantModel object
        """
        return ConstantModel(group.attrs['num_channels'])

