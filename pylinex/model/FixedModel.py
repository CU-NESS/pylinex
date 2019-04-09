"""
File: pylinex/model/FixedModel.py
Author: Keith Tauscher
Date: 21 Jul 2018

Description: File containing a simple Model which always yields a fixed curve.
             The model has no parameters
"""
import numpy as np
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value
from .LoadableModel import LoadableModel

class FixedModel(LoadableModel):
    """
    A simple Model which has no parameters and always yields a fixed curve.
    """
    def __init__(self, fixed_curve):
        """
        Initializes this FixedModel with the curve it will always return.
        
        fixed_curve: the curve that this model should always return
        """
        self.fixed_curve = fixed_curve
    
    @property
    def fixed_curve(self):
        """
        Property storing the fixed curve this model will always return.
        """
        if not hasattr(self, '_fixed_curve'):
            raise AttributeError("fixed_curve was referenced before it was " +\
                "set.")
        return self._fixed_curve
    
    @fixed_curve.setter
    def fixed_curve(self, value):
        """
        Setter for the fixed curve this model will always return.
        
        value: 1D numpy.ndarray
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._fixed_curve = value
            else:
                raise ValueError("fixed_curve was set to an array with " +\
                    "many dimensions.")
        else:
            raise TypeError("fixed_curve was set to a non-sequence.")
    
    @property
    def num_channels(self):
        """
        Property storing the integer number of channels to produce in the
        output.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.fixed_curve)
        return self._num_channels
    
    @property
    def parameters(self):
        """
        There are no parameters of this model, so this returns an empty list.
        """
        return []
    
    def __call__(self, pars):
        """
        Evaluates this model at the given parameter values.
        
        pars: only here for this to follow the Model interface
        
        returns: the fixed curve given at initialization
        """
        return self.fixed_curve
    
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
        
        returns: 0s in a numpy.ndarray of shape (num_channels, 0)
        """
        return np.zeros((self.num_channels, 0))
    
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
        
        returns: 0s in a numpy.ndarray of shape (num_channels, 0, 0)
        """
        return np.zeros((self.num_channels, 0, 0))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'FixedModel'
        create_hdf5_dataset(group, 'fixed_curve', data=self.fixed_curve)
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this constant model. If the length is
              consistent with the expanded basis, then the expanded basis is
              used. Otherwise, the unexpanded basis is attempted to be used. If
              the length is inconsistent with both of these possibilities, then
              an error is raised
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        
        returns: (parameter_mean, parameter_covariance) where parameter_mean is
                 an array of shape (0,) and parameter_covariance is an array of
                 shape (0, 0)
        """
        return (np.zeros((0,)), np.zeros((0, 0)))
    
    def __eq__(self, other):
        """
        Checks if other is the equivalent to this model.
        
        other: object to check for equality
        
        returns: False unless other is a FixedModel with the same number of
                 channels as this model.
        """
        return (isinstance(other, FixedModel) and\
            np.allclose(self.fixed_curve, other.fixed_curve))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a FixedModel from the given group. The load_from_hdf5_group of
        a given subclass model should always be called.
        
        group: the hdf5 file group from which to load the FixedModel
        
        returns: a FixedModel object
        """
        return FixedModel(get_hdf5_value(group['fixed_curve']))

