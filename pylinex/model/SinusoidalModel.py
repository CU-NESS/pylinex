"""
File: plyinex/model/SinusoidalModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing class representing a model containing a single
             sinusoidal wave characterized by three parameters: amplitude,
             frequency, phase.
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .LoadableModel import LoadableModel

class SinusoidalModel(LoadableModel):
    """
    An abstract class representing any model.
    """
    def __init__(self, x_values):
        """
        Creates a new SinusoidalModel at the given x values.
        
        x_values: 1D array of independent variable values for the sine model
        """
        self.x_values = x_values
    
    @property
    def x_values(self):
        """
        Property storing the 1D numpy.ndarray of x_values at which the Gaussian
        of this model should be evaluated.
        """
        if not hasattr(self, '_x_values'):
            raise AttributeError("x_values referenced before it was set.")
        return self._x_values
    
    @x_values.setter
    def x_values(self, value):
        """
        Setter for the x_values at which the Gaussian of this model should be
        evaluated.
        
        value: must be a 1D numpy.ndarray of numbers
        """
        value = np.array(value)
        if value.ndim == 1:
            self._x_values = value
        else:
            raise ValueError("x_values was set to a non-array.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output which is the same
        as the number of x_values given.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.x_values)
        return self._num_channels
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ['amplitude', 'angular_frequency', 'phase']
        return self._parameters
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        (amplitude, angular_frequency, phase) = parameters
        angle = (angular_frequency * self.x_values) + phase
        return amplitude * np.sin(angle)
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return True
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        (amplitude, angular_frequency, phase) = parameters
        angle = (angular_frequency * self.x_values) + phase
        gradient = np.ndarray(self.num_channels, 3)
        gradient[:,0] = np.sin(angle)
        gradient[:,2] = amplitude * np.cos(angle)
        gradient[:,1] = (gradient[:,2] * self.x_values)
        return gradient
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return True
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        (amplitude, angular_frequency, phase) = parameters
        angle = (angular_frequency * self.x_values) + phase
        hessian = np.ndarray(self.num_channels, 3, 3)
        hessian[:,0,0] = 0
        hessian[:,0,2] = np.cos(angle)
        hessian[:,0,1] = hessian[:,0,2] * self.x_values
        hessian[:,2,2] = (-amplitude) * np.sin(angle)
        hessian[:,1,2] = hessian[:,2,2] * self.x_values
        hessian[:,1,1] = hessian[:,1,2] * self.x_values
        hessian[:,1,0] = hessian[:,0,1]
        hessian[:,2,0] = hessian[:,0,2]
        hessian[:,2,1] = hessian[:,1,2]
        return hessian
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'SinusoidalModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this SinusoidalModel.
        
        other: object to check for equality
        
        returns: False unless other is a SinusoidalModel with the same x_values
        """
        return isinstance(other, SinusoidalModel) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a SinusoidalModel
        """
        return SinusoidalModel(get_hdf5_value(group['x_values']))

