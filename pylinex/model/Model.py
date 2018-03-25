"""
File: Model.py
Author: Keith Tauscher
Date: 12 Nov 2017

Description: File containing an abstract class representing a model.
"""
from ..util import Savable

# an error indicating everything which should be implemented by subclass
shouldnt_instantiate_model_error = NotImplementedError("Model shouldn't be " +\
    "substantiated directly. Each subclass must implement its own __call__ " +\
    "function.")

class Model(Savable):
    """
    An abstract class representing any model.
    """
    def __init__(self, *args, **kwargs):
        """
        Since the Model class should not be directly instantiated, an error is
        thrown if its initializer is called.
        """
        raise shouldnt_instantiate_model_error
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters necessitated by this model.
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        raise shouldnt_instantiate_model_error
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        raise shouldnt_instantiate_model_error
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        raise shouldnt_instantiate_model_error
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        raise shouldnt_instantiate_model_error
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        raise shouldnt_instantiate_model_error
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        raise shouldnt_instantiate_model_error
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        raise shouldnt_instantiate_model_error
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        raise shouldnt_instantiate_model_error
    
    def __ne__(self, other):
        """
        Checks for inequality with other. This just returns the opposite of
        __eq__ so that (a!=b)==(!(a==b)) for all a and b.
        
        other: the object to check for inequality
        
        returns: False if other is equal to this model, True otherwise
        """
        return (not self.__eq__(other))

