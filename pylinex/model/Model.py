"""
File: Model.py
Author: Keith Tauscher
Date: 12 Nov 2017

Description: File containing an abstract class representing a model.
"""
import numpy as np
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
    
    def curve_sample(self, distribution_set, ndraw):
        """
        Generates a curve sample from this model given a DistributionSet
        object.
        
        distribution_set: an instance of distpy's DistributionSet class which
                          describes a distribution for the parameters of this
                          model
        ndraw: positive integer number of curves to generate
        """
        draw = distribution_set.draw(ndraw)
        draw = np.array([draw[parameter] for parameter in self.parameters]).T
        return np.array([self(parameters) for parameters in draw])
    
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
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        
        returns: (parameter_mean, parameter_covariance) where parameter_mean is
                 a length N (number of parameters) 1D array and
                 parameter_covariance is a 2D array of shape (N,N). If no error
                 is given, parameter_covariance doesn't really mean anything
                 (especially if error is far from 1 in magnitude)
        """
        raise NotImplementedError("Either the Model subclass you are using " +\
            "does not support quick_fit or it hasn't yet been implemented.")
    
    def quick_residual(self, data, error=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        
        returns: 1D array of data's shape containing residual of quick fit
        """
        (parameter_mean, parameter_covariance) =\
            self.quick_fit(data, error=error)
        return data - self(parameter_mean)
    
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
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model. Since many models
        have no constraints, unless this property is overridden in a subclass,
        that subclass will give no bounds.
        """
        if not hasattr(self, '_bounds'):
            return {parameter: (None, None) for parameter in self.parameters}
        return self._bounds

