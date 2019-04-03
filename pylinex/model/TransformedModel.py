"""
File: pylinex/model/TransformedModel.py
Author: Keith Tauscher
Date: 22 Mar 2018

Description: File containing a class representing a model which is a simple
             transformation of another model.
"""
import numpy as np
from distpy import castable_to_transform, cast_to_transform
from .Model import Model

class TransformedModel(Model):
    """
    Class representing a model which is a simple transformation of another
    model.
    """
    def __init__(self, model, transform):
        """
        Initializes a TransformedModel based around the given underlying model
        and the transform which will affect it.
        
        model: a Model object
        transform: either a Transform object or something which can be cast to
                   a Transform object
        """
        self.model = model
        self.transform = transform
    
    @property
    def model(self):
        """
        Property storing the inner model (as a Model object) which is being
        transformed.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the inner model which is being transformed.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def transform(self):
        """
        Property storing the transform which changes the model underlying
        this one as a distpy Transform object.
        """
        if not hasattr(self, '_transform'):
            raise AttributeError("transform referenced before it was set.")
        return self._transform
    
    @transform.setter
    def transform(self, value):
        """
        Setter for the transform which will change the model underlying this
        one.
        
        value: either a Transform object or an object which can be cast to a
               Transform object (such as a string e.g. 'log10', 'exp', 'arcsin'
               or None).
        """
        if castable_to_transform(value):
            self._transform = cast_to_transform(value)
        else:
            raise TypeError("transform could not be successfully cast to a " +\
                "Transform object.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.model.parameters
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return self.transform(self.model(parameters))
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return self.model.gradient_computable
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        outer_derivative = self.transform.derivative(self.model(parameters))
        inner_gradient = self.model.gradient(parameters)
        return outer_derivative[:,np.newaxis] * inner_gradient
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return self.model.gradient_computable and self.model.hessian_computable
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        inner_value = self.model(parameters)
        inner_gradient = self.model.gradient(parameters)
        inner_hessian = self.model.hessian(parameters)
        outer_derivative = self.transform.derivative(inner_value)
        outer_second_derivative = self.transform.second_derivative(inner_value)
        return (outer_second_derivative[:,np.newaxis,np.newaxis] *\
            inner_gradient[:,:,np.newaxis] * inner_gradient[:,np.newaxis,:]) +\
            (outer_derivative[:,np.newaxis,np.newaxis] * inner_hessian)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'TransformedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        self.transform.fill_hdf5_group(group.create_group('transform'))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if isinstance(other, TransformedModel):
            return (self.model == other.model) and\
                (self.transform == other.transform)
        else:
            return False
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit to the given data.
        
        data: curve to fit with the model
        error: noise level in the data
        
        returns: (parameter_mean, parameter_covariance)
        """
        data_to_fit = self.transform.apply_inverse(data)
        if type(error) is type(None):
            error_to_fit = np.ones_like(data)
        else:
            error_to_fit =\
                error / np.abs(self.transform.derivative(data_to_fit))
        return self.model.quick_fit(data_to_fit, error=error_to_fit)
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of the underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

