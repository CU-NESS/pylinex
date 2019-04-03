"""
File: pylinex/model/ScaledModel.py
Author: Keith Tauscher
Date: 2 Aug 2018

Description: File containing a class which represents a model which simply
             scales the output of a different model.
"""
from ..util import real_numerical_types
from .Model import Model

class ScaledModel(Model):
    """
    Class which represents a model which simply scales the output of a
    different model.
    """
    def __init__(self, model, scale_factor):
        """
        Creates a ScaledModel with the given model and scale factor.
        
        model: Model object to build this model around
        scale_factor: the number by which to multiply outputs of model
        """
        self.model = model
        self.scale_factor = scale_factor
    
    @property
    def model(self):
        """
        Property storing the Model object at the core of this model.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the Model object at the core of this model.
        
        value: must be a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was not a Model object.")
    
    @property
    def scale_factor(self):
        """
        Property storing the scale factor by which all outputs of the model at
        the heart of this model will be multiplied.
        """
        if not hasattr(self, '_scale_factor'):
            raise AttributeError("scale_factor was referenced before it " +\
                "was set.")
        return self._scale_factor
    
    @scale_factor.setter
    def scale_factor(self, value):
        """
        Sets the scale_factor by which to multiply all outputs of the model at
        the core of this model.
        """
        if type(value) in real_numerical_types:
            self._scale_factor = value
        else:
            raise TypeError("scale_factor was set to a non-number.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model. These are the same as the parameters
        necessitated by the parameters of the core model.
        """
        return self.model.parameters
    
    def __call__(self, parameters):
        """
        Gets the scaled curve associated with the given parameters.
        
        returns: array of size (num_channels,)
        """
        return self.scale_factor * self.model(parameters)
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this model is computable. This
        is true as long as the gradient of the core model is computable.
        """
        return self.model.gradient_computable
    
    def gradient(self, parameters):
        """
        Function which computes the gradient of this model at the given
        parameters.
        
        parameters: numpy.ndarray of parameter values. shape: (num_parameters,)
        
        returns: numpy.ndarray of gradient values of this model of shape
                 (num_channels, num_parameters)
        """
        return self.scale_factor * self.model.gradient(parameters)
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this model is computable. This
        is true as long as the hessian of the core model is computable.
        """
        return self.model.hessian_computable
    
    def hessian(self, parameters):
        """
        Function which computes the hessian of this model at the given
        parameters.
        
        parameters: numpy.ndarray of parameter values. shape: (num_parameters,)
        
        returns: numpy.ndarray of hessian values of this model of shape
                 (num_channels, num_parameters, num_parameters)
        """
        return self.scale_factor * self.model.hessian(parameters)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information necessary to reload
        it at a later time.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ScaledModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        group.attrs['scale_factor'] = self.scale_factor
    
    def __eq__(self, other):
        """
        Checks if other is equivalent to this model.
        
        other: object to check for equality
        
        returns: False unless other is an ScaledModel with the same core model
                 and scale_factor.
        """
        if isinstance(other, ScaledModel):
            return ((self.model == other.model) and\
                (self.scale_factor == other.scale_factor))
        else:
            return False
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this scaled model.
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should a 1D array of same length as data
        
        returns: (parameter_mean, parameter_covariance) which are 1D and 2D
                 arrays respectively
        """
        if type(error) is type(None):
            error = np.ones_like(data)
        data_to_fit = data / self.scale_factor
        error_to_fit = error / self.scale_factor
        return self.model.quick_fit(data_to_fit, error=error_to_fit)
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of he underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

