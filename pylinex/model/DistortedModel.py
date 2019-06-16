"""
File: pylinex/model/DistortedModel.py
Author: Keith Tauscher
Date: 3 Aug 2018

Description: File containing a class representing a model which is the same as
             a different model except with transformed inputs.
"""
import numpy as np
from distpy import TransformList
from .Model import Model

class DistortedModel(Model):
    """
    Class representing a model which is the same as a different model except
    with transformed inputs.
    """
    def __init__(self, model, transform_list):
        """
        Initializes a TransformedModel based around the given underlying model
        and the transform_list which will affect its inputs.
        
        model: a Model object
        transform_list: either a TransformList object or something which can be
                        cast to a TransformList object with the model's number
                        of parameters. The transform list describes the
                        transform which will be applied before each parameter
                        is passed to the underlying model
        """
        self.model = model
        self.transform_list = transform_list
    
    @property
    def model(self):
        """
        Property storing the inner model (as a Model object) which is being
        distorted.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the inner model which is being distorted.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def transform_list(self):
        """
        Property storing the transform_list with which the model underlying
        this one will be distorted. This transform_list is applied to
        parameters before they are passed to the underlying model.
        """
        if not hasattr(self, '_transform_list'):
            raise AttributeError("transform_list referenced before it was " +\
                "set.")
        return self._transform_list
    
    @transform_list.setter
    def transform_list(self, value):
        """
        Setter for the TransformList which will be applied to the parameters
        before they are passed to the underlying model.
        
        value: either a TransformList object or an object which can be cast to
               a TransformList object (such as None or a list of strings which
               can each be cast to a Transform object)
        """
        if TransformList.castable(value, num_transforms=self.num_parameters):
            self._transform_list =\
                TransformList.cast(value, num_transforms=self.num_parameters)
        else:
            raise TypeError("transform_list could not be successfully cast " +\
                "to a TransformList object.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.model.parameters
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in output
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.model.num_channels
        return self._num_channels
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return self.model(self.transform_list(parameters))
    
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
        transformed_gradient =\
            self.model.gradient(self.transform_list(parameters))
        return self.transform_list.detransform_gradient(transformed_gradient,\
            parameters, axis=-1)
    
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
        transformed_point = self.transform_list(parameters)
        transformed_gradient = self.model.gradient(transformed_point)
        transformed_hessian = self.model.hessian(transformed_point)
        return self.transform_list.detransform_hessian(transformed_hessian,\
            transformed_gradient, parameters, first_axis=-2)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'DistortedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        self.transform_list.fill_hdf5_group(\
            group.create_group('transform_list'))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if isinstance(other, DistortedModel):
            return (self.model == other.model) and\
                (self.transform_list == other.transform_list)
        else:
            return False
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model,
        determined by "untransforming" the bounds of the underlying model.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for (iname, name) in enumerate(self.parameters):
                transform = self.transform_list[iname]
                (lower_bound, upper_bound) = self.model.bounds[name]
                if type(lower_bound) is type(None):
                    lower_bound = -np.inf
                try:
                    lower_bound = transform.apply_inverse(lower_bound)
                except:
                      lower_bound = None
                else:
                    if not np.isfinite(lower_bound):
                        lower_bound = None
                if type(upper_bound) is type(None):
                    upper_bound = np.inf
                try:
                    upper_bound = transform.apply_inverse(upper_bound)
                except:
                    upper_bound = None
                else:
                    if not np.isfinite(upper_bound):
                        upper_bound = None
                self._bounds[name] = (lower_bound, upper_bound)
        return self._bounds
    
    def quick_fit(self, data, error, quick_fit_parameters=[], prior=None):
        """
        Performs a quick fit to the given data.
        
        data: curve to fit with the model
        error: noise level in the data
        quick_fit_parameters: quick fit parameters to pass to underlying model
        prior: either None or a GaussianDistribution object containing priors
               (in space of underlying model)
        
        returns: (parameter_mean, parameter_covariance)
        """
        (transformed_mean, transformed_covariance) =\
            self.model.quick_fit(data, error,\
            quick_fit_parameters=quick_fit_parameters, prior=prior)
        untransformed_mean =\
            self.transform_list.apply_inverse(transformed_mean)
        derivatives = self.transform_list.derivative(untransformed_mean)
        untransformed_covariance = transformed_covariance /\
            (derivatives[:,np.newaxis] * derivatives[np.newaxis,:])
        return (untransformed_mean, untransformed_covariance)
    
    @property
    def quick_fit_parameters(self):
        """
        Property storing the parameters necessary to call quick_fit.
        """
        if not hasattr(self, '_quick_fit_parameters'):
            self._quick_fit_parameters = self.model.quick_fit_parameters
        return self._quick_fit_parameters

