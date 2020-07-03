"""
File: pylinex/model/ExpandedModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class which represents a model which simply
             expands the output of another model using an Expander from the
             pylinex.expander module.
"""
import numpy as np
from ..expander import Expander
from .Model import Model

class ExpandedModel(Model):
    """
    Class which represents a model which simply expands the output of another
    model using an Expander from the pylinex.expander module.
    """
    def __init__(self, model, expander):
        """
        Creates an ExpandedModel with the given model and expander.
        
        model: Model object to build this model around
        expander: Expander object with which to expand the output of model
        """
        self.model = model
        self.expander = expander
    
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
    def expander(self):
        """
        Property storing the Expander object which expands the output of the
        core model to the output of this model.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Setter for the expander object which expands the output of the core
        model to the output of this model.
        
        value: must be an Expander object
        """
        if isinstance(value, Expander):
            self._expander = value
        else:
            raise TypeError("expander was not an Expander object.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model. These are the same as the parameters
        necessitated by the parameters of the core model.
        """
        return self.model.parameters
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the outputs of this model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels =\
                self.expander.expanded_space_size(self.model.num_channels)
        return self._num_channels
    
    def __call__(self, parameters):
        """
        Gets the expanded curve associated with the given parameters.
        
        returns: array of size (num_channels,)
        """
        return self.expander(self.model(parameters))
    
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
        return self.expander(self.model.gradient(parameters).T).T
    
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
        return self.expander(self.model.hessian(parameters).T).T
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information necessary to reload
        it at a later time.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ExpandedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        self.expander.fill_hdf5_group(group.create_group('expander'))
    
    def __eq__(self, other):
        """
        Checks if other is equivalent to this model.
        
        other: object to check for equality
        
        returns: False unless other is an ExpandedModel with the same core
                 model and expander.
        """
        if isinstance(other, ExpandedModel):
            return ((self.model == other.model) and\
                (self.expander == other.expander))
        else:
            return False
    
    def quick_fit(self, data, error, quick_fit_parameters=[], prior=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this expanded model.
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should a 1D array of same length as data
        quick_fit_parameters: quick fit parameters to pass to underlying model
        prior: either None or a GaussianDistribution object containing priors
               (in space of underlying model)
        
        returns: (parameter_mean, parameter_covariance) which are 1D and 2D
                 arrays respectively
        """
        if type(error) is type(None):
            error = np.ones_like(data)
        try:
            smaller_data = self.expander.invert(data, error)
        except:
            raise NotImplementedError("This ExpandedModel does not have a " +\
                "quick_fit function because the Expander it was made with " +\
                "does not implement the invert method.")
        smaller_error = self.expander.contract_error(error)
        return self.model.quick_fit(smaller_data, smaller_error,\
            quick_fit_parameters=quick_fit_parameters, prior=prior)
    
    @property
    def quick_fit_parameters(self):
        """
        Property storing the parameters necessary to call quick_fit.
        """
        if not hasattr(self, '_quick_fit_parameters'):
            self._quick_fit_parameters = self.model.quick_fit_parameters
        return self._quick_fit_parameters
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of he underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

