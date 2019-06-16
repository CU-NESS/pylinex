"""
File: pylinex/model/BinnedModel.py
Author: Keith Tauscher
Date: 21 Sep 2018

Description: File containing a class representing a model which is a binned
             form of another model.
"""
import numpy as np
from ..util import sequence_types, RectangularBinner
from .Model import Model

class BinnedModel(Model):
    """
    Class representing a model which is a binned form of another model.
    """
    def __init__(self, model, binner, weights=None):
        """
        Initializes a TransformedModel based around the given underlying model
        and the binner which will Bin it.
        
        model: a Model object
        binner: a RectangularBinner object
        weights: weights to use in binning. Default: None
        """
        self.model = model
        self.binner = binner
        self.weights = weights
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output of this model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.binner.nbins_to_keep
        return self._num_channels
    
    @property
    def weights(self):
        """
        Property storing the weights to be used in binned. This is None by
        default but can be set to match data weights.
        """
        if not hasattr(self, '_weights'):
            raise AttributeError("weights was referenced before it was set.")
        return self._weights
    
    @weights.setter
    def weights(self, value):
        """
        Setter for the weights to used in binning.
        
        value: either None or a 1D array of same length as unbinned_x_values
               given to the RectangularBinner underlying this model
        """
        if type(value) is type(None):
            self._weights = value
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.binner.unbinned_x_values.shape:
                self._weights = value
            else:
                raise ValueError("weights was not of same shape as " +\
                    "unbinned_x_values of underlying RectangularBinner.")
        else:
            raise TypeError("weights was set to a non-array.")
    
    @property
    def x_values(self):
        """
        Property storing the x values (after binning) of this model.
        """
        return self.binner.binned_x_values
    
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
    def binner(self):
        """
        Property storing the Binner object which will bin the model.
        """
        if not hasattr(self, '_binner'):
            raise AttributeError("binner referenced before it was set.")
        return self._binner
    
    @binner.setter
    def binner(self, value):
        """
        Setter for the Binner object which will bin the model.
        
        value: a RectangularBinner object
        """
        if isinstance(value, RectangularBinner):
            self._binner = value
        else:
            raise TypeError("binner was set to something other than a " +\
                "RectangularBinner object.")
    
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
        return self.binner(self.model(parameters), weights=self.weights)
    
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
        transposed_gradient = self.model.gradient(parameters).T
        if type(self.weights) is type(None):
            return self.binner(transposed_gradient).T
        else:
            return self.binner(transposed_gradient,\
                weights=self.weights[np.newaxis,:]*\
                np.ones_like(transposed_gradient)).T
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return self.model.hessian_computable
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        transposed_hessian = self.model.hessian(parameters).T
        if type(self.weights) is type(None):
            return self.binner(transposed_hessian).T
        else:
            return self.binner(transposed_hessian,\
                weights=self.weights[np.newaxis,np.newaxis,:]*\
                np.ones_like(transposed_hessian)).T
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'BinnedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        self.binner.fill_hdf5_group(group.create_group('binner'))
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if isinstance(other, BinnedModel):
            return (self.model == other.model) and\
                (self.binner == other.binner)
        else:
            return False
    
    def quick_fit(self, data, error):
        """
        Performs a quick fit to the given data.
        
        data: curve to fit with the model
        error: noise level in the data
        
        returns: (parameter_mean, parameter_covariance)
        """
        raise NotImplementedError("quick_fit not implemented for " +\
            "BinnedModel objects.")
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of the underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

