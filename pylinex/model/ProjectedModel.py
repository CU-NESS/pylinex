"""
File: pylinex/model/ProjectedModel.py
Author: Keith Tauscher
Date: 13 Jul 2018

Description: File containing a class representing a model which is a simple
             basis decomposition of a submodel.
"""
import numpy as np
import numpy.linalg as la
from ..util import numerical_types, sequence_types, create_hdf5_dataset
from ..basis import Basis
from .Model import Model

class ProjectedModel(Model):
    """
    Class representing a model which is a simple basis decomposition of a
    submodel.
    """
    def __init__(self, model, basis, error=None):
        """
        Initializes a ProjectedModel based around the given underlying model
        and the basis and error which will project it.
        
        model: a Model object
        basis: a Basis object containing vectors onto which to project the
               output of the given model
        error: if None, all channels are considered equally important
        """
        self.model = model
        self.basis = basis
        self.error = error
    
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
    def basis(self):
        """
        Property storing the Basis object storing the vectors onto which to
        project the output of this model's model.
        """
        if not hasattr(self, '_basis'):
            raise AttributeError("basis was referenced before it was set.")
        return self._basis
    
    @basis.setter
    def basis(self, value):
        """
        Setter for the Basis object onto which to project each model output.
        
        value: a Basis object onto which each model output should be 
        """
        if isinstance(value, Basis):
            self._basis = value
        else:
            raise TypeError("basis was set to a non-Basis object.")
    
    @property
    def num_channels(self):
        """
        Property storing the integer number of channels.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.basis.num_larger_channel_set_indices
        return self._num_channels
    
    @property
    def error(self):
        """
        Property storing the 1D array of error values with which to do the
        projection.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error to use in projection.
        
        value: if None, all channels are equally important
               otherwise, a 1D array of length equal to number of channels
        """
        if type(value) is type(None):
            self._error = np.ones(self.num_channels)
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.num_channels,):
                self._error = value
            else:
                raise ValueError("error was set to an array of the wrong " +\
                    "shape. It should be 1D and with length equal to the " +\
                    "number of channels in the basis.")
        else:
            raise 
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.model.parameters
    
    @property
    def projection_matrix(self):
        """
        Property storing the matrix with which the model output is projected.
        """
        if not hasattr(self, '_projection_matrix'):
            normed_basis = self.basis.expanded_basis / self.error[np.newaxis,:]
            covariance = la.inv(np.dot(normed_basis, normed_basis.T))
            self._projection_matrix =\
                np.dot(covariance, normed_basis / self.error[np.newaxis,:])
        return self._projection_matrix
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return np.dot(self.projection_matrix, self.model(parameters))
    
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
        return np.dot(self.projection_matrix, self.model.gradient(parameters))
    
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
        hessian = self.model.hessian(parameters)
        hessian = np.reshape(hessian, (hessian.shape[0], -1))
        hessian = np.dot(self.projection_matrix, hessian)
        return np.reshape(hessian, (-1,) + (2 * (self.num_parameters,)))
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'ProjectedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        self.basis.fill_hdf5_group(group.create_group('basis'))
        create_hdf5_dataset(group, 'error', data=self.error)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if isinstance(other, ProjectedModel):
            return (self.model == other.model) and (self.basis == other.basis)\
                and np.allclose(self.error, other.error, rtol=1e-6)
        else:
            return False
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of he underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

