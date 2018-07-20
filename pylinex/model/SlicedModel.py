"""
File: pylinex/model/SlicedModel.py
Author: Keith Tauscher
Date: 9 Apr 2018

Description: File containing a class represening a model which has one or more
             of its parameters fixed.
"""
import numpy as np
from .Model import Model

class SlicedModel(Model):
    """
    Class represening a model which has one or more of its parameters fixed.
    """
    def __init__(self, model, **constant_parameters):
        """
        Since the Model class should not be directly instantiated, an error is
        thrown if its initializer is called.
        """
        self.model = model
        self.constant_parameters = constant_parameters
    
    @property
    def model(self):
        """
        Property storing the model which is sliced by this SlicedModel.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the model which is sliced by this SlicedModel.
        
        value: a Model object to slice
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def constant_parameters(self):
        """
        Property storing a dictionary of parameters which are held constant.
        """
        if not hasattr(self, '_constant_parameters'):
            raise AttributeError("constant_parameters was referenced " +\
                "before it was set.")
        return self._constant_parameters
    
    @constant_parameters.setter
    def constant_parameters(self, value):
        """
        Setter for the dictionary of parameters which are held constant.
        
        value: dictionary where every key is a 
        """
        if isinstance(value, dict):
            if all([(key in self.model.parameters) for key in value]):
                if len(value) == 0:
                    raise ValueError("No constant_parameters given to " +\
                        "SlicedModel, which means that the SlicedModel " +\
                        "would be essentially identical to the underlying " +\
                        "model, making it unnecessary.")
                else:
                    self._constant_parameters = value
                    (indices, values) = ([], [])
                    for key in value:
                        indices.append(self.model.parameters.index(key))
                        values.append(value[key])
                    (indices, values) = (np.array(indices), np.array(values))
                    argsort = np.argsort(indices)
                    (indices, values) = (indices[argsort], values[argsort])
                    self._indices_of_parameters = np.array([index\
                        for index in range(self.model.num_parameters)\
                        if index not in indices])
                    parameter_template =\
                        np.ndarray((self.model.num_parameters,))
                    if self.num_parameters != 0:
                        parameter_template[self.indices_of_parameters] = np.nan
                    parameter_template[indices] = values
                    self._parameter_template = parameter_template
            else:
                raise ValueError("Not all keys of constant_parameters were " +\
                    "parameters of the underlying model.")
        else:
            raise TypeError("constant_parameters was set to a non-dictionary.")
    
    @property
    def indices_of_parameters(self):
        """
        Property storing the indices of the parameters of the underlying model
        which are not held constant.
        """
        if not hasattr(self, '_indices_of_parameters'):
            raise AttributeError("indices_of_parameters was referenced " +\
                "before constant_parameters was set.")
        return self._indices_of_parameters
    
    @property
    def parameter_template(self):
        """
        Property storing the values of the parameters of the underlying model
        to replace (which correspond to the complement of indices_of_parameters
        property).
        """
        if not hasattr(self, '_parameter_template'):
            raise AttributeError("parameter_template was referenced before " +\
                "constant_parameters was set.")
        return self._parameter_template
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = [self.model.parameters[index]\
                for index in self.indices_of_parameters]
        return self._parameters
    
    def form_parameters(self, parameters):
        """
        Forms the parameter array to pass to the underlying model from the one
        to passed to this model.
        
        parameters: 1D array of length self.num_parameters
        
        returns: 1D array of length self.model.num_parameters
        """
        if self.num_parameters == 0:
            return self.parameter_template
        else:
            filled_template = self.parameter_template.copy()
            filled_template[self.indices_of_parameters] = parameters
            return filled_template
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return self.model(self.form_parameters(parameters))
    
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
        gradient = self.model.gradient(self.form_parameters(parameters))
        if self.num_parameters == 0:
            return np.zeros((gradient.shape[0], 0))
        else:
            return gradient[:,self.indices_of_parameters]
    
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
        hessian = self.model.hessian(self.form_parameters(parameters))
        if self.num_parameters == 0:
            return np.zeros((hessian.shape[0], 0, 0))
        else:
            hessian = hessian[:,:,self.indices_of_parameters]
            hessian = hessian[:,self.indices_of_parameters,:]
            return hessian
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'SlicedModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        subgroup = group.create_group('constant_parameters')
        for key in self.constant_parameters:
            subgroup.attrs[key] = self.constant_parameters[key]
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, SlicedModel):
            return False
        if self.model != other.model:
            return False
        these_keys = set(self.constant_parameters.keys())
        other_keys = set(other.constant_parameters.keys())
        if these_keys != other_keys:
            return False
        for key in these_keys:
            if self.constant_parameters[key] != other.constant_parameters[key]:
                return False
        return True
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for name in self.model.parameters:
                self._bounds[name] = self.model.bounds[name]
        return self._bounds

