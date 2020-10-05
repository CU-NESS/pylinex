"""
File: plyinex/model/SingleConditionalFitModel.py
Author: Keith Tauscher
Date: 31 May 2019

Description: File containing class representing a model where part(s) of the
             model is conditionalized and the parameters of the other part(s)
             of the model are automatically evaluated at parameters that
             maximize a likelihood.
"""
import numpy as np
from ..util import numerical_types, sequence_types
from .Model import Model
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ConditionalFitModel(Model):
    """
    An class representing a model where part(s) of the model is
    conditionalized and the parameters of the other part(s) of the model are
    automatically evaluated at parameters that maximize a likelihood.
    """
    @property
    def model(self):
        """
        Property storing the full model that is used to fit data.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the full model.
        
        value: a model with submodels
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was not a Model object.")
    
    @property
    def data(self):
        """
        Property storing the data vector to use when conditionalizing. Ideally,
        all results from this model are similar to this data vector.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data vector to use when conditionalizing.
        
        value: 1D numpy.ndarray object
        """
        if type(value) in sequence_types:
            if all([(type(element) in numerical_types) for element in value]):
                self._data = np.array(value)
            else:
                raise TypeError("data was set to a sequence whose elements " +\
                    "are not numbers.")
        else:
            raise TypeError("data was set to a non-sequence.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output which is the same
        as the number of x_values given.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.data)
        return self._num_channels
    
    @property
    def error(self):
        """
        Property storing the noise level on the data vector.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the noise level on the data vector.
        
        value: 1D numpy.ndarray object of same length as data vector
        """
        if type(value) is type(None):
            self._error = np.ones_like(self.data)
        elif type(value) in sequence_types:
            if len(value) == self.num_channels:
                if all([(type(element) in numerical_types)\
                    for element in value]):
                    self._error = np.array(value)
                else:
                    raise TypeError("error was set to a sequence whose " +\
                        "elements are not numbers.")
            else:
                raise ValueError("error does not have same length as data.")
        else:
            raise TypeError("error was set to a non-sequence.")
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return False

