"""
File: pylinex/model/RestrictedModel.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: File containing a class representing a Model which is the same as
             an underlying model except the values of its parameters are
             restricted to within given values.
"""
import numpy as np
from ..util import create_hdf5_dataset, sequence_types
from .Model import Model

class RestrictedModel(Model):
    """
    Class representing a Model which is the same as an underlying model except
    the values of its parameters are restricted to within given values.
    """
    def __init__(self, model, array_bounds):
        """
        Initializes a new RestrictedModel with the given model and
        array_bounds.
        
        model: Model object
        array_bounds: sequence of 2-tuples of form (low, high) where either can
                      be None, indicating that there is no boundary on that
                      side
        """
        self.model = model
        self.array_bounds = array_bounds
    
    @property
    def model(self):
        """
        Property storing the Model object underlying this Model.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the Model underlying this Model.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model.")
    
    @property
    def array_bounds(self):
        """
        Property storing the array_bounds of this model in a tuple (low, high)
        where low and high are numpy arrays of lower (upper) boundaries to
        impose.
        """
        if not hasattr(self, '_array_bounds'):
            raise AttributeError("array_bounds referenced before it was set.")
        return self._array_bounds
    
    @array_bounds.setter
    def array_bounds(self, value):
        """
        Setter for the array_bounds at which to impose restrictions.
        
        value: sequence of 2-tuples of form (low, high) where either can be
               None or a number. If both are numbers, high must be greater than
               low
        """
        if type(value) in sequence_types:
            if len(value) == self.model.num_parameters:
                if all([type(element) in sequence_types for element in value]):
                    if all([(len(element) == 2) for element in value]):
                        minima = []
                        maxima = []
                        for element in value:
                            (minimum, maximum) = element
                            if type(minimum) is type(None):
                                minima.append(-np.inf)
                            else:
                                minima.append(minimum)
                            if type(maximum) is type(None):
                                maxima.append(np.inf)
                            else:
                                maxima.append(maximum)
                        self._array_bounds =\
                            (np.array(minima), np.array(maxima))
                    else:
                        raise ValueError("Not all sequence elements of " +\
                            "array_bounds sequence have length 2.")
                else:
                    raise TypeError("Not all elements of array_bounds " +\
                        "sequence were sequences.")
            else:
                raise ValueError("The length of the array_bounds sequence " +\
                    "was not the same as the number of parameters in the " +\
                    "model.")
        else:
            raise TypeError("array_bounds was set to a non-sequence.")
    
    def check_parameter_bounds(self, parameters):
        """
        Checks if the given parameters are within the bounds imposed by this
        RestrictedModel. If parameters are within bounds, nothing happens.
        Otherwise, a Valueerror is thrown.
        
        parameters: array of parameter values of length model.num_parameters
        """
        (minima, maxima) = self.array_bounds
        if np.any(np.logical_or(parameters < minima, parameters > maxima)):
            raise ValueError("parameters aren't within boundaries imposed " +\
                "by this Model.")
    
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
        self.check_parameter_bounds(parameters)
        return self.model(parameters)
    
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
        self.check_parameter_bounds(parameters)
        return self.model.gradient(parameters)
    
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
        self.check_parameter_bounds(parameters)
        return self.model.hessian(parameters)
    
    def fill_hdf5_group(self, group, model_link=None):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        model_link: existing hdf5 group containing the information about the
                    Model underlying this one. If it is None, model is saved
                    fresh in this function
        """
        group.attrs['class'] = 'RestrictedModel'
        if type(model_link) is type(None):
            self.model.fill_hdf5_group(group.create_group('model'))
        else:
            group['model'] = model_link
        (minima, maxima) = self.array_bounds
        create_hdf5_dataset(group, 'minima', data=minima)
        create_hdf5_dataset(group, 'maxima', data=maxima)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if isinstance(other, RestrictedModel):
            if self.model == other.model:
                ((smin, smax), (omin, omax)) =\
                    (self.array_bounds, other.array_bounds)
                return np.all((smin == omin) & (smax == omax))
        return False
    
    @property
    def bounds(self):
        """
        Property storing the bounds in a dictionary format instead of an array
        format. The keys of the dictionary are parameter names and the values
        are tuples of the form (min, max) where either can be None.
        """
        if not hasattr(self, '_bounds'):
            (minima, maxima) = self.array_bounds
            self._bounds = {}
            for (name, minimum, maximum) in\
                zip(self.parameters, minima, maxima):
                self._bounds[name] = (minimum, maximum)
        return self._bounds

