"""
File: pylinex/model/RestrictedModel.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: File containing a class representing a Model which is the same as
             an underlying model except the values of its parameters are
             restricted to within given values.
"""
from ..util import create_hdf5_dataset
from .Model import Model

class RestrictedModel(Model):
    """
    Class representing a Model which is the same as an underlying model except
    the values of its parameters are restricted to within given values.
    """
    def __init__(self, model, bounds):
        """
        Initializes a new RestrictedModel with the given model and bounds.
        
        model: Model object
        bounds: sequence of 2-tuples of form (low, high) whether either can be
                None, indicating that there is no boundary on that side
        """
        self.model = model
        self.bounds = bounds
    
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
    def bounds(self):
        """
        Property storing the bounds of this model in a tuple (low, high) where
        low and high are numpy arrays of lower (upper) boundaries to impose.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for the bounds at which to impose restrictions.
        
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
                            if minimum is None:
                                minima.append(-np.inf)
                            else:
                                minima.append(minimum)
                            if maximum is None:
                                maxima.append(np.inf)
                            else:
                                maxima.append(maximum)
                        self._bounds = (np.array(minima), np.array(maxima))
                    else:
                        raise ValueError("Not all sequence elements of " +\
                            "bounds sequence have length 2.")
                else:
                    raise TypeError("Not all elements of bounds sequence " +\
                        "were sequences.")
            else:
                raise ValueError("The length of the bounds sequence was " +\
                    "not the same as the number of parameters in the model.")
        else:
            raise TypeError("bounds was set to a non-sequence.")
    
    def check_parameter_bounds(self, parameters):
        """
        Checks if the given parameters are within the bounds imposed by this
        RestrictedModel. If parameters are within bounds, nothing happens.
        Otherwise, a Valueerror is thrown.
        
        parameters: array of parameter values of length model.num_parameters
        """
        (minima, maxima) = self.bounds
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
        if model_link is None:
            group['model'] = model_link
        else:
            self.model.fill_hdf5_group(group.create_group('model'))
        (minima, maxima) = self.bounds
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
                ((smin, smax), (omin, omax)) = (self.bounds, other.bounds)
                return np.all((smin == omin) & (smax == omax))
        return False

