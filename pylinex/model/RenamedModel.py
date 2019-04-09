"""
File: pylinex/pylinex/model/RenamedModel.py
Author: Keith Tauscher
Date: 23 Mar 2018

Description: File containing a class representing a model whose parameters have
             been renamed.
"""
from ..util import sequence_types
from .Model import Model

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class RenamedModel(Model):
    """
    Class representing a model whose parameters have been renamed.
    """
    def __init__(self, model, new_parameter_names):
        """
        Initializes a new RenamedModel with the given model and parameter
        names.
        
        model: a Model object whose parameters need to be renamed
        new_parameter_names: sequence of strings whose length is equal to the
                             number of parameters associated with the given
                             model
        """
        self.model = model
        self.parameters = new_parameter_names
    
    @property
    def parameters(self):
        """
        Property storing the list of (new) names of this model's parameters.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters was referenced before it was " +\
                "set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the parameter names of this model.
        
        value: list of string parameter names with a length given by the number
               of parameters required by the underlying model
        """
        if type(value) in sequence_types:
            length = len(value)
            required = self.model.num_parameters
            if length == required:
                if all([isinstance(element, basestring) for element in value]):
                    self._parameters = [element for element in value]
                else:
                    raise TypeError("Not all elements of parameters list " +\
                        "were strings.")
            else:
                raise ValueError("parameters was set to a sequence of the " +\
                    "wrong length; its length was {0:d} even through the " +\
                    "underlying model requires {1:d} parameters.".format(\
                    length, required))
        else:
            raise TypeError("parameters was set to a non-sequence.")
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
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
        return self.model.hessian(parameters)
    
    def fill_hdf5_group(self, group, model_link=None):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        model_link: existing hdf5 group which contains the information about
                    the Model underlying this RenamedModel. If None, the model
                    is freshly saved here
        """
        group.attrs['class'] = 'RenamedModel'
        if type(model_link) is type(None):
            self.model.fill_hdf5_group(group.create_group('model'))
        else:
            group['model'] = model_link
        subgroup = group.create_group('parameters')
        for (iparameter, parameter) in enumerate(self.parameters):
            subgroup.attrs['{:d}'.format(iparameter)] = parameter
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if isinstance(other, RenamedModel):
            return (self.model == other.model) and\
                (self.parameters == other.parameters)
        else:
            return False
    
    def quick_fit(self, data, error=None):
        """
        Fits the given data (with error given possibly).
        
        data: the data to fit with this model
        error: the error on the data given. If not given, all points assumed
               identical and returned parameter covariance is meaningless
        
        returns: (parameter_mean, parameter_covariance)
        """
        return self.model.quick_fit(data, error=error)
    
    @property
    def bounds(self):
        """
        Property storing the bounds of parameters of this model in a
        dictionary.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for (old_name, new_name) in\
                zip(self.model.parameters, self.parameters):
                self._bounds[new_name] = self.model.bounds[old_name]
        return self._bounds

