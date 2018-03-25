"""
File: pylinex/model/ExpandedModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class which represents a model which simply
             expands the output of another model using an Expander from the
             pylinex.expander module.
"""
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
        model_hessian = self.model.hessian(parameters)
        terminal_shape = (-1,) + model_hessian.shape[1:]
        intermediate_shape = (-1, np.prod(terminal_shape))
        model_hessian = np.reshape(model_hessian, intermediate_shape)
        return np.reshape(self.expander(model_hessian.T).T, terminal_shape)
    
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

