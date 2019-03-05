"""
File: pylinex/loglikelihood/LoglikelihoodWithModel.py
Author: Keith Tauscher
Date: 23 Nov 2018

Description: File containing class representing a likelihood whose parameters
             are identical to the parameters of some Model object.
"""
from ..model import Model, load_model_from_hdf5_group
from .LoglikelihoodWithData import LoglikelihoodWithData

class LoglikelihoodWithModel(LoglikelihoodWithData):
    """
    Class representing a likelihood whose parameters are identical to the
    parameters of some Model object.
    """
    @property
    def model(self):
        """
        Property storing the Model object which models the data used by this
        likelihood.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the model of the data used by this likelihood.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model must be a Model object.")
    
    @property
    def parameters(self):
        """
        Property storing the names of the parameters of the model defined by
        this likelihood.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = self.model.parameters
        return self._parameters
    
    def save_model(self, group, **model_links):
        """
        Saves the Model of this Loglikelihood in the appropriate part of given
        hdf5 file group.
        
        group: hdf5 file group at which to save Model
        **model_links: kwargs to pass on to Model's fill_hdf5_group method
        """
        self.model.fill_hdf5_group(group.create_group('model'), **model_links)
    
    @staticmethod
    def load_model(group):
        """
        Loads the model from a given group in which a LoglikelihoodWithModel
        object was saved.
        
        group: group from which to load Model
        
        returns: Model object associated with this Loglikelihood object
        """
        return load_model_from_hdf5_group(group['model'])
    
    def change_model(self, new_model):
        """
        Returns a LoglikelihoodWithModel of the same class as this object with
        a different model. Everything else is kept constant.
        
        returns: a new LoglikelihoodWithModel of the same class as this object
        """
        raise NotImplementedError("Either the LoglikelihoodWithModel class " +\
            "was directly instantiated (not allowed, use a subclass " +\
            "instead) or the subclass of LoglikelihoodWithModel being used " +\
            "has not implemented the change_model function (in which case " +\
            "it should be implemented).")
    
    def center(self, parameters):
        """
        Finds a LoglikelihoodWithModel of the same class as this object with
        the data set to this object's model evaluated at the given parameters.
        
        parameters: the parameters at which the desired loglikelihood should be
                    maximized
        
        returns: a new LoglikelihoodWithModel of the same class as this object
                 whose data has been changed to be equal to this object's model
                 evaluated at the given parameters
        """
        return self.change_data(self.model(parameters))

