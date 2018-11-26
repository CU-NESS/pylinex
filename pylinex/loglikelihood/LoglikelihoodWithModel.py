"""
File: pylinex/loglikelihood/LoglikelihoodWithModel.py
Author: Keith Tauscher
Date: 23 Nov 2018

Description: File containing class representing a likelihood whose parameters
             are identical to the parameters of some Model object.
"""
from ..model import Model, load_model_from_hdf5_group
from .Loglikelihood import Loglikelihood

class LoglikelihoodWithModel(Loglikelihood):
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

