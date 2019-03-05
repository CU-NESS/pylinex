"""
File: pylinex/loglikelihood/LoglikelihoodWithData.py
Author: Keith Tauscher
Date: 4 Mar 2019

Description: File containing class representing a likelihood which contains a
             data vector.
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .Loglikelihood import Loglikelihood

class LoglikelihoodWithData(Loglikelihood):
    """
    Class representing a likelihood which contains a data vector.
    """
    @property
    def data(self):
        """
        Property storing the data fit by this likelihood.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data to fit with this likelihood.
        
        value: 1D numpy.ndarray of same length as error
        """
        value = np.array(value)
        if value.ndim == 1:
            self._data = value
        else:
            raise ValueError("data given was not 1D.")
    
    @property
    def num_channels(self):
        """
        Property storing the integer number of data channels in the data of
        this Loglikelihood.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.data)
        return self._num_channels
    
    @property
    def degrees_of_freedom(self):
        """
        Property storing the integer number of degrees of freedom
        (num_channels less num_parameters).
        """
        if not hasattr(self, '_degrees_of_freedom'):
            self._degrees_of_freedom = self.num_channels - self.num_parameters
        return self._degrees_of_freedom
    
    def save_data(self, group, data_link=None):
        """
        Saves the data of this Loglikelihood object.
        
        group: hdf5 file group where information about this object is being
               saved
        data_link: link to where data is already saved somewhere (if it exists)
        """
        create_hdf5_dataset(group, 'data', data=self.data, link=data_link)
    
    @staticmethod
    def load_data(group):
        """
        Loads the data of a Loglikelihood object from the given group.
        
        group: hdf5 file group where loglikelihood.save_data(group)
               has previously been called
        
        returns: data, an array
        """
        return get_hdf5_value(group['data'])
    
    def change_data(self, new_data):
        """
        Returns a LoglikelihoodWithData of the same class as this object with a
        different data vector. Everything else is kept constant.
        
        returns: a new LoglikelihoodWithData of the same class as this object
        """
        raise NotImplementedError("Either the LoglikelihoodWithData class " +\
            "was directly instantiated (not allowed, use a subclass " +\
            "instead) or the subclass of LoglikelihoodWithData being used " +\
            "has not implemented the change_data function (in which case " +\
            "it should be implemented).")

