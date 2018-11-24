"""
File: pylinex/nonlinear/loglikelihood/Loglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a base class representing a likelihood that can be
             evaluated using a data vector and a Model object (and possibly
             other things, depending on the subclass).
"""
import numpy as np
from ..util import Savable, Loadable, create_hdf5_dataset, get_hdf5_value
from ..model import Model, load_model_from_hdf5_group

cannot_instantiate_loglikelihood_error = NotImplementedError("The " +\
    "Loglikelihood class cannot be instantiated directly!")

class Loglikelihood(Savable, Loadable):
    """
    Abstract class representing a likelihood which is Gaussian in the data.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializer throws error because Loglikelihood is supposed to be an
        abstract class that is not directly instantiated.
        """
        raise cannot_instantiate_loglikelihood_error
    
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
    def parameters(self):
        """
        Property storing the names of the parameters of the model defined by
        this likelihood.
        """
        raise cannot_instantiate_loglikelihood_error
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters needed by the Model at the
        heart of this Loglikelihood.
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def degrees_of_freedom(self):
        """
        Property storing the integer number of degrees of freedom
        (num_channels less num_parameters).
        """
        if not hasattr(self, '_degrees_of_freedom'):
            self._degrees_of_freedom = self.num_channels - self.num_parameters
        return self._degrees_of_freedom
    
    def fill_hdf5_group(self, group, *args, **kwargs):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        """
        raise cannot_instantiate_loglikelihood_error
    
    def save_data(self, group, data_link=None):
        """
        Saves the data and model of this Loglikelihood object.
        
        group: hdf5 file group where information about this object is being
               saved
        data_link: link to where data is already saved somewhere (if it exists)
        """
        create_hdf5_dataset(group, 'data', data=self.data, link=data_link)
    
    @staticmethod
    def load_data(group):
        """
        Loads the model of a Loglikelihood object from the given group.
        
        group: hdf5 file group where loglikelihood.save_data_and_model(group)
               has previously been called
        
        returns: data, an array
        """
        return get_hdf5_value(group['data'])
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        raise cannot_instantiate_loglikelihood_error
    
    def check_parameter_dimension(self, pars):
        """
        Checks to ensure that the given array is 1D and has one element for
        each parameter. The only thing this function does is throw an error if
        the array is the wrong shape.
        
        pars: array to check
        """
        if pars.shape != (self.num_parameters,):
            raise ValueError("The array of parameters given to this " +\
                "Loglikelihood object was not of the correct size.")
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this Loglikelihood can be
        computed.
        """
        raise cannot_instantiate_loglikelihood_error
    
    def gradient(self, pars):
        """
        Computes the gradient of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the gradient
        return_negative: if true, the negative of the gradient of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        
        returns: 1D numpy.ndarray of length num_parameters containing gradient
                 of loglikelihood value
        """
        raise cannot_instantiate_loglikelihood_error
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this Loglikelihood can be
        computed. The hessian of this Loglikelihood is computable as long as
        the model's gradient and hessian are computable.
        """
        raise cannot_instantiate_loglikelihood_error
    
    def hessian(self, pars):
        """
        Computes the hessian of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the hessian
        return_negative: if true, the negative of the hessian of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        
        returns: square 2D numpy.ndarray of side length num_parameters
                 containing hessian of loglikelihood value
        """
        raise cannot_instantiate_loglikelihood_error
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        raise NotImplementedError("The __eq__ magic method must be defined " +\
            "by each subclass of Loglikelihood individually. The class " +\
            "being used does not have the method defined.")
    
    def __ne__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self do not have the same properties
        """
        return (not self.__eq__(other))

