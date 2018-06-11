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
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        error_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        model_links: dictionary of any other kwargs to pass on to the model's
                     fill_hdf5_group function
        """
        raise cannot_instantiate_loglikelihood_error
    
    def save_data_and_model(self, group, data_link=None, **model_links):
        """
        Saves the data and model of this Loglikelihood object.
        
        group: hdf5 file group where information about this object is being
               saved
        data_link: link to where data is already saved somewhere (if it exists)
        model_links: extra kwargs to pass on to the fill_hdf5_group of the
                     model being saved
        """
        create_hdf5_dataset(group, 'data', data=self.data, link=data_link)
        self.model.fill_hdf5_group(group.create_group('model'), **model_links)
    
    @staticmethod
    def load_data_and_model(group):
        """
        Loads the model of a Loglikelihood object from the given group.
        
        group: hdf5 file group where loglikelihood.save_data_and_model(group)
               has previously been called
        
        returns: (data, model) where data is an array and model is a Model
        """
        data = get_hdf5_value(group['data'])
        model = load_model_from_hdf5_group(group['model'])
        return (data, model)
    
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

