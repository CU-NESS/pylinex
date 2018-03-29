"""
File: pylinex/nonlinear/loglikelihood/GaussianLoglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a class which evaluates a likelihood which is
             Gaussian in the data.
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .Loglikelihood import Loglikelihood

class GaussianLoglikelihood(Loglikelihood):
    """
    class which evaluates a likelihood which is Gaussian in the data.
    """
    def __init__(self, data, error, model):
        """
        Initializes this Loglikelihood with the given data, noise level in the
        data, and Model of the data.
        
        data: 1D numpy.ndarray of data being fit
        error: 1D numpy.ndarray describing the noise level of the data
        model: the Model object with which to describe the data
        """
        self.data = data
        self.error = error
        self.model = model
    
    @property
    def error(self):
        """
        Property storing the error on the data given to this likelihood.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error used to define the likelihood.
        
        value: must be a numpy.ndarray of the same shape as the data property
        """
        value = np.array(value)
        if value.shape == self.data.shape:
            self._error = value
        else:
            raise ValueError("error given was not the same shape as the data.")
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None,\
        **model_links):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        error_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        model_links: dictionary of any other kwargs to pass on to the model's
                     fill_hdf5_group function
        """
        group.attrs['class'] = 'GaussianLoglikelihood'
        self.save_data_and_model(group, data_link=data_link, **model_links)
        create_hdf5_dataset(group, 'error', data=self.error, link=error_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'GaussianLoglikelihood'
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "GaussianLoglikelihood object.")
        (data, model) = Loglikelihood.load_data_and_model(group)
        error = get_hdf5_value(group['error'])
        return GaussianLoglikelihood(data, error, model)
    
    def weighted_bias(self, pars):
        """
        Computes the weighted difference between the data and the model
        evaluated at the given parameters.
        
        pars: array of parameter values at which to evaluate the weighted_bias
        
        returns: 1D numpy array of biases (same shape as data and error arrays)
        """
        return (self.data - self.model(pars)) / self.error
    
    def __call__(self, pars, return_negative=False):
        """
        Gets the value of the loglikelihood at the given parameters.
        
        pars: the parameter values at which to evaluate the likelihood
        return_negative: if true the negative of the loglikelihood is returned
                         (this is useful for times when the loglikelihood must
                         be maximized since scipy optimization functions only
                         deal with minimization
        
        returns: the value of this Loglikelihood (or its negative if indicated)
        """
        self.check_parameter_dimension(pars)
        try:
            logL_value = np.sum(self.weighted_bias(pars) ** 2) / (-2.)
        except ValueError:
            logL_value = -np.inf
        if return_negative:
            return -logL_value
        else:
            return logL_value
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this Loglikelihood can be
        computed. The gradient of this Loglikelihood is computable as long as
        the model's gradient is computable.
        """
        if not hasattr(self, '_gradient_computable'):
            self._gradient_computable = self.model.gradient_computable
        return self._gradient_computable
    
    def gradient(self, pars, return_negative=False):
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
        self.check_parameter_dimension(pars)
        try:
            doubly_weighted_bias = self.weighted_bias(pars) / self.error
            to_sum =\
                doubly_weighted_bias[:,np.newaxis] * self.model.gradient(pars)
        except:
            return np.nan * np.ones(self.num_parameters)
        else:
            gradient_value = np.sum(to_sum, axis=0)
            if return_negative:
                return -gradient_value
            else:
                return gradient_value
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this Loglikelihood can be
        computed. The hessian of this Loglikelihood is computable as long as
        the model's gradient and hessian are computable.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable = (self.model.gradient_computable and\
                self.model.hessian_computable)
        return self._hessian_computable
    
    def hessian(self, pars, return_negative=False):
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
        self.check_parameter_dimension(pars)
        doubly_weighted_bias = self.weighted_bias(pars) / self.error
        hessian_part = np.sum(doubly_weighted_bias[:,np.newaxis,np.newaxis] *\
            self.model.hessian(pars), axis=0)
        gradient = self.model.gradient(pars)
        squared_gradient_part = np.sum(\
            (gradient[:,:,np.newaxis] * gradient[:,np.newaxis,:]) /\
            (self.error ** 2)[:,np.newaxis,np.newaxis], axis=0)
        hessian_value = hessian_part - squared_gradient_part
        if return_negative:
            return -hessian_value
        else:
            return hessian_value
