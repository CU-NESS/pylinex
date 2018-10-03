"""
File: pylinex/nonlinear/loglikelihood/PoissonLoglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a class which evaluates a likelihood whose data
             are Poisson-distributed.
"""
import numpy as np
from scipy.special import gammaln as log_gamma
from ..model import load_model_from_hdf5_group, Model
from .Loglikelihood import Loglikelihood

class PoissonLoglikelihood(Loglikelihood):
    """
    Class which evaluates a likelihood whose data are Poisson-distributed.
    """
    def __init__(self, data, model):
        """
        Initializes this Loglikelihood with the given data, noise level in the
        data, and Model of the data.
        
        data: 1D numpy.ndarray of data being fit
        error: 1D numpy.ndarray describing the noise level of the data
        model: the Model object with which to describe the data
        """
        self.data = data
        self.model = model
    
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
    
    def fill_hdf5_group(self, group, data_link=None, **model_links):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        model_links: dictionary of any other kwargs to pass on to the model's
                     fill_hdf5_group function
        """
        group.attrs['class'] = 'PoissonLoglikelihood'
        self.save_data(group, data_link=data_link)
        self.model.fill_hdf5_group(group.create_group('model'), **model_links)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'PoissonLoglikelihood'
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "PoissonLoglikelihood object.")
        data = Loglikelihood.load_data(group)
        model = load_model_from_hdf5_group(group['model'])
        return PoissonLoglikelihood(data, model)
    
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
        mean = self.model(pars)
        if np.any(mean <= 0):
            raise ValueError("models for Poisson likelihoods can only " +\
                "return positive numbers.")
        data = self.data
        logL_value = np.sum((data * np.log(mean)) - mean - log_gamma(data + 1))
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
        mean = self.model(pars)
        if np.any(mean <= 0):
            raise ValueError("models for Poisson likelihoods can only " +\
                "return positive numbers.")
        gradient_value = np.sum(self.model.gradient(pars) *\
            ((self.data / mean) - 1)[:,np.newaxis], axis=0)
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
        mean = self.model(pars)
        if np.any(mean <= 0):
            raise ValueError("models for Poisson likelihoods can only " +\
                "return positive numbers.")
        hess = self.model.hessian(pars)
        data_over_mean = (self.data / mean)[:,np.newaxis,np.newaxis]
        hessian_part = np.sum(hess * (data_over_mean - 1), axis=0)
        del hess
        grad = self.model.gradient(pars)
        data_over_mean2 = (data_over_mean / mean[:,np.newaxis,np.newaxis])
        gradient_part = np.sum(grad[:,:,np.newaxis] * grad[:,np.newaxis,:] *\
            data_over_mean2, axis=0)
        del grad
        hessian_value = hessian_part - gradient_part
        if return_negative:
            return -hessian_value
        else:
            return hessian_value

