"""
File: pylinex/nonlinear/loglikelihood/GammaLoglikelihood.py
Author: Keith Tauscher
Date: 23 Nov 2018

Description: File containing a class which evaluates a likelihood whose data
             are Gamma-distributed.
"""
import numpy as np
from ..util import real_numerical_types, sequence_types
from ..model import load_model_from_hdf5_group, Model
from .Loglikelihood import Loglikelihood

class GammaLoglikelihood(Loglikelihood):
    """
    Class which evaluates a likelihood whose data are Gamma-distributed.
    """
    def __init__(self, data, model, num_averaged):
        """
        Initializes this Loglikelihood with the given data, noise level in the
        data, and Model of the data.
        
        data: 1D numpy.ndarray of data being fit
        model: the Model object with which to describe the data
        num_averaged: Gamma distributed data can come about from averaging an
                      integer amount of exponential random variables. This
                      parameter should be set to that integer, although it can
                      have the same array dependence as data and model.
        """
        self.data = data
        self.model = model
        self.num_averaged = num_averaged
    
    @property
    def num_averaged(self):
    """
    Property storing the integer numbers of data points which were averaged
    into the data of this likelihood.
    """
    if not hasattr(self, '_num_averaged'):
        raise AttributeError("num_averaged was referenced before it was set.")
    return self._num_averaged
    
    @num_averaged.setter
    def num_averaged(self, value):
        """
        Setter for the number of curves which were averaged in 
        """
        if type(value) in real_numerical_types:
            self._num_averaged = value * np.ones_like(self.data)
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == self.data.shape:
                self._num_averaged = value
            else:
                raise ValueError("num_averaged shape was not the same as " +\
                    "data shape.")
        else:
            raise TypeError("num_averaged was set to an object which was " +\
                "neither a real number or a sequence of real numbers.")
    
    @property
    def summed_data(self):
        """
        Property storing the data property multiplied by the num_averaged
        property.
        """
        if not hasattr(self, '_summed_data'):
            self._summed_data = self.data * self.num_averaged
        return self._summed_data
    
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
        group.attrs['class'] = 'GammaLoglikelihood'
        self.save_data(group, data_link=data_link)
        group.create_dataset('num_averaged', data=self.num_averaged)
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
            assert group.attrs['class'] == 'GammaLoglikelihood'
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "GammaLoglikelihood object.")
        data = Loglikelihood.load_data(group)
        model = load_model_from_hdf5_group(group['model'])
        num_averaged = group['num_averaged'].value
        return GammaLoglikelihood(data, model, num_averaged)
    
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
            raise ValueError("models for Gamma likelihoods can only " +\
                "return positive numbers.")
        mean_normalized_data = (self.summed_data / mean)
        logL_value = np.sum(\
            ((self.num_averaged - 1) * np.log(mean_normalized_data)) -\
            np.log(mean) - mean_normalized_data)
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
        model_gradient = self.model.gradient(pars)
        if np.any(mean <= 0):
            raise ValueError("models for Gamma likelihoods can only " +\
                "return positive numbers.")
        gradient_value = np.sum(((self.num_averaged * (self.data - mean)) /\
            (mean ** 2))[:,np.newaxis] * model_gradient, axis=0)
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
        model_gradient = self.model.gradient(pars)
        model_hessian = self.model.hessian(pars)
        hessian_part = np.sum(((self.num_averaged * (self.data - mean)) /\
            (mean ** 2))[:,np.newaxis,np.newaxis] * model_hessian, axis=0)
        squared_gradient_part = np.sum(model_gradient[:,:,np.newaxis] *\
            model_gradient[:,np.newaxis,:] *\
            ((mean - (2 * self.data)) / mean)[:,np.newaxis,np.newaxis], axis=0)
        hessian_value = hessian_part + squared_gradient_part
        if return_negative:
            return -hessian_value
        else:
            return hessian_value

