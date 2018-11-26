"""
File: pylinex/nonlinear/loglikelihood/GaussianLoglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a class which evaluates a likelihood which is
             Gaussian in the data.
"""
import numpy as np
import numpy.linalg as la
from distpy import cast_to_transform_list, WindowedDistribution,\
    GaussianDistribution, DistributionSet, DistributionList
from ..util import create_hdf5_dataset, get_hdf5_value
from .Loglikelihood import Loglikelihood
from .LoglikelihoodWithModel import LoglikelihoodWithModel

class GaussianLoglikelihood(LoglikelihoodWithModel):
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
        elif value.shape == (self.data.shape * 2):
            self._error = value
        else:
            raise ValueError("error given was not the same shape as the data.")
    
    def save_error(self, group, error_link=None):
        """
        Saves the error of this Loglikelihood object.
        
        group: hdf5 file group where information about this object is being
               saved
        error_link: link to where error is already saved somewhere (if it
                    exists)
        """
        create_hdf5_dataset(group, 'error', data=self.error, link=error_link)
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None,\
        **model_links):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        error_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        model_links: extra kwargs to pass on to the fill_hdf5_group of the
                     model being saved
        """
        group.attrs['class'] = 'GaussianLoglikelihood'
        self.save_data(group, data_link=data_link)
        self.save_model(group, **model_links)
        self.save_error(group, error_link=error_link)
    
    @staticmethod
    def load_error(group):
        """
        Loads the error of a Loglikelihood object from the given group.
        
        group: hdf5 file group where loglikelihood.save_error(group)
               has previously been called
        
        returns: error, an array
        """
        return get_hdf5_value(group['error'])
    
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
        data = Loglikelihood.load_data(group)
        model = LoglikelihoodWithModel.load_model(group)
        error = GaussianLoglikelihood.load_error(group)
        return GaussianLoglikelihood(data, error, model)
    
    @property
    def weighting_matrix(self):
        """
        Property storing the matrix to use for weighting if error is given as
        2D array.
        """
        if not hasattr(self, '_weighting_matrix'):
            if self.error.ndim == 1:
                raise AttributeError("The weighting_matrix property only " +\
                    "makes sense if the error given was a covariance matrix.")
            else:
                (eigenvalues, eigenvectors) = la.eigh(self.error)
                eigenvalues = np.power(eigenvalues, -0.5)
                self._weighting_matrix = np.dot(\
                    eigenvectors * eigenvalues[np.newaxis,:], eigenvectors.T)
        return self._weighting_matrix
    
    def weight(self, quantity):
        """
        Meant to generalize weighting by the inverse square root of the
        covariance matrix so that it is efficient when the error is 1D
        
        quantity: quantity whose 0th axis is channel space which should be
                  weighted
        
        returns: numpy.ndarray of same shape as quantity containing weighted
                 quantity
        """
        if self.error.ndim == 1:
            error_index =\
                ((slice(None),) + ((np.newaxis,) * (quantity.ndim - 1)))
            return quantity / self.error[error_index]
        elif quantity.ndim in [1, 2]:
            return np.dot(self.weighting_matrix, quantity)
        else:
            quantity_shape = quantity.shape
            quantity = np.reshape(quantity, (quantity_shape[0], -1))
            quantity = np.dot(self.weighting_matrix, quantity)
            return np.reshape(quantity, quantity_shape)
    
    def weighted_bias(self, pars):
        """
        Computes the weighted difference between the data and the model
        evaluated at the given parameters.
        
        pars: array of parameter values at which to evaluate the weighted_bias
        
        returns: 1D numpy array of biases (same shape as data and error arrays)
        """
        return self.weight(self.data - self.model(pars))
    
    def weighted_gradient(self, pars):
        """
        Computes the weighted version of the gradient of the model in this
        likelihood.
        
        pars: array of parameter values at which to evaluate model gradient
        
        returns: 2D array of shape (num_channels, num_parameters)
        """
        return self.weight(self.model.gradient(pars))
    
    def weighted_hessian(self, pars):
        """
        Computes the weighted version of the hessian of the model in this
        likelihood.
        
        pars: array of parameter values at which to evaluate model hessian
        
        returns: 2D array of shape
                 (num_channels, num_parameters, num_parameters)
        """
        return self.weight(self.model.hessian(pars))
    
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
            logL_value = np.sum(np.abs(self.weighted_bias(pars)) ** 2) / (-2.)
        except (ValueError, ZeroDivisionError):
            logL_value = -np.inf
        if np.isnan(logL_value):
            logL_value = -np.inf
        if return_negative:
            return -logL_value
        else:
            return logL_value
    
    def reduced_chi_squared(self, parameters):
        """
        Computes the reduced chi squared statistic. It should follow a
        chi2_reduced distribution with the correct number of degrees of
        freedom.
        
        pars: the parameter values at which to evaluate the likelihood
        
        returns: single number statistic proportional to the value of this
                 GaussianLoglikelihood object (since additive constant
                 corresponding to normalization constant is not included)
        """
        return ((-2.) * self(parameters, return_negative=False)) /\
            self.degrees_of_freedom
    
    def fisher_information_no_hessian(self, maximum_likelihood_parameters,\
        transform_list=None, differences=1e-6):
        """
        Calculates the Fisher information matrix of this likelihood assuming
        that the argument associated with the maximum of this likelihood is
        reasonably approximated by the given parameters.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed. No matter what,
                        maximum_likelihood_parameters should be the parameters
                        that maximize the likelihood when plugged into the
                        model of this likelihood untransformed.
        differences: either single number of 1D array of numbers to use as the
                     numerical difference in each parameter. Default: 10^(-6)
                     Only necessary if this likelihood's model does not have an
                     analytically implemented gradient
        
        returns: numpy.ndarray of shape (num_parameters, num_parameters)
                 containing the Fisher information matrix
        """
        if self.gradient_computable:
            gradient = self.model.gradient(maximum_likelihood_parameters)
            transform_list = cast_to_transform_list(transform_list,\
                num_transforms=self.num_parameters)
            gradient = transform_list.transform_gradient(gradient,\
                maximum_likelihood_parameters)
        else:
            gradient = self.model.numerical_gradient(\
                maximum_likelihood_parameters, differences=differences,\
                transform_list=transform_list)
        weighted_gradient = self.weight(gradient)
        return np.real(np.dot(np.conj(weighted_gradient.T), weighted_gradient))
    
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
    
    def auto_gradient(self, pars, return_negative=False, differences=1e-6,\
        transform_list=None):
        """
        Computes the gradient of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the gradient
        return_negative: if true, the negative of the gradient of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        differences: either single number or 1D array of numbers to use as the
                     numerical difference in parameter. Default: 10^(-6)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: 1D numpy.ndarray of length num_parameters containing gradient
                 of loglikelihood value
        """
        self.check_parameter_dimension(pars)
        try:
            gradient_value = np.real(np.dot(self.weight(\
                self.model.auto_gradient(pars, differences=differences,\
                transform_list=transform_list)).T,\
                np.conj(self.weighted_bias(pars))))
        except:
            gradient_value = np.nan * np.ones((self.num_parameters,))
        else:
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
    
    def auto_hessian(self, pars, return_negative=False,\
        larger_differences=1e-5, smaller_differences=1e-6,\
        transform_list=None):
        """
        Computes the hessian of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the hessian
        return_negative: if true, the negative of the hessian of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        larger_differences: either single number or 1D array of numbers to use
                            as the numerical difference in parameters.
                            Default: 10^(-5). This is the amount by which the
                            parameters are shifted between evaluations of the
                            gradient. Only used if gradient is not explicitly
                            computable.
        smaller_differences: either single_number or 1D array of numbers to use
                             as the numerical difference in parameters.
                             Default: 10^(-6). This is the amount by which the
                             parameters are shifted during each approximation
                             of the gradient. Only used if hessian is not
                             explicitly computable
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: square 2D numpy.ndarray of side length num_parameters
                 containing hessian of loglikelihood value
        """
        self.check_parameter_dimension(pars)
        try:
            weighted_bias = self.weighted_bias(pars)
            weighted_gradient = self.weight(self.model.auto_gradient(\
                pars, differences=smaller_differences,\
                transform_list=transform_list))
            weighted_hessian = self.weight(self.model.auto_hessian(\
                pars, larger_differences=larger_differences,\
                smaller_differences=smaller_differences,\
                transform_list=transform_list))
            hessian_part =\
                np.real(np.dot(weighted_hessian.T, np.conj(weighted_bias)))
            squared_gradient_part = np.real(\
                np.dot(np.conj(weighted_gradient).T, weighted_gradient))
            hessian_value = hessian_part - squared_gradient_part
        except:
            hessian_value = np.nan * np.ones((self.num_parameters,) * 2)
        if return_negative:
            return -hessian_value
        else:
            return hessian_value
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        if not isinstance(other, GaussianLoglikelihood):
            return False
        return ((self.model == other.model) and\
            np.allclose(self.data, other.data) and\
            np.allclose(self.error, other.error))

