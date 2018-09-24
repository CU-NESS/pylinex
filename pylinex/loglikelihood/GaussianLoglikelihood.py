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
        elif value.shape == (self.data.shape * 2):
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
            logL_value = np.sum(self.weighted_bias(pars) ** 2) / (-2.)
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
    
    def fisher_information(self, maximum_likelihood_parameters,\
        differences=1e-6, transform_list=None):
        """
        Calculates the Fisher information matrix of this likelihood assuming
        that the argument associated with the maximum of this likelihood is
        reasonably approximated by the given parameters.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
        differences: either single number of 1D array of numbers to use as the
                     numerical difference in each parameter. Default: 10^(-6)
                     Only necessary if this likelihood's model does not have an
                     analytically implemented gradient
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed. No matter what,
                        maximum_likelihood_parameters should be the parameters
                        that maximize the likelihood when plugged into the
                        model of this likelihood untransformed.
        
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
        return np.dot(weighted_gradient.T, weighted_gradient)
    
    def parameter_covariance_fisher_formalism(self,\
        maximum_likelihood_parameters, differences=1e-6, transform_list=None,\
        max_standard_deviations=np.inf):
        """
        Finds the parameter covariance assuming maximum_likelihood_parameters
        contains a reasonable approximation of the true maximum likelihood
        parameter vector.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
        differences: either single number of 1D array of numbers to use as the
                     numerical difference in each parameter. Default: 10^(-6)
                     Only necessary if this likelihood's model does not have an
                     analytically implemented gradient
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed. No matter what,
                        maximum_likelihood_parameters should be the parameters
                        that maximize the likelihood when plugged into the
                        model of this likelihood untransformed.
        max_standard_deviations: single value or array of values containing the
                                 maximum allowable standard deviations of each
                                 parameter. This will stop the covariance from
                                 producing extremely wide results in the case
                                 of an unconstrained parameter. The default
                                 value is numpy.inf, which causes this
                                 correction to be unimportant in all cases.
        
        returns: numpy.ndarray of shape (num_parameters, num_parameters)
                 containing the inverse Fisher information matrix
        """
        inverse_covariance = self.fisher_information(\
            maximum_likelihood_parameters, differences=differences,\
            transform_list=transform_list)
        if np.any(max_standard_deviations == 0):
            raise ValueError("At least one of the max_standard deviations " +\
                "was set to 0, which implies the existence of at least one " +\
                "element in the null space of the covariance matrix, which " +\
                "does not make sense.")
        max_standard_deviations =\
            max_standard_deviations * np.ones(self.num_parameters)
        inverse_covariance = inverse_covariance +\
            np.diag(np.power(max_standard_deviations, -2))
        return la.inv(inverse_covariance)
    
    def parameter_distribution_fisher_formalism(self,\
        maximum_likelihood_parameters, differences=1e-6, transform_list=None,\
        max_standard_deviations=np.inf,\
        prior_to_impose_in_transformed_space=None):
        """
        Finds the parameter distribution assuming maximum_likelihood_parameters
        contains a reasonable approximation of the true maximum likelihood
        parameter vector.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
        differences: either single number of 1D array of numbers to use as the
                     numerical difference in each parameter. Default: 10^(-6)
                     Only necessary if this likelihood's model does not have an
                     analytically implemented gradient
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed. No matter what,
                        maximum_likelihood_parameters should be the parameters
                        that maximize the likelihood when plugged into the
                        model of this likelihood untransformed.
        max_standard_deviations: single value or array of values containing the
                                 maximum allowable standard deviations of each
                                 parameter. This will stop the covariance from
                                 producing extremely wide results in the case
                                 of an unconstrained parameter. The default
                                 value is numpy.inf, which causes this
                                 correction to be unimportant in all cases.
        prior_to_impose_in_transformed_space: if None (default), no prior is
                                                                 imposed and a
                                                                 Gaussian is
                                                                 returned
                                                                 through the
                                                                 Fisher matrix
                                                                 formalism
                                              otherwise, prior_to_impose should
                                                         be a Distribution
                                                         object whose log_value
                                                         function returns
                                                         -np.inf in disallowed
                                                         regions. The prior has
                                                         no effect inside the
                                                         region in which it is
                                                         finite.
        
        returns: DistributionSet object containing GaussianDistribution object
                 approximating distribution in transformed space
        """
        transform_list = cast_to_transform_list(transform_list,\
            num_transforms=self.num_parameters)
        mean = transform_list(maximum_likelihood_parameters)
        covariance = self.parameter_covariance_fisher_formalism(\
            maximum_likelihood_parameters, differences=differences,\
            transform_list=transform_list,\
            max_standard_deviations=max_standard_deviations)
        distribution = GaussianDistribution(mean, covariance)
        if prior_to_impose_in_transformed_space is not None:
            distribution = WindowedDistribution(distribution,\
                prior_to_impose_in_transformed_space)
        return DistributionSet([(distribution, self.parameters, transform_list)])
    
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
            gradient_value = np.dot(\
                self.weighted_gradient(pars).T, self.weighted_bias(pars))
        except:
            return np.nan * np.ones(self.num_parameters)
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
        try:
            weighted_bias = self.weighted_bias(pars)
            weighted_gradient = self.weighted_gradient(pars)
            weighted_hessian = self.weighted_hessian(pars)
            hessian_part = np.dot(weighted_hessian.T, weighted_bias)
            squared_gradient_part =\
                np.dot(weighted_gradient.T, weighted_gradient)
            hessian_value = hessian_part - squared_gradient_part
        except:
            hessian_value = np.nan * np.ones((self.num_parameters,) * 2)
        if return_negative:
            return -hessian_value
        else:
            return hessian_value

