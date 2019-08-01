"""
File: Model.py
Author: Keith Tauscher
Date: 12 Nov 2017

Description: File containing an abstract class representing a model.
"""
from __future__ import division
import numpy as np
from distpy import TransformList
from ..util import Savable

# an error indicating everything which should be implemented by subclass
shouldnt_instantiate_model_error = (lambda name, kind:\
    NotImplementedError(("Model shouldn't be substantiated directly. Each " +\
    "subclass must implement its own {0!s} {1!s}.").format(name, kind)))

class Model(Savable):
    """
    An abstract class representing any model.
    """
    def __init__(self, *args, **kwargs):
        """
        Since the Model class should not be directly instantiated, an error is
        thrown if its initializer is called.
        """
        raise shouldnt_instantiate_model_error('__init__', 'function')
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in outputs of this model.
        """
        raise shouldnt_instantiate_model_error('num_channels', 'property')
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters necessitated by this model.
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        raise shouldnt_instantiate_model_error('parameters', 'property')
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        raise shouldnt_instantiate_model_error('__call__', 'function')
    
    def curve_sample(self, distribution_set, ndraw, return_parameters=False):
        """
        Generates a curve sample from this model given a DistributionSet
        object.
        
        distribution_set: an instance of distpy's DistributionSet class which
                          describes a distribution for the parameters of this
                          model
        ndraw: positive integer number of curves to generate
        return_parameters: if True, parameters are returned alongside training
                           set
        
        returns: curve_set as 2D array of form (ndraw, nchannel) or
                 (curve_set, parameters) where parameters is a 2D array of
                 form (ndraw, npar)
        """
        draw = distribution_set.draw(ndraw)
        draw = np.array([draw[parameter] for parameter in self.parameters]).T
        curve_set = np.array([self(parameters) for parameters in draw])
        if return_parameters:
            return (curve_set, draw)
        else:
            return curve_set
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        raise shouldnt_instantiate_model_error('gradient_computable',\
            'property')
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        if self.gradient_computable:
            raise shouldnt_instantiate_model_error('gradient', 'function')
        else:
            raise NotImplementedError("gradient is not implemented because " +\
                "gradient_computable is False.")
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        raise shouldnt_instantiate_model_error('hessian_computable',\
            'property')
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        if self.hessian_computable:
            raise shouldnt_instantiate_model_error('hessian', 'function')
        else:
            raise NotImplementedError("hessian is not implemented because " +\
                "hessian_computable is False.")
    
    def numerical_gradient(self, parameters, differences=1e-6,\
        transform_list=None):
        """
        Numerically approximates the gradient of this model. parameters should
        not be within differences of any bounds.
        
        parameters: the 1D parameter vector at which to approximate the
                    gradient it shouldn't be in the neighborhood of any bounds
        differences: either single number or 1D array of numbers to use as the
                     numerical difference in parameter. Default: 10^(-6)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: array of shape (num_channels, num_parameters) containing
                 gradient values
        """
        differences = differences * np.ones(self.num_parameters)
        center_model_value = self(parameters)
        transform_list = TransformList.cast(transform_list,\
            num_transforms=self.num_parameters)
        vectors = transform_list.I(transform_list(parameters)[np.newaxis,:] +\
            np.diag(differences))
        outer_model_values =\
            np.stack([self(vector) for vector in vectors], axis=-1)
        return (outer_model_values - center_model_value[:,np.newaxis]) /\
            differences[np.newaxis,:]
    
    def auto_gradient(self, parameters, differences=1e-6, transform_list=None):
        """
        Computes a default gradient procedure. If the gradient of this Model is
        computable, it is computed directly. Otherwise, it is ascertained
        through numerical approximation.
        
        parameters: parameter values at which to evaluate gradient
        differences: either single number or 1D array of numbers to use as the
                     numerical difference in parameter. Default: 10^(-6)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: array of shape (num_channels, num_parameters) containing
                 gradient values
        """
        if self.gradient_computable:
            transform_list = TransformList.cast(transform_list,\
                num_transforms=self.num_parameters)
            return transform_list.transform_gradient(\
                self.gradient(parameters), parameters)
        else:
            return self.numerical_gradient(parameters,\
                differences=differences, transform_list=transform_list)
    
    def seminumerical_hessian(self, parameters, differences=1e-6,\
        transform_list=None):
        """
        Numerically approximates the gradient of this model. parameters should
        not be within differences of any bounds.
        
        parameters: the 1D parameter vector at which to approximate the
                    gradient it shouldn't be in the neighborhood of any bounds
        differences: either single number or 1D array of numbers to use as the
                     numerical difference in parameter. Default: 10^(-6)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
                 containing hessian values
        """
        differences = differences * np.ones(self.num_parameters)
        transform_list = TransformList.cast(transform_list,\
            num_transforms=self.num_parameters)
        center_model_gradient = transform_list.transform_gradient(\
            self.gradient(parameters), parameters)
        vectors = transform_list.I(transform_list(parameters)[np.newaxis,:] +\
            np.diag(differences))
        outer_gradient_values = np.stack([\
            transform_list.transform_gradient(self.gradient(vector), vector)\
            for vector in vectors], axis=-1)
        approximate_hessian =\
            (outer_gradient_values - center_model_gradient[:,:,np.newaxis]) /\
            differences[np.newaxis,np.newaxis,:]
        return (approximate_hessian +\
            np.swapaxes(approximate_hessian, -2, -1)) / 2
    
    def numerical_hessian(self, parameters, larger_differences=1e-5,\
        smaller_differences=1e-6, transform_list=None):
        """
        Numerically approximates the gradient of this model. parameters should
        not be within differences of any bounds.
        
        parameters: the 1D parameter vector at which to approximate the
                    gradient it shouldn't be in the neighborhood of any bounds
        larger_differences: either single number or 1D array of numbers to use
                            as the numerical difference in parameters.
                            Default: 10^(-5). This is the amount by which the
                            parameters are shifted between evaluations of the
                            gradient
        smaller_differences: either single_number or 1D array of numbers to use
                             as the numerical difference in parameters.
                             Default: 10^(-6). This is the amount by which the
                             parameters are shifted during each approximation
                             of the gradient.
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
                 containing hessian values
        """
        larger_differences = larger_differences * np.ones(self.num_parameters)
        center_model_gradient = self.numerical_gradient(parameters,\
            differences=smaller_differences, transform_list=transform_list)
        transform_list = TransformList.cast(transform_list,\
            num_transforms=self.num_parameters)
        vectors = transform_list.I(transform_list(parameters)[np.newaxis,:] +\
            np.diag(larger_differences))
        outer_gradient_values = np.stack([self.numerical_gradient(vector,\
            differences=smaller_differences, transform_list=transform_list)\
            for vector in vectors], axis=-1)
        approximate_hessian =\
            (outer_gradient_values - center_model_gradient[:,:,np.newaxis]) /\
            larger_differences[np.newaxis,np.newaxis,:]
        return (approximate_hessian +\
            np.swapaxes(approximate_hessian, -2, -1)) / 2
    
    def auto_hessian(self, parameters, larger_differences=1e-5,\
        smaller_differences=1e-6, transform_list=None):
        """
        Computes a default gradient procedure. If the gradient of this Model is
        computable, it is computed directly. Otherwise, it is ascertained
        through numerical approximation.
        
        parameters: the 1D parameter vector at which to approximate the
                    gradient it shouldn't be in the neighborhood of any bounds
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
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
                 containing hessian values
        """
        if self.gradient_computable:
            if self.hessian_computable:
                transform_list = TransformList.cast(transform_list,\
                    num_transforms=self.num_parameters)
                untransformed_gradient = self.gradient(parameters)
                untransformed_hessian = self.hessian(parameters)
                transformed_gradient = transform_list.transform_gradient(\
                    untransformed_gradient, parameters)
                return transform_list.transform_hessian(untransformed_hessian,\
                    transformed_gradient, parameters)
            else:
                return self.seminumerical_hessian(parameters,\
                    differences=smaller_differences,\
                    transform_list=transform_list)
        else:
            return self.numerical_hessian(parameters,\
                larger_differences=larger_differences,\
                smaller_differences=smaller_differences,\
                transform_list=transform_list)
    
    def quick_fit(self, data, error, quick_fit_parameters=[], prior=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        quick_fit_parameters: quick fit parameters to pass to underlying model
        prior: either None or a GaussianDistribution object containing priors
               (in space of underlying model)
        
        returns: (parameter_mean, parameter_covariance) where parameter_mean is
                 a length N (number of parameters) 1D array and
                 parameter_covariance is a 2D array of shape (N,N). If no error
                 is given, parameter_covariance doesn't really mean anything
                 (especially if error is far from 1 in magnitude)
        """
        raise NotImplementedError("Either the Model subclass you are using " +\
            "does not support quick_fit or it hasn't yet been implemented.")
    
    @property
    def quick_fit_parameters(self):
        """
        Property storing the quick_fit_parameters
        """
        if not hasattr(self, '_quick_fit_parameters'):
            self._quick_fit_parameters = []
        return self._quick_fit_parameters
    
    @property
    def num_quick_fit_parameters(self):
        """
        Property storing the number of quick_fit_parameters in this model.
        """
        if not hasattr(self, '_num_quick_fit_parameters'):
            self._num_quick_fit_parameters = len(self.quick_fit_parameters)
        return self._num_quick_fit_parameters
    
    def quick_residual(self, data, error, *quick_fit_parameters):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        
        returns: 1D array of data's shape containing residual of quick fit
        """
        (parameter_mean, parameter_covariance) =\
            self.quick_fit(data, error, *quick_fit_parameters)
        return data - self(parameter_mean)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        raise shouldnt_instantiate_model_error('fill_hdf5_group', 'function')
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        raise shouldnt_instantiate_model_error('__eq__', 'function')
    
    def __ne__(self, other):
        """
        Checks for inequality with other. This just returns the opposite of
        __eq__ so that (a!=b)==(!(a==b)) for all a and b.
        
        other: the object to check for inequality
        
        returns: False if other is equal to this model, True otherwise
        """
        return (not self.__eq__(other))
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model. Since many models
        have no constraints, unless this property is overridden in a subclass,
        that subclass will give no bounds.
        """
        if not hasattr(self, '_bounds'):
            return {parameter: (None, None) for parameter in self.parameters}
        return self._bounds

