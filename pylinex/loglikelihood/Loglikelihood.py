"""
File: pylinex/nonlinear/loglikelihood/Loglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a base class representing a likelihood that can be
             evaluated using a data vector and a Model object (and possibly
             other things, depending on the subclass).
"""
import numpy as np
import numpy.linalg as la
from distpy import cast_to_transform_list, GaussianDistribution,\
    WindowedDistribution, DistributionSet
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
    
    def __call__(self, pars, return_negative=False):
        """
        Gets the value of this Loglikelihood at the given parameters.
        """
        raise cannot_instantiate_loglikelihood_error
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this Loglikelihood can be
        computed.
        """
        raise cannot_instantiate_loglikelihood_error
    
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
        if self.gradient_computable:
            return self.auto_gradient(pars, return_negative=return_negative)
        else:
            raise NotImplementedError("gradient is not computable in an " +\
                "exact form. Use auto_gradient instead to do a numerical " +\
                "approximation.")
    
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
        raise cannot_instantiate_loglikelihood_error
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this Loglikelihood can be
        computed. The hessian of this Loglikelihood is computable as long as
        the model's gradient and hessian are computable.
        """
        raise cannot_instantiate_loglikelihood_error
    
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
        if self.hessian_computable:
            return self.auto_hessian(pars, return_negative=return_negative)
        else:
            raise NotImplementedError("hessian is not computable in an " +\
                "exact form. Use auto_hessian instead to do a numerical " +\
                "approximation.")
    
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
    
    def fisher_information(self, maximum_likelihood_parameters,\
        larger_differences=1e-5, smaller_differences=1e-6,\
        transform_list=None):
        """
        Calculates the Fisher information matrix of this likelihood assuming
        that the argument associated with the maximum of this likelihood is
        reasonably approximated by the given parameters.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
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
                        space is not transformed. No matter what,
                        maximum_likelihood_parameters should be the parameters
                        that maximize the likelihood when plugged into the
                        model of this likelihood untransformed.
        
        returns: numpy.ndarray of shape (num_parameters, num_parameters)
                 containing the Fisher information matrix
        """
        return self.auto_hessian(maximum_likelihood_parameters,\
            larger_differences=larger_differences,\
            smaller_differences=smaller_differences,\
            transform_list=transform_list, return_negative=True)
    
    def parameter_covariance_fisher_formalism(self,\
        maximum_likelihood_parameters, transform_list=None,\
        max_standard_deviations=np.inf, larger_differences=1e-5,\
        smaller_differences=1e-6):
        """
        Finds the parameter covariance assuming maximum_likelihood_parameters
        contains a reasonable approximation of the true maximum likelihood
        parameter vector.
        
        maximum_likelihood_parameters: the maximum likelihood parameter vector
                                       (or some approximation of it), given in
                                       untransformed space, no matter the value
                                       of the transform_list argument
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
        
        returns: numpy.ndarray of shape (num_parameters, num_parameters)
                 containing the inverse Fisher information matrix
        """
        inverse_covariance = self.fisher_information(\
            maximum_likelihood_parameters, transform_list=transform_list,\
            larger_differences=larger_differences,\
            smaller_differences=smaller_differences)
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
        maximum_likelihood_parameters, transform_list=None,\
        max_standard_deviations=np.inf,\
        prior_to_impose_in_transformed_space=None,\
        larger_differences=1e-5, smaller_differences=1e-6):
        """
        Finds the parameter distribution assuming maximum_likelihood_parameters
        contains a reasonable approximation of the true maximum likelihood
        parameter vector.
        
        maximum_likelihood_parameters: the maximum likelihood  parameter vector
                                       (or some approximation of it)
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
        
        returns: DistributionSet object containing GaussianDistribution object
                 approximating distribution in transformed space
        """
        transform_list = cast_to_transform_list(transform_list,\
            num_transforms=self.num_parameters)
        mean = transform_list(maximum_likelihood_parameters)
        covariance = self.parameter_covariance_fisher_formalism(\
            maximum_likelihood_parameters, transform_list=transform_list,\
            max_standard_deviations=max_standard_deviations,\
            larger_differences=larger_differences,\
            smaller_differences=smaller_differences)
        distribution = GaussianDistribution(mean, covariance)
        if prior_to_impose_in_transformed_space is not None:
            distribution = WindowedDistribution(distribution,\
                prior_to_impose_in_transformed_space)
        return\
            DistributionSet([(distribution, self.parameters, transform_list)])

