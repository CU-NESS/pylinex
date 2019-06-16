"""
File: pylinex/nonlinear/loglikelihood/ConditionalFitGaussianLoglikelihood.py
Author: Keith Tauscher
Date: 5 Jun 2019

Description: File containing a class which evaluates a likelihood which is
             Gaussian in the data by using a ConditionalFitModel, which
             computes conditional distributions of a given submodel.
"""
from __future__ import division
import numpy as np
import numpy.linalg as la
from ..util import numerical_types
from ..model import ConditionalFitModel, load_model_from_hdf5_group
from .LoglikelihoodWithData import LoglikelihoodWithData
from .GaussianLoglikelihood import GaussianLoglikelihood

cannot_call_function_error = NotImplementedError("This function is not " +\
    "implemented by the ConditionalFitGaussianLoglikelihood class, and " +\
    "thus cannot be called.")

class ConditionalFitGaussianLoglikelihood(LoglikelihoodWithData):
    """
    A class which evaluates a likelihood which is Gaussian in the data using a
    ConditionalFitModel.
    """
    def __init__(self, conditional_fit_model):
        """
        Initializes this Loglikelihood with the given ConditionalFitModel.
        """
        self.conditional_fit_model = conditional_fit_model
    
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
    def conditional_fit_model(self):
        """
        Property storing the ConditionalFitModel defining this Loglikelihood.
        """
        if not hasattr(self, '_conditional_fit_model'):
            raise AttributeError("conditional_fit_model was referenced " +\
                "before it was set.")
        return self._conditional_fit_model
    
    @conditional_fit_model.setter
    def conditional_fit_model(self, value):
        """
        Setter for the ConditionalFitModel defining this Loglikelihood.
        
        value: a ConditionalFitModel object
        """
        if isinstance(value, ConditionalFitModel):
            self._conditional_fit_model = value
        else:
            raise TypeError("conditional_fit_model was set to an object " +\
                "that was not of the ConditionalFitModel class.")
    
    @property
    def model(self):
        """
        Alias property for the ConditionalFitModel upon which this
        loglikelihood is based.
        """
        return self.conditional_fit_model
    
    @property
    def data(self):
        """
        Property storing the data given to this likelihood.
        """
        return self.conditional_fit_model.data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data vector. Cannot be used in this class because the
        data is taken directly from a ConditionalFitModel object.
        """
        raise RuntimeError("data cannot be set with the " +\
            "ConditionalFitGaussianLoglikelihood class. It is taken " +\
            "directly from a ConditionalFitModel object.")
    
    @property
    def error(self):
        """
        Property storing the error on the data given to this likelihood.
        """
        return self.conditional_fit_model.error
    
    @property
    def unknown_name_chain(self):
        """
        Property storing the chain of unknown names in the ConditionalFitModel.
        """
        return self.conditional_fit_model.unknown_name_chain
    
    @property
    def full_model(self):
        """
        Property storing the full model of the data given to this likelihood.
        """
        return self.conditional_fit_model.model
    
    @property
    def prior(self):
        """
        Property storing the prior on the parameters which are not
        conditionalized over (i.e. the ones which are marginalized over).
        """
        return self.conditional_fit_model.prior
    
    @property
    def full_loglikelihood(self):
        """
        Property storing the loglikelihood with the same data, error, and model
        that is not conditioned on any model.
        """
        if not hasattr(self, '_full_loglikelihood'):
            self._full_loglikelihood =\
                GaussianLoglikelihood(self.data, self.error, self.full_model)
        return self._full_loglikelihood
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        """
        group.attrs['class'] = 'ConditionalFitGaussianLoglikelihood'
        self.conditional_fit_model.fill_hdf5_group(\
            group.create_group('conditional_fit_model'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        try:
            assert(group.attrs['class'] ==\
                'ConditionalFitGaussianLoglikelihood')
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "ConditionalFitGaussianLoglikelihood object.")
        return ConditionalFitGaussianLoglikelihood(\
            load_model_from_hdf5_group(group['conditional_fit_model']))
    
    def __call__(self, parameters, return_negative=False):
        """
        Gets the value of the loglikelihood at the given parameters.
        
        parameters: the parameter values at which to evaluate the likelihood
        return_negative: if true the negative of the loglikelihood is returned
                         (this is useful for times when the loglikelihood must
                         be maximized since scipy optimization functions only
                         deal with minimization
        
        returns: the value of this Loglikelihood (or its negative if indicated)
        """
        self.check_parameter_dimension(parameters)
        try:
            (recreation, conditional_mean, conditional_covariance) =\
                self.model(parameters, return_conditional_mean=True,\
                return_conditional_covariance=True)
            weighted_bias = np.abs((self.data - recreation) / self.error)
            bias_term = np.sum(weighted_bias ** 2) / (-2.)
            covariance_term = (la.slogdet(conditional_covariance)[1] / 2)
            if type(self.prior) is type(None):
                prior_term = 0
            else:
                prior_term = self.prior.log_value(conditional_mean)
            logL_value = bias_term + covariance_term + prior_term
        except (ValueError, ZeroDivisionError):
            logL_value = -np.inf
        if np.isnan(logL_value):
            logL_value = -np.inf
        if return_negative:
            return -logL_value
        else:
            return logL_value
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this Loglikelihood can be
        computed. The gradient of this Loglikelihood is not computable.
        """
        return False
    
    def auto_gradient(self, pars, return_negative=False, differences=1e-6,\
        transform_list=None):
        """
        This function cannot be called on the
        ConditionalFitGaussianLoglikelihood class.
        """
        raise cannot_call_function_error
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this Loglikelihood can be
        computed. The hessian of this Loglikelihood is not computable.
        """
        return False
    
    def auto_hessian(self, pars, return_negative=False,\
        larger_differences=1e-5, smaller_differences=1e-6,\
        transform_list=None):
        """
        This function cannot be called on the
        ConditionalFitGaussianLoglikelihood class.
        """
        raise cannot_call_function_error
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        if isinstance(other, ConditionalFitGaussianLoglikelihood):
            return self.conditional_fit_model == other.conditional_fit_model
        else:
            return False
    
    def change_data(self, new_data):
        """
        Finds the ConditionalFitGaussianLoglikelihood with a different data
        vector with everything else kept constant.
        
        returns: a new ConditionalFitGaussianLoglikelihood with the given data
        """
        return ConditionalFitGaussianLoglikelihood(\
            self.conditional_fit_model.change_data(new_data))

