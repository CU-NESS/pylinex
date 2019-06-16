"""
File: pylinex/loglikelihood/NonlinearTruncationLoglikelihood.py
Author: Keith Tauscher
Date: 29 Sep 2018

Description: File containing a class which represents a DIC-like loglikelihood
             which uses the number of coefficients to use in each of a number
             of bases as the parameters of the likelihood.
"""
import numpy as np
from distpy import Expression
from ..util import int_types, real_numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from ..basis import Basis, BasisSet
from ..fitter import Fitter
from ..model import TruncatedBasisHyperModel, CompositeModel
from .LoglikelihoodWithData import LoglikelihoodWithData
from .LoglikelihoodWithModel import LoglikelihoodWithModel

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class NonlinearTruncationLoglikelihood(LoglikelihoodWithModel):
    """
    Class which represents a DIC-like loglikelihood which uses the number of
    coefficients to use in each of a number of bases as the parameters of the
    likelihood.
    """
    def __init__(self, basis_set, data, error, expression,\
        parameter_penalty=1):
        """
        Initializes a new TruncationLoglikelihood with the given basis_sum,
        data, and error.
        
        basis_set: BasisSet objects containing basis with the largest number
                   of basis vectors allowed for each component
        data: 1D data vector to fit
        error: 1D vector of noise level estimates for data
        expression: Expression object which forms full model from submodels.
                    The ith submodel (with i starting at 0) should be
                    represented by {i} in the expression string
        parameter_penalty: the logL parameter penalty for adding a parameter in
                           any given model. Should be a non-negative constant.
                           It defaults to 1, which is the penalty used for the
                           Deviance Information Criterion (DIC)
        """
        self.basis_set = basis_set
        self.data = data
        self.error = error
        self.expression = expression
        self.parameter_penalty = parameter_penalty
        self.model =\
            CompositeModel(self.expression, self.basis_set.names, self.models)
    
    @property
    def basis_set(self):
        """
        Property storing the BasisSet object 
        """
        if not hasattr(self, '_basis_set'):
            raise AttributeError("basis_set was referenced before it was set.")
        return self._basis_set
    
    @basis_set.setter
    def basis_set(self, value):
        """
        Setter for the basis_set object.
        
        value: a BasisSet object
        """
        if isinstance(value, BasisSet):
            self._basis_set = value
        else:
            raise TypeError("basis_set was set to a non-BasisSet object.")
    
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
    
    @property
    def expression(self):
        """
        Property storing the Expression object which allows for the combination
        of all of the sets of basis vectors.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression was referenced before it was " +\
                "set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for the Expression object which allows for the combination of
        all of the sets of basis vectors.
        
        value: an Expression object which has as many arguments as the
               basis_set has names.
        """
        if isinstance(value, Expression):
            if value.num_arguments == len(self.basis_set.names):
                self._expression = value
            else:
                raise ValueError("expression had a different number of " +\
                    "arguments than the basis_set had sets of basis vectors.")
        else:
            raise TypeError("expression was set to a non-Expression object.")
    
    @property
    def parameter_penalty(self):
        """
        Property storing the penalty imposed on the log-likelihood when an
        extra parameter is included in any given model.
        """
        if not hasattr(self, '_parameter_penalty'):
            raise AttributeError("parameter_penalty was referenced before " +\
                "it was set.")
        return self._parameter_penalty
    
    @parameter_penalty.setter
    def parameter_penalty(self, value):
        """
        Setter for the penalty assessed when an extra parameter is included in
        any given model.
        
        value: a non-negative number
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._parameter_penalty = value
            else:
                raise ValueError("parameter_penalty was set to a negative " +\
                    "number.")
        else:
            raise TypeError("parameter_penalty was set to a non-number.")
    
    @property
    def models(self):
        """
        Property storing the underlying models which are combined into the
        composite model.
        """
        if not hasattr(self, '_models'):
            self._models = [TruncatedBasisHyperModel(self.basis_set[name])\
                for name in self.basis_set.names]
        return self._models
    
    def save_error(self, group, error_link=None):
        """
        Saves the error of this Loglikelihood object.
        
        group: hdf5 file group where information about this object is being
               saved
        error_link: link to where error is already saved somewhere (if it
                    exists)
        """
        create_hdf5_dataset(group, 'error', data=self.error, link=error_link)
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        error_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        """
        group.attrs['class'] = 'NonlinearTruncationLoglikelihood'
        self.save_data(group, data_link=data_link)
        self.save_error(group, error_link=error_link)
        self.basis_set.fill_hdf5_group(group.create_group('basis_set'))
        self.expression.fill_hdf5_group(group.create_group('expression'))
        group.attrs['parameter_penalty'] = self.parameter_penalty
    
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
            assert group.attrs['class'] == 'NonlinearTruncationLoglikelihood'
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "NonlinearTruncationLoglikelihood object.")
        data = LoglikelihoodWithData.load_data(group)
        error = NonlinearTruncationLoglikelihood.load_error(group)
        basis_set = BasisSet.load_from_hdf5_group(group['basis_set'])
        expression = Expression.load_from_hdf5_group(group['expression'])
        parameter_penalty = group.attrs['parameter_penalty']
        return NonlinearTruncationLoglikelihood(basis_set, data, error,\
            expression, parameter_penalty=parameter_penalty)
    
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
            logL_value =\
                np.sum(np.abs(self.weighted_bias(pars)) ** 2) / (-2.) -\
                (self.parameter_penalty * self.num_used_parameters(pars))
        except (ValueError, ZeroDivisionError):
            logL_value = -np.inf
        if np.isnan(logL_value):
            logL_value = -np.inf
        if return_negative:
            return -logL_value
        else:
            return logL_value
    
    def chi_squared(self, parameters):
        """
        Computes the (non-reduced) chi squared statistic. It should follow a
        chi squared distribution with the correct number of degrees of freedom.
        
        parameters: the parameter values at which to evaluate chi squared
        
        returns: single number statistic equal to the negative of twice the
                 loglikelihood
        """
        return ((-2.) * self(parameters, return_negative=False))
    
    def num_used_parameters(self, parameters):
        """
        Finds effective number of parameters given the given parameter vector.
        
        parameters: parameter vector at which to find the number of effective
                    parameters
        
        returns: integer number of effective parameters
        """
        return sum([int(round(parameters[index])) for (index, name) in\
            enumerate(self.parameters) if ('nterms' in name)])
    
    def chi_squared_z_score(self, parameters):
        """
        Computes the z-score of the chi squared value computed at the given
        parameters.
        
        parameters: the parameter values at which to evaluate chi squared
        
        returns: single value which should be roughly Gaussian with mean 0 and
                 stdv 1 if degrees_of_freedom is very large.
        """
        degrees_of_freedom =\
            self.num_channels - self.num_used_parameters(parameters)
        return (self.chi_squared(parameters) - degrees_of_freedom) /\
            np.sqrt(2 * degrees_of_freedom)
    
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
        degrees_of_freedom =\
            self.num_channels - self.num_used_parameters(parameters)
        return self.chi_squared(parameters) / degrees_of_freedom
    
    @property
    def gradient_computable(self):
        """
        Returns False because NonlinearTruncationLoglikelihood has some
        discrete and some continuous parameters.
        """
        return False
    
    def auto_gradient(self, *args, **kwargs):
        """
        Cannot compute the gradient of NonlinearTruncationLoglikelihood objects
        because they have some discrete parameters.
        """
        raise NotImplementedError("gradient cannot be computed for " +\
            "NonlinearTruncationLoglikelihood because some parameters are " +\
            "discrete.")
    
    @property
    def hessian_computable(self):
        """
        Returns False because NonlinearTruncationLoglikelihood has some
        discrete and some continuous parameters.
        """
        return False
    
    def auto_hessian(self, *args, **kwargs):
        """
        Cannot compute the hessian of NonlinearTruncationLoglikelihood objects
        because they have some discrete parameters.
        """
        raise NotImplementedError("hessian cannot be computed for " +\
            "NonlinearTruncationLoglikelihood because some parameters are " +\
            "discrete.")
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        if not isinstance(other, NonlinearTruncationLoglikelihood):
            return False
        if self.basis_set != other.basis_set:
            return False
        if not np.allclose(self.data, other.data):
            return False
        if not np.allclose(self.error, other.error):
            return False
        if self.expression != other.expression:
            return False
        return (self.parameter_penalty == other.parameter_penalty)
    
    def change_data(self, new_data):
        """
        Finds the NonlinearTruncationLoglikelihood with a different data vector
        with everything else kept constant.
        
        new_data: data to use for new NonlinearTruncationLoglikelihood object
        
        returns: a new NonlinearTruncationLoglikelihood with the given data
                 property
        """
        return NonlinearTruncationLoglikelihood(self.basis_set, new_data,\
            self.error, self.expression,\
            parameter_penalty=self.parameter_penalty)
    
    def change_model(self, new_model):
        """
        This function is not implemented for the
        NonlinearTruncationLoglikelihood class.
        """
        raise NotImplementedError("The NonlinearTruncationLoglikelihood " +\
            "class does not implement the change_model class because the " +\
            "model is internally defined.")

