"""
File: TruncatedBasisHyperModel.py
Author: Keith Tauscher
Date: 29 Dec 2017

Description: A basis model where the number of terms used is decided by a
             parameter instead of being fixed.
"""
import numpy as np
from ..util import int_types, numerical_types
from ..basis import Basis
from ..fitter import Fitter
from .LoadableModel import LoadableModel
from .BasisModel import BasisModel

class TruncatedBasisHyperModel(LoadableModel):
    """
    A BasisModel where the number of terms used is decided by a parameter
    instead of being fixed.
    """
    def __init__(self, basis, min_terms=None, max_terms=None,\
        default_num_terms=None):
        """
        Initializes a new TruncatedBasisHyperModel with the given basis at its
        core.
        
        basis: the Basis object at the heart of this model
        min_terms: the minimum number of terms allowed
        max_terms: the maximum number of terms allowed
        default_num_terms: this is the number of terms used for the quick_fit
                           function. If None (default), max_terms is used
        """
        self.basis = basis
        self.min_terms = min_terms
        self.max_terms = max_terms
        self.default_num_terms = default_num_terms
    
    @property
    def basis(self):
        """
        Property storing the Basis object at the core of this model.
        """
        if not hasattr(self, '_basis'):
            raise AttributeError("basis referenced before it was set.")
        return self._basis
    
    @basis.setter
    def basis(self, value):
        """
        Setter for the basis property.
        
        value: must be a Basis object
        """
        if isinstance(value, Basis):
            self._basis = value
        else:
            raise TypeError("basis was set to something which was not a " +\
                "Basis object.")
    
    @property
    def expander(self):
        """
        Property which simply returns the expander property of the basis
        underlying this model.
        """
        return self.basis.expander
    
    @property
    def min_terms(self):
        """
        Property storing the minimum integer number of terms allowed.
        """
        if not hasattr(self, '_min_terms'):
            raise AttributeError("min_terms referenced before it was set.")
        return self._min_terms
    
    @min_terms.setter
    def min_terms(self, value):
        """
        Setter for the minimum number of terms allowed. It must be at least 1
        and at most basis.num_basis_vectors-1.
        
        value: if None, min_terms is set to 1
               otherwise, must be an integer satisfying
                          1<=value<basis.num_basis_vectors
        """
        if value is None:
            self._min_terms = 1
        elif type(value) in int_types:
            if value <= 0:
               raise ValueError("min_terms was set to a non-positive integer.") 
            elif value >= self.basis.num_basis_vectors:
                raise ValueError("min_terms was set to an integer greater " +\
                    "than or equal to the number of basis vectors in the " +\
                    "given basis.")
            else:
                self._min_terms = value
        else:
            raise TypeError("min_terms was sent to a non-int.")
    
    @property
    def max_terms(self):
        """
        Property storing the maximum integer number of terms allowed.
        """
        if not hasattr(self, '_max_terms'):
            raise AttributeError("max_terms referenced before it was set.")
        return self._max_terms
    
    @max_terms.setter
    def max_terms(self, value):
        """
        Setter for the maximum number of terms to allow.
        
        value: if None, max_terms is set to basis.num_basis_vectors
               otherwise, must be an int satisfying
                          min_terms<value<=basis.num_basis_vectors
        """
        if value is None:
            self._max_terms = self.basis.num_basis_vectors
        elif type(value) in int_types:
            if value < self.min_terms:
               raise ValueError("max_terms was set to an integer lower " +\
                   "than min_terms.")
            elif value == self.min_terms:
                raise ValueError("max_terms was set to the same integer as " +\
                    "min_terms. This doesn't make sense. You should use a " +\
                    "simple BasisModel instead if this is what you wanted.")
            elif value > self.basis.num_basis_vectors:
                raise ValueError("max_terms was set to an integer greater " +\
                    "than the number of basis vectors in the given basis.")
            else:
                self._max_terms = value
        else:
            raise TypeError("max_terms was sent to a non-int.")
    
    @property
    def default_num_terms(self):
        """
        Property storing the number of terms used in the quick_fit function.
        """
        if not hasattr(self, '_default_num_terms'):
            raise AttributeError("default_num_terms was referenced before " +\
                "it was set.")
        return self._default_num_terms
    
    @default_num_terms.setter
    def default_num_terms(self, value):
        """
        Setter for the number of terms 
        
        value: if None (default for __init__), set to max_terms
               otherwise, integer between min_terms and max_terms (inclusive)
        """
        if value is None:
            self._default_num_terms = self.max_terms
        elif isinstance(value, int):
            if (value >= self.min_terms) and (value <= self.max_terms):
                self._default_num_terms = value
            else:
                raise ValueError("default_num_terms was set to a number " +\
                    "outside of [min_terms,max_terms].")
        else:
            raise TypeError("default_num_terms was set to neither None nor " +\
                "an integer.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters =\
                ['a{}'.format(iterm) for iterm in range(self.max_terms)]
            self._parameters.append('nterms')
        return self._parameters
    
    def __call__(self, parameters):
        """
        Gets the curve associated with the given parameters. Warning: if nterms
        is not an integer, it is cast to one (which essentially means the floor
        is taken of the float.
        
        parameters: must be an array of size self.max_terms+1 where the last
                    parameter is the number of the terms and the other
                    parameters are the coefficients of the basis vectors of the
                    terms
        
        returns: array of size (num_channels)
        """
        terms = int(parameters[-1])
        if np.sign(terms - self.min_terms) == np.sign(terms - self.max_terms):
            raise ValueError("min_terms and max_terms were on the same " +\
                "side of terms, implying that terms does not satisfy " +\
                "min_terms<=terms<=max_terms.")
        return np.dot(parameters[:terms], self.basis.expanded_basis[:terms])
    
    def __getitem__(self, key):
        """
        Gets the simple BasisModel object which keeps the number of terms
        constant at key.
        
        key: must be an integer satisfying self.min_terms<=key<=self.max_terms
        
        returns: a BasisModel object with number of parameters given by key
        """
        if type(key) in int_types:
            if np.sign(key - self.min_terms) == np.sign(key - self.max_terms):
                raise ValueError("min_terms and max_terms were on the same " +\
                    "side of key, implying that key does not satisfy " +\
                    "min_terms<=key<=max_terms.")
            else:
                return BasisModel(self.basis[:key])
        else:
            raise TypeError("Submodel can only be taken when the given key " +\
                "is an integer.")
    
    @property
    def gradient_computable(self):
        """
        A gradient normally implies continuity in the parameters. This model
        has integer parameters so the gradient and hessian are not calculable.
        """
        return False
    
    def gradient(self, parameters):
        """
        Since gradient_computable is False, this function simply throws a
        NotImplementedError.
        """
        raise NotImplementedError("The gradient doesn't exist!")
    
    @property
    def hessian_computable(self):
        """
        A gradient normally implies continuity in the parameters. This model
        has integer parameters so the gradient and hessian are not calculable.
        """
        return False
    
    def hessian(self, parameters):
        """
        Since hessian_computable is False, this function simply throws a
        NotImplementedError.
        """
        raise NotImplementedError("The hessian doesn't exist.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this model, allowing
        it to be loaded at a later time from disk.
        
        group: hdf5 group to fill with information about this model
        """
        group.attrs['class'] = 'TruncatedBasisHyperModel'
        self.basis.fill_hdf5_group(group.create_group('basis'))
        group.attrs['min_terms'] = self.min_terms
        group.attrs['max_terms'] = self.max_terms
        group.attrs['default_num_terms'] = self.default_num_terms
    
    def __eq__(self, other):
        """
        Checks for equality between other and this TruncatedBasisHyperModel
        
        other: object to check for equality
        
        returns: False unless other is a TruncatedBasisHyperModel with the same
                 basis and the same min_terms and max_terms
        """
        if not isinstance(other, TruncatedBasisHyperModel):
            return False
        if self.basis != other.basis:
            return False
        if self.min_terms != other.min_terms:
            return False
        if self.max_terms != other.max_terms:
            return False
        return (self.default_num_terms == other.default_num_terms)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        basis = Basis.load_from_hdf5_group(group['basis'])
        min_terms = group.attrs['min_terms']
        max_terms = group.attrs['max_terms']
        if 'default_num_terms' in group.attrs:
            default_num_terms = group.attrs['default_num_terms']
        else:
            default_num_terms = None
        return TruncatedBasisHyperModel(basis, min_terms=min_terms,\
            max_terms=max_terms, default_num_terms=default_num_terms)
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit of this model to the given data with (or without)
        a given noise level.
        
        data: 1D array to fit with this constant model
        error: if None, the unweighted least square fit is given for
                        parameter_mean and parameter_covariance will be
                        nonsense
               otherwise, error should either be a single number or a 1D array
                          of same length as data
        
        returns: (parameter_mean, parameter_covariance) where parameter_mean is
                 a length N (number of basis vectors) 1D array and
                 parameter_covariance is a 2D array of shape (N,N). If no error
                 is given, parameter_covariance doesn't really mean anything
                 (especially if error is far from 1 in magnitude)
        """
        if error is None:
            error = 1
        if type(error) in numerical_types:
            error = error * np.ones_like(data)
        fitter = Fitter(self.basis[:self.max_terms], data, error=error)
        mean = np.concatenate([fitter.parameter_mean, [self.max_terms]])
        covariance = fitter.parameter_covariance
        column_to_add = np.zeros((covariance.shape[0], 1))
        covariance = np.concatenate([covariance, column_to_add], axis=-1)
        row_to_add =\
            (np.arange(covariance.shape[-1]) == (covariance.shape[-1] - 1))
        covariance =\
            np.concatenate([covariance, row_to_add[None,:]], axis=0)
        fitter = Fitter(self.basis[:self.default_num_terms], date, error=error)
        mean = np.concatenate([fitter.parameter_mean,\
            np.zeros(len(mean) - len(fitter.parameter_mean) - 1),\
            [self.default_num_terms]])
        return (mean, covariance)

