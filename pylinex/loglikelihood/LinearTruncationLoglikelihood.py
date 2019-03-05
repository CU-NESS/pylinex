"""
File: pylinex/loglikelihood/LinearTruncationLoglikelihood.py
Author: Keith Tauscher
Date: 29 Sep 2018

Description: File containing a class which represents a DIC-like loglikelihood
             which uses the number of coefficients to use in each of a number
             of bases as the parameters of the likelihood.
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from ..basis import Basis, BasisSum
from ..fitter import Fitter
from .LoglikelihoodWithData import LoglikelihoodWithData

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class LinearTruncationLoglikelihood(LoglikelihoodWithData):
    """
    Class which represents a DIC-like loglikelihood which uses the number of
    coefficients to use in each of a number of bases as the parameters of the
    likelihood.
    """
    def __init__(self, basis_sum, data, error,\
        information_criterion='deviance_information_criterion'):
        """
        Initializes a new TruncationLoglikelihood with the given basis_sum,
        data, and error.
        
        basis_sum: BasisSum containing Basis objects which contain the largest
                   number of basis vectors allowed.
        data: 1D data vector to fit
        error: 1D vector of noise level estimates for data
        information_criterion: the name of the quantity returned by this
                               Loglikelihood. Must be a valid property of the
                               Fitter class from pylinex.
                               Default: 'deviance_information_criterion'
        """
        self.basis_sum = basis_sum
        self.data = data
        self.error = error
        self.information_criterion = information_criterion
    
    @property
    def information_criterion(self):
        """
        Property storing the string name of the information criterion to return
        when called (must be a valid property of the Fitter class from
        pylinex).
        """
        if not hasattr(self, '_information_criterion'):
            raise AttributeError("information_criterion was referenced " +\
                "before it was set.")
        return self._information_criterion
    
    @information_criterion.setter
    def information_criterion(self, value):
        """
        Setter for the information criterion returned by this Loglikelihood.
        
        value: string name of a valid property of the Fitter class from pylinex
        """
        if isinstance(value, basestring):
            self._information_criterion = value
        else:
            raise TypeError("information_criterion was set to a non-string.")
    
    @property
    def basis_sum(self):
        """
        Property storing the BasisSum object whose basis vectors will be
        used by this object in the fit.
        """
        if not hasattr(self, '_basis_sum'):
            raise AttributeError("basis_sum was referenced before it was " +\
                                 "set. This shouldn't happen. Something is " +\
                                 "wrong.")
        return self._basis_sum
    
    @basis_sum.setter
    def basis_sum(self, value):
        """
        Allows user to set basis_sum property.
        
        value: BasisSum object or, more generally, a Basis object containing
               the basis vectors with which to perform the fit
        """
        if isinstance(value, BasisSum):
            self._basis_sum = value
        elif isinstance(value, Basis):
            self._basis_sum = BasisSum('sole', value)
        else:
            raise TypeError("basis_sum was neither a BasisSum or a " +\
                            "different Basis object.")
    
    @property
    def names(self):
        """
        Property storing the names of the component bases at play with this
        Loglikelihood.
        """
        if not hasattr(self, '_names'):
            return [name for name in self.basis_sum.names]
        return self._names
    
    @property
    def nterms_maxima(self):
        """
        Property storing the maximum number of terms in each subbasis.
        """
        if not hasattr(self, '_nterms_maxima'):
            self._nterms_maxima =\
                np.array([self.basis_sum[name].num_basis_vectors\
                for name in self.names])
        return self._nterms_maxima
    
    def truncated_basis_sum(self, truncations):
        """
        Finds the BasisSum corresponding to the given truncations
        
        truncations: array of integers of same length as self.basis_sum_names
        
        returns: a BasisSum object corresponding to the given truncations.
        """
        new_bases = [self.basis_sum[name][:truncation]\
            for (name, truncation) in zip(self.names, truncations)]
        return BasisSum(self.names, new_bases)
    
    @property
    def error(self):
        """
        Property storing the 1D error vector for fit.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error wasn't set before it was " +\
                                 "referenced. Something is wrong. This " +\
                                 "shouldn't happen.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Allows user to set the error property.
        
        value: must be a 1D numpy.ndarray with the same length as the basis
               vectors.
        """
        value = np.array(value)
        if value.shape == (self.num_channels,):
            self._error = value
        else:
            raise ValueError(("error was not of the expected shape, " +\
                "({0:d},).").format(self.num_channels))
    
    @property
    def computed(self):
        """
        Property storing the loglikelihood's computed values.
        """
        if not hasattr(self, '_computed'):
            self._computed = {}
        return self._computed
    
    def hash(self, point):
        """
        Hashes the given point of integers to summarize it into a single
        integer.
        
        point: the point, consisting of integers, to hash into a single integer
        
        returns: single integer, summarizing point
        """
        return np.ravel_multi_index(point - 1, self.nterms_maxima)
    
    def unhash(self, hash_value):
        """
        Unhashes the given value to give the point which hashed to it.
        
        hash_value: the hash value to which the desired point maps
        
        returns: the point which maps to the given hash value
        """
        return np.array(np.unravel_index(hash_value, self.nterms_maxima)) + 1
    
    def __call__(self, point):
        """
        Calls this loglikelihood by evaluating the DIC achieved by the
        truncations specified by point.
        
        point: array of integers containing truncation numbers
        """
        point = point.astype(int)
        hash_value = self.hash(point)
        if hash_value not in self.computed:
            self.computed[hash_value] =\
                getattr(Fitter(self.truncated_basis_sum(point), self.data,\
                error=self.error), self.information_criterion) / (-2.)
        return self.computed[hash_value]
    
    @property
    def parameters(self):
        """
        Property storing the names of the parameters of the model defined by
        this likelihood.
        """
        if not hasattr(self, '_parameters'):
            self._parameters =\
                ['{!s}_nterms'.format(name) for name in self.names]
        return self._parameters
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        data_link: link to data, if applicable
        error_link: link to error, if applicable
        """
        group.attrs['class'] = 'TruncationLoglikelihood'
        group.attrs['information_criterion'] = self.information_criterion
        self.save_data(group, data_link=data_link)
        create_hdf5_dataset(group, 'error', data=self.error, link=error_link)
        self.basis_sum.fill_hdf5_group(group.create_group('basis_sum'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        group.attrs['class'] = 'LinearTruncationLoglikelihood'
        data = LoglikelihoodWithData.load_data(group)
        error = get_hdf5_value(group['error'])
        basis_sum = BasisSum.load_from_hdf5_group(group['basis_sum'])
        information_criterion = group.attrs['information_criterion']
        return TruncationLoglikelihood(basis_sum, data, error,\
            information_criterion=information_criterion)
    
    @property
    def gradient_computable(self):
        """
        The gradient of this Loglikelihood is not computable because it exists
        in a discrete parameter space.
        """
        return False
    
    def gradient(self, point):
        """
        Raises an error because the gradient of this loglikelihood doesn't
        exist.
        """
        raise NotImplementedError("gradient of this loglikelihood cannot " +\
            "be computed because it exists in a discrete parameter space.")
    
    @property
    def hessian_computable(self):
        """
        The hessian of this Loglikelihood is not computable because it exists
        in a discrete parameter space.
        """
        return False
    
    def hessian(self, point):
        """
        Raises an error because the hessian of this loglikelihood doesn't
        exist.
        """
        raise NotImplementedError("hessian of this loglikelihood cannot " +\
            "be computed because it exists in a discrete parameter space.")
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        if not isinstance(other, TruncationLoglikelihood):
            return False
        if self.basis_sum != other.basis_sum:
            return False
        if not np.allclose(self.data, other.data):
            return False
        if not np.allclose(self.error, other.error):
            return False
        return (self.information_criterion == other.information_criterion)
    
    def change_data(self, new_data):
        """
        Finds the LinearTruncationLoglikelihood with a different data vector
        with everything else kept constant.
        
        returns: new LinearTruncationLoglikelihood with the given data property
        """
        return LinearTruncationLoglikelihood(self.basis_sum, new_data,\
            self.error, information_criterion=self.information_criterion)
    
