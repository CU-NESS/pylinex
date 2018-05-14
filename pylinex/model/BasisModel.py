"""
File: pylinex/model/BasisModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class representing a model which can be
             encapsulated by a single matrix (i.e. one described entirely by an
             ordered set of basis vectors).
"""
import numpy as np
from ..basis import Basis
from ..fitter import Fitter
from .LoadableModel import LoadableModel

class BasisModel(LoadableModel):
    """
    Class representing a model which can be encapsulated by a single matrix
    (i.e. one described entirely by an ordered set of basis vectors).
    """
    def __init__(self, basis):
        """
        Initializes a BasisModel from the given Basis object.
        
        basis: Basis object on which to base this model
        """
        self.basis = basis
    
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
        Setter for the basis at the core of this model.
        
        value: must be a Basis object
        """
        if isinstance(value, Basis):
            self._basis = value
        else:
            raise TypeError("basis was set to a non-Basis object.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the outputs of this model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.basis.expanded_basis.shape[1]
        return self._num_channels
    
    @property
    def parameters(self):
        """
        Property storing the sequence of strings describing these parameters.
        They are of the form a* where * counts up from 0.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ['a{}'.format(ipar)\
                for ipar in range(self.basis.num_basis_vectors)]
        return self._parameters
    
    def __call__(self, pars):
        """
        Call this model on the given parameters. Returns the expanded output of
        the Basis at the core of this model.
        
        pars: 1D numpy.ndarray of parameters values
        
        returns: 1D numpy.ndarray of length num_channels
        """
        return self.basis(pars, expanded=True)
    
    @property
    def gradient_computable(self):
        """
        The gradient of a BasisModel is computable. It is the expanded basis
        matrix itself.
        
        returns: True
        """
        return True
    
    def gradient(self, pars):
        """
        The gradient of a BasisModel is simply the expanded basis matrix
        itself.
        
        pars: unused array of parameter values
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        return self.basis.expanded_basis.T
    
    @property
    def hessian_computable(self):
        """
        The hessian of a BasisModel is computable. It is simply 0.
        
        returns: True
        """
        return True
    
    def hessian(self, pars):
        """
        The hessian of a linear model is 0.
        
        pars: unused array of parameter values
        
        returns: numpy.ndarray of zeros of shape
                 (num_channels, num_parameters, num_parameters)
        """
        return np.zeros((self.num_channels,) + ((self.num_parameters,) * 2))
    
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
        fitter = Fitter(self.basis, data, error=error)
        return (fitter.parameter_mean, fitter.parameter_covariance)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with complete information about this
        BasisModel (which is just the Basis object at its core).
        
        group: hdf5 file group to fill with information about this BasisModel
        """
        group.attrs['class'] = 'BasisModel'
        self.basis.fill_hdf5_group(group.create_group('basis'))
    
    def __eq__(self, other):
        """
        Checks whether this BasisModel is equivalent to other.
        
        other: object to check for equality
        
        returns: True if other is a BasisModel with the same Basis at its core
        """
        return isinstance(other, BasisModel) and (self.basis == other.basis)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BasisModel from the given group. The load_from_hdf5_group of a
        given subclass model should always be called.
        
        group: the hdf5 file group from which to load the BasisModel
        
        returns: a BasisModel object
        """
        return BasisModel(Basis.load_from_hdf5_group(group['basis']))

