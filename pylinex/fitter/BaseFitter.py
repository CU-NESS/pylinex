"""
File: pylinex/fitter/BaseFitter.py
Author: Keith Tauscher 
Date: 14 Jul 2020

Description: File containing base class containing common properties for any
             Fitter class, such as a basis, priors, and data and error vectors.
"""
import numpy as np
import scipy.linalg as scila
from distpy import GaussianDistribution
from ..util import create_hdf5_dataset, get_hdf5_value
from ..basis import Basis, BasisSum

class BaseFitter(object):
    """
    Base class containing common properties for any Fitter class, such as a
    basis, priors, and data and error vectors.
    """
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
    def num_channels(self):
        """
        Property storing the number of data channels in this fit. This should
        be the length of the data and error vectors.
        """
        return self.basis_sum.num_larger_channel_set_indices
    
    @property
    def sizes(self):
        """
        Property storing a dictionary with basis names as keys and the number
        of basis vectors in that basis as values.
        """
        if not hasattr(self, '_sizes'):
            self._sizes = self.basis_sum.sizes
        return self._sizes
    
    @property
    def names(self):
        """
        Property storing the names of the component Bases of the BasisSum.
        """
        if not hasattr(self, '_names'):
            self._names = self.basis_sum.names
        return self._names
    
    @property
    def data(self):
        """
        Property storing the 1D (2D) data vector (matrix) to fit.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data wasn't set before it was " +\
                                 "referenced. Something is wrong. This " +\
                                 "shouldn't happen.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Allows user to set the data property.
        
        value: must be a 1D numpy.ndarray with the same length as the basis
               vectors.
        """
        value = np.array(value)
        if value.ndim in [1, 2]:
            if value.shape[-1] == self.num_channels:
                self._data = value
            else:
                raise ValueError("data curve(s) did not have the same " +\
                                 "length as the basis functions.")
        else:
            raise ValueError("data was neither 1- or 2-dimensional.")
    
    @property
    def multiple_data_curves(self):
        """
        Property storing a bool describing whether this Fitter contains
        multiple data curves (True) or not (False).
        """
        if not hasattr(self, '_multiple_data_curves'):
            self._multiple_data_curves = (self.data.ndim == 2)
        return self._multiple_data_curves
    
    @property
    def error(self):
        """
        Property storing 1D error vector with which to weight the least square
        fit.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error wasn't set before it was " +\
                                 "referenced. Something is wrong. This " +\
                                 "shouldn't happen.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property.
        
        value: 1D vector of positive numbers with which to weight the fit
        """
        if type(value) is type(None):
            self._error = np.ones(self.num_channels)
        else:
            value = np.array(value)
            if value.shape == (self.num_channels,):
                self._error = value
            else:
                raise ValueError("error didn't have the same length as the " +\
                                 "basis functions.")
    
    @property
    def translated_data(self):
        """
        Property storing the data with the translation vector subtracted (i.e.
        the part of the data that will be fit with a linear model).
        """
        if not hasattr(self, '_translated_data'):
            self._translated_data = self.data - self.basis_sum.translation
        return self._translated_data
    
    @property
    def weighted_data(self):
        """
        Property storing the data vector weighted down by the error. This is
        the one that actually goes into the calculation of the mean and
        covariance of the posterior distribution.
        """
        if not hasattr(self, '_weighted_data'):
            if self.multiple_data_curves:
                self._weighted_data = self.data / self.error[np.newaxis,:]
            else:
                self._weighted_data = self.data / self.error
        return self._weighted_data
    
    @property
    def weighted_translated_data(self):
        """
        Property storing the weighted version of the data translated by the
        translation of the basis_sum.
        """
        if not hasattr(self, '_weighted_translated_data'):
            if self.multiple_data_curves:
                self._weighted_translated_data =\
                    self.translated_data / self.error[np.newaxis,:]
            else:
                self._weighted_translated_data =\
                    self.translated_data / self.error
        return self._weighted_translated_data
    
    @property
    def weighted_basis(self):
        """
        Property storing the basis functions of the basis set weighted down by
        the error. This is the one actually used in the calculations.
        """
        if not hasattr(self, '_weighted_basis'):
            self._weighted_basis =\
                self.basis_sum.basis / self.error[np.newaxis,:]
        return self._weighted_basis
    
    @property
    def basis_overlap_matrix(self):
        """
        Property storing the matrix of overlaps between the weighted basis
        vectors. It is an NxN numpy.ndarray where N is the total number of
        basis vectors.
        """
        if not hasattr(self, '_basis_overlap_matrix'):
            self._basis_overlap_matrix =\
                np.dot(self.weighted_basis, self.weighted_basis.T)
        return self._basis_overlap_matrix
    
    @property
    def has_priors(self):
        """
        Property storing whether or not priors will be used in the fit.
        """
        if not hasattr(self, '_has_priors'):
            self._has_priors = bool(self.priors)
        return self._has_priors
    
    @property
    def priors(self):
        """
        Property storing the priors provided to this Fitter at initialization.
        It should be a dictionary with keys of the form (name+'_prior') and
        values which are GaussianPrior objects.
        """
        if not hasattr(self, '_priors'):
            self._priors = {}
        return self._priors
    
    @priors.setter
    def priors(self, value):
        """
        Sets the priors property.
        
        value: should be a dictionary which is either a) empty, or b) filled
               with keys of the form (name+'_prior') and values which are
               GaussianPrior objects.
        """
        self._priors = value
        self._has_all_priors = False
        if self.has_priors:
            self._has_all_priors = True
            self._prior_mean = []
            self._prior_covariance = []
            self._prior_inverse_covariance = []
            for name in self.names:
                key = '{!s}_prior'.format(name)
                if key in self._priors:
                    self._prior_mean.append(\
                        self._priors[key].internal_mean.A[0])
                    self._prior_covariance.append(\
                        self._priors[key].covariance.A)
                    self._prior_inverse_covariance.append(\
                        self._priors[key].inverse_covariance.A)
                else:
                    nparams = self.basis_sum[name].num_basis_vectors
                    self._prior_mean.append(np.zeros(nparams))
                    self._prior_covariance.append(np.zeros((nparams, nparams)))
                    self._prior_inverse_covariance.append(\
                        np.zeros((nparams, nparams)))
                    self._has_all_priors = False
            self._prior_mean = np.concatenate(self._prior_mean)
            self._prior_covariance = scila.block_diag(*self._prior_covariance)
            self._prior_inverse_covariance =\
                scila.block_diag(*self._prior_inverse_covariance)
    
    @property
    def has_all_priors(self):
        """
        Property storing boolean describing whether all basis sets have priors.
        """
        if not hasattr(self, '_has_all_priors'):
            raise AttributeError("has_all_priors was referenced before it " +\
                "was set. It can't be referenced until the priors dict " +\
                "exists.")
        return self._has_all_priors
    
    @property
    def prior_mean(self):
        """
        Property storing the mean parameter vector of the prior distribution.
        It is a 1D numpy.ndarray with an element for each basis vector.
        """
        if not hasattr(self, '_prior_mean'):
            raise AttributeError("prior_mean was referenced before it was " +\
                                 "set. Something is wrong. This shouldn't " +\
                                 "happen.")
        return self._prior_mean
    
    @property
    def prior_channel_mean(self):
        """
        Property storing the prior mean in the space of the data. Represented
        mathematically as F mu.
        """
        if not hasattr(self, '_prior_channel_mean'):
            self._prior_channel_mean = self.basis_sum.translation +\
                np.dot(self.prior_mean, self.basis_sum.basis)
        return self._prior_channel_mean
    
    @property
    def weighted_prior_channel_mean(self):
        """
        Property storing the error-weighted channel mean. Represented
        mathematically as C^{-1/2} F mu
        """
        if not hasattr(self, '_weighted_prior_channel_mean'):
            self._weighted_prior_channel_mean =\
                self.prior_channel_mean / self.error
        return self._weighted_prior_channel_mean
    
    @property
    def weighted_shifted_data(self):
        """
        Property storing an error-weighted version of the data vector, shifted
        by the prior mean, represented mathematically as C^{-1/2} (y - F mu).
        """
        if not hasattr(self, '_weighted_shifted_data'):
            if self.has_priors:
                if self.multiple_data_curves:
                    self._weighted_shifted_data = self.weighted_data -\
                        self.weighted_prior_channel_mean[np.newaxis,:]
                else:
                    self._weighted_shifted_data =\
                        self.weighted_data - self.weighted_prior_channel_mean
            else:
                self._weighted_shifted_data = self.weighted_translated_data
        return self._weighted_shifted_data
    
    @property
    def prior_inverse_covariance(self):
        """
        Property storing the inverse covariance matrix of the prior
        distribution. It is a square 2D numpy.ndarray with diagonal elements
        for each basis vector.
        """
        if not hasattr(self, '_prior_inverse_covariance'):
            raise AttributeError("prior_inverse_covariance was referenced " +\
                                 "before it was set. Something is wrong. " +\
                                 "This shouldn't happen.")
        return self._prior_inverse_covariance
    
    @property
    def prior_inverse_covariance_times_mean(self):
        """
        Property storing the vector result of the matrix multiplication of the
        prior inverse covariance matrix and the prior mean vector. This
        quantity appears in the formula for the posterior mean parameter
        vector.
        """
        if not hasattr(self, '_prior_inverse_covariance_times_mean'):
            self._prior_inverse_covariance_times_mean =\
                np.dot(self.prior_inverse_covariance, self.prior_mean)
        return self._prior_inverse_covariance_times_mean
    
    @property
    def prior_covariance(self):
        """
        Property storing the prior covariance matrix. It is a square 2D
        numpy.ndarray with a diagonal element for each basis vector.
        """
        if not hasattr(self, '_prior_covariance'):
            raise AttributeError("prior_covariance was referenced before " +\
                                 "it was set. Something is wrong. This " +\
                                 "shouldn't happen.")
        return self._prior_covariance
    
    def save_data(self, root_group, data_link=None):
        """
        Saves data using the given root h5py.Group object.
        
        data_link: link to existing data dataset, if it exists (see
                   create_hdf5_dataset docs for info about accepted formats)
        """
        create_hdf5_dataset(root_group, 'data', data=self.data, link=data_link)
    
    @staticmethod
    def load_data(root_group):
        """
        Loads data saved when Fitter was saved.
        
        root_group: the group in which Fitter was saved
        
        returns: data vector(s) loaded
        """
        return get_hdf5_value(root_group['data'])
    
    def save_error(self, root_group, error_link=None):
        """
        Saves error using the given root h5py.Group object.
        
        error_link: link to existing error dataset, if it exists (see
                    create_hdf5_dataset docs for info about accepted formats)
        """
        create_hdf5_dataset(root_group, 'error', data=self.error,\
            link=error_link)
    
    @staticmethod
    def load_error(root_group):
        """
        Loads error saved when Fitter was saved.
        
        root_group: the group in which Fitter was saved
        
        returns: error vector loaded
        """
        return get_hdf5_value(root_group['error'])
    
    def save_basis_sum(self, root_group, basis_links=None,\
        expander_links=None):
        """
        Saves priors using the given root h5py.Group object.
        
        basis_links: list of links to basis functions saved elsewhere (see
                     create_hdf5_dataset docs for info about accepted formats)
        expander_links: list of links to existing saved Expander (see
                        create_hdf5_dataset docs for info about accepted
                        formats)
        """
        self.basis_sum.fill_hdf5_group(root_group.create_group('basis_sum'),\
            basis_links=basis_links, expander_links=expander_links)
    
    @staticmethod
    def load_basis_sum(root_group):
        """
        Loads BasisSum saved when Fitter was saved.
        
        root_group: the h5py.Group object in which the Fitter was saved
        
        returns: BasisSum object
        """
        return BasisSum.load_from_hdf5_group(root_group['basis_sum'])
    
    def save_priors(self, root_group, prior_mean_links=None,\
        prior_covariance_links=None):
        """
        Saves priors using the given root h5py.Group object.
        
        root_group: the h5py.Group object in which the Fitter was saved
        prior_mean_links: dict of links to existing saved prior means (see
                          create_hdf5_dataset docs for info about accepted
                          formats)
        prior_covariance_links: dict of links to existing saved prior
                                covariances (see create_hdf5_dataset docs for
                                info about accepted formats)
        """
        if self.has_priors:
            group = root_group.create_group('prior')
            if type(prior_mean_links) is type(None):
                prior_mean_links = {name: None for name in self.names}
            if type(prior_covariance_links) is type(None):
                prior_covariance_links = {name: None for name in self.names}
            for name in self.names:
                key = '{!s}_prior'.format(name)
                if key in self.priors:
                    subgroup = group.create_group(name)
                    self.priors[key].fill_hdf5_group(subgroup,\
                        mean_link=prior_mean_links[name],\
                        covariance_link=prior_covariance_links[name])
    
    @staticmethod
    def load_priors(root_group):
        """
        Loads priors dictionary saved when Fitter was saved.
        
        root_group: the h5py.Group object in which the Fitter was saved
        
        returns: dictionary of priors, with string keys and
                 GaussianDistribution values
        """
        priors = {}
        if 'prior' in root_group:
            group = root_group['prior']
            for name in group:
                key = '{!s}_prior'.format(name)
                subgroup = group[name]
                distribution =\
                    GaussianDistribution.load_from_hdf5_group(subgroup)
                priors[key] = distribution
        return priors

