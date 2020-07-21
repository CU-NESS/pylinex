"""
File: pylinex/fitter/MAAFitter.py
Author: Keith Tauscher
Date: 15 Jul 2020

Description: File containing class which represents a Minimum Assumption
             Analysis (MAA) fitting class. It assumes there is a desired
             component and undesired component(s) in the data and that the
             model of the desired component is specified entirely by an
             Expander object while the undesired component is specified by a
             (possibly expanded with an Expander) basis vector set.
"""
from __future__ import division
import numpy as np
import numpy.linalg as la
from distpy import ChiSquaredDistribution
from ..util import Savable, Loadable, create_hdf5_dataset
from ..expander import Expander, load_expander_from_hdf5_group
from .BaseFitter import BaseFitter

class MAAFitter(BaseFitter, Savable, Loadable):
    """
    Class which represents a Minimum Assumption Analysis (MAA) fitting class.
    It assumes there is a desired component and undesired component(s) in the
    data and that the model of the desired component is specified entirely by
    an Expander object while the undesired component is specified by a
    (possibly expanded with an Expander) basis vector set.
    """
    def __init__(self, expander, basis_sum, data, error=None, **priors):
        """
        Initializes a new MAAFitter.
        
        expander: an Expander object that encodes the minimal assumption on the
                  desired component
        basis_sum: a BasisSum object (or a Basis object, which is converted
                   internally to a BasisSum of one Basis with the name 'sole')
                   that represents the undesired components
        data: 1D vector of same length as vectors in basis_sum or 2D
              numpy.ndarray of shape (ncurves, nchannels)
        error: 1D vector of same length as vectors in basis_sum containing only
               positive numbers
        **priors: keyword arguments where the keys are exactly the names of the
                  basis sets with '_prior' appended to them and the values are
                  GaussianDistribution objects. If only one basis is given as
                  the basis_sum, then the priors should either be empty or a
                  dictionary of the form {'sole_prior': gaussian_distribution}
        """
        self.basis_sum = basis_sum
        self.priors = priors
        self.data = data
        self.error = error
        self.expander = expander
    
    @property
    def expander(self):
        """
        Property storing the Expander object that encodes the minimal
        assumption on the desired component.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Setter for the property representing the Expander object encoding the
        minimal assumption on the desired component.
        
        value: an Expander object that is compatible with an
               expanded_space_size given by the length of the data and error
               vectors
        """
        if isinstance(value, Expander):
            try:
                self._num_desired_channels =\
                    value.original_space_size(self.num_channels)
            except:
                raise ValueError("expander is not compatible with an " +\
                    "expanded space size equal to the length of the basis " +\
                    "vectors, the data vector(s), and the error vector.")
            else:
                self._expander = value
        else:
            raise TypeError("expander was set to a non-Expander object.")
    
    @property
    def num_desired_channels(self):
        """
        Property storing the number of channels in the desired component.
        """
        if not hasattr(self, '_num_desired_channels'):
            raise AttributeError("num_desired_channels was referenced " +\
                "before expander was set.")
        return self._num_desired_channels
    
    @property
    def expansion_matrix(self):
        """
        Property storing the expansion matrix. This is the linear
        transformation that maps the desired component to its effect on the
        full data vector. It has shape (num_channels, num_desired_channels). It
        is represented mathematically as Psi.
        """
        if not hasattr(self, '_expansion_matrix'):
            self._expansion_matrix =\
                self.expander.expansion_matrix(self.num_desired_channels)
        return self._expansion_matrix
    
    @property
    def weighted_expansion_matrix(self):
        """
        Property storing the expansion matrix weighted by the error vector. It
        has the same shape as the expansion matrix. Represented mathematically
        as C^{-1/2} Psi
        """
        if not hasattr(self, '_weighted_expansion_matrix'):
            self._weighted_expansion_matrix =\
                self.expansion_matrix / self.error[:,np.newaxis]
        return self._weighted_expansion_matrix
    
    @property
    def expansion_gram_matrix(self):
        """
        Property storing the self-overlap of the expansion matrix. Represented
        mathematically as Psi^T C^{-1} Psi.
        """
        if not hasattr(self, '_expansion_gram_matrix'):
            self._expansion_gram_matrix =\
                np.dot(self.weighted_expansion_matrix.T,\
                self.weighted_expansion_matrix)
        return self._expansion_gram_matrix
    
    @property
    def expander_basis_overlap_matrix(self):
        """
        Property storing the overlap of the expansion matrix and the basis
        matrix. This is represented mathematically as Psi^T C^{-1} F.
        """
        if not hasattr(self, '_expander_basis_overlap_matrix'):
            self._expander_basis_overlap_matrix =\
                np.dot(self.weighted_basis, self.weighted_expansion_matrix).T
        return self._expander_basis_overlap_matrix
    
    @property
    def degrees_of_freedom(self):
        """
        Property storing the integer number of degrees of freedom left in
        residuals of this fit. This is the number of data channels minus the
        number of undesired component modes minus the number of channels in the
        unexpanded space of the desired component.
        """
        if not hasattr(self, '_degrees_of_freedom'):
            self._degrees_of_freedom = self.num_channels -\
                (self.basis_sum.num_basis_vectors + self.num_desired_channels)
        return self._degrees_of_freedom
    
    @property
    def inverse_undesired_only_covariance(self):
        """
        Property storing the inverse mode covariance of a fit that does not
        include the desired component.
        """
        if not hasattr(self, '_inverse_undesired_only_covariance'):
            self._inverse_undesired_only_covariance = self.basis_overlap_matrix
            if self.has_priors:
                self._inverse_undesired_only_covariance =\
                    self._inverse_undesired_only_covariance +\
                    self.prior_inverse_covariance
        return self._inverse_undesired_only_covariance
    
    @property
    def undesired_only_covariance(self):
        """
        Property storing the mode covariance of a fit that does not include the
        desired component.
        """
        if not hasattr(self, '_undesired_only_covariance'):
            self._undesired_only_covariance =\
                la.inv(self.inverse_undesired_only_covariance)
        return self._undesired_only_covariance
    
    @property
    def desired_only_covariance(self):
        """
        Property storing the covariance when fitting with only the desired
        component.
        """
        if not hasattr(self, '_desired_only_covariance'):
            self._desired_only_covariance = la.inv(self.expansion_gram_matrix)
        return self._desired_only_covariance
    
    @property
    def desired_only_variances(self):
        """
        Property storing the variances when fitting with only the desired
        component.
        """
        if not hasattr(self, '_desired_only_variances'):
            self._desired_only_variances =\
                np.diag(self.desired_only_covariance)
        return self._desired_only_variances
    
    @property
    def desired_only_error(self):
        """
        Property storing the error (i.e. standard deviations) when fitting with
        only the desired component.
        """
        if not hasattr(self, '_desired_only_error'):
            self._desired_only_error = np.sqrt(self.desired_only_variances)
        return self._desired_only_error
    
    @property
    def desired_noise_level(self):
        """
        Property referring to the same quantity as the desired_only_error
        property.
        """
        return self.desired_only_error
    
    @property
    def desired_only_correlation(self):
        """
        Property storing the channel correlation matrix when fitting with only
        the desired component.
        """
        if not hasattr(self, '_desired_only_correlation'):
            self._desired_only_correlation = self.desired_only_covariance /\
                (self.desired_only_error[np.newaxis,:] *\
                self.desired_only_error[:,np.newaxis])
        return self._desired_only_correlation
    
    @property
    def inverse_desired_covariance(self):
        """
        Property storing the inverse channel covariance of the desired
        component (the one described by the Expander given by initialization
        and not any of those described by the basis).
        """
        if not hasattr(self, '_inverse_desired_covariance'):
            self._inverse_desired_covariance = self.expansion_gram_matrix -\
                np.dot(self.expander_basis_overlap_matrix,\
                np.dot(self.undesired_only_covariance,\
                self.expander_basis_overlap_matrix.T))
        return self._inverse_desired_covariance
    
    @property
    def desired_covariance(self):
        """
        Property storing the channel covariance of the desired component (the
        one described by the Expander given by initialization and not any of
        those described by the basis).
        """
        if not hasattr(self, '_desired_covariance'):
            self._desired_covariance =\
                la.inv(self.inverse_desired_covariance)
        return self._desired_covariance
    
    @property
    def desired_variances(self):
        """
        Property storing the variances of the channels in the desired
        component's space.
        """
        if not hasattr(self, '_desired_variances'):
            self._desired_variances = np.diag(self.desired_covariance)
        return self._desired_variances
    
    @property
    def desired_error(self):
        """
        Property storing the standard deviations of the channels in the desired
        component's space.
        """
        if not hasattr(self, '_desired_error'):
            self._desired_error = np.sqrt(self.desired_variances)
        return self._desired_error
    
    @property
    def desired_correlation(self):
        """
        Property storing the channel correlation matrix of the desired
        component.
        """
        if not hasattr(self, '_desired_correlation'):
            self._desired_correlation = self.desired_covariance /\
                (self.desired_error[np.newaxis,:] *\
                self.desired_error[:,np.newaxis])
        return self._desired_correlation
    
    @property
    def undesired_covariance(self):
        """
        Property storing the covariance of the undesired modes (the ones
        defined by the basis).
        """
        if not hasattr(self, '_undesired_covariance'):
            self._undesired_covariance = np.dot(self.desired_covariance,\
                self.expander_basis_overlap_matrix)
            self._undesired_covariance =\
                np.dot(self.expander_basis_overlap_matrix.T,\
                self._undesired_covariance)
            self._undesired_covariance = np.dot(\
                self.undesired_covariance, self.undesired_only_covariance)
            self._undesired_covariance = np.dot(\
                self.undesired_only_covariance, self.undesired_covariance)
            self._undesired_covariance =\
                self.undesired_only_covariance + self._undesired_covariance
        return self._undesired_covariance
    
    @property
    def covariance_of_desired_and_undesired_components(self):
        """
        Property storing the covariance of desired component channels (first
        index) and undesired component modes (second index).
        """
        if not hasattr(self,\
            '_covariance_of_desired_and_undesired_components'):
            self._covariance_of_desired_and_undesired_components = (-1.) *\
                np.dot(self.desired_covariance,\
                np.dot(self.expander_basis_overlap_matrix,\
                self.undesired_only_covariance))
        return self._covariance_of_desired_and_undesired_components
    
    @property
    def overlap_of_shifted_data_with_expansion_matrix(self):
        """
        Property storing the overlap of the shifted data and the expansion
        matrix. Represented mathematically as Psi^T C^{-1} (y - F mu)
        """
        if not hasattr(self, '_overlap_of_shifted_data_with_expansion_matrix'):
            self._overlap_of_shifted_data_with_expansion_matrix = np.dot(\
                self.weighted_shifted_data, self.weighted_expansion_matrix)
        return self._overlap_of_shifted_data_with_expansion_matrix
    
    @property
    def overlap_of_shifted_data_with_basis(self):
        """
        Property storing the overlap of the shifted data and the basis matrix.
        Represented mathematically as F^T C^{-1} (y - F mu).
        """
        if not hasattr(self, '_overlap_of_shifted_data_with_basis'):
            self._overlap_of_shifted_data_with_basis =\
                np.dot(self.weighted_shifted_data, self.weighted_basis.T)
        return self._overlap_of_shifted_data_with_basis
    
    @property
    def desired_mean(self):
        """
        Property storing the channel mean of the desired component.
        """
        if not hasattr(self, '_desired_mean'):
            self._desired_mean = np.dot(self.undesired_only_covariance,\
                self.overlap_of_shifted_data_with_basis.T)
            self._desired_mean =\
                np.dot(self.expander_basis_overlap_matrix, self._desired_mean)
            self._desired_mean =\
                self.overlap_of_shifted_data_with_expansion_matrix.T -\
                self._desired_mean
            self._desired_mean =\
                np.dot(self.desired_covariance, self._desired_mean).T
        return self._desired_mean
    
    @property
    def expanded_desired_mean(self):
        """
        Property storing the desired mean in the expanded (data) space.
        """
        if not hasattr(self, '_desired_mean_channel_space'):
            self._desired_mean_channel_space =\
                np.dot(self.desired_mean, self.expansion_matrix.T)
        return self._desired_mean_channel_space
    
    @property
    def undesired_mode_mean(self):
        """
        Property storing the mean of the undesired modes (the ones described by
        the basis).
        """
        if not hasattr(self, '_undesired_mode_mean'):
            self._undesired_mode_mean =\
                self.overlap_of_shifted_data_with_basis.T
            self._undesired_mode_mean = self._undesired_mode_mean -\
                np.dot(self.expander_basis_overlap.T,\
                np.dot(self.desired_covariance,\
                self.overlap_of_shifted_data_with_expansion_matrix.T))
            self._undesired_mode_mean = self._undesired_mode_mean +\
                np.dot(self.expander_basis_overlap.T,\
                np.dot(self.desired_covariance,\
                np.dot(self.expander_basis_overlap,\
                np.dot(self.undesired_only_covariance,\
                self.overlap_of_shifted_data_with_basis.T))))
            self._undesired_mode_mean = np.dot(self.undesired_only_covariance,\
                self._undesired_mode_mean).T
            if self.has_priors:
                if self.multiple_data_curves:
                    self._undesired_mode_mean = self._undesired_mode_mean +\
                        self.prior_mean[np.newaxis,:]
                else:
                    self._undesired_mode_mean =\
                        self._undesired_mode_mean + self.prior_mean
        return self._undesired_mode_mean
    
    @property
    def undesired_channel_mean(self):
        """
        Property storing the mean of the undesired component in the space of
        the data.
        """
        if not hasattr(self, '_undesired_channel_mean'):
            self._undesired_channel_mean =\
                np.dot(self.undesired_mode_mean, self.basis_sum.basis)
        return self._undesired_channel_mean
    
    @property
    def data_reconstruction(self):
        """
        Property storing the full reconstruction of the data using both desired
        and undesired components.
        """
        if not hasattr(self, '_data_reconstruction'):
            self._data_reconstruction =\
                self.undesired_channel_mean + self.expanded_desired_mean
        return self._data_reconstruction
    
    @property
    def channel_bias(self):
        """
        Property storing the bias of the full data reconstruction, i.e.
        data-reconstruction.
        """
        if not hasattr(self, '_channel_bias'):
            self._channel_bias = self.data - self.data_reconstruction
        return self._channel_bias
    
    @property
    def weighted_channel_bias(self):
        """
        Property storing the weighted channel bias.
        """
        if not hasattr(self, '_weighted_channel_bias'):
            if self.multiple_data_curves:
                self._weighted_channel_bias =\
                    self.channel_bias / self.error[np.newaxis,:]
            else:
                self._weighted_channel_bias = self.channel_bias / self.error
        return self._weighted_channel_bias
    
    @property
    def chi_squared(self):
        """
        Property storing the non-reduced chi-squared value(s) associated with
        the fit(s) performed in this object.
        """
        if not hasattr(self, '_chi_squared'):
            self._chi_squared =\
                np.sum(self.weighted_channel_bias ** 2, axis=-1)
        return self._chi_squared
    
    @property
    def reduced_chi_squared(self):
        """
        Property storing the chi squared value(s) divided by the number of
        degrees of freedom. This should be around one, with "around" being
        defined by the error given in the reduced_chi_squared_error property.
        """
        if not hasattr(self, '_reduced_chi_squared'):
            self._reduced_chi_squared =\
                self.chi_squared / self.degrees_of_freedom
        return self._reduced_chi_squared
    
    @property
    def reduced_chi_squared_error(self):
        """
        Property storing the expected error of the reduced chi squared
        value(s).
        """
        if not hasattr(self, '_reduced_chi_squared_error'):
            self._reduced_chi_squared_error =\
                np.sqrt(2 / self.degrees_of_freedom)
        return self._reduced_chi_squared_error
    
    @property
    def reduced_chi_squared_expected_distribution(self):
        """
        Property storing the expected distribution (in the form of a distpy
        Distribution object) of the reduced chi squared value(s) from this
        fitter.
        """
        if not hasattr(self, '_reduced_chi_squared_expected_distribution'):
            self._reduced_chi_squared_expected_distribution =\
                ChiSquaredDistribution(self.degrees_of_freedom, reduced=True)
        return self._reduced_chi_squared_expected_distribution
    
    @property
    def desired_reduced_chi_squared_expected_distribution(self):
        """
        Property storing the expected distribution of the reduced chi squared
        statistic of the desired component.
        """
        if not hasattr(self,\
            '_desired_reduced_chi_squared_expected_distribution'):
            self._desired_reduced_chi_squared_expected_distribution =\
                ChiSquaredDistribution(self.num_desired_channels, reduced=True)
        return self._desired_reduced_chi_squared_expected_distribution
    
    def desired_reduced_chi_squared(self, desired_component):
        """
        Function which computes the reduced chi squared statistic of the
        desired component.
        
        desired_component: numpy array of shape (num_channels,) if there is a
                           single data vector or
                           (num_fits, num_desired_channels) if there are
                           num_fits data vectors
        
        returns: a single number if there is a single data vector or an array
                 of length num_fits if there are num_fits data vectors
        """
        bias = self.desired_mean - desired_component
        doubly_weighted_bias = np.dot(bias, self.inverse_desired_covariance)
        return np.sum(bias * doubly_weighted_bias, axis=-1) /\
            self.num_desired_channels
    
    def save_expander(self, root_group, expander_link=None):
        """
        Saves the Expander that describes the desired component.
        
        root_group: the group into which this fitter is being saved
        expander_link: link to existing saved Expander describing desired
                       component
        """
        try:
            create_hdf5_dataset(root_group, 'expander', link=expander_link)
        except ValueError:
            self.expander.fill_hdf5_group(root_group.create_group('expander'))
    
    @staticmethod
    def load_expander(root_group):
        """
        Loads the expander saved when a MAAFitter was saved.
        
        root_group: the group into which an MAAFitter was once saved
        
        returns: Expander object describing desired component
        """
        return load_expander_from_hdf5_group(root_group['expander'])
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None,\
        expander_link=None, basis_links=None, basis_expander_links=None,\
        prior_mean_links=None, prior_covariance_links=None):
        """
        Fills the given hdf5 group with information from this MAAFitter.
        
        group: the hdf5 file group to fill (only required argument)
        data_link: link to existing data dataset, if it exists (see
                   create_hdf5_dataset docs for info about accepted formats)
        error_link: link to existing error dataset, if it exists (see
                    create_hdf5_dataset docs for info about accepted formats)
        expander_link: link to existing saved Expander describing desired
                       component
        basis_links: list of links to basis functions saved elsewhere (see
                     create_hdf5_dataset docs for info about accepted formats)
        basis_expander_links: list of links to existing saved Expander (see
                        create_hdf5_dataset docs for info about accepted
                        formats)
        prior_mean_links: dict of links to existing saved prior means (see
                          create_hdf5_dataset docs for info about accepted
                          formats)
        prior_covariance_links: dict of links to existing saved prior
                                covariances (see create_hdf5_dataset docs for
                                info about accepted formats)
        """
        self.save_data(group, data_link=data_link)
        self.save_error(group, error_link=error_link)
        self.save_basis_sum(group, basis_links=basis_links,\
            expander_links=basis_expander_links)
        self.save_expander(group, expander_link=expander_link)
        self.save_priors(group, prior_mean_links=prior_mean_links,\
            prior_covariance_links=prior_covariance_links)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an MAAFitter from the given hdf5 group.
        
        group: h5py Group object in which MAAFitter was saved in the past
        
        returns: an MAAFitter object that was previously saved in this group
        """
        data = BaseFitter.load_data(group)
        error = BaseFitter.load_error(group)
        expander = MAAFitter.load_expander(group)
        basis_sum = BaseFitter.load_basis_sum(group)
        priors = BaseFitter.load_priors(group)
        return MAAFitter(expander, basis_sum, data, error=error, **priors)

