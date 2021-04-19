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
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
from distpy import GaussianDistribution, ChiSquaredDistribution
from ..util import Savable, Loadable, create_hdf5_dataset, real_numerical_types
from ..expander import Expander, load_expander_from_hdf5_group
from ..basis import TrainedBasis
from .BaseFitter import BaseFitter
from .Fitter import Fitter

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
                self._undesired_covariance, self.undesired_only_covariance)
            self._undesired_covariance = np.dot(\
                self.undesired_only_covariance, self._undesired_covariance)
            self._undesired_covariance =\
                self.undesired_only_covariance + self._undesired_covariance
        return self._undesired_covariance
    
    @property
    def inverse_undesired_covariance(self):
        """
        Property storing the inverse of the posterior undesired covariance.
        """
        if not hasattr(self, '_inverse_undesired_covariance'):
            self._inverse_undesired_covariance =\
                la.inv(self.undesired_covariance)
        return self._inverse_undesired_covariance
    
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
        if not hasattr(self, '_expanded_desired_mean'):
            self._expanded_desired_mean =\
                np.dot(self.desired_mean, self.expansion_matrix.T)
        return self._expanded_desired_mean
    
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
                np.dot(self.expander_basis_overlap_matrix.T,\
                np.dot(self.desired_covariance,\
                self.overlap_of_shifted_data_with_expansion_matrix.T))
            self._undesired_mode_mean = self._undesired_mode_mean +\
                np.dot(self.expander_basis_overlap_matrix.T,\
                np.dot(self.desired_covariance,\
                np.dot(self.expander_basis_overlap_matrix,\
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
            self._undesired_channel_mean = self.basis_sum.translation +\
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
        defined by the error given by the square root of the
        reduced_chi_squared_variance property.
        """
        if not hasattr(self, '_reduced_chi_squared'):
            self._reduced_chi_squared =\
                self.chi_squared / self.degrees_of_freedom
        return self._reduced_chi_squared
    
    @property
    def reduced_chi_squared_expected_mean(self):
        """
        Property storing the expected mean of the reduced chi squared value(s).
        """
        if not hasattr(self, '_reduced_chi_squared_expected_mean'):
            if self.has_priors:
                mean = np.sum(self.undesired_covariance *\
                    self.prior_inverse_covariance)
            else:
                mean = 0
            self._reduced_chi_squared_expected_mean =\
                (mean + self.degrees_of_freedom) / self.degrees_of_freedom
        return self._reduced_chi_squared_expected_mean
    
    @property
    def reduced_chi_squared_expected_variance(self):
        """
        Property storing the expected variance of the reduced chi squared
        value(s).
        """
        if not hasattr(self, '_reduced_chi_squared_variance'):
            if self.has_priors:
                variance = np.dot(self.prior_inverse_covariance,\
                    self.undesired_covariance)
                variance = np.sum(variance.T * variance)
            else:
                variance = 0
            self._reduced_chi_squared_expected_variance =\
                (2 * (variance + self.degrees_of_freedom)) /\
                (self.degrees_of_freedom ** 2)
        return self._reduced_chi_squared_expected_variance
    
    @property
    def reduced_chi_squared_expected_distribution(self):
        """
        Property storing the expected distribution (in the form of a distpy
        Distribution object) of the reduced chi squared value(s) from this
        fitter.
        """
        if not hasattr(self, '_reduced_chi_squared_expected_distribution'):
            if self.has_priors:
                self._reduced_chi_squared_expected_distribution =\
                    GaussianDistribution(\
                    self.reduced_chi_squared_expected_mean,\
                    self.reduced_chi_squared_expected_variance)
            else:
                self._reduced_chi_squared_expected_distribution =\
                    ChiSquaredDistribution(self.degrees_of_freedom,\
                    reduced=True)
        return self._reduced_chi_squared_expected_distribution
    
    @property
    def desired_reduced_chi_squared_expected_mean(self):
        """
        Property storing the expected mean of the desired reduced chi squared
        value(s).
        """
        if not hasattr(self, '_desired_reduced_chi_squared_expected_mean'):
            self._desired_reduced_chi_squared_expected_mean = 1
        return self._desired_reduced_chi_squared_expected_mean
    
    @property
    def desired_reduced_chi_squared_expected_variance(self):
        """
        Property storing the expected variance of the desired reduced chi
        squared value(s).
        """
        if not hasattr(self, '_desired_reduced_chi_squared_variance'):
            self._desired_reduced_chi_squared_expected_variance =\
                2 / self.num_desired_channels
        return self._desired_reduced_chi_squared_expected_variance
    
    @property
    def desired_reduced_chi_squared_expected_distribution(self):
        """
        Property storing the expected distribution of the reduced chi squared
        statistic of the desired component.
        """
        if not hasattr(self,\
            '_desired_reduced_chi_squared_expected_distribution'):
            self._desired_reduced_chi_squared_expected_distribution =\
                ChiSquaredDistribution(self.num_desired_channels,\
                reduced=True)
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
    
    @property
    def undesired_reduced_chi_squared_expected_mean(self):
        """
        Property storing the expected mean of the undesired reduced chi squared
        value(s).
        """
        if not hasattr(self, '_undesired_reduced_chi_squared_expected_mean'):
            self._undesired_reduced_chi_squared_expected_mean = 1
        return self._undesired_reduced_chi_squared_expected_mean
    
    @property
    def undesired_reduced_chi_squared_expected_variance(self):
        """
        Property storing the expected variance of the undesired reduced chi
        squared value(s).
        """
        if not hasattr(self, '_undesired_reduced_chi_squared_variance'):
            self._undesired_reduced_chi_squared_expected_variance =\
                2 / self.basis_sum.num_basis_vectors
        return self._undesired_reduced_chi_squared_expected_variance
    
    @property
    def undesired_reduced_chi_squared_expected_distribution(self):
        """
        Property storing the expected distribution of the reduced chi squared
        statistic of the undesired component.
        """
        if not hasattr(self,\
            '_undesired_reduced_chi_squared_expected_distribution'):
            self._undesired_reduced_chi_squared_expected_distribution =\
                ChiSquaredDistribution(self.basis_sum.num_basis_vectors,\
                reduced=True)
        return self._undesired_reduced_chi_squared_expected_distribution
    
    def undesired_reduced_chi_squared(self, undesired_component):
        """
        Function which computes the reduced chi squared statistic of the
        undesired component. This is given by (b^T S^{-1} b + d^T C^{-1} d)/N_F
        where b is the bias of the mean from the input in undesired coefficient
        space, S is undesired_covariance, d is the portion of the input that is
        unfittable with the basis, C is the noise covariance (given by error),
        and N_F is the number of basis vectors in the systematic basis.
        
        undesired_component: numpy array of shape (num_channels,) if there is a
                             single data vector or (num_fits, num_channels) if
                             there are num_fits data vectors
        
        returns: a single number if there is a single data vector or an array
                 of length num_fits if there are num_fits data vectors
        """
        fitter = Fitter(self.basis_sum, undesired_component, error=self.error)
        parameter_bias = self.undesired_mode_mean - fitter.parameter_mean
        doubly_weighted_parameter_bias =\
            np.dot(parameter_bias, self.inverse_undesired_covariance)
        parameter_bias_term =\
            np.sum(parameter_bias * doubly_weighted_parameter_bias, axis=-1)
        unfittable_component_term = fitter.chi_squared
        return (parameter_bias_term + unfittable_component_term) /\
            self.basis_sum.num_basis_vectors
    
    def plot_reduced_chi_squared_histogram(self, fig=None, ax=None,\
        figsize=(12,9), xlabel='', ylabel='', title='', fontsize=24,\
        show=False, **kwargs):
        """
        Plots a histogram of the reduced chi squared statistic.
        
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist
        figsize: size of figure to create if fig is None
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string label to use to title the plot
        fontsize: size of font for axis labels, tick labels, and title
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        kwargs: extra keyword arguments to pass to matplotlib.pyplot.hist
        
        returns: None if show is True, axes if show is False
        """
        if not self.multiple_data_curves:
            raise NotImplementedError("Cannot plot histogram of reduced " +\
                "chi squared because there is only one data curve and " +\
                "therefore only one chi squared value.")
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        kwargs['density'] = True
        kwargs['label'] = '$\chi^2$ values'
        ax.hist(self.reduced_chi_squared, **kwargs)
        xlim = ax.get_xlim()
        ax = self.reduced_chi_squared_expected_distribution.plot(\
            np.linspace(xlim[0], xlim[1], 1001)[1:], ax=ax, color='k',\
            fontsize=fontsize, xlabel=xlabel, ylabel=ylabel, title=title,\
            label='$\chi^2$ distribution')
        ax.set_xlim(xlim)
        ax.legend(fontsize=fontsize, frameon=False)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_desired_mean(self, nsigmas=1, alphas=0.5, which=None,\
        desired_component=None, channels=None, fig=None, ax=None,\
        figsize=(12,9), xlabel='', ylabel='', title='', fontsize=24,\
        show=False):
        """
        Plots means of the desired component. Requires the true desired
        component to be known.
        
        nsigmas: either a single number of sigma or a list of numbers of sigmas
        which: either an integer index or a slice (only used if there are
               multiple data curves fit by this object)
        desired_component: the true desired components to plot (if given),
                           should be 1D if there is only one curve being fit or
                           if which is an integer and should be the same shape
                           as self.desired_mean[which] otherwise
        channels: channels to use as x-values for plot.
                  If None (default), set to numpy.arange(num_desired_channels)
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist (one is used if 
        figsize: size of figure to create if fig is None
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string label to use to title the plot
        fontsize: size of font for axis labels, tick labels, and title
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        
        returns: if show is True, returns None
                 if show is False, returns axes on which plot was drawn
        """
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        if type(channels) is type(None):
            channels = np.arange(self.num_desired_channels)
        if self.multiple_data_curves and (type(which) is not type(None)):
            to_plot = self.desired_mean[which,:]
        else:
            to_plot = self.desired_mean
        if type(nsigmas) in real_numerical_types:
            nsigmas = [nsigmas]
        if type(alphas) in real_numerical_types:
            alphas = [alphas]
        if to_plot.ndim == 2:
            for index in range(len(to_plot)):
                if type(desired_component) is not type(None):
                    ax.plot(channels, desired_component[index], color='k',\
                        **({'label': 'inputs'} if (index == 0) else {}))
                ax.plot(channels, to_plot[index], color='C3',\
                    **({'label': 'means'} if (index == 0) else {}))
                for (nsigma, alpha) in zip(nsigmas, alphas):
                    if isinstance(nsigma, int):
                        label = '${:d}\sigma$ bands'.format(nsigma)
                    else:
                        label = '${:.1f}\sigma$ bands'.format(nsigma)
                    ax.fill_between(channels,\
                        to_plot[index] - (nsigma * self.desired_error),\
                        to_plot[index] + (nsigma * self.desired_error),\
                        color='C3', alpha=alpha, **({'label': label}\
                        if (index == 0) else {}))
        else:
            if type(desired_component) is not type(None):
                ax.plot(channels, desired_component, color='k', label='input')
            ax.plot(channels, to_plot, color='C3', label='mean')
            for (nsigma, alpha) in zip(nsigmas, alphas):
                if isinstance(nsigma, int):
                    label = '${:d}\sigma$ band'.format(nsigma)
                else:
                    label = '${:.1f}\sigma$ band'.format(nsigma)
                ax.fill_between(channels,\
                    to_plot - (nsigma * self.desired_error),\
                    to_plot + (nsigma * self.desired_error), color='C3',\
                    alpha=alpha, label=label)
        ax.legend(fontsize=fontsize, frameon=False)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.set_xlim((channels[0], channels[-1]))
        if show:
            pl.show()
        else:
            return ax
    
    def plot_desired_bias(self, desired_component, which=None,\
        plot_desired_error=True, plot_desired_noise_level=True, channels=None,\
        fig=None, ax=None, figsize=(12,9), xlabel='', ylabel='', title='',\
        fontsize=24, show=False, **kwargs):
        """
        Plots biases in the desired component. Requires the true desired
        component to be known.
        
        desired_component: the true desired components to subtract to compute
                           biases, should be 1D if there is only one curve
                           being fit or if which is an integer and should be
                           the same shape as self.desired_mean[which] otherwise
        which: either an integer index or a slice (only used if there are
               multiple data curves fit by this object)
        plot_desired_error: if True, +1 and -1 sigma uncertainties are plotted
        plot_desired_noise_level: if True, +1 and -1 sigma noise levels are
                                  plotted
        channels: channels to use as x-values for plot.
                  If None (default), set to numpy.arange(num_desired_channels)
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist
        figsize: size of figure to create if fig is None
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string label to use to title the plot
        fontsize: size of font for axis labels, tick labels, and title
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        kwargs: keyword arguments to pass onto matplotlib.pyplot.plot when
                plotting biases (not used when plotting desired error and
                desired noise level)
        
        returns: if show is True, returns None
                 if show is False, returns axes on which plot was drawn
        """
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        if type(channels) is type(None):
            channels = np.arange(self.num_desired_channels)
        if self.multiple_data_curves and (type(which) is not type(None)):
            bias = self.desired_mean[which,:] - desired_component
        else:
            bias = self.desired_mean - desired_component
        if bias.ndim == 2:
            ax.plot(channels, bias[0], label='biases', **kwargs)
            if len(bias) > 1:
                ax.plot(channels, bias[1:].T, **kwargs)
        else:
            ax.plot(channels, bias, label='bias', **kwargs)
        if plot_desired_noise_level:
            ax.plot(channels, self.desired_noise_level, color='C1',\
                label='$1\sigma$ noise level')
            ax.plot(channels, -self.desired_noise_level, color='C1')
        if plot_desired_error:
            ax.plot(channels, self.desired_error, color='C3',\
                label='$1\sigma$ error')
            ax.plot(channels, -self.desired_error, color='C3')
        ax.legend(fontsize=fontsize, frameon=False)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.set_xlim((channels[0], channels[-1]))
        if show:
            pl.show()
        else:
            return ax
    
    def plot_desired_reduced_chi_squared_histogram(self, desired_component,\
        fig=None, ax=None, figsize=(12,9), xlabel='', ylabel='', title='',\
        fontsize=24, show=False, **kwargs):
        """
        Plots a histogram of the reduced chi squared statistic on the desired
        component.
        
        desired_component: the true desired components of the data curves
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist 
        figsize: size of figure to create if fig is None
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string label to use to title the plot
        fontsize: size of font for axis labels, tick labels, and title
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        kwargs: extra keyword arguments to pass to matplotlib.pyplot.hist
        
        returns: None if show is True, axes if show is False
        """
        if not self.multiple_data_curves:
            raise NotImplementedError("Cannot plot histogram of desired " +\
                "reduced chi squared because there is only one data curve " +\
                "and therefore only one chi squared value.")
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        kwargs['density'] = True
        kwargs['label'] = '$\chi^2$ values'
        ax.hist(self.desired_reduced_chi_squared(desired_component), **kwargs)
        xlim = ax.get_xlim()
        ax = self.desired_reduced_chi_squared_expected_distribution.plot(\
            np.linspace(xlim[0], xlim[1], 1001)[1:], ax=ax, color='k',\
            fontsize=fontsize, xlabel=xlabel, ylabel=ylabel, title=title,\
            label='$\chi^2$ distribution')
        ax.set_xlim(xlim)
        ax.legend(fontsize=fontsize, frameon=False)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_desired_error(self, plot_desired_noise_level=False, fig=None,\
        ax=None, figsize=(12,9), channels=None, yscale='linear', xlabel='',\
        ylabel='', title='', fontsize=24, show=False, **kwargs):
        """
        Plots the error on the desired component.
        
        plot_desired_noise_level: if True, desired noise level
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist
        figsize: size of figure to create if fig is None
        channels: channels to use as x-values for plot.
                  If None (default), set to numpy.arange(num_desired_channels)
        yscale: 'linear', 'log', etc.
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string label to use to title the plot
        fontsize: size of font for axis labels, tick labels, and title
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        kwargs: keyword arguments to pass to matplotlib.pyplot.plot
        
        returns: None if show is True, axes if show is False
        """
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        if type(channels) is type(None):
            channels = np.arange(self.num_desired_channels)
        if plot_desired_noise_level:
            kwargs['linestyle'] = '-'
            kwargs['label'] = '$1\sigma$ uncertainty'
        ax.plot(channels, self.desired_error, **kwargs)
        if plot_desired_noise_level:
            kwargs['linestyle'] = '--'
            kwargs['label'] = '$1\sigma$ noise level'
            ax.plot(channels, self.desired_noise_level, **kwargs)
        ax.set_xlim((channels[0], channels[-1]))
        ax.set_xlabel(xlabel, size=fontsize)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        ax.set_yscale(yscale)
        if plot_desired_noise_level:
            ax.legend(fontsize=fontsize, frameon=False)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_desired_correlation(self, fig=None, ax=None, figsize=(12,9),\
        channel_lim=None, axis_label='', colorbar_label='', title='',\
        fontsize=24, major_tick_locator=None, minor_tick_locator=None,\
        show=False, **kwargs):
        """
        Plots the correlation matrix of the desired component.
        
        fig: figure to plot onto if it exists (only used if ax is None)
        ax: axes to plot onto if they exist
        figsize: size of figure to create if fig is None
        channel_lim: tuple of form (start, end) where start is the x-value of
                     the first channel and end is the x-value of the last
                     channel
        axis_label: string label to put on the x-axis and y-axis
        colorbar_label: string label to use on colorbar
        title: string label to use to title the plot
        major_tick_locator: either a tick Locator object or a number with which
                            to make a MultipleLocator object for major ticks
        minor_tick_locator: either a tick Locator object or a number with which
                            to make a MultipleLocator object for minor ticks
        fontsize: size of the fonts to use for title, axis and tick labels
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
              if False (default), axes on which plot was made was returned
        kwargs: keyword arguments to pass to matplotlib.pyplot.imshow. If
                'extent' kwarg is set and channel_lim is given, then 'extent'
                will be overwritten. 'origin' will be set to 'upper' regardless
                of its value here.
        
        returns: None if show is True, axes if show is False
        """
        if type(ax) is type(None):
            if type(fig) is type(None):
                fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        if type(channel_lim) is not type(None):
            (start, end) = channel_lim
            half_channel_width =\
                (end - start) / (2 * (self.num_desired_channels - 1))
            (start, end) =\
                (start - half_channel_width, end + half_channel_width)
            kwargs['extent'] = (start, end, end, start)
        kwargs['origin'] = 'upper'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -1
        if 'vmax' not in kwargs:
            kwargs['vmax'] = 1
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'coolwarm'
        image = ax.imshow(self.desired_correlation, **kwargs)
        colorbar = pl.colorbar(image)
        if type(major_tick_locator) is not type(None):
            if type(major_tick_locator) in real_numerical_types:
                major_tick_locator = MultipleLocator(major_tick_locator)
            ax.xaxis.set_major_locator(major_tick_locator)
            ax.yaxis.set_major_locator(major_tick_locator)
        if type(minor_tick_locator) is not type(None):
            if type(minor_tick_locator) in real_numerical_types:
                minor_tick_locator = MultipleLocator(minor_tick_locator)
            ax.xaxis.set_minor_locator(minor_tick_locator)
            ax.yaxis.set_minor_locator(minor_tick_locator)
        ax.set_xlabel(axis_label, size=fontsize)
        ax.set_ylabel(axis_label, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        colorbar.ax.tick_params(labelsize=24, width=2.5, length=7.5,\
            which='major')
        colorbar.ax.tick_params(labelsize=24, width=1.5, length=4.5,\
            which='minor')
        colorbar.ax.set_ylabel(colorbar_label, size=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
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
    

def MAA_bias_statistic_offsets(basis, training_set, target_expander, error,\
    return_means=True, return_maxima=True, return_trace_covariances=True):
    """
    Finds the offset of the bias statistic mean (over noise realizations) from
    1 for different numbers of terms. This is determined by the overlap between
    the basis vectors and the expansion matrix and by the amount of the
    training set curves that cannot be fit by the basis vectors. The means
    returned if return_means is True are the means of these quantities over all
    training set curves, while the maxima returned if return_maxima is True is
    the value of this mean for the worst training set curve. The
    trace_covariances returned if return_trace_covariances are the RMS ratios
    of desired covariance to desired only covariance.
    
    basis: the basis with which to fit the training set
    training_set: the nuisance curves that need to be fit in the presence of
                  the desired component which is described by the
                  target_expander argument. Should be an numpy array of shape
                  (num_curves, num_channels)
    target_expander: the expander that describes how the desired component
                     appears in the space of the training set
    error: the noise level that will exist in the data
    return_means: if True, means of bias statistic offsets over training set
                           curves are returned
    return_maxima: if True, bias statistic offsets for worst training set curve
                            are returned
    return_trace_covariances: if True, RMS ratio of desired covariance to
                                       desired only covariance is returned
    
    returns: ((means,) if return_means else ()) +
             ((maxima,) if return_maxima else ()) +
             ((trace_covariances,) if return_trace_covariances else ())
    """
    if not (return_means or return_maxima or return_trace_covariances):
        raise ValueError("At least one of return_means, return_maxima, and " +\
            "return_trace_covariances must be True; otherwise, this " +\
            "function wouldn't do anything!")
    possible_num_terms = 1 + np.arange(basis.num_basis_vectors)
    contracted_covariance = target_expander.contracted_covariance(error)
    if return_means or return_maxima:
        overlap = target_expander.overlap(training_set - basis.translation,\
            error=error)
        statistics =\
            np.mean(overlap * np.dot(overlap, contracted_covariance), axis=-1)
        means = [np.mean(statistics)]
        maxima = [np.max(statistics)]
    if return_trace_covariances:
        inverse_contracted_covariance = la.inv(contracted_covariance)
        trace_covariances = [np.mean(np.diag(\
            np.dot(contracted_covariance, inverse_contracted_covariance)))]
    for num_terms in possible_num_terms:
        fitter = MAAFitter(target_expander, basis[:num_terms], training_set,\
            error=error)
        if return_means or return_maxima:
            statistics =\
                np.mean(fitter.desired_mean * np.dot(fitter.desired_mean,\
                fitter.inverse_desired_covariance), axis=-1)
            means.append(np.mean(statistics))
            maxima.append(np.max(statistics))
        if return_trace_covariances:
            trace_covariances.append(np.mean(np.diag(np.dot(\
                fitter.desired_covariance, inverse_contracted_covariance))))
    if return_means or return_maxima:
        (means, maxima) = (np.array(means), np.array(maxima))
    if return_trace_covariances:
        trace_covariances = np.array(trace_covariances)
    return_value = ()
    if return_means:
        return_value = return_value + (means,)
    if return_maxima:
        return_value = return_value + (maxima,)
    if return_trace_covariances:
        return_value = return_value + (trace_covariances,)
    if len(return_value) == 1:
        return_value = return_value[0]
    return return_value

def MAA_self_offsets(training_set, target_expander, error, num_terms,\
    mean_translation=False, return_means=True, return_maxima=True,\
    return_trace_covariances=True,):
    """
    Same as MAA_bias_statistic_offsets function except the basis is generated
    through SVD on the training set.
    
    training_set: the nuisance curves that need to be fit in the presence of
                  the desired component which is described by the
                  target_expander argument. Should be an numpy array of shape
                  (num_curves, num_channels)
    target_expander: the expander that describes how the desired component
                     appears in the space of the training set
    error: the noise level that will exist in the data
    num_terms: the number of terms to get from SVD on the training set
    mean_translation: if True (default False), means are subtracted from
                      training set before SVD is taken and the mean is stored
                      as the basis' translation property
    return_means: if True, means of bias statistic offsets over training set
                           curves are returned
    return_maxima: if True, bias statistic offsets for worst training set curve
                            are returned
    return_trace_covariances: if True, RMS ratio of desired covariance to
                              desired only covariance is returned
    
    returns: ((means,) if return_means else ()) +
             ((maxima,) if return_maxima else ()) +
             ((trace_covariances,) if return_trace_covariances else ())
    """
    return MAA_bias_statistic_offsets(TrainedBasis(training_set, num_terms,\
        error=error, mean_translation=mean_translation), training_set,\
        target_expander, error, return_means=return_means,\
        return_maxima=return_maxima,\
        return_trace_covariances=return_trace_covariances)

def plot_training_set_MAA_quantities(training_set, target_expander, error,\
    num_terms, mean_translation=False, fig=None, figsize=(12, 9), ax=None,\
    fontsize=24, show=False):
    """
    Plots the quantities returned by the MAA_self_offsets function.
    
    training_set: the nuisance curves that need to be fit in the presence of
                  the desired component which is described by the
                  target_expander argument. Should be an numpy array of shape
                  (num_curves, num_channels)
    target_expander: the expander that describes how the desired component
                     appears in the space of the training set
    error: the noise level that will exist in the data
    num_terms: the number of terms to get from SVD on the training set
    mean_translation: if True (default False), means are subtracted from
                      training set before SVD is taken and the mean is stored
                      as the basis' translation property
    fig: the Figure object on which to make axes
    figsize: size of figure to create if fig is None
    ax: Axes object on which to plot quantities
    fontsize: size of font for labels and ticks
    
    returns: (() if show else (ax,)) +\
             (means, maxima, MS_spectrum, trace_covariances)
    """
    num_terms_array = np.arange(1 + num_terms)
    (means, maxima, trace_covariances) = MAA_self_offsets(training_set,\
        target_expander, error, num_terms, mean_translation=mean_translation,\
        return_means=True, return_maxima=True, return_trace_covariances=True)
    MS_spectrum = TrainedBasis(training_set, num_terms, error=error,\
        mean_translation=mean_translation).RMS_spectrum ** 2
    quantities = [means, maxima, MS_spectrum ** 2, trace_covariances]
    labels = ['mean bias statistic offset', 'maximum bias statistic offset',\
        'mean-square training set bias', 'mean-square uncertainty expansion']
    minimum = min([min(quantity[np.logical_and(np.isfinite(quantity),\
        quantity != 0)]) for quantity in quantities])
    maximum = max([max(quantity[np.logical_and(np.isfinite(quantity),\
        quantity != 0)]) for quantity in quantities])
    middle = np.sqrt(minimum * maximum)
    half_width = np.sqrt(maximum / minimum)
    buffer_percentage = 5
    new_half_width = np.power(half_width, 1 + (buffer_percentage / 100))
    new_minimum = middle / new_half_width
    new_maximum = middle * new_half_width
    ylim = (new_minimum, new_maximum)
    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['o', 'v', 'P', 'D']
    point_size = 30
    if type(fig) is type(None):
        fig = pl.figure(figsize=figsize)
    if type(ax) is type(None):
        ax = fig.add_subplot(111)
    for (quantity, label, color, marker) in\
        zip(quantities, labels, colors, markers):
        ax.scatter(num_terms_array, quantity, color=color, marker=marker,\
            s=point_size, label=label)
    ax.legend(fontsize=fontsize, frameon=False)
    ax.set_xlabel('# of terms', size=fontsize)
    ax.set_ylabel('Bias statistic mean offset / Squared # of $\sigma$',\
        size=fontsize)
    ax.set_title('Training set MAA statistics', size=fontsize)
    ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
    ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
    ax.set_yscale('log')
    ax.set_xlim((0, num_terms))
    ax.set_ylim(ylim)
    if show:
        pl.show()
        return (means, maxima, MS_spectrum, trace_covariances)
    else:
        return (ax, means, maxima, MS_spectrum, trace_covariances)

