"""
File: pylinex/basis/TrainedBasis.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing Basis subclass whose basis vectors are calculated
             through (weighted) SVD.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from ..util import real_numerical_types, sequence_types, bool_types
from ..expander import NullExpander
from .Basis import Basis

def weighted_SVD(matrix, error=None, full_matrices=False):
    """
    Finds the most important modes of the given matrix given the weightings
    given by the error.
    
    matrix a horizontal rectangular matrix
    error weighting applied to the dimension corresponding to the rows
    """
    if type(error) is type(None):
        error = np.ones(matrix.shape[0])
    expanded_error = error[:,np.newaxis]
    to_svd = matrix / expanded_error
    (SVD_U, SVD_S, SVD_V_transpose) =\
        la.svd(to_svd, full_matrices=full_matrices)
    SVD_U = SVD_U * expanded_error
    return SVD_U, SVD_S, SVD_V_transpose.T

def weighted_SVD_basis(curves, error=None, Neigen=None):
    """
    Finds a basis using weighted SVD performed on the given curves.
    
    curves: 2D numpy.ndarray of curves with which to define the basis
    error: if None, no weights are used
           otherwise, it should be a 1D numpy.ndarray of the same length as the
                      curves (Nchannel)
    Neigen: number of basis vectors to return
    
    returns: 2D numpy.ndarray of shape (Neigen, Nchannel)
    """
    if type(error) is type(None):
        error = np.ones(curves.shape[-1])
    if type(Neigen) is type(None):
        Neigen = curves.shape[-1]
    SVD_U, SVD_S, SVD_V = weighted_SVD(curves.T, error=error)
    return (SVD_U.T[:Neigen], SVD_S, SVD_V.T[:Neigen])

class TrainedBasis(Basis):
    """
    Class which derives a basis from a training set using weighted Singular
    Value Decomposition (SVD).
    """
    def __init__(self, training_set, num_basis_vectors, error=None,\
        expander=None, mean_translation=False):
        """
        Initializes this TranedBasis using the given training_set and number of
        basis vectors.
        
        training_set: numpy.ndarray of shape (ncurves, nchannels)
        num_basis_vectors: number of basis vectors to keep from SVD
        error: if None, no weighting is used
               if error is a single number, it is set to a constant array with
                                            that value. This will yield the
                                            same basis vectors as setting error
                                            to None but will affect things like
                                            the scaling of the RMS spectrum
               if error is a numpy.ndarray, it should be 1D and of length
                                            nchannels (i.e. expanded training
                                            set curve length)
        expander: Expander object which expands from training set space to
                  final basis vector space
        mean_translation: if True (default False), the mean is subtracted from
                          the training set before SVD is computed. The mean is
                          then stored in the translation property of the basis.
                          This argument can also be given as an array so that
                          the translation can be specifically given by user
        """
        self._training_set_curve_length = training_set.shape[-1]
        self.expander = expander
        self.error = error
        if type(mean_translation) in bool_types:
            if mean_translation:
                translation = np.mean(training_set, axis=0)
            else:
                translation = np.zeros(self.training_set_curve_length)
        else:
            translation = mean_translation
        SVD_basis =\
            weighted_SVD_basis(training_set - translation[np.newaxis,:],\
            error=self.error, Neigen=num_basis_vectors)
        self.basis = SVD_basis[0]
        self.translation = translation
        self.training_set_space_singular_vectors = SVD_basis[2]
        self.full_importances = SVD_basis[1]
    
    @property
    def training_set_curve_length(self):
        """
        Property storing the length of the training set curves that made this
        basis.
        """
        if not hasattr(self, '_training_set_curve_length'):
            raise AttributeError("training_set_curve_length was referenced " +\
                "before it was set.")
        return self._training_set_curve_length
    
    @property
    def error(self):
        """
        Property storing the error used in computing SVD for this basis.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property (error will be contracted to size of
        training set curves inside this function).
        
        value: 1D array of same length as expanded training set curves
        """
        expanded_space_size =\
            self.expander.expanded_space_size(self.training_set_curve_length)
        if type(value) is type(None):
            value = np.ones(expanded_space_size)
        elif type(value) in real_numerical_types:
            value = value * np.ones(expanded_space_size)
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape != (expanded_space_size,):
                raise ValueError("error did not have the same shape as an " +\
                    "expanded training set curve.")
        else:
            raise TypeError("error was set to neither None, a number, or " +\
                "an array.")
        if np.all(value > 0):
            self._error = self.expander.contract_error(value)
        else:
            raise ValueError("At least one value in the error array was " +\
                "negative.")
    
    @property
    def training_set_space_singular_vectors(self):
        """
        Finds the right singular vectors (i.e. the ones that don't exist in
        data channel space) of the training set.
        """
        if not hasattr(self, '_training_set_space_singular_vectors'):
            raise AttributeError("training_set_space_singular_vectors were " +\
                                 "referenced before they were set.")
        return self._training_set_space_singular_vectors
    
    @training_set_space_singular_vectors.setter
    def training_set_space_singular_vectors(self, value):
        """
        Setter for the training_set_space_singular_vectors property.
        
        value: should be a 2D numpy.ndarray of shape (n_basis_vectors, n_curve)
               where each row is a unit vector in n_curve dimensional space
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[0] == self.num_basis_vectors:
                self._training_set_space_singular_vectors = value
            else:
                raise ValueError("The number of " +\
                                 "training_set_space_singular_vectors was " +\
                                 "not the same as the number of basis " +\
                                 "functions.")
        else:
            raise ValueError("training_set_space_singular_vectors was not " +\
                             "a 2D numpy.ndarray.")
    
    @property
    def num_curves(self):
        """
        Property storing the number of training set curves used to seed this
        SVD basis.
        """
        if not hasattr(self, '_num_curves'):
            self._num_curves =\
                self.training_set_space_singular_vectors.shape[1]
        return self._num_curves
    
    @property
    def training_set_size(self):
        """
        Property storing the total number of numbers supplied in the training
        set.
        """
        if not hasattr(self, '_training_set_size'):
            self._training_set_size =\
                self.num_curves * self.num_smaller_channel_set_indices
        return self._training_set_size
    
    @property
    def summed_training_set_space_singular_vectors(self):
        """
        Property storing the training set space singular vectors summed over
        the ncurve-dimensional axis. This quantity appears in the priors.
        """
        if not hasattr(self, '_summed_training_set_space_singular_vectors'):
            self._summed_training_set_space_singular_vectors =\
                np.sum(self.training_set_space_singular_vectors, axis=1)
        return self._summed_training_set_space_singular_vectors
    
    @property
    def full_importances(self):
        """
        Property storing the full importances array (including modes that are
        not in this basis).
        """
        if not hasattr(self, '_full_importances'):
            raise AttributeError("full_importances was referenced before " +\
                "it was set.")
        return self._full_importances
    
    @full_importances.setter
    def full_importances(self, value):
        """
        Setter for the full_importances property.
        
        value: 1D array of min(num_curves, num_channels) positive numbers in
               decreasing order
        """
        value = np.array(value)
        if value.shape ==\
            (min(self.num_curves, self.num_smaller_channel_set_indices),):
            if np.all(value >= 0):
                self._full_importances = value
            else:
                raise ValueError("Some of the importances given aren't " +\
                                 "nonnegative, which makes no sense.")
        else:
            raise ValueError("importances don't have the correct number of " +\
                "elements.")
    
    @property
    def importances(self):
        """
        Property storing a 1D numpy.ndarray giving the importances of the modes
        in this TrainedBasis.
        """
        if not hasattr(self, '_importances'):
            self._importances = self.full_importances[:self.num_basis_vectors]
        return self._importances
    
    @property
    def training_set_fit_coefficients(self):
        """
        Property storing the coefficients of the fit to each training set
        curve.
        """
        if not hasattr(self, '_training_set_fit_coefficients'):
            self._training_set_fit_coefficients =\
                self.training_set_space_singular_vectors.T *\
                self.importances[np.newaxis,:]
        return self._training_set_fit_coefficients
    
    @property
    def prior_mean(self):
        """
        Property storing the mean vector of the prior parameter distribution.
        It is the mean of the coefficient vectors when the basis is used to fit
        the training set.
        """
        if not hasattr(self, '_prior_mean'):
            self._prior_mean = (self.importances *\
                self.summed_training_set_space_singular_vectors) /\
                self.num_curves
        return self._prior_mean
    
    @property
    def prior_covariance(self):
        """
        Property storing the covariance matrix of the prior parameter
        distribution. It is the covariance of the coefficient vectors when the
        basis is used to fit the training set.
        """
        if not hasattr(self, '_prior_covariance'):
            self._prior_covariance =\
                self.summed_training_set_space_singular_vectors[:,np.newaxis]
            self._prior_covariance =\
                (self._prior_covariance * self._prior_covariance.T)
            self._prior_covariance = self._prior_covariance / self.num_curves
            self._prior_covariance =\
                np.identity(self.num_basis_vectors) - self._prior_covariance
            self._prior_covariance =\
                (self.importances[:,np.newaxis] * self._prior_covariance)
            self._prior_covariance =\
                (self.importances[np.newaxis,:] * self._prior_covariance)
            self._prior_covariance =\
                self._prior_covariance / (self.num_curves - 1)
        return self._prior_covariance
    
    @property
    def prior_inverse_covariance(self):
        """
        Property storing the inverse of the prior covariance matrix. See the
        prior_covariance property for more details.
        """
        if not hasattr(self, '_prior_inverse_covariance'):
            self._prior_inverse_covariance = la.inv(self.prior_covariance)
        return self._prior_inverse_covariance
    
    @property
    def prior_inverse_covariance_times_mean(self):
        """
        Property storing the vector given by the matrix multiplication of the
        prior inverse covariance by the prior mean.
        """
        if not hasattr(self, '_prior_inverse_covariance_times_mean'):
            self._prior_inverse_covariance_times_mean =\
                np.dot(self.prior_inverse_covariance, self.prior_mean)
        return self._prior_inverse_covariance_times_mean
    
    @property
    def gaussian_prior(self):
        """
        Property storing an ares.inference.Priors.GaussianPrior object
        describing the priors.
        """
        if not hasattr(self, '_gaussian_prior'):
            self.generate_gaussian_prior()
        return self._gaussian_prior
    
    def generate_gaussian_prior(self, covariance_expansion_factor=1.,\
        diagonal=False):
        """
        Generates a new Gaussian prior from the training set given an expansion
        factor of the covariance matrix.
        
        covariance_expansion_factor: positive number by which to multiply the
                                     prior covariance matrix
        diagonal: if True, all off-diagonal elements of the covariance matrix
                           are set to 0.
                  otherwise, off-diagonal elements are retained
        """
        cov_to_return = self.prior_covariance * covariance_expansion_factor
        if diagonal:
            cov_to_return = np.diag(np.diag(cov_to_return))
        self._gaussian_prior =\
            GaussianDistribution(self.prior_mean, cov_to_return)
    
    @property
    def normed_importances(self):
        """
        Property storing the importances of the modes in this TrainedBasis
        normalized by dividing by the sum of all importances.
        """
        if not hasattr(self, '_normed_importances'):
            self._normed_importances = self.importances / self.total_importance
        return self._normed_importances
    
    @property
    def total_importance(self):
        """
        Property storing the sum of the importances of all modes.
        """
        if not hasattr(self, '_total_importance'):
            self._total_importance = np.sum(self.full_importances)
        return self._total_importance
    
    @property
    def truncated_normed_importance_loss(self):
        """
        Property storing 1D numpy.ndarray containing values corresponding to
        the sum of importances of modes of higher order than the given index.
        """
        if not hasattr(self, '_truncated_normed_importance_loss'):
            self._truncated_normed_importance_loss =\
                1 - np.cumsum(self.normed_importances)
        return self._truncated_normed_importance_loss
    
    def terms_necessary_to_reach_noise_level_multiple(self, multiple):
        """
        Function that computes the approximate number of terms needed to reach
        the given multiple  noise level of the training set that sourced this
        basis. It is approximate because it is the number needed for the mean
        squared error across all channels and all training set curves to be
        equal to multiple**2. This allows some individual training set curves
        to be fit slightly worse than the noise level.
        
        multiple: multiple of noise level under consideration
        
        returns: 
        """
        if np.all(self.RMS_spectrum > multiple):
            return None
        else:
            return np.argmax(self.RMS_spectrum < multiple)
    
    @property
    def terms_necessary_to_reach_noise_level(self):
        """
        Property storing the approximate number of terms needed to reach the
        noise level of the training set that sourced this basis. It is
        approximate because it is the number needed for the mean squared error
        across all channels and all training set curves to be equal to 1. This
        allows some individual training set curves to be fit slightly worse
        than the noise level.
        """
        if not hasattr(self, '_terms_necessary_to_reach_noise_level'):
            self._terms_necessary_to_reach_noise_level =\
                self.terms_necessary_to_reach_noise_level_multiple(1)
        return self._terms_necessary_to_reach_noise_level
    
    def weighted_basis_similarity_statistic(self, basis):
        """
        Computes the weighted basis similarity statistic between this
        TrainedBasis and the given Basis. Note that this is different than the
        similarity statistic between the given Basis and this TrainedBasis.
        
        basis: a Basis object that will be used to fit curves similar to those
               in this training set
        
        returns: a statistic between 0 and 1 indicating a weighted similarity
                 based on the training set at the heart of this basis
        """
        GT_sqrt_Cinv = self.basis / self.error[np.newaxis,:]
        FT_sqrt_Cinv = basis.basis / self.error[np.newaxis,:]
        GT_Cinv_F = np.dot(GT_sqrt_Cinv, FT_sqrt_Cinv.T)
        _FT_Cinv_F_inv = la.inv(np.dot(FT_sqrt_Cinv, FT_sqrt_Cinv.T))
        GT_Cinv_F__FT_Cinv_F_inv__FT_Cinv_G = np.dot(GT_Cinv_F,\
            np.dot(_FT_Cinv_F_inv, GT_Cinv_F.T))
        numerator_mu_term = np.dot(self.prior_mean,\
            np.dot(GT_Cinv_F__FT_Cinv_F_inv__FT_Cinv_G, self.prior_mean))
        denominator_mu_term = np.sum(np.power(self.prior_mean, 2))
        numerator_sigma_term = np.trace(\
            np.dot(self.prior_covariance, GT_Cinv_F__FT_Cinv_F_inv__FT_Cinv_G))
        denominator_sigma_term = np.trace(self.prior_covariance)
        numerator = numerator_mu_term + numerator_sigma_term
        denominator = denominator_mu_term + denominator_sigma_term
        return numerator / denominator
    
    @property
    def RMS_spectrum(self):
        """
        Property storing the spectrum of normalized RMS values when the
        training set at the heart of this basis is fit by this basis.
        """
        if not hasattr(self, '_RMS_spectrum'):
            cumulative_squared_importance_loss = np.cumsum(np.concatenate(\
                [[0], np.power(self.full_importances[-1::-1], 2)]))[-1::-1]
            cumulative_squared_importance_loss =\
                cumulative_squared_importance_loss[:1+self.num_basis_vectors]
            self._RMS_spectrum = np.sqrt(cumulative_squared_importance_loss /\
                self.training_set_size)
        return self._RMS_spectrum
    
    def plot_RMS_spectrum(self, threshold=1, ax=None, show=False, title='',\
        fontsize=24, plot_reference_lines=True, **kwargs):
        """
        Plots the RMS values (in number of noise levels) achieved in the
        training set as a function of number of modes.
        
        show: if True, matplotlib.pyplot.show() is called before this function
                       is returned
        title: string title of the plot (can have LaTeX inside)
        kwargs: extra keyword arguments to pass on to matplotlib.pyplot.scatter
        """
        plot_xs = np.arange(1 + self.num_basis_vectors)
        if ax is None:
            fig = pl.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
        ax.scatter(plot_xs, self.RMS_spectrum, **kwargs)
        ylim = (10 ** int(np.log10(\
            np.min(self.RMS_spectrum[self.RMS_spectrum > 0])) - 1),\
            10 ** int(np.log10(np.max(self.RMS_spectrum)) + 1))
        if plot_reference_lines:
            ax.plot(plot_xs, np.ones_like(plot_xs) * threshold, color='k',\
                linestyle='--')
            ax.plot([plot_xs[np.argmax(self.RMS_spectrum < threshold)]] * 2,\
                ylim, color='k', linestyle='--')
        ax.set_xlim((plot_xs[0], plot_xs[-1]))
        ax.set_ylim(ylim)
        ax.set_yscale('log')
        ax.set_xlabel('# of modes', size=fontsize)
        ax.set_ylabel('Total RMS in noise levels', size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return ax

    def plot_importance_spectrum(self, normed=False,\
        plot_importance_loss=False, ax=None, show=False, title='',\
        fontsize=24, **kwargs):
        """
        Plots the spectrum of importances of the modes in this TrainedBasis.
        
        normed: if True, importances shown add to 1
        plot_importance_loss: if True, importance loss from truncation at
                                       various points is plotted
        show: if True, matplotlib.pyplot.show() is called before this function
                       is returned
        title: string title of the plot (can have LaTeX inside)
        kwargs: extra keyword arguments to pass on to matplotlib.pyplot.scatter
        """
        plot_xs = np.arange(self.num_basis_vectors)
        if ax is None:
            fig = pl.figure(figsize=(12,9))
            ax = fig.add_subplot(111)
        to_show = np.where(self.normed_importances > 0)[0]
        if plot_importance_loss:
            if normed:
                plot_ys = self.truncated_normed_importance_loss
                y_max = 1
                y_min = min(self.truncated_normed_importance_loss[to_show])
                ylabel = 'Normalized importance of modes above index'
            else:
                plot_ys = self.truncated_normed_importance_loss *\
                    self.total_importance
                y_max = self.total_importance
                y_min = min(self.truncated_normed_importance_loss[to_show]) *\
                     self.total_importance
                ylabel = 'Importance of modes above index'
        else:
            if normed:
                plot_ys = self.normed_importances
                y_min = min(self.normed_importances[to_show])
                y_max = max(self.normed_importances[to_show])
                ylabel = 'Normalized mode importance'
            else:
                plot_ys = self.importances
                y_min = min(self.importances[to_show])
                y_max = max(self.importances[to_show])
                ylabel = 'Mode importance'
        ax.scatter(plot_xs, plot_ys, **kwargs)
        ylim = (10 ** int(np.log10(y_min) - 1), 10 ** int(np.log10(y_max) + 1))
        ax.set_ylim(ylim)
        ax.set_yscale('log')
        ax.set_xlabel('Mode index', size=fontsize)
        ax.set_ylabel(ylabel, size=fontsize)
        ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return ax

