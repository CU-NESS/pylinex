"""
File: pylinex/basis/PartiallyTrainedBasis.py
Author: Neil Bassett
Update date: 3 May 2020

Description: File containing a Basis subclass whose basis vectors are a
             combination of fixed vectors that are supplied by the user and
             vectors calculated through (weighted) SVD.
"""
from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from ..util import real_numerical_types, sequence_types, bool_types
from ..expander import NullExpander
from .Basis import Basis, weighted_SVD, weighted_SVD_basis

class PartiallyTrainedBasis(Basis):
    """
    Class that defines a basis created through the combination of fixed vectors
    supplied by the user and vectors derived from performing (weighted)
    Singular Value Decomposition (SVD) on a training set.
    """
    def __init__(self, fixed_vectors, training_set, num_SVD_basis_vectors,\
        error=None, expander=None, mean_translation=False):
        """
        Initializes a PartiallyTrainedBasis using the given fixed_vectors and
        training_set.

        fixed_vectors: numpy.ndarray of shape (nvectors, nchannels) which
                       contains the fixed vectors to use for the basis
        training_set: numpy.ndarray of shape (ncurves, nchannels) which contains
                      the training set curves on which to perform SVD
        num_SVD_basis_vectors: number of basis vectors from SVD to use in the
                               basis
        error: if None, no weighting is used
               if error is a single number, it is set to a constant array with
                                            that value. This will yield the
                                            same basis vectors as setting error
                                            to None but will affect things like
                                            the scaling of the RMS spectrum
               if error is a numpy.ndarray, it should be 1D and of length
                                            nchannels (i.e. expanded training
                                            set curve length)
        expander: Expander object which expands from fixed vector/training set
                  space to final basis vector space
        mean_translation: if True (default False), the mean is subtracted from
                          the training set before SVD is computed. The mean is
                          then stored in the translation property of the basis.
                          This argument can also be given as an array so that
                          the translation can be specifically given by user
        """
        if fixed_vectors.shape[-1] != training_set.shape[-1]:
            raise ValueError("Each curve in fixed_vectors and training_set" +\
                "must have the same length (i.e. number of channels)")
        self._basis_vector_length = fixed_vectors.shape[-1]
        self.num_fixed_basis_vectors = fixed_vectors.shape[0]
        self.num_SVD_basis_vectors = num_SVD_basis_vectors
        self.expander = expander
        self.error = error
        self.fixed_basis = np.array(fixed_vectors)
        weighted_fixed_basis = self.fixed_basis / self.error[np.newaxis,:]
        inverse_self_overlap =\
                la.inv(np.dot(weighted_fixed_basis, weighted_fixed_basis.T))
        training_set_overlap =\
                np.dot(weighted_fixed_basis, training_set.T)
        fit_parameters = np.dot(inverse_self_overlap, training_set_overlap).T
        residual_training_set = training_set -\
                np.dot(fit_parameters, weighted_fixed_basis)
        mean_translated = False
        if type(mean_translation) in bool_types:
            if mean_translation:
                mean_translated = True
                translation = np.mean(residual_training_set, axis=0)
            else:
                translation = np.zeros(self.training_set_curve_length)
        else:
            translation = mean_translation
        self._mean_translated = mean_translated
        SVD_basis =\
            weighted_SVD_basis(residual_training_set -\
            translation[np.newaxis,:], error=self.error,\
            Neigen=num_SVD_basis_vectors)
        self.SVD_basis = SVD_basis[0]
        self.basis = self.fixed_basis + self.SVD_basis
        self.translation = translation
        self.training_set_space_singular_vectors = SVD_basis[2]
        self.full_importances = SVD_basis[1]

    @property
    def basis_vector_length(self):
        """
        Property storing the length of the basis vectors that make up this
        basis. This property is set in the initializer and should not be changed
        by the user directly.
        """
        if not hasattr(self, '_basis_vector_length'):
            raise AttributeError("basis_vector_length was referenced " +\
                "before it was set.")
        return self._basis_vector_length

    @property
    def num_fixed_basis_vectors(self):
        """
        Property storing the number of basis vectors that are fixed (i.e.
        provided by the user instead of obtained through SVD). This property is
        set in the initializer and should not be changed by the user directly.
        """
        if not hasattr(self, '_num_fixed_basis_vectors'):
            raise AttributeError("num_fixed_basis_vectors was referenced " +\
                "before it was set.")
        return self._num_fixed_basis_vectors

    @num_fixed_basis_vectors.setter
    def num_fixed_basis_vectors(self, value):
        """
        Setter for the num_fixed_basis_vectors property.

        value: number of fixed vectors that will be used in the basis
        """
        if isinstance(value, int):
            if value >= 0:
                self._num_fixed_basis_vectors = value
            else:
                raise ValueError("num_fixed_basis_vectors must be greater " +\
                    "than or equal to 0.")
        else:
            raise ValueError("num_fixed_basis_vectors must be an integer.")

    @property
    def num_SVD_basis_vectors(self):
        """
        Property storing the number of basis vectors from SVD. This property is
        set in the initializer and should not be changed by the user directly.
        """
        if not hasattr(self, '_num_SVD_basis_vectors'):
            raise AttributeError("num_SVD_basis_vectors was referenced " +\
                "before it was set.")self.fixed_basis / self.error[np.newaxis,:]
        return self._num_SVD_basis_vectors

    @num_SVD_basis_vectors.setter
    def num_SVD_basis_vectors(self, value):
        """
        Setter for the num_SVD_basis_vectors property.

        value: number of SVD vectors that will be used in the basis
        """
        if isinstance(value, int):
            if value >= 0:
                self._num_SVD_basis_vectors = value
            else:
                raise ValueError("num_SVD_basis_vectors must be greater " +\
                    "than or equal to 0.")
        else:
            raise ValueError("num_SVD_basis_vectors must be an integer.")

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
            raise TypeError("error was set to neither None, a number, nor " +\
                "an array.")
        if np.all(value > 0):
            self._error = self.expander.contract_error(value)
        else:
            raise ValueError("At least one value in the error array was " +\
                "negative.")

    @property
    def fixed_basis(self):
        """
        Property storing the basis containing only the fixed vectors supplied
        by the user.
        """
        if not hasattr(self, '_fixed_basis'):
            raise AttributeError("fixed_basis was referenced before it was set")
        return self._fixed_basis

    @fixed_basis.setter
    def fixed_basis(self, value):
        """
        Setter for the fixed_basis property.

        value: 2D numpy.ndarray of shape (n_basis_vectors, n_channels)
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[0] == self.num_fixed_basis_vectors:
                if value.shape[-1] == self.basis_vector_length:
                    self._fixed_basis = value
                else:
                    raise ValueError("The vectors in fixed_basis did not " +\
                        "have the correct length (i.e. number of channels)")
            else:
                raise ValueError("fixed_basis did not have the same number " +\
                    "of curves as num_fixed_basis_vectors.")
        else:
            raise ValueError("fixed_basis was not a 2d numpy.ndarray")

    @property
    def mean_translated(self):
        """
        Property storing a boolean describing whether the mean was subtracted
        before SVD was done. This property is set automatically in the
        initializer and shouldn't be changed by the user directly.
        """
        if not hasattr(self, '_mean_translated'):
            raise AttributeError("Something went wrong. The " +\
                "mean_translated property apparently was not set " +\
                "automatically in the initializer as it should have been.")
        return self._mean_translated

    @property
    def SVD_basis(self):
        """
        Property storing the basis containing only the SVD vectors from the
        training set.
        """
        if not hasattr(self, '_SVD_basis'):
            raise AttributeError("SVD_basis was referenced before it was set")
        return self._SVD_basis

    @SVD_basis.setter
    def SVD_basis(self, value):
        """
        Setter for the SVD_basis property.

        value: 2D numpy.ndarray of shape (n_basis_vectors, n_channels)
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[0] == self.num_SVD_basis_vectors:
                if value.shape[-1] == self.basis_vector_length:
                    self._SVD_basis = value
                else:
                    raise ValueError("The vectors in SVD_basis did not " +\
                        "have the correct length (i.e. number of channels)")
            else:
                raise ValueError("SVD_basis did not have the same number " +\
                    "of curves as num_SVD_basis_vectors.")
        else:
            raise ValueError("SVD_basis was not a 2d numpy.ndarray")

    @property
    def training_set_space_singular_vectors(self):
        """
        Property storing the right singular vectors (i.e. the ones that don't
        exist in data channel space) of the training set.
        """
        if not hasattr(self, '_training_set_space_singular_vectors'):
            raise AttributeError("training_set_space_singular_vectors were " +\
                                 "referenced before they were set.")
        return self._training_set_space_singular_vectors
    
    @training_set_space_singular_vectors.setter
    def training_set_space_singular_vectors(self, value):
        """
        Setter for the training_set_space_singular_vectors property.
        
        value: 2D numpy.ndarray of shape (n_basis_vectors, n_curve)
               where each row is a unit vector in n_curve dimensional space
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[0] == self.num_SVD_basis_vectors:
                self._training_set_space_singular_vectors = value
            else:
                raise ValueError("The number of " +\
                                 "training_set_space_singular_vectors was " +\
                                 "not the same as the number of SVD basis " +\
                                 "vectors.")
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
    def prior_mean(self):
        """
        TODO
        """
        # TODO

   @property
    def prior_covariance(self):
        """
        TODO
        """
        # TODO

    @property
    def prior_inverse_covariance(self):
        """
        TODO
        """
        # TODO

    @property
    def prior_inverse_covariance_times_mean(self):
        """
        TODO
        """
        # TODO

    @property
    def gaussian_prior(self):
        """
        TODO
        """
        # TODO
    
    def generate_gaussian_prior(self, covariance_expansion_factor=1.,\
        diagonal=False):
        """
        TODO
        """
        # TODO

    @property
    def full_importances(self):
        """
        Property storing the full importances array for the SVD modes (including
        modes that are not in this basis).
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
        Property storing a 1D numpy.ndarray giving the importances of the SVD
        modes in this PartiallyTrainedBasis.
        """
        if not hasattr(self, '_importances'):
            self._importances =\
                self.full_importances[:self.num_SVD_basis_vectors]
        return self._importances

    @property
    def total_importance(self):
        """
        Property storing the sum of the importances of all modes.
        """
        if not hasattr(self, '_total_importance'):
            self._total_importance = np.sum(self.full_importances)
        return self._total_importance

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
    def truncated_normed_importance_loss(self):
        """
        Property storing 1D numpy.ndarray containing values corresponding to
        the sum of importances of modes of higher order than the given index.
        """
        if not hasattr(self, '_truncated_normed_importance_loss'):
            self._truncated_normed_importance_loss =\
                1 - np.cumsum(self.normed_importances)
        return self._truncated_normed_importance_loss

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

    def SVD_terms_necessary_to_reach_noise_level_multiple(self, multiple):
        """
        Function that computes the approximate number of SVD terms needed to
        reach the given multiple noise level of the training set that sourced
        this basis. It is approximate because it is the number needed for the
        mean squared error across all channels and all training set curves to be
        equal to multiple**2. This allows some individual training set curves
        to be fit slightly worse than the noise level.
        
        multiple: multiple of noise level under consideration
        
        returns: Number of SVD modes needed to reach noise level multiple. Note
                 that this number is the total number of basis vectors minus
                 the number of fixed basis vectors. If noise level multiple is
                 never reached, returns None.
        """
        if np.all(self.RMS_spectrum > multiple):
            return None
        else:
            return np.argmax(self.RMS_spectrum < multiple) -\
                self.num_fixed_basis_vectors

    @property
    def SVD_terms_necessary_to_reach_noise_level(self):
        """
        Property storing the approximate number of SVD terms needed to reach the
        noise level of the training set that sourced this basis. It is
        approximate because it is the number needed for the mean squared error
        across all channels and all training set curves to be equal to 1. This
        allows some individual training set curves to be fit slightly worse
        than the noise level. Note that this number is the total number of basis
        vectors minus the number of fixed basis vectors
        """
        if not hasattr(self, '_SVD_terms_necessary_to_reach_noise_level'):
            self._SVD_terms_necessary_to_reach_noise_level =\
                self.SVD_terms_necessary_to_reach_noise_level_multiple(1)
        return self._SVD_terms_necessary_to_reach_noise_level

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
        # TODO

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
        Plots the spectrum of importances of the modes in this
        PartiallyTrainedBasis.
        
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

