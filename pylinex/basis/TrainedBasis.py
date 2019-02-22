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
from ..util import real_numerical_types
from ..expander import NullExpander
from .Basis import Basis

def weighted_SVD(matrix, error=None, full_matrices=False):
    """
    Finds the most important modes of the given matrix given the weightings
    given by the error.
    
    matrix a horizontal rectangular matrix
    error weighting applied to the dimension corresponding to the rows
    """
    if error is None:
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
    if error is None:
        error = np.ones(curves.shape[-1])
    if Neigen is None:
        Neigen = curves.shape[-1]
    SVD_U, SVD_S, SVD_V = weighted_SVD(curves.T, error=error)
    total_importance = np.sum(SVD_S)
    return SVD_U.T[:Neigen], SVD_S[:Neigen], SVD_V.T[:Neigen], total_importance

class TrainedBasis(Basis):
    """
    Class which derives a basis from a training set using weighted Singular
    Value Decomposition (SVD).
    """
    def __init__(self, training_set, num_basis_vectors, error=None,\
        expander=None):
        """
        Initializes this TranedBasis using the given training_set and number of
        basis vectors.
        
        training_set: numpy.ndarray of shape (ncurves, nchannels)
        num_basis_vectors: number of basis vectors to keep from SVD
        error: if None, no weighting is used
               if error is a numpy.ndarray, it should be 1D and of length
                                            nchannels
        expander: Expander object which expands from training set space to
                  final basis vector space
        """
        if error is None:
            error = np.ones(training_set.shape[-1])
        if expander is None:
            expander = NullExpander()
        error = expander.contract_error(error)
        SVD_basis = weighted_SVD_basis(training_set,\
            error=error, Neigen=num_basis_vectors)
        self.basis = SVD_basis[0]
        self.importances = SVD_basis[1]
        self.training_set_space_singular_vectors = SVD_basis[2]
        self.total_importance = SVD_basis[3]
        self.expander = expander
    
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
    def importances(self):
        """
        Property storing a 1D numpy.ndarray giving the importances of the modes
        in this TrainedBasis.
        """
        if not hasattr(self, '_importances'):
            raise AttributeError("importances haven't been set.")
        return self._importances
    
    @importances.setter
    def importances(self, value):
        """
        Setter for the importances property.
        
        value: 1D numpy.ndarray of length num_basis_vectors
        """
        value = np.array(value)
        if value.shape == (self.num_basis_vectors,):
            if np.all(value >= 0):
                self._importances = value
            else:
                raise ValueError("Some of the importances given aren't " +\
                                 "nonnegative, which makes no sense.")
        else:
            raise ValueError("importances don't have the same number of " +\
                             "elements as the basis.")
    
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
    def training_set_length(self):
        """
        Property storing the number of curves in the training set used to
        generate this basis.
        """
        if not hasattr(self, '_training_set_length'):
            self._training_set_length =\
                self.training_set_space_singular_vectors.shape[-1]
        return self._training_set_length
    
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
                self.training_set_length
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
            self._prior_covariance =\
                self._prior_covariance / self.training_set_length
            self._prior_covariance =\
                np.identity(self.num_basis_vectors) - self._prior_covariance
            self._prior_covariance =\
                (self.importances[:,np.newaxis] * self._prior_covariance)
            self._prior_covariance =\
                (self.importances[np.newaxis,:] * self._prior_covariance)
            self._prior_covariance =\
                self._prior_covariance / (self.training_set_length - 1)
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
            raise AttributeError("total_importance was not set before " +\
                                 "being referenced.")
        return self._total_importance
    
    @total_importance.setter
    def total_importance(self, value):
        """
        Setter for the total_importance property.
        
        value: must be a single positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._total_importance = value
            else:
                raise ValueError("total_importance must be a positive number.")
        else:
            raise TypeError("total_importance must be a single number.")
    
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

    def plot_importance_spectrum(self, normed=False,\
        plot_importance_loss=False, plot_xs=None, show=False, title='',\
        **kwargs):
        """
        Plots the spectrum of importances of the modes in this TrainedBasis.
        
        normed: if True, importances shown add to 1
        plot_importance_loss: if True, importance loss from truncation at
                                       various points is plotted
        plot_xs: if None, x-values for plot are assumed to be the first
                          num_basis_vectors natural numbers
                 otherwise, plot_xs should be a 1D numpy.ndarray of length
                            num_basis_vectors containing x-values for the plot
        show: if True, matplotlib.pyplot.show() is called before this function
                       is returned
        title: string title of the plot (can have LaTeX inside)
        kwargs: extra keyword arguments to pass on to matplotlib.pyplot.scatter
        """
        if plot_xs is None:
            plot_xs = np.arange(self.num_basis_vectors)
        fig = pl.figure()
        ax = fig.add_subplot(111)
        to_show = np.where(self.normed_importances > 0)[0]
        if plot_importance_loss:
            if normed:
                ax.scatter(plot_xs, self.truncated_normed_importance_loss,\
                    **kwargs)
                y_max = 1
                y_min = min(self.truncated_normed_importance_loss[to_show])
            else:
                ax.scatter(plot_xs, self.truncated_normed_importance_loss *\
                    self.total_importance, **kwargs)
                y_max = self.total_importance
                y_min = min(self.truncated_normed_importance_loss[to_show]) *\
                     self.total_importance
        else:
            if normed:
                ax.scatter(plot_xs, self.normed_importances, **kwargs)
                y_min = min(self.normed_importances[to_show])
                y_max = max(self.normed_importances[to_show])
            else:
                ax.scatter(plot_xs, self.importances, **kwargs)
                y_min = min(self.importances[to_show])
                y_max = max(self.importances[to_show])
        y_max = 10 ** int(np.log10(y_max) + 1)
        y_min = 10 ** int(np.log10(y_min) - 1)
        ax.set_ylim((y_min, y_max))
        ax.set_yscale('log')
        pl.xlabel('Mode index')
        if plot_importance_loss:
            if normed:
                ylabel = 'Normalized importance of modes above index'
            else:
                ylabel = 'Importance of modes above index'
        else:
            if normed:
                ylabel = 'Normalized mode importance'
            else:
                ylabel = 'Mode importance'
        pl.ylabel(ylabel)
        pl.title(title)
        if show:
            pl.show()

