"""
File: extractpy/fitter/Fitter.py
Author: Keith Tauscher
Date: 27 Aug 2017

Description: File containing a class which computes fits of data using linear
             models through analytical calculations. It has functions to output
             the signal estimate (with error), parameter covariance, and more.
"""
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scila
import matplotlib.pyplot as pl
from ..util import Savable, TrainingSetIterator
from ..basis import Basis, BasisSet

class Fitter(Savable):
    """
    An object which takes as inputs a BasisSet object and data and error
    vectors and outputs statistics about the fit of the data assuming the error
    and using the BasisSet.
    """
    def __init__(self, basis_set, data, error=None, **priors):
        """
        Initializes a new Analyzer object using the given inputs.
        
        basis_set: a BasisSet object (or a Basis object, which is converted
                   internally to a BasisSet of one Basis with the name 'sole')
        data: 1D vector of same length as vectors in basis_set or 2D
              numpy.ndarray of shape (ncurves, )
        error: 1D vector of same length as vectors in basis_set containing only
               positive numbers
        **priors: keyword arguments where the keys are exactly the names of the
                  basis sets with '_prior' appended to them
        """
        self.basis_set = basis_set
        self.priors = priors
        self.data = data
        self.error = error
    
    @property
    def basis_set(self):
        """
        Property storing the BasisSet object whose basis vectors will be
        used by this object in the fit.
        """
        if not hasattr(self, '_basis_set'):
            raise AttributeError("basis_set was referenced before it was " +\
                                 "set. This shouldn't happen. Something is " +\
                                 "wrong.")
        return self._basis_set
    
    @basis_set.setter
    def basis_set(self, value):
        """
        Allows user to set basis_set property.
        
        value: BasisSet object or, more generally, a Basis object containing
               the basis vectors with which to perform the fit
        """
        if isinstance(value, BasisSet):
            self._basis_set = value
        elif isinstance(value, Basis):
            self._basis_set = BasisSet('sole', value)
        else:
            raise TypeError("basis_set was neither a BasisSet or a " +\
                            "different Basis object.")
    
    @property
    def sizes(self):
        if not hasattr(self, '_sizes'):
            self._sizes = self.basis_set.sizes
        return self._sizes
    
    @property
    def names(self):
        """
        Property storing the names of the component Bases of the BasisSet.
        """
        if not hasattr(self, '_names'):
            self._names = self.basis_set.names
        return self._names
    
    @property
    def total_normed_importance(self):
        """
        Property allowing for quick access of the total_normed_importance
        property of the BasisSet object at the heart of this Fitter.
        """
        if not hasattr(self, '_total_normed_importance'):
            self._total_normed_importance =\
                self.basis_set.total_normed_importance
        return self._total_normed_importance
    
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
        if self.has_priors:
            self._prior_mean = np.concatenate(\
                [self._priors[key + '_prior'].mean.A[0] for key in self.names])
            self._prior_covariance = scila.block_diag(*\
                [self._priors[key + '_prior'].covariance.A\
                for key in self.names])
            self._prior_inverse_covariance = scila.block_diag(*\
                [self._priors[key + '_prior'].invcov.A\
                for key in self.names])
    
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
    
    @property
    def prior_significance(self):
        """
        Property storing the quantity, mu^T Lambda^{-1} mu, where mu is the
        prior mean and Lambda is the prior covariance matrix.
        """
        if not hasattr(self, '_prior_significance'):
            self._prior_significance = np.dot(self.prior_mean,\
                np.dot(self.prior_covariance, self.prior_mean))
        return self._prior_significance
    
    @property
    def log_prior_covariance_determinant(self):
        """
        Property storing the logarithm (base e) of the determinant of the prior
        parameter covariance matrix.
        """
        if not hasattr(self, '_log_prior_covariance_determinant'):
            self._log_prior_covariance_determinant =\
                npla.slogdet(self.prior_covariance)[1]
        return self._log_prior_covariance_determinant
        
    
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
        if value is None:
            self._error = np.ones(self.num_channels)
        else:
            value = np.array(value)
            if value.shape == (self.num_channels,):
                self._error = value
            else:
                raise ValueError("error didn't have the same length as the " +\
                                 "basis functions.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of data channels in this fit. This should
        be the length of the data and error vectors.
        """
        return self.basis_set.num_larger_channel_set_indices
    
    @property
    def data_significance(self):
        if not hasattr(self, '_data_significance'):
            if self.data.ndim == 2:
                self._data_significance = np.einsum('ij,ij->i',\
                    self.weighted_data, self.weighted_data)
            else:
                self._data_significance =\
                    np.dot(self.weighted_data, self.weighted_data.T)
        return self._data_significance
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters of the fit. This is the same
        as the number of basis vectors in the basis_set.
        """
        return self.basis_set.num_basis_vectors
    
    @property
    def posterior_covariance_times_prior_inverse_covariance(self):
        if not hasattr(self,\
            '_posterior_covariance_times_prior_inverse_covariance'):
            self._posterior_covariance_times_prior_inverse_covariance =\
                np.dot(self.parameter_covariance,\
                       self.prior_inverse_covariance)
        return self._posterior_covariance_times_prior_inverse_covariance
    
    @property
    def model_complexity_mean_to_peak_logL(self):
        """
        Returns a measure of the model complexity which is computed by taking
        the difference between the mean and peak values of the log likelihood.
        If this Fitter has no priors, then this property will always simply
        return the number of parameters.
        """
        if not hasattr(self, '_model_complexity_mean_to_peak_logL'):
            self._model_complexity_mean_to_peak_logL = self.num_parameters
            if self.has_priors:
                self._model_complexity_mean_to_peak_logL -= np.trace(\
                    self.posterior_covariance_times_prior_inverse_covariance)
        return self._model_complexity_mean_to_peak_logL
    
    @property
    def model_complexity_logL_variance(self):
        """
        Returns a measure of the model complexity which is computed by finding
        the variance of the log likelihood function.
        """
        if not hasattr(self, '_model_complexity_logL_variance'):
            self._model_complexity_logL_variance = self.num_parameters
            bias_term = np.dot(self.weighted_basis, self.weighted_bias.T).T
            if self.data.ndim == 2:
                bias_term = np.einsum('ij,ik,jk->i', bias_term, bias_term,\
                    self.parameter_covariance)
            else:
                bias_term = np.dot(bias_term,\
                    np.dot(self.parameter_covariance, bias_term))
            self._model_complexity_logL_variance += (2 * bias_term)
            if self.has_priors:
                self._model_complexity_logL_variance += np.trace(np.dot(\
                    self.posterior_covariance_times_prior_inverse_covariance,\
                    self.posterior_covariance_times_prior_inverse_covariance))
                self._model_complexity_logL_variance -= (2 * np.trace(\
                    self.posterior_covariance_times_prior_inverse_covariance))
        return self._model_complexity_logL_variance
    
    @property
    def weighted_basis(self):
        """
        Property storing the basis functions of the basis set weighted down by
        the error. This is the one actually used in the calculations.
        """
        if not hasattr(self, '_weighted_basis'):
            self._weighted_basis =\
                self.basis_set.basis / self.error[np.newaxis,:]
        return self._weighted_basis
    
    @property
    def weighted_data(self):
        """
        Property storing the data vector weighted down by the error. This is
        the one that actually goes into the calculation of the mean and
        covariance of the posterior distribution.
        """
        if not hasattr(self, '_weighted_data'):
            if self.data.ndim == 1:
                self._weighted_data = self.data / self.error
            else:
                self._weighted_data = self.data / self.error[np.newaxis,:]
        return self._weighted_data
    
    @property
    def overlap_matrix(self):
        """
        Property storing the matrix of overlaps between the weighted basis
        vectors. It is an NxN numpy.ndarray where N is the total number of
        basis vectors.
        """
        if not hasattr(self, '_overlap_matrix'):
            self._overlap_matrix =\
                np.dot(self.weighted_basis, self.weighted_basis.T)
        return self._overlap_matrix
    
    @property
    def basis_dot_products(self):
        """
        Property storing the dot products between the Basis objects underlying
        the BasisSet this object stores.
        """
        if not hasattr(self, '_basis_dot_products'):
            self._basis_dot_products =\
                self.basis_set.basis_dot_products(error=self.error)
        return self._basis_dot_products
    
    @property
    def basis_dot_product_sum(self):
        """
        Property storing the sum of all basis_dot_products
        """
        if not hasattr(self, '_basis_dot_product_sum'):
            self._basis_dot_product_sum = np.sum(self.basis_dot_products)
            self._basis_dot_product_sum = self._basis_dot_product_sum -\
                np.trace(self.basis_dot_products)
            self._basis_dot_product_sum = self._basis_dot_product_sum / 2.
        return self._basis_dot_product_sum

    @property
    def parameter_inverse_covariance(self):
        """
        Property storing the inverse of the posterior distribution's covariance
        matrix.
        """
        if not hasattr(self, '_parameter_inverse_covariance'):
            self._parameter_inverse_covariance = self.overlap_matrix
            if self.has_priors:
                self._parameter_inverse_covariance =\
                    self._parameter_inverse_covariance +\
                    self.prior_inverse_covariance
        return self._parameter_inverse_covariance
    
    @property
    def parameter_covariance(self):
        """
        Property storing the covariance matrix of the posterior parameter
        distribution.
        """
        if not hasattr(self, '_parameter_covariance'):
            self._parameter_covariance =\
                npla.inv(self.parameter_inverse_covariance)
        return self._parameter_covariance
    
    @property
    def log_parameter_covariance_determinant(self):
        """
        Property storing the logarithm (base e) of the determinant of the
        posterior parameter covariance matrix.
        """
        if not hasattr(self, '_log_parameter_covariance_determinant'):
            self._log_parameter_covariance_determinant =\
                npla.slogdet(self.parameter_covariance)[1]
        return self._log_parameter_covariance_determinant
    
    @property
    def log_parameter_covariance_determinant_ratio(self):
        """
        Property storing the logarithm (base e) of the ratio of the determinant
        of the posterior parameter covariance matrix to the determinant of the
        prior parameter covariance matrix.
        """
        if not hasattr(self, '_log_parameter_covariance_determinant_ratio'):
            self._log_parameter_covariance_determinant_ratio =\
                self.log_parameter_covariance_determinant -\
                self.log_prior_covariance_determinant
        return self._log_parameter_covariance_determinant_ratio
    
    @property
    def channel_error(self):
        """
        Property storing the error on the estimate of the full data in channel
        space.
        """
        if not hasattr(self, '_channel_error'):
            self._channel_error = np.sqrt(np.diag(\
                np.dot(self.basis_set.basis.T,\
                np.dot(self.parameter_covariance, self.basis_set.basis))))
        return self._channel_error
    
    @property
    def channel_RMS(self):
        """
        Property storing the RMS error on the estimate of the full data in
        channel space.
        """
        if not hasattr(self, '_channel_RMS'):
            self._channel_RMS =\
                np.sqrt(np.mean(np.power(self.channel_error, 2)))
        return self._channel_RMS
    
    @property
    def parameter_mean(self):
        """
        Property storing the posterior mean parameter vector/matrix. The shape
        of the result is either (nparams,) or (ncurves, nparams).
        """
        if not hasattr(self, '_parameter_mean'):
            self._parameter_mean =\
                np.dot(self.weighted_basis, self.weighted_data.T)
            if self.has_priors:
                if self.data.ndim == 2:
                    self._parameter_mean = self._parameter_mean +\
                        self.prior_inverse_covariance_times_mean[:,np.newaxis]
                else:
                    self._parameter_mean = self._parameter_mean +\
                        self.prior_inverse_covariance_times_mean
            self._parameter_mean =\
                np.dot(self.parameter_covariance, self._parameter_mean).T
        return self._parameter_mean
    
    @property
    def posterior_significance(self):
        """
        Property storing the quantity, x^T S^{-1} x, where x is the posterior
        mean and S is the posterior parameter covariance matrix.
        """
        if not hasattr(self, '_posterior_significance'):
            if self.data.ndim == 2:
                self._posterior_significance = np.einsum('ij,ik,jk->i',\
                    self.parameter_mean, self.parameter_inverse_covariance,\
                    self.parameter_mean)
            else:
                self._posterior_significance =\
                    np.dot(self.parameter_mean,\
                    np.dot(self.parameter_inverse_covariance,\
                    self.parameter_mean))
        return self._posterior_significance
    
    @property
    def channel_mean(self):
        """
        Property storing the maximum likelihood estimate of the data in channel
        space.
        """
        if not hasattr(self, '_channel_mean'):
            self._channel_mean =\
                np.dot(self.basis_set.basis.T, self.parameter_mean.T).T
        return self._channel_mean
    
    @property
    def channel_bias(self):
        """
        Property storing the bias of the estimate of the data (i.e. the maximum
        likelihood estimate of the data minus the data)
        """
        if not hasattr(self, '_channel_bias'):
            self._channel_bias = self.channel_mean - self.data
        return self._channel_bias
    
    @property
    def channel_bias_RMS(self):
        """
        """
        if not hasattr(self, '_channel_bias_RMS'):
            if self.data.ndim == 2:
                self._channel_bias_RMS = np.sqrt(np.einsum('ij,ij->i',\
                    self.channel_bias, self.channel_bias) / self.num_channels)
            else:
                self._channel_bias_RMS =\
                    np.sqrt(np.dot(self.channel_bias, self.channel_bias) /\
                    self.num_channels)
        return self._channel_bias_RMS
    
    @property
    def weighted_bias(self):
        """
        Property storing the basis functions weighted down by the errors. This
        is the basis actually used in the calculation of the mean and
        covariance of the parameters.
        """
        if not hasattr(self, '_weighted_bias'):
            if self.data.ndim == 2:
                self._weighted_bias =\
                    self.channel_bias / self.error[np.newaxis,:]
            else:
                self._weighted_bias = self.channel_bias / self.error
        return self._weighted_bias
    
    @property
    def bias_statistic(self):
        """
        Property which stores a statistic known as the "bias statistic". It is
        a measure of the bias of the full model being fit. It should have a
        chi^2(N) distribution where N is the number of data points.
        """
        if not hasattr(self, '_bias_statistic'):
            if self.data.ndim == 2:
                self._bias_statistic = np.einsum('ij,ij->i',\
                    self.weighted_bias, self.weighted_bias)
            else:
                self._bias_statistic =\
                    np.dot(self.weighted_bias, self.weighted_bias)
        return self._bias_statistic
    
    @property
    def normalized_bias_statistic(self):
        """
        Property which stores a normalized version of the bias statistic. The
        expectation value of this normed version is 1.
        """
        if not hasattr(self, '_normalized_bias_statistic'):
            self._normalized_bias_statistic = self.bias_statistic /\
                (self.num_channels - self.num_parameters - 1)
        return self._normalized_bias_statistic
    
    @property
    def significance_difference(self):
        """
        Property storing the difference between the posterior significance and
        the sum of the data significance and prior significance. It is a term
        in the log evidence.
        """
        if not hasattr(self, '_significance_difference'):
            if self.data.ndim == 2:
                parameter_mean_sum =\
                    self.parameter_mean + self.prior_mean[np.newaxis,:]
                parameter_mean_difference =\
                    self.parameter_mean - self.prior_mean[np.newaxis,:]
                prior_covariance_part = np.einsum('ij,ik,jk->i',\
                    parameter_mean_sum, parameter_mean_difference,\
                    self.prior_inverse_covariance)
                weighted_channel_mean_sum = self.weighted_data +\
                    (self.channel_mean / self.error[np.newaxis,:])
                likelihood_covariance_part = np.einsum('ij,ij->i',\
                    weighted_channel_mean_sum, self.weighted_bias)
            else:
                parameter_mean_sum = self.parameter_mean + self.prior_mean
                parameter_mean_difference =\
                    self.parameter_mean - self.prior_mean
                prior_covariance_part = np.dot(parameter_mean_sum,\
                    np.dot(self.prior_inverse_covariance,\
                    parameter_mean_difference))
                weighted_channel_mean_sum =\
                    (self.data + self.channel_mean) / self.error
                likelihood_covariance_part =\
                    np.dot(weighted_channel_mean_sum, self.weighted_bias)
            self._significance_difference =\
                prior_covariance_part + likelihood_covariance_part
        return self._significance_difference
    
    @property
    def log_evidence(self):
        """
        Property storing the natural logarithm of the evidence (a.k.a. marginal
        likelihood) of this fit. The evidence is the integral over parameter
        space of the likelihood and is often very large.
        """
        if not hasattr(self, '_log_evidence'):
            self._log_evidence =\
                (self.log_parameter_covariance_determinant_ratio +\
                self.significance_difference) / 2.
            # only constants added below, ignore if facing numerical problems
            self._log_evidence = self._log_evidence -\
                ((self.num_channels * np.log(2 * np.pi)) / 2.) -\
                np.sum(np.log(self.error))
        return self._log_evidence
    
    @property
    def log_evidence_per_data_channel(self):
        """
        Property storing the log_evidence divided by the number of channels.
        """
        if not hasattr(self, '_log_evidence_per_data_channel'):
            self._log_evidence_per_data_channel =\
                self.log_evidence / self.num_channels
        return self._log_evidence_per_data_channel
    
    @property
    def evidence(self):
        """
        Property storing the evidence (a.k.a. marginal likelihood) of this fit.
        Beware: the evidence is often extremely in magnitude, with log
        evidences sometimes approaching +-10^7. In these cases, the evidence
        will end up NaN.
        """
        if not hasattr(self, '_evidence'):
            self._evidence = np.exp(self.log_evidence)
        return self._evidence
    
    @property
    def evidence_per_data_channel(self):
        """
        Finds the factor by which each data channel multiplies the Bayesian
        evidence on average (more precisely, geometric mean).
        """
        if not hasattr(self, '_evidence_per_data_channel'):
            self._evidence_per_data_channel =\
                np.exp(self.log_evidence_per_data_channel)
        return self._evidence_per_data_channel
    
    @property
    def bayesian_information_criterion(self):
        """
        Property storing the Bayesian Information Criterion (BIC) which is
        essentially the same as the bias statistic except it includes
        information about the complexity of the model.
        """
        if not hasattr(self, '_bayesian_information_criterion'):
            self._bayesian_information_criterion = self.bias_statistic +\
                (self.num_parameters * np.log(self.num_channels))
        return self._bayesian_information_criterion
    
    @property
    def BIC(self):
        """
        Alias for bayesian_information_criterion property.
        """
        return self.bayesian_information_criterion
    
    @property
    def akaike_information_criterion(self):
        """
        An information criterion given by -2ln(L_max)+2p where L_max is the
        maximum likelihood and p is the number of parameters.
        """
        if not hasattr(self, '_akaike_information_criterion'):
            self._akaike_information_criterion =\
               self.bias_statistic + 2 * self.num_parameters
        return self._akaike_information_criterion
    
    @property
    def AIC(self):
        """
        Alias for akaike_information_criterion property.
        """
        return self.akaike_information_criterion
    
    @property
    def deviance_information_criterion(self):
        """
        An information criterion given by -4 ln(L_max) + <2 ln(L)> where L is
        the likelihood, <> denotes averaging over the posterior, and L_max is
        the maximum likelihood.
        """
        if not hasattr(self, '_deviance_information_criterion'):
            self._deviance_information_criterion = self.bias_statistic +\
                self.model_complexity_mean_to_peak_logL
        return self._deviance_information_criterion
    
    @property
    def DIC(self):
        """
        Alias for deviance_information_criterion property.
        """
        return self.deviance_information_criterion
    
    @property
    def deviance_information_criterion_logL_variance(self):
        """
        Version of the Deviance Information Criterion (DIC) which estimates the
        model complexity through computation of the variance of the log
        likelihood (with respect to the posterior).
        """
        if not hasattr(self, '_deviance_information_criterion_logL_variance'):
            self._deviance_information_criterion_logL_variance =\
                self.bias_statistic + self.model_complexity_logL_variance
        return self._deviance_information_criterion_logL_variance
    
    @property
    def DIC2(self):
        """
        Alias for the deviance_information_criterion_logL_variance property.
        """
        return self.deviance_information_criterion_logL_variance

    @property
    def posterior_prior_mean_difference(self):
        """
        Property storing the difference between the posterior parameter mean
        and the prior parameter mean.
        """
        if not hasattr(self, '_posterior_prior_mean_difference'):
            if self.data.ndim == 2:
                self._posterior_prior_mean_difference =\
                    self.parameter_mean - self.prior_mean[np.newaxis,:]
            else:
                self._posterior_prior_mean_difference =\
                    self.parameter_mean - self.prior_mean
        return self._posterior_prior_mean_difference
    
    @property
    def bayesian_predictive_information_criterion(self):
        """
        Property storing the Bayesian Predictive Information Criterion (BPIC),
        a statistic which gives relatives goodness of fit values.
        """
        if not hasattr(self, '_bayesian_predictive_information_criterion'):
            self._bayesian_predictive_information_criterion =\
                self.num_parameters + self.bias_statistic
            self._bayesian_predictive_information_criterion +=\
                (self.num_channels * np.log(2 * np.pi))
            self._bayesian_predictive_information_criterion +=\
                (2. * np.sum(np.log(self.error)))
            if self.has_priors:
                self._bayesian_predictive_information_criterion -= np.trace(\
                    self.posterior_covariance_times_prior_inverse_covariance)
                term_v1 = np.dot(\
                    self.posterior_covariance_times_prior_inverse_covariance,\
                    self.posterior_prior_mean_difference.T).T
                term_v2 = np.dot(self.prior_inverse_covariance,\
                    self.posterior_prior_mean_difference.T).T +\
                    (2 * np.dot(self.weighted_basis, self.weighted_bias.T).T)
            squared_weighted_bias = ()
            if self.data.ndim == 2:
                if self.has_priors:
                    self._bayesian_predictive_information_criterion +=\
                        (np.einsum('ij,ij->i', term_v1, term_v2) /\
                        self.num_channels)
                self._bayesian_predictive_information_criterion += np.einsum(\
                    'ij,kj,mj,km->i', self.weighted_bias ** 2,\
                    self.weighted_basis, self.weighted_basis,\
                    self.parameter_covariance)
            else:
                if self.has_priors:
                    self._bayesian_predictive_information_criterion +=\
                        (np.dot(term_v1, term_v2) / self.num_channels)
                self._bayesian_predictive_information_criterion += np.einsum(\
                    'j,kj,mj,km', self.weighted_bias ** 2,\
                    self.weighted_basis, self.weighted_basis,\
                    self.parameter_covariance)
        return self._bayesian_predictive_information_criterion
    
    @property
    def BPIC(self):
        """
        Alias for the bayesian_predictive_information_criterion property.
        """
        return self.bayesian_predictive_information_criterion
    def subbasis_log_separation_evidence(self, name=None, per_channel=True):
        """
        Calculates the subbasis_log_separation evidence. This is the same as
        the evidence with the log covariance determinant ratio replaced by the
        log covariance determinant ratio for the given subbasis.
        
        name: string identifying subbasis under concern
        per_channel: if True, normalizes the log_separation_evidence by
                              dividing by the nuiber of data channels.
        
        returns: single float number
        """
        answer = self.log_evidence -\
            (self.log_parameter_covariance_determinant_ratio / 2.) +\
            (self.subbasis_log_parameter_covariance_determinant_ratio(\
            name=name) / 2.)
        if per_channel:
            return answer / self.num_channels
        else:
            return answer
    
    def subbasis_separation_evidence_per_channel(self, name=None):
        """
        Finds the subbasis separation evidence per data channel.
        
        name: string identifying subbasis under concern
        
        returns: single non-negative float number
        """
        return np.exp(self.subbasis_log_separation_evidence(name=name,\
            per_channel=True))
    
    @property
    def log_separation_evidence(self):
        """
        Property storing the logarithm (base e) of the separation evidence, a
        version of the evidence where the log of the ratio of the determinants
        of the posterior to prior covariance matrices is replaced by the sum
        over all subbases of such logs of ratios.
        """
        if not hasattr(self, '_log_separation_evidence'):
            self._log_separation_evidence = self.log_evidence -\
                (self.log_parameter_covariance_determinant_ratio / 2.) +\
                (self.subbasis_log_parameter_covariance_determinant_ratios_sum\
                / 2.)
        return self._log_separation_evidence
    
    @property
    def log_separation_evidence_per_data_channel(self):
        """
        Property storing the log_separation_evidence divided by the number of
        data channels. For more information, see the log_separation_evidence
        property.
        """
        if not hasattr(self, '_log_separation_evidence_per_data_channel'):
            self._log_separation_evidence_per_data_channel =\
                self.log_separation_evidence / self.num_channels
        return self._log_separation_evidence_per_data_channel
    
    @property
    def separation_evidence(self):
        """
        Property storing the separation evidence, a version of the evidence
        where the log of the ratio of the determinants of the posterior to
        prior covariance matrices is replaced by the sum over all subbases of
        such logs of ratios.
        """
        if not hasattr(self, '_separation_evidence'):
            self._separation_evidence = np.exp(self.log_separation_evidence)
        return self._separation_evidence
    
    @property
    def separation_evidence_per_data_channel(self):
        """
        Property storing the average (geometric mean) factor by which each data
        channel affects the separation evidence.
        """
        if not hasattr(self, '_separation_evidence_per_data_channel'):
            self._separation_evidence_per_data_channel =\
                np.exp(self.log_separation_evidence_per_data_channel)
        return self._separation_evidence_per_data_channel
    
    @property
    def subbasis_log_parameter_covariance_determinant_ratios_sum(self):
        """
        Property storing the sum of the logarithms (base e) of the ratios of
        the posterior parameter covariance matrices to the prior parameter
        covariance matrices.
        """
        if not hasattr(self,\
            '_subbasis_log_parameter_covariance_determinant_ratios_sum'):
            self._subbasis_log_parameter_covariance_determinant_ratios_sum =\
                sum([self.subbasis_log_parameter_covariance_determinant_ratio(\
                name=name) for name in self.names])
        return self._subbasis_log_parameter_covariance_determinant_ratios_sum
    
    def subbasis_prior_significance(self, name=None):
        """
        Finds and returns the quantity: mu^T Lambda^{-1} mu, where mu is the
        prior subbasis parameter mean and Lambda is the prior subbasis
        parameter covariance.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        prior = self.priors[name + '_prior']
        mean = prior.mean.A[0]
        invcov = prior.invcov.A
        return np.dot(mean, np.dot(invcov, mean))
        
    
    def subbasis_parameter_inverse_covariance(self, name=None):
        """
        Finds the inverse of the marginalized covariance matrix corresponding
        to the given subbasis.
        
        name: string identifying subbasis under concern
        """
        return npla.inv(self.subbasis_parameter_covariance(name=name))

    def subbases_overlap_matrix(self, row_name=None, column_name=None):
        """
        Creates a view into the overlap matrix between the given subbases.
        
        row_name: the (string) name of the subbasis whose parameter number will
                  be represented by the row of the returned matrix.
        column_name: the (string) name of the subbasis whose parameter number
                     will be represented by the column of the returned matrix
        
        returns: n x m matrix where n is the number of basis vectors in the row
                 subbasis and m is the number of basis vectors in the column
                 subbasis in the form of a 2D numpy.ndarray
        """
        row_slice = self.basis_set.slices_by_name[row_name]
        column_slice = self.basis_set.slices_by_name[column_name]
        return self.overlap_matrix[:,column_slice][row_slice]
    
    def subbasis_parameter_covariance(self, name=None):
        """
        Finds and returns the portion of the parameter covariance matrix
        associated with the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns 2D numpy.ndarray of shape (k, k) where k is the number of basis
                vectors in the subbasis
        """
        subbasis_slice = self.basis_set.slices_by_name[name]
        return self.parameter_covariance[:,subbasis_slice][subbasis_slice]
    
    def subbasis_log_parameter_covariance_determinant(self, name=None):
        """
        Finds the logarithm (base e) of the determinant of the posterior
        parameter covariance matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        return npla.slogdet(self.subbasis_parameter_covariance(name=name))[1]
    
    def subbasis_log_prior_covariance_determinant(self, name=None):
        """
        Finds the logarithm (base e) of the determinant of the prior parameter
        covariance matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        if name is None:
            return self.log_prior_covariance_determinant
        else:
            return npla.slogdet(self.priors[name + '_prior'].covariance.A)[1]
    
    def subbasis_log_parameter_covariance_determinant_ratio(self, name=None):
        """
        Finds logarithm (base e) of the ratio of the determinant of the
        posterior covariance matrix to the determinant of the prior covariance
        matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        return self.subbasis_log_parameter_covariance_determinant(name=name) -\
            self.subbasis_log_prior_covariance_determinant(name=name)
    
    def subbasis_parameter_covariance_determinant_ratio(self, name=None):
        """
        Finds the ratio of the determinant of the posterior covariance matrix
        to the determinant of the prior covariance matrix for the given
        subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single non-negative float number
        """
        if name is None:
            return\
                self.subbasis_log_parameter_covariance_determinant_ratios_sum
        else:
            return self.subbasis_log_parameter_covariance_determinant_ratio(\
                name=name)
    
    def subbasis_channel_error(self, name=None):
        """
        Finds the error (in data channel space) of the fit by a given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray of the same length as the basis vectors of
                 the subbasis (which may or may not be different than the
                 length of the expanded basis vectors).
        """
        subbasis = self.basis_set[name]
        subbasis_channel_covariance = np.dot(subbasis.basis.T,\
            np.dot(self.subbasis_parameter_covariance(name=name),\
            subbasis.basis))
        return np.sqrt(np.diag(subbasis_channel_covariance))
    
    def subbasis_parameter_mean(self, name=None):
        """
        Finds the posterior parameter mean for a subbasis. This is just a view
        into the view posterior parameter mean.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray containing the parameters for the given
                 subbasis
        """
        return self.parameter_mean[...,self.basis_set.slices_by_name[name]]
    
    def subbasis_channel_mean(self, name=None, expanded=False):
        """
        The estimate of the contribution to the data from the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray containing the channel-space estimate from
                 the given subbasis
        """
        if expanded:
            return np.dot(self.subbasis_parameter_mean(name=name),\
                self.basis_set[name].expanded_basis)
        else:
            return np.dot(self.subbasis_parameter_mean(name=name),\
                self.basis_set[name].basis)
    
    def subbasis_channel_bias(self, name=None, true_curve=None):
        """
        Calculates and returns the bias on the estimate from the given subbasis
        using the given curve as a reference.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        
        returns: 1D numpy.ndarray in channel space containing the difference
                 between the estimate of the data's contribution from the given
                 subbasis and the given true curve
        """
        if true_curve is None:
            expanded = False
        elif true_curve.shape[-1] ==\
            self.basis_set[name].num_smaller_channel_set_indices:
            expanded = False
        elif true_curve.shape[-1] ==\
            self.basis_set[name].num_larger_channel_set_indices:
            expanded = True
        if name is None:
            if true_curve is None:
                return self.channel_bias
            else:
                raise ValueError("true_curve should only be given to " +\
                                 "subbasis_channel_bias if the name of a " +\
                                 "subbasis is specified.")
        else:
            if true_curve is None:
                raise ValueError("true_curve must be given to " +\
                                 "subbasis_channel_bias if the name of a " +\
                                 "subbasis is specified.")
            elif self.data.ndim == 2:
                if true_curve.ndim == 1:
                    return self.subbasis_channel_mean(name=name,\
                        expanded=expanded) - true_curve[np.newaxis,:]
                else:
                    return self.subbasis_channel_mean(name=name,\
                        expanded=expanded) - true_curve
            else:
                return self.subbasis_channel_mean(name=name,\
                    expanded=expanded) - true_curve
    
    def subbasis_channel_RMS(self, name=None):
        """
        Calculates and returns the RMS channel error on the estimate of the
        contribution to the data from the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: single float number RMS
        """
        return np.sqrt(\
            np.mean(np.power(self.subbasis_channel_error(name=name), 2)))
    
    def subbasis_separation_statistic(self, name=None):
        """
        Finds the separation statistic associated with the given subbasis. The
        separation statistic is essentially an RMS'd error expansion factor.
        
        name: name of the subbasis for which to find the separation statistic
        """
        weighted_expanded_basis =\
            self.basis_set[name].expanded_basis / self.error
        return np.sqrt(np.trace(np.dot(\
            self.subbasis_parameter_covariance(name=name),\
            np.dot(weighted_expanded_basis, weighted_expanded_basis.T))) /\
            self.num_channels)
    
    def subbasis_weighted_basis(self, name=None):
        """
        Returns the weighted form of one of the sets of basis vectors.
        
        name: name of subbasis under concern
        """
        return self.basis_set[name].basis / self.error[np.newaxis,:]
    
    def subbasis_weighted_bias(self, name=None, true_curve=None,\
        compare_likelihood=True):
        """
        The bias of the contribution of a given subbasis to the data. This
        function requires knowledge of the "truth".
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        compare_likelihood: if True, 
                            if False, compares estimated expanded curve to
                                      given true curve using extracted error
        
        returns: 1D numpy.ndarray of weighted bias values
        """
        subbasis_channel_bias =\
            self.subbasis_channel_bias(name=name, true_curve=true_curve)
        if compare_likelihood:
            subbasis_channel_error = self.error
        else:
            subbasis_channel_error = self.subbasis_channel_error(name=name)
        if self.data.ndim == 2:
            return subbasis_channel_bias / subbasis_channel_error[np.newaxis,:]
        else:
            return subbasis_channel_bias / subbasis_channel_error
        
    
    def subbasis_bias_statistic(self, name=None, true_curve=None,\
        compare_likelihood=True):
        """
        The bias statistic of the fit to the contribution of the given
        subbasis. The bias statistic is the difference between the given true
        curve and the maximum likelihood estimate
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        compare_likelihood: if True, 
                            if False, compares estimated expanded curve to
                                      given true curve using extracted error
        
        returns: single float number representing roughly 
        """
        weighted_bias = self.subbasis_weighted_bias(name=name,\
            true_curve=true_curve, compare_likelihood=compare_likelihood)
        if self.data.ndim == 2:
            return np.einsum('ij,ij->i', weighted_bias, weighted_bias)
        else:
            return np.dot(weighted_bias, weighted_bias)
    
    def bias_score(self, training_sets, max_block_size=2**20,\
        bases_to_score=None):
        """
        Evaluates the candidate basis_set given the available training sets.
        
        training_sets: dictionary of training_sets indexed by basis name
        max_block_size: number of floats in the largest possible training set
                        block
        
        returns: scalar value of Delta
        """
        num_channels = training_sets.values()[0].shape[1]
        if any([ts.shape[1] != num_channels for ts in training_sets.values()]):
            raise ValueError("The lengths of the curves in the training " +\
                             "sets must be equal.")
        if set(self.basis_set.names) != set(training_sets.keys()):
            raise ValueError("There must be the same number of basis sets " +\
                             "as training sets.")
        if (bases_to_score is None) or (not bases_to_score):
            bases_to_score = self.basis_set.names
        score = 0.
        ordered_training_sets =\
            [training_sets[key] for key in self.basis_set.names]
        iterator = TrainingSetIterator(ordered_training_sets,\
            return_constituents=True, max_block_size=max_block_size)
        for (block, constituents) in iterator:
            fitter = Fitter(self.basis_set, block, self.error)
            for basis_to_score in bases_to_score:
                true_curve =\
                    constituents[self.basis_set.names.index(basis_to_score)]
                score += np.sum(fitter.subbasis_bias_statistic(\
                    name=basis_to_score, true_curve=true_curve,\
                    compare_likelihood=True))
        num_training_set_curves =\
            np.prod([len(ts) for ts in ordered_training_sets])
        return score / (num_training_set_curves * num_channels)
    
    def fill_hdf5_group(self, root_group):
        """
        Fills the given hdf5 file group with data about the inputs and results
        of this Fitter.
        
        root_group: the hdf5 file group to fill
        """
        group = root_group.create_group('posterior')
        group.create_dataset('parameter_mean', data=self.parameter_mean)
        group.create_dataset('parameter_covariance',\
            data=self.parameter_covariance)
        for name in self.names:
            subgroup = group.create_group(name)
            subgroup.create_dataset('parameter_covariance',\
                data=self.subbasis_parameter_covariance(name=name))
            subgroup.create_dataset('parameter_mean',\
                data=self.subbasis_parameter_mean(name=name))
            subgroup.create_dataset('channel_mean',\
                data=self.subbasis_channel_mean(name=name))
            subgroup.create_dataset('channel_error',\
                data=self.subbasis_channel_error(name=name))
        self.basis_set.fill_hdf5_group(root_group.create_group('basis_set'))
        root_group.attrs['BPIC'] = self.BPIC
        root_group.attrs['DIC'] = self.DIC
        root_group.attrs['AIC'] = self.AIC
        root_group.attrs['BIC'] = self.BIC
        if self.has_priors:
            root_group.attrs['log_evidence_per_data_channel'] =\
                self.log_evidence_per_data_channel
            group = root_group.create_group('prior')
            for name in self.names:
                subgroup = group.create_group(name)
                self.priors[name + '_prior'].fill_hdf5_group(subgroup)
    
    @property
    def sizes(self):
        """
        Property storing a dictionary with basis names as keys and the number
        of basis vectors in that basis as values.
        """
        if not hasattr(self, '_sizes'):
            self._sizes = self.basis_set.sizes
        return self._sizes
    
    def plot_overlap_matrix(self, title='Overlap matrix', fig=None, ax=None,\
        show=True, **kwargs):
        """
        Plots the overlap matrix of the total basis.
        
        title: (Optional) the title of the plot. default: 'Overlap matrix'
        fig: the matplotlib.figure object on which the plot should appear
        ax: the matplotlib.axes object on which the plot should appear
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to supply to matplotlib.pyplot.imshow()
        """
        def_kwargs = {'interpolation': None}
        def_kwargs.update(**kwargs)
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        ax.imshow(self.overlap_matrix, **def_kwargs)
        pl.colorbar()
        pl.title(title)
        if show:
            pl.show()
    
    def plot_parameter_covariance(self, title='Covariance matrix', fig=None,\
        ax=None, show=True, **kwargs):
        """
        Plots the posterior parameter covariance matrix.
        
        title: (Optional) the title of the plot. default: 'Overlap matrix'
        fig: the matplotlib.figure object on which the plot should appear
        ax: the matplotlib.axes object on which the plot should appear
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to supply to matplotlib.pyplot.imshow()
        """
        def_kwargs = {'interpolation': None}
        def_kwargs.update(**kwargs)
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        ax.imshow(self.parameter_covariance, **def_kwargs)
        pl.colorbar()
        pl.title(title)
        if show:
            pl.show()
    
    def plot_subbasis_fit(self, nsigma=1, name=None, which_data=None,\
        true_curve=None, subtract_truth=False, shorter_error=None,\
        x_values=None, title=None, xlabel='x', ylabel='y', fig=None, ax=None,\
        show_noise_level=False, noise_level_alpha=0.5, full_error_alpha=0.2,\
        colors='b', full_error_first=True, show=False):
        """
        Plots the fit of the contribution to the data from a given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        subtract_truth: Boolean which determines whether the residuals of a fit
                        are plotted or just the curves. Can only be True if
                        true_curve is given or name is None.
        shorter_error: 1D numpy.ndarray of the same length as the vectors of
                       the subbasis containing the error on the given subbasis
        x_values: (Optional) x_values to use for plot
        title: (Optional) the title of the plot
        fig: the matplotlib.figure object on which the plot should appear
        ax: the matplotlib.axes object on which the plot should appear
        show: If True, matplotlib.pyplot.show() is called before this function
              returns.
        """
        if (self.data.ndim == 2) and (which_data is None):
            which_data = 0
        if name is None:
            mean = self.channel_mean
            error = self.channel_error
        else:
            mean = self.subbasis_channel_mean(name=name)
            error = self.subbasis_channel_error(name=name)
        if self.data.ndim == 2:
            mean = mean[which_data]
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if x_values is None:
            x_values = np.arange(len(mean))
        if (true_curve is None) and (name is None):
            true_curve = self.data
            if self.data.ndim == 2:
                true_curve = true_curve[which_data]
        if (true_curve is None) and (name is not None) and subtract_truth:
            raise ValueError("Truth cannot be subtracted because it is not " +\
                             "known. Supply it as the true_curve argument " +\
                             "if you wish for it to be subtracted.")
        if subtract_truth:
            to_subtract = true_curve
            ax.plot(x_values, np.zeros_like(x_values), color='k', linewidth=2,\
                label='true')
        else:
            to_subtract = np.zeros_like(x_values)
            if true_curve is not None:
                ax.plot(x_values, true_curve, color='k', linewidth=2,\
                    label='true')
        ax.plot(x_values, mean - to_subtract, color=colors[0], linewidth=2,\
            label='mean')
        if full_error_first:
            ax.fill_between(x_values, mean - to_subtract - (nsigma * error),\
                mean - to_subtract + (nsigma * error), alpha=full_error_alpha,\
                color=colors[1])
        if show_noise_level:
            if shorter_error is not None:
                ax.fill_between(x_values,\
                    mean - to_subtract - (nsigma * shorter_error),\
                    mean - to_subtract + (nsigma * shorter_error),\
                    alpha=noise_level_alpha, color=colors[2])
            elif len(mean) == self.num_channels:
                ax.fill_between(x_values,\
                    mean - to_subtract - (nsigma * self.error),\
                    mean - to_subtract + (nsigma * self.error),\
                    alpha=noise_level_alpha, color=colors[2])
        if not error_first:
            ax.fill_between(x_values, mean - to_subtract - (nsigma * error),\
                mean - to_subtract + (nsigma * error), alpha=full_error_alpha,\
                color=colors[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            if subtract_truth:
                ax.set_title('Fit residual')
            else:
                ax.set_title('Fit curve')
        else:
            ax.set_title(title)
        if show:
            pl.show()
    
    def plot_overlap_matrix(self, row_name=None, column_name=None,\
        title='Overlap matrix', fig=None, ax=None, show=True, **kwargs):
        """
        Plots the overlap matrix between the given subbases.
        
        row_name: the (string) name of the subbasis whose parameter number will
                  be represented by the row of the returned matrix.
        column_name: the (string) name of the subbasis whose parameter number
                     will be represented by the column of the returned matrix
        title: (Optional) the title of the plot. default: 'Overlap matrix'
        fig: the matplotlib.figure object on which the plot should appear
        ax: the matplotlib.axes object on which the plot should appear
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to supply to matplotlib.pyplot.imshow()
        """
        def_kwargs = {'interpolation': None}
        def_kwargs.update(**kwargs)
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        to_show = self.subbases_overlap_matrix(row_name=row_name,\
            column_name=column_name)
        ax.imshow(to_show, **def_kwargs)
        pl.colorbar()
        pl.title(title)
        if show:
            pl.show()
    
    def plot_parameter_covariance(self, name=None, title='Covariance matrix',\
        fig=None, ax=None, show=True, **kwargs):
        """
        Plots the posterior parameter covariance matrix.
        
        name: the (string) name of the subbasis whose parameter number
              will be represented by the rows and columns of the returned
              matrix. If None, full parameter covariance is plotted.
              Default: None
        title: (Optional) the title of the plot. default: 'Overlap matrix'
        fig: the matplotlib.figure object on which the plot should appear
        ax: the matplotlib.axes object on which the plot should appear
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to supply to matplotlib.pyplot.imshow()
        """
        def_kwargs = {'interpolation': None}
        def_kwargs.update(**kwargs)
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        to_show = self.subbasis_parameter_covariances[name]
        ax.imshow(to_show, **def_kwargs)
        pl.colorbar()
        pl.title(title)
        if show:
            pl.show()

