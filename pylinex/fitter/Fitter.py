"""
File: pylinex/fitter/Fitter.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class which computes fits of data using linear
             models through analytical calculations. It has functions to output
             the signal estimate (with error), parameter covariance, and more.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from ..util import Savable, create_hdf5_dataset, psi_squared
from .TrainingSetIterator import TrainingSetIterator
from .BaseFitter import BaseFitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class Fitter(BaseFitter, Savable):
    """
    An object which takes as inputs a BasisSum object and data and error
    vectors and outputs statistics about the fit of the data assuming the error
    and using the BasisSum.
    """
    def __init__(self, basis_sum, data, error=None, **priors):
        """
        Initializes a new Analyzer object using the given inputs.
        
        basis_sum: a BasisSum object (or a Basis object, which is converted
                   internally to a BasisSum of one Basis with the name 'sole')
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
    
    @property
    def prior_significance(self):
        """
        Property storing the quantity, mu^T Lambda^{-1} mu, where mu is the
        prior mean and Lambda is the prior covariance matrix.
        """
        if not hasattr(self, '_prior_significance'):
            self._prior_significance = np.dot(self.prior_mean,\
                np.dot(self.prior_inverse_covariance, self.prior_mean))
        return self._prior_significance
    
    @property
    def log_prior_covariance_determinant(self):
        """
        Property storing the logarithm (base e) of the determinant of the prior
        parameter covariance matrix. Note that if a given prior is not given,
        it is simply not used here (to avoid getting 0 as the determinant).
        """
        if not hasattr(self, '_log_prior_covariance_determinant'):
            self._log_prior_covariance_determinant = 0
            for key in self.priors:
                this_prior_covariance = self.priors[key].covariance.A
                self._log_prior_covariance_determinant +=\
                    la.slogdet(this_prior_covariance)[1]
        return self._log_prior_covariance_determinant
    
    @property
    def data_significance(self):
        if not hasattr(self, '_data_significance'):
            if self.multiple_data_curves:
                self._data_significance =\
                    np.sum(self.weighted_translated_data ** 2, axis=1)
            else:
                self._data_significance =\
                    np.dot(self.weighted_translated_data,\
                    self.weighted_translated_data)
        return self._data_significance
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters of the fit. This is the same
        as the number of basis vectors in the basis_sum.
        """
        return self.basis_sum.num_basis_vectors
    
    @property
    def posterior_covariance_times_prior_inverse_covariance(self):
        """
        Property storing the posterior covariance multiplied on the right by
        the prior inverse covariance. This is a matrix measure of the effect of
        the data on the distribution of parameters (i.e. it approaches the zero
        matrix if the data constrains parameters much more powerfully than the
        prior and approaches the identity matrix if the prior constrains
        parameters much more powerfully than the data)
        """
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
            if self.multiple_data_curves:
                covariance_times_bias_term =\
                    np.dot(bias_term, self.parameter_covariance)
                bias_term =\
                    np.sum(bias_term * covariance_times_bias_term, axis=1)
                del covariance_times_bias_term
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
    def basis_dot_products(self):
        """
        Property storing the dot products between the Basis objects underlying
        the BasisSum this object stores.
        """
        if not hasattr(self, '_basis_dot_products'):
            self._basis_dot_products =\
                self.basis_sum.basis_dot_products(error=self.error)
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
            self._parameter_inverse_covariance = self.basis_overlap_matrix
            if self.has_priors:
                self._parameter_inverse_covariance =\
                    self._parameter_inverse_covariance +\
                    self.prior_inverse_covariance
        return self._parameter_inverse_covariance
    
    @property
    def likelihood_parameter_covariance(self):
        """
        Property storing the parameter covariance implied by the likelihood
        (always has bigger determinant than the posterior covariance).
        """
        if not hasattr(self, '_likelihood_parameter_covariance'):
            if self.has_priors:
                self._likelihood_parameter_covariance =\
                    la.inv(self.basis_overlap_matrix)
            else:
                self._likelihood_parameter_covariance =\
                    self.parameter_covariance
        return self._likelihood_parameter_covariance
    
    @property
    def likelihood_parameter_mean(self):
        """
        Property storing the parameter mean implied by the likelihood (i.e.
        disregarding priors).
        """
        if not hasattr(self, '_likelihood_parameter_mean'):
            if self.has_priors:
                self._likelihood_parameter_mean =\
                    np.dot(self.likelihood_parameter_covariance,\
                    np.dot(self.weighted_basis,\
                    self.weighted_translated_data.T)).T
            else:
                self._likelihood_parameter_mean = self.parameter_mean
        return self._likelihood_parameter_mean
    
    @property
    def likelihood_channel_mean(self):
        """
        Property storing the channel mean associated with the likelihood
        parameter mean (i.e. the result if there are no priors).
        """
        if not hasattr(self, '_likelihood_channel_mean'):
            if self.has_priors:
                self._likelihood_channel_mean = self.basis_sum.translation +\
                    np.dot(self.basis_sum.basis.T,\
                    self.likelihood_parameter_mean.T).T
            else:
                self._likelihood_channel_mean = self.channel_mean
        return self._likelihood_channel_mean
    
    @property
    def likelihood_channel_bias(self):
        """
        Property storing the channel-space bias associated with the likelihood
        parameter mean (i.e. the result if there are no priors).
        """
        if not hasattr(self, '_likelihood_channel_bias'):
            if self.has_priors:
                self._likelihood_channel_bias =\
                    self.data - self.likelihood_channel_mean
            else:
                self._likelihood_channel_bias = self.channel_bias
        return self._likelihood_channel_bias
    
    @property
    def likelihood_weighted_bias(self):
        """
        Property storing the likelihood_channel_bias weighted by the error.
        """
        if not hasattr(self, '_likelihood_weighted_bias'):
            if self.has_priors:
                if self.multiple_data_curves:
                    self._likelihood_weighted_bias =\
                        self.likelihood_channel_bias / self.error[np.newaxis,:]
                else:
                    self._likelihood_weighted_bias =\
                        self.likelihood_channel_bias / self.error
            else:
                self._likelihood_weighted_bias = self.weighted_bias
        return self._likelihood_weighted_bias
    
    @property
    def likelihood_bias_statistic(self):
        """
        Property storing the maximum value of the loglikelihood. This is a
        number equal to (delta^T C^{-1} delta) where delta is the
        likelihood_channel_bias. It is equal to -2 times the loglikelihood.
        """
        if not hasattr(self, '_likelihood_bias_statistic'):
            if self.has_priors:
                if self.multiple_data_curves:
                    self._likelihood_bias_statistic =\
                        np.sum(self.likelihood_weighted_bias ** 2, axis=1)
                else:
                    self._likelihood_bias_statistic = np.dot(\
                        self.likelihood_weighted_bias,\
                        self.likelihood_weighted_bias)
            else:
                self._likelihood_bias_statistic = self.bias_statistic
        return self._likelihood_bias_statistic
    
    @property
    def degrees_of_freedom(self):
        """
        Property storing the difference between the number of channels and the
        number of parameters.
        """
        if not hasattr(self, '_degrees_of_freedom'):
            self._degrees_of_freedom = self.num_channels - self.num_parameters
        return self._degrees_of_freedom
    
    @property
    def normalized_likelihood_bias_statistic(self):
        """
        Property storing the normalized version of the likelihood bias
        statistic. This is a statistic that should be close to 1 which measures
        how well the total data is fit.
        """
        if not hasattr(self, '_normalized_likelihood_bias_statistic'):
            self._normalized_likelihood_bias_statistic =\
                self.likelihood_bias_statistic / self.degrees_of_freedom
        return self._normalized_likelihood_bias_statistic
    
    @property
    def chi_squared(self):
        """
        Property that returns the reduced chi-squared values of the fit(s) in
        this Fitter.
        """
        return self.normalized_bias_statistic
    
    @property
    def psi_squared(self):
        """
        Property storing the reduced psi-squared values of the fit(s) in this
        Fitter.
        """
        if not hasattr(self, '_psi_squared'):
            if self.multiple_data_curves:
                self._psi_squared =\
                    np.array([psi_squared(bias, error=self.error)\
                    for bias in self.channel_bias])
            else:
                self._psi_squared =\
                    psi_squared(self.channel_bias, error=self.error)
        return self._psi_squared
    
    @property
    def maximum_loglikelihood(self):
        """
        Property storing the maximum value of the Gaussian loglikelihood (when
        the normalizing constant outside the exponential is left off).
        """
        if not hasattr(self, '_maximum_loglikelihood'):
            self._maximum_loglikelihood =\
                (-(self.likelihood_bias_statistic / 2.))
        return self._maximum_loglikelihood
    
    @property
    def parameter_covariance(self):
        """
        Property storing the covariance matrix of the posterior parameter
        distribution.
        """
        if not hasattr(self, '_parameter_covariance'):
            self._parameter_covariance =\
                la.inv(self.parameter_inverse_covariance)
        return self._parameter_covariance
    
    @property
    def log_parameter_covariance_determinant(self):
        """
        Property storing the logarithm (base e) of the determinant of the
        posterior parameter covariance matrix.
        """
        if not hasattr(self, '_log_parameter_covariance_determinant'):
            self._log_parameter_covariance_determinant =\
                la.slogdet(self.parameter_covariance)[1]
        return self._log_parameter_covariance_determinant
    
    @property
    def log_parameter_covariance_determinant_ratio(self):
        """
        Property storing the logarithm (base e) of the ratio of the determinant
        of the posterior parameter covariance matrix to the determinant of the
        prior parameter covariance matrix. This can be thought of as the log of
        the ratio of the hypervolume of the 1 sigma posterior ellipse to the
        hypervolume of the 1 sigma prior ellipse.
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
            SAT = np.dot(self.parameter_covariance, self.basis_sum.basis)
            self._channel_error =\
                np.sqrt(np.einsum('ab,ab->b', self.basis_sum.basis, SAT))
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
                np.dot(self.weighted_basis, self.weighted_translated_data.T).T
            if self.has_priors:
                if self.multiple_data_curves:
                    self._parameter_mean = self._parameter_mean +\
                        self.prior_inverse_covariance_times_mean[np.newaxis,:]
                else:
                    self._parameter_mean = self._parameter_mean +\
                        self.prior_inverse_covariance_times_mean
            self._parameter_mean =\
                np.dot(self.parameter_covariance, self._parameter_mean.T).T
        return self._parameter_mean
    
    @property
    def parameter_distribution(self):
        """
        Property storing a GaussianDistribution object describing the
        distribution of the parameters of this fitter.
        """
        if not hasattr(self, '_parameter_distribution'):
            if self.multiple_data_curves:
                raise ValueError("parameter_distribution only makes sense " +\
                    "if the Fitter has only one data curve.")
            else:
                self._parameter_distribution = GaussianDistribution(\
                    self.parameter_mean, self.parameter_covariance)
        return self._parameter_distribution
    
    @property
    def posterior_significance(self):
        """
        Property storing the quantity, x^T S^{-1} x, where x is the posterior
        mean and S is the posterior parameter covariance matrix.
        """
        if not hasattr(self, '_posterior_significance'):
            if self.multiple_data_curves:
                inverse_covariance_times_mean = np.dot(self.parameter_mean,\
                    self.parameter_inverse_covariance)
                self._posterior_significance = np.sum(\
                    self.parameter_mean * inverse_covariance_times_mean,\
                    axis=1)
                del inverse_covariance_times_mean
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
            self._channel_mean = self.basis_sum.translation +\
                np.dot(self.basis_sum.basis.T, self.parameter_mean.T).T
        return self._channel_mean
    
    @property
    def channel_bias(self):
        """
        Property storing the bias of the estimate of the data (i.e. the maximum
        likelihood estimate of the data minus the data)
        """
        if not hasattr(self, '_channel_bias'):
            self._channel_bias = self.data - self.channel_mean
        return self._channel_bias
    
    @property
    def channel_bias_RMS(self):
        """
        Property storing the RMS of the channel bias.
        """
        if not hasattr(self, '_channel_bias_RMS'):
            if self.multiple_data_curves:
                self._channel_bias_RMS = np.sqrt(\
                    np.sum(self.channel_bias ** 2, axis=1) / self.num_channels)
            else:
                self._channel_bias_RMS =\
                    np.sqrt(np.dot(self.channel_bias, self.channel_bias) /\
                    self.num_channels)
        return self._channel_bias_RMS
    
    @property
    def weighted_bias(self):
        """
        Property storing the posterior channel bias weighted down by the
        errors.
        """
        if not hasattr(self, '_weighted_bias'):
            if self.multiple_data_curves:
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
            if self.multiple_data_curves:
                self._bias_statistic = np.sum(self.weighted_bias ** 2, axis=1)
            else:
                self._bias_statistic =\
                    np.dot(self.weighted_bias, self.weighted_bias)
        return self._bias_statistic
    
    @property
    def loglikelihood_at_posterior_maximum(self):
        """
        Property storing the value of the Gaussian loglikelihood (without the
        normalizing factor outside the exponential) at the maximum of the
        posterior distribution.
        """
        if not hasattr(self, '_loglikelihood_at_posterior_maximum'):
            self._loglikelihood_at_posterior_maximum =\
                (-(self.bias_statistic / 2.))
        return self._loglikelihood_at_posterior_maximum
    
    @property
    def normalized_bias_statistic(self):
        """
        Property which stores a normalized version of the bias statistic. The
        expectation value of this normed version is 1.
        """
        if not hasattr(self, '_normalized_bias_statistic'):
            self._normalized_bias_statistic =\
                self.bias_statistic / self.degrees_of_freedom
        return self._normalized_bias_statistic
    
    @property
    def likelihood_significance_difference(self):
        """
        Property storing the likelihood covariance part of the significance
        difference. This is equal to (gamma^T C^-1 gamma - y^T C^-1 y) where
        gamma is the posterior channel mean, C is the likelihood channel
        covariance (i.e. data error), y is the data.
        """
        if not hasattr(self, '_likelihood_significance_difference'):
            if self.multiple_data_curves:
                error_to_divide = self.error[np.newaxis,:]
            else:
                error_to_divide = self.error
            mean_sum = (self.channel_mean + self.data -\
                (2 * self.basis_sum.translation)) / error_to_divide
            mean_difference = (self.channel_mean - self.data) / error_to_divide
            if self.multiple_data_curves:
                self._likelihood_significance_difference =\
                    np.sum(mean_sum * mean_difference, axis=1)
            else:
                self._likelihood_significance_difference =\
                    np.dot(mean_sum, mean_difference)
        return self._likelihood_significance_difference
    
    @property
    def prior_significance_difference(self):
        """
        Property storing the prior covariance part of the significance
        difference. This is equal to (nu^T Lambda^-1 nu - mu^T Lambda^-1 mu)
        where mu is the prior mean, nu is the posterior mean, and Lambda is the
        prior covariance matrix.
        """
        if not hasattr(self, '_prior_significance_difference'):
            if self.multiple_data_curves:
                self._prior_significance_difference =\
                    np.zeros(self.data.shape[:-1])
            else:
                self._prior_significance_difference = 0
            for name in self.names:
                key = '{!s}_prior'.format(name)
                if key in self.priors:
                    prior = self.priors[key]
                    prior_mean = prior.internal_mean.A[0]
                    prior_inverse_covariance = prior.inverse_covariance.A
                    posterior_mean = self.subbasis_parameter_mean(name=name)
                    mean_sum = posterior_mean + prior_mean
                    mean_difference = posterior_mean - prior_mean
                    if self.multiple_data_curves:
                        this_term =\
                            np.dot(mean_difference, prior_inverse_covariance)
                        this_term = np.sum(this_term * mean_sum, axis=1)
                    else:
                        this_term = np.dot(mean_sum,\
                            np.dot(prior_inverse_covariance, mean_difference))
                    self._prior_significance_difference =\
                        self._prior_significance_difference + this_term
                            
        return self._prior_significance_difference
    
    @property
    def significance_difference(self):
        """
        Property storing the difference between the posterior significance and
        the sum of the data significance and prior significance. It is a term
        in the log evidence.
        """
        if not hasattr(self, '_significance_difference'):
            self._significance_difference =\
                self.likelihood_significance_difference +\
                self.prior_significance_difference
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
            if self.has_all_priors:
                # only constants added below, ignore if numerical problems
                self._log_evidence = self._log_evidence -\
                    (((self.num_channels * np.log(2 * np.pi)) / 2.) -\
                    np.sum(np.log(self.error)))
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
            self._bayesian_information_criterion =\
                self.likelihood_bias_statistic +\
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
               self.likelihood_bias_statistic + (2 * self.num_parameters)
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
            self._deviance_information_criterion =\
                self.likelihood_bias_statistic +\
                (2 * self.model_complexity_mean_to_peak_logL)
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
                self.likelihood_bias_statistic +\
                self.model_complexity_logL_variance
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
            if self.multiple_data_curves:
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
            if self.has_priors: # TODO
                self._bayesian_predictive_information_criterion -= np.trace(\
                    self.posterior_covariance_times_prior_inverse_covariance)
                term_v1 = np.dot(\
                    self.posterior_covariance_times_prior_inverse_covariance,\
                    self.posterior_prior_mean_difference.T).T
                term_v2 = np.dot(self.prior_inverse_covariance,\
                    self.posterior_prior_mean_difference.T).T +\
                    (2 * np.dot(self.weighted_basis, self.weighted_bias.T).T)
                if self.multiple_data_curves:
                    self._bayesian_predictive_information_criterion +=\
                        (np.sum(term_v1 * term_v2, axis=1) / self.num_channels)
                else:
                    self._bayesian_predictive_information_criterion +=\
                        (np.dot(term_v1, term_v2) / self.num_channels)
            weighted_error = self.channel_error / self.error
            if self.multiple_data_curves:
                weighted_error = weighted_error[np.newaxis,:]
            to_sum = ((weighted_error * self.weighted_bias) ** 2)
            self._bayesian_predictive_information_criterion +=\
                (2 * np.sum(to_sum, axis=-1))
            del to_sum
        return self._bayesian_predictive_information_criterion
    
    @property
    def BPIC(self):
        """
        Alias for the bayesian_predictive_information_criterion property.
        """
        return self.bayesian_predictive_information_criterion
    
    def subbasis_log_separation_evidence(self, name=None):
        """
        Calculates the subbasis_log_separation evidence per degree of freedom.
        This is the same as the evidence with the log covariance determinant
        ratio replaced by the log covariance determinant ratio for the given
        subbasis (normalized by the degrees of freedom).
        
        name: string identifying subbasis under concern
        per_channel: if True, normalizes the log_separation_evidence by
                              dividing by the nuiber of data channels.
        
        returns: single float number
        """
        if not hasattr(self, '_subbasis_log_separation_evidences'):
            self._subbasis_log_separation_evidences = {}
        if name not in self._subbasis_log_separation_evidences:
            self._subbasis_log_separation_evidences[name] =\
                (self.log_evidence -\
                (self.log_parameter_covariance_determinant_ratio / 2.) +\
                (self.subbasis_log_parameter_covariance_determinant_ratio(\
                name=name) / 2.)) / self.degrees_of_freedom
        return self._subbasis_log_separation_evidences[name]
    
    def subbasis_separation_evidence_per_degree_of_freedom(self, name=None):
        """
        Finds the subbasis separation evidence per degree of freedom.
        
        name: string identifying subbasis under concern
        
        returns: single non-negative float number
        """
        if not hasattr(self,\
            '_subbasis_separation_evidences_per_degree_of_freedom'):
            self._subbasis_separation_evidences_per_degree_of_freedom = {}
        if name not in\
            self._subbasis_separation_evidences_per_degree_of_freedom:
            self._subbasis_separation_evidences_per_degree_of_freedom[name] =\
                np.exp(self.subbasis_log_separation_evidence(name=name))
        return self._subbasis_separation_evidences_per_degree_of_freedom[name]
    
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
        if not hasattr(self, '_subbasis_prior_significances'):
            self._subbasis_prior_significances = {}
        if name not in self._subbasis_prior_significances:
            prior = self.priors[name + '_prior']
            mean = prior.internal_mean.A[0]
            inverse_covariance = prior.inverse_covariance.A
            self._subbasis_prior_significances[name] =\
                np.dot(mean, np.dot(inverse_covariance, mean))
        return self._subbasis_prior_significances[name]
    
    def subbasis_parameter_inverse_covariance(self, name=None):
        """
        Finds the inverse of the marginalized covariance matrix corresponding
        to the given subbasis.
        
        name: string identifying subbasis under concern
        """
        if not hasattr(self, '_subbasis_parameter_inverse_covariances'):
            self._subbasis_parameter_inverse_covariances = {}
        if name not in self._subbasis_parameter_inverse_covariances:
            self._subbasis_parameter_inverse_covariances[name] =\
                la.inv(self.subbasis_parameter_covariance(name=name))
        return self._subbasis_parameter_inverse_covariances[name]

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
        row_slice = self.basis_sum.slices_by_name[row_name]
        column_slice = self.basis_sum.slices_by_name[column_name]
        return self.basis_overlap_matrix[:,column_slice][row_slice]
    
    def subbasis_parameter_covariance(self, name=None):
        """
        Finds and returns the portion of the parameter covariance matrix
        associated with the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns 2D numpy.ndarray of shape (k, k) where k is the number of basis
                vectors in the subbasis
        """
        if not hasattr(self, '_subbasis_parameter_covariances'):
            self._subbasis_parameter_covariances = {}
        if name not in self._subbasis_parameter_covariances:
            subbasis_slice = self.basis_sum.slices_by_name[name]
            self._subbasis_parameter_covariances[name] =\
                self.parameter_covariance[:,subbasis_slice][subbasis_slice]
        return self._subbasis_parameter_covariances[name]
    
    def subbasis_log_parameter_covariance_determinant(self, name=None):
        """
        Finds the logarithm (base e) of the determinant of the posterior
        parameter covariance matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        if not hasattr(self,\
            '_subbasis_log_parameter_covariance_determinants'):
            self._subbasis_log_parameter_covariance_determinants = {}
        if name not in self._subbasis_log_parameter_covariance_determinants:
            self._subbasis_log_parameter_covariance_determinants[name] =\
                la.slogdet(self.subbasis_parameter_covariance(name=name))[1]
        return self._subbasis_log_parameter_covariance_determinants[name]
    
    def subbasis_log_prior_covariance_determinant(self, name=None):
        """
        Finds the logarithm (base e) of the determinant of the prior parameter
        covariance matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        if type(name) is type(None):
            return self.log_prior_covariance_determinant
        if not hasattr(self, '_subbasis_log_prior_covariance_determinants'):
            self._subbasis_log_prior_covariance_determinants = {}
        if name not in self._subbasis_log_prior_covariance_determinants:
            self._subbasis_log_prior_covariance_determinants[name] =\
                la.slogdet(self.priors[name + '_prior'].covariance.A)[1]
        return self._subbasis_log_prior_covariance_determinants[name]
    
    def subbasis_log_parameter_covariance_determinant_ratio(self, name=None):
        """
        Finds logarithm (base e) of the ratio of the determinant of the
        posterior covariance matrix to the determinant of the prior covariance
        matrix for the given subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single float number
        """
        if not hasattr(self,\
            '_subbasis_log_parameter_covariance_determinant_ratios'):
            self._subbasis_log_parameter_covariance_determinant_ratios = {}
        if name not in\
            self._subbasis_log_parameter_covariance_determinant_ratios:
            self._subbasis_log_parameter_covariance_determinant_ratios[name] =\
                self.subbasis_log_parameter_covariance_determinant(name=name)-\
                self.subbasis_log_prior_covariance_determinant(name=name)
        return self._subbasis_log_parameter_covariance_determinant_ratios[name]
    
    def subbasis_parameter_covariance_determinant_ratio(self, name=None):
        """
        Finds the ratio of the determinant of the posterior covariance matrix
        to the determinant of the prior covariance matrix for the given
        subbasis.
        
        name: string identifying subbasis under concern
        
        returns: single non-negative float number
        """
        if not hasattr(self,\
            '_subbasis_parameter_covariance_determinant_ratios'):
            self._subbasis_parameter_covariance_determinant_ratios = {}
        if type(name) is type(None):
            self._subbasis_parameter_covariance_determinant_ratios[name] =\
                np.exp(\
                self.subbasis_log_parameter_covariance_determinant_ratios_sum)
        elif name not in\
             self._subbasis_parameter_covariance_determinant_ratios:
             self._subbasis_parameter_covariance_determinant_ratios[name] =\
                 np.exp(\
                 self.subbasis_log_parameter_covariance_determinant_ratio(\
                 name=name))
        return self._subbasis_parameter_covariance_determinant_ratios[name]
    
    def subbasis_channel_error(self, name=None):
        """
        Finds the error (in data channel space) of the fit by a given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray of the same length as the basis vectors of
                 the subbasis (which may or may not be different than the
                 length of the expanded basis vectors).
        """
        if type(name) is type(None):
            return self.channel_error
        if not hasattr(self, '_subbasis_channel_errors'):
            self._subbasis_channel_errors = {}
        if name not in self._subbasis_channel_errors:
            basis = self.basis_sum[name].basis
            covariance_times_basis =\
                np.dot(self.subbasis_parameter_covariance(name=name), basis)
            self._subbasis_channel_errors[name] =\
                np.sqrt(np.sum(covariance_times_basis * basis, axis=0))
        return self._subbasis_channel_errors[name]
    
    def subbasis_parameter_mean(self, name=None):
        """
        Finds the posterior parameter mean for a subbasis. This is just a view
        into the view posterior parameter mean.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray containing the parameters for the given
                 subbasis
        """
        if not hasattr(self, '_subbasis_parameter_means'):
            self._subbasis_parameter_means = {}
        if name not in self._subbasis_parameter_means:
            self._subbasis_parameter_means[name] =\
                self.parameter_mean[...,self.basis_sum.slices_by_name[name]]
        return self._subbasis_parameter_means[name]
    
    def subbasis_channel_mean(self, name=None):
        """
        The estimate of the contribution to the data from the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: 1D numpy.ndarray containing the channel-space estimate from
                 the given subbasis
        """
        if not hasattr(self, '_subbasis_channel_means'):
            self._subbasis_channel_means = {}
        if name not in self._subbasis_channel_means:
            self._subbasis_channel_means[name] =\
                np.dot(self.subbasis_parameter_mean(name=name),\
                self.basis_sum[name].basis) + self.basis_sum[name].translation
        return self._subbasis_channel_means[name]
    
    def subbasis_channel_RMS(self, name=None):
        """
        Calculates and returns the RMS channel error on the estimate of the
        contribution to the data from the given subbasis.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        
        returns: single float number RMS
        """
        if not hasattr(self, '_subbasis_channel_RMSs'):
            self._subbasis_channel_RMSs = {}
        if name not in self._subbasis_channel_RMSs:
            self._subbasis_channel_RMSs[name] = np.sqrt(\
                np.mean(np.power(self.subbasis_channel_error(name=name), 2)))
        return self._subbasis_channel_RMSs[name]
    
    def subbasis_separation_statistic(self, name=None):
        """
        Finds the separation statistic associated with the given subbasis. The
        separation statistic is essentially an RMS'd error expansion factor.
        
        name: name of the subbasis for which to find the separation statistic
        """
        if not hasattr(self, '_subbasis_separation_statistics'):
            self._subbasis_separation_statistics = {}
        if name not in self._subbasis_separation_statistics:
            weighted_basis =\
                self.basis_sum[name].expanded_basis / self.error[np.newaxis,:]
            stat = np.dot(weighted_basis, weighted_basis.T)
            stat = np.sum(stat * self.subbasis_parameter_covariance(name=name))
            stat = np.sqrt(stat / self.degrees_of_freedom)
            self._subbasis_separation_statistics[name] = stat
        return self._subbasis_separation_statistics[name]
    
    def subbasis_channel_bias(self, name=None, true_curve=None):
        """
        Calculates and returns the bias on the estimate from the given subbasis
        using the given curve as a reference.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis channel space
        
        returns: 1D numpy.ndarray in channel space containing the difference
                 between the estimate of the data's contribution from the given
                 subbasis and the given true curve
        """
        if type(name) is type(None):
            if type(true_curve) is type(None):
                return self.channel_bias
            else:
                raise ValueError("true_curve should only be given to " +\
                                 "subbasis_channel_bias if the name of a " +\
                                 "subbasis is specified.")
        else:
            if type(true_curve) is type(None):
                raise ValueError("true_curve must be given to " +\
                                 "subbasis_channel_bias if the name of a " +\
                                 "subbasis is specified.")
            if self.multiple_data_curves and (true_curve.ndim == 1):
                return true_curve[np.newaxis,:] -\
                    self.subbasis_channel_mean(name=name)
            else:
                return true_curve - self.subbasis_channel_mean(name=name)
    
    def subbasis_weighted_bias(self, name=None, true_curve=None):
        """
        The bias of the contribution of a given subbasis to the data. This
        function requires knowledge of the "truth".
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        
        returns: 1D numpy.ndarray of weighted bias values
        """
        subbasis_channel_bias =\
            self.subbasis_channel_bias(name=name, true_curve=true_curve)
        subbasis_channel_error = self.subbasis_channel_error(name=name)
        if self.multiple_data_curves:
            return subbasis_channel_bias / subbasis_channel_error[np.newaxis,:]
        else:
            return subbasis_channel_bias / subbasis_channel_error
    
    def subbasis_bias_statistic(self, name=None, true_curve=None,\
        norm_by_dof=False):
        """
        The bias statistic of the fit to the contribution of the given
        subbasis. The bias statistic is delta^T C^-1 delta where delta is the
        difference between the true curve(s) and the channel mean(s) normalized
        by the degrees of freedom.
        
        name: (string) name of the subbasis under consideration. if None is
              given, the full basis is used.
        true_curve: 1D numpy.ndarray of the same length as the basis vectors in
                    the subbasis
        norm_by_dof: if True, summed squared subbasis error weighted subbasis
                              bias is normalized by the subbasis degrees of
                              freedom
                     if False (default), summed squared subbasis error weighted
                                         subbasis bias is returned is
                                         normalized by the number of channels
                                         in the subbasis
        
        returns: single float number representing roughly 
        """
        weighted_bias = self.subbasis_weighted_bias(name=name,\
            true_curve=true_curve)
        normalization_factor = weighted_bias.shape[-1]
        if norm_by_dof:
            normalization_factor -= self.basis_sum[name].num_basis_vectors
        if self.multiple_data_curves:
            unnormalized = np.sum(weighted_bias ** 2, axis=1)
        else:
            unnormalized = np.dot(weighted_bias, weighted_bias)
        return unnormalized / normalization_factor
    
    def bias_score(self, training_sets, max_block_size=2**20,\
        num_curves_to_score=None, bases_to_score=None):
        """
        Evaluates the candidate basis_sum given the available training sets.
        
        training_sets: dictionary of training_sets indexed by basis name
        max_block_size: number of floats in the largest possible training set
                        block
        num_curves_to_score: total number of training set curves to consider
        bases_to_score: the names of the subbases to include in the scoring
                        (all bases are always used, the names not in
                        bases_to_score simply do not have their
                        subbasis_bias_statistic calculated/included)
        
        returns: scalar value of Delta
        """
        if len(self.basis_sum.names) != len(training_sets):
            raise ValueError("There must be the same number of basis sets " +\
                "as training sets.")
        if (type(bases_to_score) is type(None)) or (not bases_to_score):
            bases_to_score = self.basis_sum.names
        score = 0.
        expanders = [basis.expander for basis in self.basis_sum]
        iterator = TrainingSetIterator(training_sets, expanders=expanders,\
            max_block_size=max_block_size, mode='add',\
            curves_to_return=num_curves_to_score, return_constituents=True)
        for (block, constituents) in iterator:
            num_channels = block.shape[1]
            fitter = Fitter(self.basis_sum, block, self.error, **self.priors)
            for basis_to_score in bases_to_score:
                true_curve =\
                    constituents[self.basis_sum.names.index(basis_to_score)]
                result = fitter.subbasis_bias_statistic(\
                    name=basis_to_score, true_curve=true_curve)
                score += np.sum(result)
        if type(num_curves_to_score) is type(None):
            num_curves_to_score =\
                np.prod([ts.shape[0] for ts in training_sets])
        score = score / (num_curves_to_score * num_channels)
        return score
    
    def fill_hdf5_group(self, root_group, data_link=None, error_link=None,\
        basis_links=None, expander_links=None, prior_mean_links=None,\
        prior_covariance_links=None, save_channel_estimates=False):
        """
        Fills the given hdf5 file group with data about the inputs and results
        of this Fitter.
        
        root_group: the hdf5 file group to fill (only required argument)
        data_link: link to existing data dataset, if it exists (see
                   create_hdf5_dataset docs for info about accepted formats)
        error_link: link to existing error dataset, if it exists (see
                    create_hdf5_dataset docs for info about accepted formats)
        basis_links: list of links to basis functions saved elsewhere (see
                     create_hdf5_dataset docs for info about accepted formats)
        expander_links: list of links to existing saved Expander (see
                        create_hdf5_dataset docs for info about accepted
                        formats)
        prior_mean_links: dict of links to existing saved prior means (see
                          create_hdf5_dataset docs for info about accepted
                          formats)
        prior_covariance_links: dict of links to existing saved prior
                                covariances (see create_hdf5_dataset docs for
                                info about accepted formats)
        """
        self.save_data(root_group, data_link=data_link)
        self.save_error(root_group, error_link=error_link)
        group = root_group.create_group('sizes')
        for name in self.names:
            group.attrs[name] = self.sizes[name]
        group = root_group.create_group('posterior')
        create_hdf5_dataset(group, 'parameter_mean', data=self.parameter_mean)
        create_hdf5_dataset(group, 'parameter_covariance',\
            data=self.parameter_covariance)
        if save_channel_estimates:
            create_hdf5_dataset(group, 'channel_mean', data=self.channel_mean)
        create_hdf5_dataset(group, 'channel_error', data=self.channel_error)
        for name in self.names:
            subgroup = group.create_group(name)
            subbasis_slice = self.basis_sum.slices_by_name[name]
            create_hdf5_dataset(subgroup, 'parameter_covariance',\
                link=(group['parameter_covariance'],[subbasis_slice]*2))
            mean_slices =\
                (((slice(None),) * (self.data.ndim - 1)) + (subbasis_slice,))
            create_hdf5_dataset(subgroup, 'parameter_mean',\
                link=(group['parameter_mean'],mean_slices))
            if save_channel_estimates:
                create_hdf5_dataset(subgroup, 'channel_mean',\
                    data=self.subbasis_channel_mean(name=name))
            create_hdf5_dataset(subgroup, 'channel_error',\
                data=self.subbasis_channel_error(name=name))
        self.save_basis_sum(root_group, basis_links=basis_links,\
            expander_links=expander_links)
        root_group.attrs['degrees_of_freedom'] = self.degrees_of_freedom
        root_group.attrs['BPIC'] = self.BPIC
        root_group.attrs['DIC'] = self.DIC
        root_group.attrs['AIC'] = self.AIC
        root_group.attrs['BIC'] = self.BIC
        root_group.attrs['normalized_likelihood_bias_statistic'] =\
            self.normalized_likelihood_bias_statistic
        root_group.attrs['normalized_bias_statistic'] =\
            self.normalized_bias_statistic
        self.save_priors(root_group, prior_mean_links=prior_mean_links,\
            prior_covariance_links=prior_covariance_links)
        if self.has_priors:
            root_group.attrs['log_evidence_per_data_channel'] =\
                self.log_evidence_per_data_channel
    
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
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        image = ax.imshow(self.basis_overlap_matrix, **def_kwargs)
        pl.colorbar(image)
        ax.set_title(title)
        if show:
            pl.show()
        else:
            return ax
    
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
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        image = ax.imshow(self.parameter_covariance, **def_kwargs)
        pl.colorbar(image)
        ax.set_title(title)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_subbasis_fit(self, nsigma=1, name=None, which_data=None,\
        true_curve=None, subtract_truth=False, shorter_error=None,\
        x_values=None, title=None, xlabel='x', ylabel='y', fig=None, ax=None,\
        show_noise_level=False, noise_level_alpha=0.5, full_error_alpha=0.2,\
        colors='b', full_error_first=True, yscale='linear', show=False):
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
        if self.multiple_data_curves and (type(which_data) is type(None)):
            which_data = 0
        if type(name) is type(None):
            mean = self.channel_mean
            error = self.channel_error
        else:
            mean = self.subbasis_channel_mean(name=name)
            error = self.subbasis_channel_error(name=name)
        if isinstance(colors, basestring):
            colors = [colors] * 3
        if self.multiple_data_curves:
            mean = mean[which_data]
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if type(x_values) is type(None):
            x_values = np.arange(len(mean))
        if (type(true_curve) is type(None)) and (type(name) is type(None)):
            if self.multiple_data_curves:
                true_curve = self.data[which_data]
            else:
                true_curve = self.data
        if (type(true_curve) is type(None)) and subtract_truth:
            raise ValueError("Truth cannot be subtracted because it is not " +\
                             "known. Supply it as the true_curve argument " +\
                             "if you wish for it to be subtracted.")
        if subtract_truth:
            to_subtract = true_curve
            ax.plot(x_values, np.zeros_like(x_values), color='k', linewidth=2,\
                label='true')
        else:
            to_subtract = np.zeros_like(x_values)
            if type(true_curve) is not type(None):
                ax.plot(x_values, true_curve, color='k', linewidth=2,\
                    label='true')
        ax.plot(x_values, mean - to_subtract, color=colors[0], linewidth=2,\
            label='mean')
        if full_error_first:
            ax.fill_between(x_values, mean - to_subtract - (nsigma * error),\
                mean - to_subtract + (nsigma * error), alpha=full_error_alpha,\
                color=colors[1])
        if show_noise_level:
            if type(shorter_error) is not type(None):
                ax.fill_between(x_values,\
                    mean - to_subtract - (nsigma * shorter_error),\
                    mean - to_subtract + (nsigma * shorter_error),\
                    alpha=noise_level_alpha, color=colors[2])
            elif len(mean) == self.num_channels:
                ax.fill_between(x_values,\
                    mean - to_subtract - (nsigma * self.error),\
                    mean - to_subtract + (nsigma * self.error),\
                    alpha=noise_level_alpha, color=colors[2])
        if not full_error_first:
            ax.fill_between(x_values, mean - to_subtract - (nsigma * error),\
                mean - to_subtract + (nsigma * error), alpha=full_error_alpha,\
                color=colors[1])
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if type(title) is type(None):
            if subtract_truth:
                ax.set_title('Fit residual')
            else:
                ax.set_title('Fit curve')
        else:
            ax.set_title(title)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_overlap_matrix_block(self, row_name=None, column_name=None,\
        title='Overlap matrix', fig=None, ax=None, show=True, **kwargs):
        """
        Plots a block of the overlap matrix between the given subbases.
        
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
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        to_show = self.subbases_overlap_matrix(row_name=row_name,\
            column_name=column_name)
        image = ax.imshow(to_show, **def_kwargs)
        pl.colorbar(image)
        ax.set_title(title)
        if show:
            pl.show()
        else:
            return ax
    
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
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        to_show = self.subbasis_parameter_covariances[name]
        image = ax.imshow(to_show, **def_kwargs)
        pl.colorbar(image)
        ax.set_title(title)
        if show:
            pl.show()
        else:
            return ax

