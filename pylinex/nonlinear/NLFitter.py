"""
File: pylinex/nonlinear/NLFitter.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: File containing class which analyzes MCMC chains in order to infer
             things about the parameter distributions they describe.
"""
from __future__ import division
import re, h5py
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ..util import sequence_types, real_numerical_types, bool_types,\
    int_types, univariate_histogram, bivariate_histogram, triangle_plot
from ..model import CompoundModel
from ..loglikelihood import load_loglikelihood_from_hdf5_group,\
    GaussianLoglikelihood
from .BurnRule import BurnRule

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class NLFitter(object):
    """
    Class which analyzes MCMC chains in order to infer things about the
    parameter distributions they describe.
    """
    def __init__(self, file_name, burn_rule=None, load_all_chunks=True):
        """
        Initializes an NLFitter to analyze the chain in the hdf5 file at
        file_name.
        
        file_name: the hdf5 file in which a chain was saved by a Sampler object
        burn_rule: if None, none of the chain is excluded as a burn-in
                   otherwise, must be a BurnRule object
        load_all_chunks: if True, all chunks are considered by this fitter
                         if False, only the last saved chunk is considered
        """
        self.file_name = file_name
        self.burn_rule = burn_rule
        self.load_all_chunks = load_all_chunks
    
    @property
    def load_all_chunks(self):
        """
        Property storing a boolean describing whether all chunks (as opposed to
        just the last one) should be loaded.
        """
        if not hasattr(self, '_load_all_chunks'):
            raise AttributeError("load_all_chunks referenced before it was " +\
                "set.")
        return self._load_all_chunks
    
    @load_all_chunks.setter
    def load_all_chunks(self, value):
        """
        Setter for the load_all_chunks property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._load_all_chunks = value
        else:
            raise TypeError("load_all_chunks was set to a non-bool.")
    
    def __enter__(self):
        """
        Enters a with statement by binding the variable to this object.
        """
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits a with statement by closing the relevant hdf5 file.
        
        exc_type: type of exception thrown in with-statement (if any)
        exc_value: the exception thrown in with-statement (if any)
        traceback: traceback of the exception thrown in with-statement (if any)
        """
        self.close()
    
    @property
    def burn_rule(self):
        """
        Property storing the BurnRule object describing how the chain will be
        trimmed.
        """
        if not hasattr(self, '_burn_rule'):
            raise AttributeError("burn_rule was referenced before it was set.")
        return self._burn_rule
    
    @burn_rule.setter
    def burn_rule(self, value):
        """
        Setter for the BurnRule describing how the chain is trimmed.
        
        value: if None, a BurnRule that never burns any of the chain is set
               otherwise, value must be a BurnRule object
        """
        if value is None:
            self._burn_rule =\
                BurnRule(min_checkpoints=1, desired_fraction=1.)
        elif isinstance(value, BurnRule):
            self._burn_rule = value
        else:
            raise TypeError("burn_rule was sent to something other than a " +\
                "BurnRule object.")
    
    @property
    def file_name(self):
        """
        Property storing the string name of the file with the chain to be read.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the name of the file with the chain to be read.
        
        value: string file name
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was set to a non-string.")
    
    @property
    def file(self):
        """
        Property storing the hdf5 file with information to be read.
        """
        if not hasattr(self, '_file'):
            self._file = h5py.File(self.file_name, 'r')
        return self._file
    
    def close(self):
        """
        Closes the file at the heart of this fitter.
        """
        self.file.close()
        
    @property
    def loglikelihood(self):
        """
        Property storing the Loglikelihood object which is explored in the
        chain being analyzed.
        """
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood =\
                load_loglikelihood_from_hdf5_group(self.file['loglikelihood'])
        return self._loglikelihood
    
    @property
    def model(self):
        """
        Property storing the model being used for the fit being done by this
        fitter.
        """
        return self.loglikelihood.model
    
    @property
    def chunks_to_load(self):
        """
        Property storing the chunk numbers to load for this NLFitter.
        """
        if not hasattr(self, '_chunks_to_load'):
            num_chunks = self.file.attrs['max_chunk_index'] + 1
            if self.load_all_chunks:
                self._chunks_to_load = np.arange(num_chunks)
            else:
                self._chunks_to_load = np.array([num_chunks - 1])
        return self._chunks_to_load
    
    @property
    def num_chunks_to_load(self):
        """
        Property storing the number of chunks to load for this NLFitter.
        """
        if not hasattr(self, '_num_chunks_to_load'):
            self._num_chunks_to_load = len(self.chunks_to_load)
        return self._num_chunks_to_load
    
    @property
    def num_checkpoints(self):
        """
        Property storing a sequence of length num_chunks containing the number
        of checkpoints in each chunk of this NLFitter's file.
        """
        if not hasattr(self, '_num_checkpoints'):
            self._num_checkpoints = []
            for ichunk in range(self.num_chunks_to_load):
                group = self.file['checkpoints/chunk{:d}'.format(ichunk)]
                self._num_checkpoints.append(\
                   group.attrs['max_checkpoint_index'] + 1)
        return self._num_checkpoints
    
    @property
    def total_num_checkpoints(self):
        """
        Property storing the integer total number of checkpoints to be
        considered by the BurnRule of this NLFitter.
        """
        if not hasattr(self, '_total_num_checkpoints'):
            self._total_num_checkpoints = sum(self.num_checkpoints)
        return self._total_num_checkpoints
    
    @property
    def total_num_checkpoints_to_load(self):
        """
        Property storing the integer number of checkpoints to load from the end
        of the chunks under consideration.
        """
        if not hasattr(self, '_total_num_checkpoints_to_load'):
            self._total_num_checkpoints_to_load =\
                len(self.burn_rule(self.total_num_checkpoints))
        return self._total_num_checkpoints
    
    @property
    def checkpoints_to_load(self):
        """
        Property storing a sequence of 1D numpy.ndarrays of (specified by the
        BurnRule) of checkpoints which should be loaded into the final chain.
        """
        if not hasattr(self, '_checkpoints_to_load'):
            left_to_load = len(self.burn_rule(self.total_num_checkpoints))
            backwards_answer = []
            for iichunk in range(self.num_chunks_to_load-1, -1, -1):
                ichunk = self.chunks_to_load[iichunk]
                this_num_checkpoints = self.num_checkpoints[iichunk]
                if left_to_load == 0:
                    backwards_answer.append(np.array([]))
                elif left_to_load > this_num_checkpoints:
                    left_to_load = left_to_load - this_num_checkpoints
                    backwards_answer.append(np.arange(this_num_checkpoints))
                else:
                    backwards_answer.append(\
                        np.arange(this_num_checkpoints)[-left_to_load:])
                    left_to_load = 0
            self._checkpoints_to_load = backwards_answer[-1::-1]
        return self._checkpoints_to_load
    
    def _load_checkpoints(self):
        """
        Loads the desired checkpoints of the chain, lnprobability, and
        acceptance fraction.
        """
        (chain_chunks, lnprobability_chunks, acceptance_fraction_chunks) =\
            ([], [], [])
        for (iichunk, ichunk) in enumerate(self.chunks_to_load):
            these_checkpoints_to_load = self.checkpoints_to_load[iichunk]
            if these_checkpoints_to_load.size != 0:
                checkpoints_group =\
                    self.file['checkpoints/chunk{:d}'.format(ichunk)]
                (chain_chunk, lnprobability_chunk,\
                    acceptance_fraction_chunk) = ([], [], [])
                for checkpoint_index in these_checkpoints_to_load:
                    checkpoint_group =\
                        checkpoints_group['{}'.format(checkpoint_index)]
                    chain_chunk.append(checkpoint_group['chain'].value)
                    lnprobability_chunk.append(\
                        checkpoint_group['lnprobability'].value)
                    acceptance_fraction_chunk.append(\
                        checkpoint_group['acceptance_fraction'].value)
                chain_chunks.append(np.concatenate(chain_chunk, axis=1))
                lnprobability_chunks.append(\
                    np.concatenate(lnprobability_chunk, axis=1))
                acceptance_fraction_chunks.append(\
                    np.stack(acceptance_fraction_chunk, axis=1))
                del chain_chunk, lnprobability_chunk, acceptance_fraction_chunk
        self._chain = np.concatenate(chain_chunks, axis=1)
        self._lnprobability = np.concatenate(lnprobability_chunks, axis=1)
        self._acceptance_fraction =\
            np.concatenate(acceptance_fraction_chunks, axis=1)
        del chain_chunks, lnprobability_chunks, acceptance_fraction_chunks
    
    @property
    def chain(self):
        """
        Property storing the MCMC chain in a numpy.ndarray of shape
        (nwalkers, nsteps, ndim)
        """
        if not hasattr(self, '_chain'):
            self._load_checkpoints()
        return self._chain
    
    @property
    def lnprobability(self):
        """
        Property storing the log_probability in a numpy.ndarray of shape
        (nwalkers, nsteps)
        """
        if not hasattr(self, '_lnprobability'):
            self._load_checkpoints()
        return self._lnprobability
    
    @property
    def maximum_probability_multi_index(self):
        """
        Property storing a tuple of the form (iwalker, istep) where iwalker and
        istep describe the maximizing index of the lnprobability array.
        """
        if not hasattr(self, '_maximum_probability_multi_index'):
            self._maximum_probability_multi_index = np.unravel_index(\
                np.argmax(self.lnprobability), self.lnprobability.shape)
        return self._maximum_probability_multi_index
    
    @property
    def maximum_probability_parameters(self):
        """
        Property storing the maximum probability set of parameters found by the
        MCMC sampler upon which this NLFitter is based.
        """
        if not hasattr(self, '_maximum_probability_parameters'):
            self._maximum_probability_parameters =\
                self.chain[self.maximum_probability_multi_index]
        return self._maximum_probability_parameters
    
    @property
    def maximum_probability_parameter_dictionary(self):
        """
        Property storing the maximum probability parameters in the form of a
        dictionary whose keys are the parameter names and whose values are the
        maximum probability values.
        """
        if not hasattr(self, '_maximum_probability_parameter_dictionary'):
            self._maximum_probability_parameter_dictionary =\
                {name: value for (name, value) in\
                zip(self.parameters, self.maximum_probability_parameters)}
        return self._maximum_probability_parameter_dictionary
    
    @property
    def acceptance_fraction(self):
        """
        Property storing the acceptance fraction per walker per checkpoint in
        an array of the form (nwalkers, ncheckpoints).
        """
        if not hasattr(self, '_acceptance_fraction'):
            self._load_checkpoints()
        return self._acceptance_fraction
    
    @property
    def parameters(self):
        """
        Property storing the sequence of names of the parameters being fit by
        this fitter.
        """
        if not hasattr(self, '_parameters'):
            group = self.file['parameters']
            iparameter = 0
            parameters = []
            while '{:d}'.format(iparameter) in group.attrs:
                parameters.append(group.attrs['{:d}'.format(iparameter)])
                iparameter += 1
            self._parameters = parameters
        return self._parameters
    
    @property
    def num_parameters(self):
        """
        Property storing the integer number of parameters being fit by this
        fitter.
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def nwalkers(self):
        """
        Property storing the integer number of walkers used in the MCMC chain
        being analyzed by this fitter.
        """
        if not hasattr(self, '_nwalkers'):
            self._nwalkers = self.chain.shape[0]
        return self._nwalkers
    
    @property
    def nsteps(self):
        """
        Property storing the integer number of steps in the burned MCMC chain.
        """
        if not hasattr(self, '_nsteps'):
            self._nsteps = self.chain.shape[1]
        return self._nsteps
    
    @property
    def nsamples(self):
        """
        Property storing the total number of samples in the burned chain.
        """
        if not hasattr(self, '_nsamples'):
            self._nsamples = (self.nwalkers * self.nsteps)
        return self._nsamples
    
    @property
    def flattened_chain(self):
        """
        Property storing the flattened chain (i.e. the chain with walker
        information lost) in a 2D numpy.ndarray of shape (nsamples, ndim)
        """
        if not hasattr(self, '_flattened_chain'):
            self._flattened_chain = np.swapaxes(self.chain, 0, 1)
            self._flattened_chain =\
                np.concatenate(self._flattened_chain, axis=0)
        return self._flattened_chain
    
    @property
    def flattened_lnprobability(self):
        """
        Property storing the flattened log_probability (i.e. the
        log_probabilitywith walker information lost) in a 1D numpy.ndarray of
        length nsamples.
        """
        if not hasattr(self, '_flattened_lnprobability'):
            self._flattened_lnprobability =\
                np.concatenate(self.lnprobability.T, axis=0)
        return self._flattened_probability
    
    @property
    def parameter_mean(self):
        """
        Property storing the mean parameter vector (of length ndim) of the
        flattened chain.
        """
        if not hasattr(self, '_parameter_mean'):
            self._parameter_mean = np.mean(self.flattened_chain, axis=0)
        return self._parameter_mean
    
    @property
    def parameter_covariance(self):
        """
        Property storing the parameter covariance matrix described by the
        flattened chain in a 2D numpy.ndarray of shape (ndim, ndim).
        """
        if not hasattr(self, '_parameter_covariance'):
            self._parameter_covariance =\
                np.cov(self.flattened_chain, rowvar=False)
        return self._parameter_covariance
    
    @property
    def parameter_correlation(self):
        """
        Property storing the parameter correlation matrix described by the
        flattened chain in a 2D numpy.ndarray of shape (ndim, ndim) with 1's on
        the diagonal.
        """
        if not hasattr(self, '_parameter_correlation'):
            covariance = self.parameter_covariance
            variances = np.diag(covariance)
            self._parameter_correlation = covariance /\
                np.sqrt(variances[np.newaxis,:] * variances[:,np.newaxis])
        return self._parameter_correlation
    
    @property
    def mean_deviance(self):
        """
        Property storing the mean value of -2 lnL = delta^T C^{-1} delta where
        delta is the weighted bias vector.
        """
        if not hasattr(self, '_mean_deviance'):
            self._mean_deviance =\
                -2 * np.mean(self.flattened_lnprobability)
        return self._mean_deviance
    
    @property
    def deviance_at_mean(self):
        """
        Property storing the value of -2lnL = delta^T C^{-1} delta where delta
        is the weighted bias vector evaluated at the mean parameter vector.
        """
        if not hasattr(self, '_deviance_at_mean'):
            self._deviance_at_mean =\
                -2 * self.loglikelihood(self.parameter_mean)
        return self._deviance_at_mean
    
    @property
    def effective_number_of_parameters(self):
        """
        Property storing the effective number of parameters in the model, as
        measured by the difference between the mean deviance and the deviance
        at the mean. In the Gaussian limit, this is the same as the true number
        of parameters.
        """
        if not hasattr(self, '_effective_number_of_parameters'):
            self._effective_number_of_parameters =\
                self.mean_deviance - self.deviance_at_mean
        return self._effective_number_of_parameters
    
    @property
    def half_variance_of_deviance(self):
        """
        Property storing half of the variance of the deviance. This is another
        measure of the effective number of parameters.
        """
        if not hasattr(self, '_half_variance_of_deviance'):
            self._half_variance_of_deviance =\
                (np.var(self.flattened_lnprobability) / 2.)
        return self._half_variance_of_deviance
    
    @property
    def akaike_information_criterion(self):
        """
        Property storing Akaike's information criterion given by -2ln(L_max)+2p
        where L is the likelihood and p is the number of parameters.
        """
        if not hasattr(self, '_akaike_information_criterion'):
            self._akaike_information_criterion = self.deviance_at_mean +\
                (2 * self.num_parameters)
        return self._akaike_information_criterion
    
    @property
    def AIC(self):
        """
        Alias for the akaike_information_criterion property.
        """
        return self.akaike_information_criterion
    
    @property
    def bayesian_information_criterion(self):
        """
        Property storing the Bayesian information criterion, given by
        -2ln(L_max)+p*ln(k) where L is the likelihood, p is the number of
        parameters, and k is the number of data channels.
        """
        if not hasattr(self, '_bayesian_information_criterion'):
            self._bayesian_information_criterion = self.deviance_at_mean +\
                (self.num_parameters * np.log(self.num_channels))
        return self._bayesian_information_criterion
    
    @property
    def BIC(self):
        """
        Alias for the bayesian_information_criterion property.
        """
        return self.bayesian_information_criterion
    
    @property
    def bayesian_predictive_information_criterion(self):
        """
        Property storing the Bayesian predictive information criterion, which
        is not yet implemented.
        """
        # TODO
        raise NotImplementedError("BPIC not implemented yet because it " +\
            "requires taking derivatives of the (at this point completely " +\
            "generalized) model.")
    
    @property
    def BPIC(self):
        """
        Alias for the bayesian_predictive_information_criterion property.
        """
        return self.bayesian_predictive_information_criterion
    
    @property
    def deviance_information_criterion(self):
        """
        Property storing the deviance information criterion given by the
        deviance at the mean plus twice the effective number of parameters.
        """
        if not hasattr(self, '_deviance_information_criterion'):
            self._deviance_information_criterion = self.deviance_at_mean +\
                (2 * self.effective_number_of_parameters)
        return self._deviance_information_criterion
    
    @property
    def DIC(self):
        """
        Alias for the deviance_information_criterion property.
        """
        return self.deviance_information_criterion
    
    @property
    def modified_deviance_information_criterion(self):
        """
        Property storing the deviance information criterion where the effective
        number of parameters is given by half the variance of the deviance.
        """
        if not hasattr(self, '_modified_deviance_information_criterion'):
            self._modified_deviance_information_criterion =\
                (self.deviance_at_mean + (2 * self.half_variance_of_deviance))
        return self._modified_deviance_information_criterion
    
    @property
    def DIC2(self):
        """
        Alias for the modified_deviance_information_criterion property.
        """
        return self.modified_deviance_information_criterion
    
    @property
    def rescaling_factors(self):
        """
        Property storing the rescaling factors (R-values) of Gelman Rubin et al
        for all parameters in a 1D numpy.ndarray of shape (num_parameters,)
        """
        if not hasattr(self, '_rescaling_factors'):
            half_nsteps = (self.nsteps // 2)
            first_half = self.chain[:,:half_nsteps,:]
            second_half = self.chain[:,-half_nsteps:,:]
            split_chains = np.concatenate((first_half, second_half), axis=0)
            averaged_over_chain_steps = np.mean(split_chains, axis=1)
            between_chain_variance = half_nsteps *\
                np.var(averaged_over_chain_steps, axis=0, ddof=1)
            averaged_over_chains_and_steps =\
                np.mean(averaged_over_chain_steps, axis=0)
            vared_over_chain_steps = np.var(split_chains, axis=1, ddof=1)
            within_chain_variance = np.mean(vared_over_chain_steps, axis=0)
            self._rescaling_factors =\
                (between_chain_variance / within_chain_variance)
            self._rescaling_factors =\
                np.sqrt(1. + ((self._rescaling_factors - 1.) / half_nsteps))
        return self._rescaling_factors
    
    @property
    def effective_sample_sizes(self):
        """
        Property storing the effective sample size when computed with each of
        the parameters.
        """
        if not hasattr(self, '_effective_sample_size'):
            half_nsteps = (self.nsteps // 2)
            first_half = self.chain[:,:half_nsteps,:]
            second_half = self.chain[:,-half_nsteps:,:]
            split_chains = np.concatenate((first_half, second_half), axis=0)
            # TODO define correlations (rho(t) from equation 11.7 of Gelman)
            raise NotImplementedError("effective_sample_size property not " +\
                "yet implemented.")
            self._effective_sample_size =\
                (self.nsamples / (1 + (2 * np.sum(correlations))))
        return self._effective_sample_size
    
    def get_parameter_indices(self, parameters=None):
        """
        Gets the indices associated with the given parameters input.
        
        parameters: if None, all parameter indices are returned
                    if string, it is interpreted as a regular expression with
                               which to search through the parameter list
                    if sequence of ints, describes the very indices of the
                                         parameters to be returned
                    if sequence of strings, describes parameters
                                            whose indices should be returned
        
        returns: numpy.array of parameter indices
        """
        if parameters is None:
            # return indices of all parameters
            return np.arange(self.num_parameters)
        elif isinstance(parameters, basestring):
            # assume that this is a regex for searching through parameters
            parameter_indices = []
            for (iparameter, parameter) in enumerate(self.parameters):
                if re.search(parameters, parameter):
                    parameter_indices.append(iparameter)
            return np.array(parameter_indices)
        elif type(parameters) in sequence_types:
            # parameters must be either a list of individual
            # string parameters or a list of indices
            if all([isinstance(value, basestring) for value in parameters]):
                return np.array(\
                    [self.parameters.index(value) for value in parameters])
            elif all([(type(value) in int_types) for value in parameters]):
                return np.array(parameters)
            else:
                raise TypeError("parameters was set to a sequence, but the " +\
                    "elements of that sequence were neither all strings or " +\
                    "all ints.")
        else:
            raise TypeError("parameters was set to something other than " +\
                "None, a string, a sequence of strings, and a sequence of " +\
                "ints")
    
    def sample(self, number, parameters=None):
        """
        Takes number samples from the chain associated with this fitter.
        
        number: the number of samples in the returned array
        parameters: if None, sample all parameters
                    if string, sample all parameters which match this regular
                               expression
                    if list of indices, sample parameters with these indices
                    if list of strings, sample these parameters
        
        returns: array of the form (number, nparameters)
        """
        index_sample = np.random.randint(0, high=self.nsamples, size=number)
        full_parameter_sample = self.flattened_chain[index_sample,:]
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        return full_parameter_sample[:,parameter_indices]
    
    def subbasis_parameter_mean(self, parameters):
        """
        Finds a subset of the parameter mean.
        
        parameters: if None, full parameter mean is returned
                    if string, parameter mean is returned for all parameters
                               matching the regular expression given by
                               parameters
                    if list of ints, the mean of the parameters with these
                                     indices is returned
                    if list of strings, mean of these parameters is returned
                
        returns: 1D numpy.ndarray of parameter values
        """
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        return self.parameter_mean[parameter_indices]
    
    def subbasis_parameter_covariance(self, parameters):
        """
        Finds a subset of the parameter covariance
        
        parameters: if None, full parameter covariance is returned
                    if string, parameter covariance is returned for all
                               parameters matching the regular expression given
                               by parameters
                    if list of ints, the covariance of the parameters with
                                     these indices is returned
                    if list of strings, the covariance of these parameters is
                                        returned
        
        returns: symmetric 2D numpy.ndarray of parameter covariances
        """
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        return\
            self.parameter_covariance[:,parameter_indices][parameter_indices,:]
    
    def reconstructions(self, number, parameters=None, model=None):
        """
        Computes reconstructions of a quantity by applying model to a sample of
        parameters from this fitter's chain.
        
        number: the number of reconstructions to create
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), the full model in the loglikelihood is used
               otherwise, the model with which to create reconstructions from
                          parameters
        
        returns: 2D numpy.ndarray of shape (number, nchannels) containing
                 reconstructions created from the given model and parameters
        """
        parameter_sample = self.sample(number, parameters=parameters)
        if model is None:
            model = self.model
        return np.array([model(args) for args in parameter_sample])
    
    def bias(self, number, parameters=None, model=None, true_curve=None):
        """
        Computes differences between a true_curve and reconstructions of the
        quantity made by applying model to a sample of parameters from this
        fitter's chain.
        
        number: the number of reconstructions to create and compare to data
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), full likelihood's model is used
               otherwise, the model with which to create reconstructions from
                          parameters
        true_curve: if None (default), model and parameters must also be None.
                                       true_curve is replaced with the
                                       loglikelihood's data curve.
                    Otherwise, curve to subtract from all reconstructions to
                               compute bias
        
        returns: 2D numpy.ndarray of (number, nchannels) containing differences
                 between true_curve and reconstructions created by applying
                 model to parameters
        """
        reconstructions =\
            self.reconstructions(number, parameters=parameters, model=model)
        if true_curve is None:
            if (parameters is None) and (model is None):
                true_curve = self.loglikelihood.data
            else:
                raise NotImplementedError("The bias cannot be computed if " +\
                    "no true_curve is given unless the full " +\
                    "loglikelihood's model and all parameters are used, in " +\
                    "which case, if true_curve is None, it is replaced the " +\
                    "loglikelihood's data vector.")
        return reconstructions - true_curve
    
    def reconstruction_confidence_intervals(self, number, probabilities,\
        parameters=None, model=None):
        """
        Computes confidence intervals on reconstructions of quantities created
        by applying the given model to the given parameters from this fitter's
        chain.
        
        number: the number of curves to use in calculating the confidence
                intervals
        probabilities: the confidence levels of the bands to compute
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), full likelihood's model is used
               otherwise, the model with which to create reconstructions from
                          parameters
        
        returns: list of tuples of the form (band_min, band_max) representing
                 confidence_intervals with the given confidence levels
        """
        reconstructions =\
            self.reconstructions(number, parameters=parameters, model=model)
        sorted_channel_values = np.sort(reconstructions, axis=0)
        single_input = (type(probabilities) in real_numerical_types)
        if single_input:
            probabilities = [probabilities]
        if type(probabilities) in sequence_types:
            probabilities = np.array(probabilities)
        numbers_to_exclude =\
            np.around(number * (1 - probabilities)).astype(int)
        numbers_to_exclude_from_left = (numbers_to_exclude // 2)
        numbers_to_exclude_from_right =\
            (numbers_to_exclude - numbers_to_exclude_from_left)
        left_bands = sorted_channel_values[numbers_to_exclude_from_left,:]
        right_bands = sorted_channel_values[-numbers_to_exclude_from_right,:]
        if single_input:
            return (left_bands[0], right_bands[0])
        else:
            return [(left_bands[index], right_bands[index])\
                for index in range(len(probabilities))]
    
    def bias_confidence_intervals(self, number, probabilities,\
        parameters=None, model=None, true_curve=None):
        """
        Computes confidence intervals on differences between reconstructions of
        quantities created by applying the given model to the given parameters
        from this fitter's chain and the given true curve.
        
        number: the number of curves to use in calculating the confidence
                intervals
        probabilities: the confidence levels of the bands to compute
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), full likelihood's model is used
               otherwise, the model with which to create reconstructions from
                          parameters
        true_curve: curve to subtract from all reconstructions to compute bias
        
        returns: list of tuples of the form (band_min, band_max) representing
                 confidence_intervals with the given confidence levels
        """
        intervals = self.reconstruction_confidence_intervals(\
            number, probabilities, parameters=parameters, model=model)
        if true_curve is None:
            if parameters is None and model is None:
                true_curve = self.loglikelihood.data
            else:
                raise NotImplementedError("The bias cannot be computed if " +\
                    "no true_curve is given unless the full " +\
                    "loglikelihood's model and all parameters are used, in " +\
                    "which case, if true_curve is None, it is replaced the " +\
                    "loglikelihood's data vector.")
        if type(probabilities) in real_numerical_types:
            return (intervals[0] - true_curve, intervals[1] - true_curve)
        else:
            return [(interval[0] - true_curve, interval[1] - true_curve)\
                for interval in intervals]
    
    def plot_maximum_probability_reconstruction(self, parameters=None,\
        model=None, true_curve=None, subtract_truth=False, x_values=None,\
        ax=None, xlabel=None, ylabel=None, title=None, fontsize=28,\
        scale_factor=1., show=False):
        """
        Plots maximum probability reconstruction with the given model and
        parameters.
        
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), full model is used
               otherwise, the model with which to create reconstructions from
                          parameters
        true_curve: true form of the quantity being reconstructed
        subtract_truth: if True, true_curve is subtracted from all plotted
                                 curve(s)
        x_values: the array to use as the x_values of all plots.
                  If None, x_values start at 0 and increment up in steps of 1
        ax: the Axes instance on which to plot the confidence intervals
        xlabel: string to place on x axis
        ylabel: string to place on y axis
        title: title to place on the Axes with this plot
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        maximum_probability_parameter_subset =\
            self.maximum_probability_parameters[parameter_indices]
        if (parameters is None) and (model is None) and (true_curve is None):
            true_curve = self.loglikelihood.data * scale_factor
        if model is None:
            model = self.model
        curve = model(maximum_probability_parameter_subset)
        if x_values is None:
            x_values = np.arange(len(curve))
            to_plot = curve * scale_factor
        if subtract_truth:
            if true_curve is None:
                raise NotImplementedError("Cannot subtract truth if true " +\
                    "curve is None.")
            else:
                to_plot = to_plot - true_curve
        ax.plot(x_values, to_plot, color='r', label='Max probability')
        if true_curve is not None:
            if subtract_truth:
                ax.plot(x_values, np.zeros_like(true_curve), color='k',\
                    label='input')
            else:
                ax.plot(x_values, true_curve, color='k', label='input')
        ax.legend(fontsize=fontsize)
        if title is None:
            title = 'Maximum probability reconstruction{!s}'.format(\
                ' bias' if subtract_truth else '')
        ax.set_title(title, size=fontsize)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if show:
            pl.show()
        else:
            return ax
    
    def plot_reconstruction_confidence_intervals(self, number, probabilities,\
        parameters, model, true_curve=None, x_values=None, ax=None,\
        alphas=None, xlabel=None, ylabel=None, title=None, fontsize=28,\
        scale_factor=1., show=False):
        """
        Plots reconstruction confidence intervals with the given model,
        parameters, and confidence levels.
        
        number: number of curves used to compute of confidence intervals
        parameters: if None (default), all parameters are used
                    if string, parameter which match the regular expression
                               given by parameters are used
                    if list of ints, the parameters with these indices are used
                    if list of strings, these parameters are used
        model: if None (default), full model is used
               otherwise, the model with which to create reconstructions from
                          parameters
        true_curve: true form of the quantity being reconstructed
        x_values: the array to use as the x_values of all plots.
                  If None, x_values start at 0 and increment up in steps of 1
        ax: the Axes instance on which to plot the confidence intervals
        alphas: the alpha values with which to fill in each interval (must be
                sequence of values between 0 and 1 of length greater than or
                equal to the length of probabilities.
        xlabel: string to place on x axis
        ylabel: string to place on y axis
        title: title to place on the Axes with this plot
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        intervals = self.reconstruction_confidence_intervals(number,\
            probabilities, parameters=parameters, model=model)
        if type(probabilities) in real_numerical_types:
            probabilities = [probabilities]
            intervals = [intervals]
            if alphas is None:
                alphas = [0.3]
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if x_values is None:
            x_values = np.arange(intervals[0][0].shape[0])
        for index in range(len(intervals)):
            pconfidence = int(round(probabilities[index] * 100))
            ax.fill_between(x_values, intervals[index][0] * scale_factor,\
                intervals[index][1] * scale_factor, alpha=alphas[index],\
                color='r', label='{:d}% confidence'.format(pconfidence))
        if true_curve is not None:
            ax.plot(x_values, true_curve, linewidth=2, color='k',\
                label='input')
        ax.set_xlim((x_values[0], x_values[-1]))
        ax.legend(fontsize=fontsize)
        if title is None:
            title = 'Reconstruction confidence intervals'
        ax.set_title(title, size=fontsize)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_chain(self, parameters=None, walkers=None, thin=1,\
        figsize=(8, 8), show=False, **reference_values):
        """
        Plots the chain of this MCMC.
        
        parameters: if None, the chain is plotted for all parameters
                    if string, the chain is plotted for all parameters which
                               match this regular expression
                    if sequence of ints, describes the indices of the
                                         parameters to be plotted
                    if sequence of strings, describes parameters to be plotted
        walkers: if None, all walkers are shown.
                 if int, describes the number of walkers shown in the plot
                 if sequence, describes which walkers are shown in the plot
        thin: factor by which to thin the chain
        figsize: size of figure on which to plot chains
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        reference_values: if applicable, value to plot on top of chains for
                          each parameter
        """
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        num_parameter_plots = len(parameter_indices)
        if thin is None:
            thin = 1
        if walkers is None:
            walkers = np.arange(self.nwalkers)
        elif type(walkers) in int_types:
            walkers = np.arange(walkers)
        trimmed_chain =\
            self.chain[walkers,:,:][:,::thin,:][:,:,parameter_indices]
        steps = np.arange(0, self.nsteps, thin)
        axes_per_side = int(np.ceil(np.sqrt(num_parameter_plots)))
        fig = pl.figure(figsize=figsize)
        for index in range(num_parameter_plots):
            parameter_name = self.parameters[parameter_indices[index]]
            ax = fig.add_subplot(axes_per_side, axes_per_side, index + 1)
            ax.plot(steps, trimmed_chain[:,:,index].T, linewidth=1)
            if parameter_name in reference_values:
                ax.plot(steps,\
                    reference_values[parameter_name] * np.ones_like(steps),\
                    linewidth=2, color='k', linestyle='--')
            ax.set_title(parameter_name)
            ax.set_xlim((steps[0], steps[-1]))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,\
            wspace=0.25, hspace=0.25)
        if show:
            pl.show()
    
    def get_parameter_index(self, parameter):
        """
        Gets the integer index associated with the given parameter or index.
        
        parameter: either a string name of a parameter of a valid parameter
                   index
        
        returns: the integer index associated with the given parameter
        """
        if isinstance(parameter, basestring):
            return self.parameters.index(parameter)
        elif type(parameter) in int_types:
            if (parameter >= 0) and (parameter < self.num_parameters):
                return parameter
            else:
                raise ValueError(("parameter was {0}, even though it " +\
                    "should be 0<=parameter<{1}").format(parameter,\
                    self.num_parameters))
        else:
            raise TypeError("parameter must be a string name of a " +\
                "parameter or a valid index of a parameter.")
    
    def triangle_plot(self, parameters=None, walkers=None, thin=1,\
        figsize=(8, 8), show=False, kwargs_1D={}, kwargs_2D={}, fontsize=28,\
        nbins=100, plot_type='contour', parameter_renamer=None,\
        reference_value_mean=None, reference_value_covariance=None):
        """
        Makes a triangle plot.
        
        parameters: key describing parameters to appear in subplots
        walkers: either None or indices of walkers to include chain samples
        thin: integer stride with which to thin chain samples
        figsize: size of figure on which the triangle plot will be placed
        show: if True, matplotlib.pyplot.show is called before this function
              returns
        kwargs_1D: kwargs to pass on to univariate_histogram function
        kwargs_2D: kwargs to pass on to bivariate_histogram function
        fontsize: the size of the label/tick fonts
        nbins: the number of bins in all dimensions in all subplots
        plot_type: 'contourf', 'contour', or 'histogram'
        parameter_renamer: function to apply to each parameter name to
                           determine the label which will correspond to it
        reference_value_mean: either None or a 1D array containing reference
                              values
        reference_value_covariance: either None, a 2D array of rank
                                    num_parameters, or a tuple of length
                                    greater than 1 of the form (model, error)
                                    or (model, error, fisher_kwargs) to use for
                                    estimating the covariance matrix under the
                                    Fisher matrix formalism. This covariance
                                    will be used to plot ellipses
        """
        parameter_indices = self.get_parameter_indices(parameters=parameters)
        labels = [self.parameters[parameter_index]\
            for parameter_index in parameter_indices]
        if parameter_renamer is not None:
            labels = [parameter_renamer(label) for label in labels]
        samples = [\
            self.chain[:,:,parameter_index][walkers,:][:,::thin].flatten()\
            for parameter_index in parameter_indices]
        num_samples = len(samples)
        if reference_value_covariance is not None:
            if isinstance(reference_value_covariance, np.ndarray):
                if reference_value_covariance.shape != ((num_samples,) * 2):
                    raise ValueError("reference_value_covariance was a " +\
                        "numpy.ndarray but it was not square with rank " +\
                        "given by the number of parameters in the triangle " +\
                        "plot.")
            else:
                (model, error, fisher_kwargs) =\
                    (reference_value_covariance[0],\
                    reference_value_covariance[1],\
                    reference_value_covariance[2:])
                if len(fisher_kwargs) == 0:
                    fisher_kwargs = {}
                else:
                    fisher_kwargs = fisher_kwargs[0]
                likelihood =\
                    GaussianLoglikelihood(np.zeros(len(error)), error, model)
                reference_value_covariance =\
                    likelihood.parameter_covariance_fisher_formalism(\
                    reference_value_mean, **fisher_kwargs)
        triangle_plot(samples, labels, figsize=figsize, show=show,\
            kwargs_1D=kwargs_1D, kwargs_2D=kwargs_2D, fontsize=fontsize,\
            nbins=nbins, plot_type=plot_type,\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance)
    
    def plot_univariate_histogram(self, parameter_index, walkers=None, thin=1,\
        ax=None, show=False, reference_value=None, fontsize=28,\
        matplotlib_function='fill_between', show_intervals=False, bins=None,\
        xlabel='', ylabel='', title='', **kwargs):
        """
        Plots a 1D histogram of the given parameter.
        
        parameter_index: either a string parameter name or a valid integer
                         index of the parameter
        walkers: if None, all walkers are shown.
                 if int, describes the number of walkers shown in the plot
                 if sequence, describes which walkers are shown in the plot
        thin: factor by which to thin the chain
        ax: if None, new Figure and Axes are created
            otherwise, this Axes object is plotted on
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
        reference_value: a point at which to plot a dashed reference line
        fontsize: the size of the tick label font
        matplotlib_function: either 'fill_between' or 'plot'
        bins: bins to pass to numpy.histogram: default, None
        xlabel: string for labeling x axis
        ylabel: string for labeling y axis
        title: string for top of plot
        kwargs: keyword arguments to pass on to matplotlib.Axes.plot or
                matplotlib.Axes.fill_between
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        parameter_index = self.get_parameter_index(parameter_index)
        if thin is None:
            thin = 1
        if walkers is None:
            walkers = np.arange(self.nwalkers)
        elif type(walkers) in int_types:
            walkers = np.arange(walkers)
        sample = self.chain[:,:,parameter_index][walkers,:][:,::thin].flatten()
        univariate_histogram(sample, reference_value=reference_value,\
            bins=bins, matplotlib_function=matplotlib_function,\
            show_intervals=show_intervals, xlabel=xlabel, ylabel=ylabel,\
            title=title, fontsize=fontsize, ax=ax, show=show, **kwargs)
    
    def plot_bivariate_histogram(self, parameter_index1, parameter_index2,\
        walkers=None, thin=1, ax=None, show=False, reference_value_mean=None,\
        reference_value_covariance=None, fontsize=28, bins=None,\
        matplotlib_function='imshow', xlabel='', ylabel='', title='',\
        **kwargs):
        """
        Plots a 2D histogram of the given parameters.
        
        parameter_index1: either a string parameter name or a valid integer
                          index of the parameter (x axis)
        parameter_index2: either a string parameter name or a valid integer
                          index of the parameter (y axis)
        walkers: if None, all walkers are shown.
                 if int, describes the number of walkers shown in the plot
                 if sequence, describes which walkers are shown in the plot
        thin: factor by which to thin the chain
        ax: if None, new Figure and Axes are created
            otherwise, this Axes object is plotted on
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
        reference_value_mean: point at which to plot a dashed reference lines
                              for the axes
        reference_value_covariance: if not None, used to plot reference ellipse
        fontsize: the size of the tick label font
        bins: bins to pass to numpy.histogram2d, default: None
        xlabel: string for labeling x axis
        ylabel: string for labeling y axis
        title: string for top of plot
        matplotlib_function: function to use in plotting. One of ['imshow',
                             'contour', 'contourf']. default: 'imshow'
        kwargs: keyword arguments to pass on to matplotlib.Axes.imshow (any but
                'origin', 'extent', or 'aspect') or matplotlib.Axes.contour or
                matplotlib.Axes.contourf (any)
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        parameter_index1 = self.get_parameter_index(parameter_index1)
        parameter_index2 = self.get_parameter_index(parameter_index2)
        if thin is None:
            thin = 1
        if walkers is None:
            walkers = np.arange(self.nwalkers)
        elif type(walkers) in int_types:
            walkers = np.arange(walkers)
        xsample =\
            self.chain[:,:,parameter_index1][walkers,:][:,::thin].flatten()
        ysample =\
            self.chain[:,:,parameter_index2][walkers,:][:,::thin].flatten()
        bivariate_histogram(xsample, ysample,\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance, bins=bins,\
            matplotlib_function=matplotlib_function, xlabel=xlabel,\
            ylabel=ylabel, title=title, fontsize=fontsize, ax=ax, show=show,\
            **kwargs)
    
    def plot_lnprobability(self, walkers=None, log_scale=False,\
        title='Log probability', ax=None, show=False):
        """
        Plots the log probability values accessed by this MCMC chain.
        
        walkers: if None, all walkers are shown.
                 if int, describes the number of walkers shown in the plot
                 if sequence, describes which walkers are shown in the plot
        log_scale: if False (default), everything is plotted on linear scale
                   if True, difference from maximum likelihood is plotted on a
                            log scale (on y-axis)
        ax: Axes instance on which to plot the log_probability chain
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        if walkers is None:
            walkers = np.arange(self.nwalkers)
        elif type(walkers) in int_types:
            walkers = np.arange(walkers)
        trimmed_lnprobability = self.lnprobability[walkers,:]
        steps = np.arange(self.nsteps)
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if log_scale:
            ax.semilogy(steps,\
                (np.max(trimmed_lnprobability) - trimmed_lnprobability).T)
            ax.set_ylabel('$\ln(p_{max})-\ln(p)$')
        else:
            ax.plot(steps, trimmed_lnprobability.T)
            ax.set_ylabel('$\ln(p)$')
        ax.set_xlim((steps[0], steps[-1]))
        ax.set_title(title)
        ax.set_xlabel('Step number')
        if show:
            pl.show()
        else:
            return ax
    
    def plot_covariance_matrix(self, normalize_by_variances=False, fig=None,\
        ax=None, show=False, **imshow_kwargs):
        """
        Plots a covariance (or correlation) matrix of the parameters.
        
        normalize_by_variances: if True, correlation matrix is plotted
                                otherwise, covariance matrix is plotted
        fig: matplotlib.Figure object containing axes, or None if one should be
             created
        ax: matplotlib.axes.Axes object 
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
        imshow_kwargs: keyword arguments to pass on to the
                       matplotlib.axes.Axes.imshow function. Common keywords
                       include 'norm' and 'cmap'. imshow_kwargs should not
                       include 'extent'. It is worked out internally
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if normalize_by_variances:
            to_plot = self.parameter_correlation
        else:
            to_plot = self.parameter_covariance
        low = 0.5
        high = self.num_parameters + 0.5
        image =\
            ax.imshow(to_plot, extent=[low, high, high, low], **imshow_kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(image, cax=cax, orientation='vertical')
        if show:
            pl.show()
        else:
            return ax
    
    def plot_rescaling_factors(self, log_scale=False, ax=None, show=False):
        """
        Plots the rescaling factors for each of the parameters. These values
        should be close to 1.
        
        ax: Axes instance on which to plot the acceptance fraction
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        x_values = np.arange(self.num_parameters) + 1
        ax.scatter(x_values, self.rescaling_factors, color='b')
        x_ones = np.ones(self.num_parameters)
        ax.plot(x_values, x_ones, color='r')
        ax.plot(x_values, x_ones * 1.1, color='r', linestyle='--')
        if log_scale:
            ax.set_yscale('log')
        ax.set_title('Rescaling factors')
        ax.set_xlabel('Parameter number')
        ax.set_ylabel('Rescaling factor, $R$')
        ax.set_xlim((x_values[0] - 0.5, x_values[-1] + 0.5))
        if show:
            pl.show()
        else:
            return ax
    
    def plot_acceptance_fraction(self, walkers=None, average=False, ax=None,\
        log_scale=False, show=False):
        """
        Plots the fraction of MCMC steps which are accepted by walker.
        
        walkers: if None, all walkers are shown.
                 if int, describes the number of walkers shown in the plot
                 if sequence, describes which walkers are shown in the plot
        average: if True, the acceptance fraction is averaged across the given
                          walkers before being plotted
                 if False, all of the given walkers' acceptance fractions are
                           plotted
        ax: Axes instance on which to plot the acceptance fraction
        log_scale: if True, the acceptance fraction is shown in a semilogy plot
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, otherwise Axes instance with plot
        """
        if walkers is None:
            walkers = np.arange(self.nwalkers)
        elif type(walkers) in int_types:
            walkers = np.arange(walkers)
        trimmed_acceptance_fraction = self.acceptance_fraction[walkers,:]
        steps = np.arange(trimmed_acceptance_fraction.shape[1]) + 1
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if average:
            title = '$f_{acc}$ averaged across walkers'
            trimmed_acceptance_fraction =\
               np.mean(trimmed_acceptance_fraction, axis=0)
        else:
            title = '$f_{acc}$ by walker'
        if log_scale:
            ylim = (1e-3, 1)
            ax.semilogy(steps, trimmed_acceptance_fraction.T)
        else:
            ylim = (0, 1)
            ax.plot(steps, trimmed_acceptance_fraction.T)
        ax.set_ylim(ylim)
        loaded_so_far = 0
        ax.plot([1, 1], ylim, color='k', linestyle='--')
        for these_checkpoints_to_load in self.checkpoints_to_load:
            loaded_so_far += len(these_checkpoints_to_load)
            if loaded_so_far == 0:
                ax.plot([1, 1], ylim, color='k', linestyle='--')
            elif loaded_so_far == self.total_num_checkpoints_to_load:
                ax.plot([loaded_so_far] * 2, ylim, color='k', linestyle='--')
            else:
                ax.plot([0.5 + loaded_so_far] * 2, ylim, color='k',\
                    linestyle='--')
        ax.set_title(title)
        ax.set_xlim((steps[0] - 0.5, steps[-1] + 0.5))
        if show:
            pl.show()
        else:
            return ax

