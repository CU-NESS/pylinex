"""
File: Sampler.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class which wraps around the
             MetropolisHastingsSampler defined in the distpy package.
"""
import os, time, h5py
import numpy as np
import numpy.linalg as la
from emcee import EnsembleSampler
from distpy import GaussianDistribution, CustomDiscreteDistribution,\
    DistributionSet, GaussianJumpingDistribution, JumpingDistributionSet,\
    MetropolisHastingsSampler
from ..util import bool_types, int_types, real_numerical_types, sequence_types
from ..loglikelihood import Loglikelihood

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class Sampler(object):
    """
    Class which wraps around the MetropolisHastingsSampler defined in the
    distpy package.
    """
    def __init__(self, file_name, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, steps_per_checkpoint=100, verbose=True,\
        restart_mode=None, nthreads=1, args=[], kwargs={},\
        use_ensemble_sampler=False, proposal_covariance_reduction_factor=1.):
        """
        Initializes a new sampler with the given file_name, loglikelihood,
        jumping distribution set, and guess distribution set.
        
        file_name: name of hdf5 file to save (if extant, a restart process is
                   triggered)
        nwalkers: number of MCMC chains being iterated together
        loglikelihood: likelihood to explore
        jumping_distribution_set: JumpingDistributionSet object which can be
                                  used to draw new points from any given source
                                  point in the MCMC. If None during a
                                  non-restart, throws an error. If None during
                                  a restart with restart_mode=='continue',
                                  jumping_distribution_set is copied over from
                                  last run. If None during a restart with
                                  restart_mode=='reinitialize', the continuous
                                  parameters' proposal distribution is updated
                                  to a GaussianJumpingDistribution approximated
                                  from the last checkpoint's worth of samples
                                  from the last saved chunk and the discrete
                                  parameters' proposal distribution is carried
                                  over from the last saved chunk.
        guess_distribution_set: DistributionSet object from which guesses can
                                be drawn. If None during a non-restart, this
                                will throw an error. This value is not used in
                                the case of a restart with
                                restart_mode=='continue'. If None during a
                                restart with restart_mode='reinitialize', the
                                continuous parameters are described by a single
                                high dimensional Gaussian approximated by the
                                last checkpoint and the discrete parameters are
                                each individually described by 1D
                                CustomDiscreteDistribution objects approximated
                                by the last checkpoint
        prior_distribution_set: optional extra DistributionSet object with
                                priors to include in the posterior to explore.
                                If None and this is a restart, priors are
                                copied from last run. if None and this is not a
                                restart, uses no priors.
        steps_per_checkpoint: integer number of steps taken between saving to
                              the hdf5 file. (if None, it is set to 100)
        verbose: if True, the time is printed after the completion of each
                          checkpoint (and before the first checkpoint)
        nthreads: the number of threads to use in log likelihood calculations
                  for walkers. Default: 1, 1 is best unless loglikelihood is
                  very slow
        args: extra positional arguments to pass on to the likelihood
        kwargs: extra keyword arguments to pass on to the likelihood
        use_ensemble_sampler: if True, EnsembleSampler of emcee is used
                              otherwise, MetropolisHastingsSampler from distpy
                                         is used
        proposal_covariance_reduction_factor: only used if this is a restart,
                                              restart_mode=='reinitialize', and
                                              jumping_distribution_set is None
        """
        self.use_ensemble_sampler = use_ensemble_sampler
        self.nthreads = nthreads
        self.proposal_covariance_reduction_factor =\
            proposal_covariance_reduction_factor
        self.restart_mode = restart_mode
        self.verbose = verbose
        self.nwalkers = nwalkers
        self.steps_per_checkpoint = steps_per_checkpoint
        self.loglikelihood = loglikelihood
        self.jumping_distribution_set = jumping_distribution_set
        self.guess_distribution_set = guess_distribution_set
        self.prior_distribution_set = prior_distribution_set
        self.file_name = file_name
        self.args = args
        self.kwargs = kwargs
        self.file # loads in things for restart if necessary
        self.close()
    
    @property
    def nthreads(self):
        """
        Property storing the number of threads to use in calculating log
        likelihood values.
        """
        if not hasattr(self, '_nthreads'):
            raise AttributeError("nthreads referenced before it was set.")
        return self._nthreads
    
    @nthreads.setter
    def nthreads(self, value):
        """
        Setter for the number of threads to use in log likelihood calculations.
        
        value: a positive integer; 1 is best unless loglikelihood is very slow
        """
        if type(value) in int_types:
            if value > 0:
                self._nthreads = value
            else:
                raise ValueError("nthreads must be non-negative.")
        else:
            raise TypeError("nthreads was set to a non-int.")
    
    @property
    def use_ensemble_sampler(self):
        """
        Property storing a boolean deciding whether the EnsembleSampler of
        emcee is used at the core of this Sampler.
        """
        if not hasattr(self, '_use_ensemble_sampler'):
            raise AttributeError("use_ensemble_sampler referenced before " +\
                "it was set.")
        return self._use_ensemble_sampler
    
    @use_ensemble_sampler.setter
    def use_ensemble_sampler(self, value):
        """
        Setter for the boolean deciding whether the EnsembleSampler of emcee is
        used at the core of this Sampler.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._use_ensemble_sampler = value
        else:
            raise TypeError("use_ensemble_sampler was set to a non-bool.")
    
    @property
    def proposal_covariance_reduction_factor(self):
        """
        Property storing the proposal_covariance_reduction_factor used if this
        is a restart, restart_mode=='reinitialize', and
        self.jumping_distribution_set is None
        """
        if not hasattr(self, '_proposal_covariance_reduction_factor'):
            raise AttributeError("proposal_covariance_reduction_factor was " +\
                "referenced before it was set.")
        return self._proposal_covariance_reduction_factor
    
    @proposal_covariance_reduction_factor.setter
    def proposal_covariance_reduction_factor(self, value):
        """
        Setter for the proposal_covariance_reduction_factor.
        
        value: must be a single positive number
        """
        if (type(value) in real_numerical_types) and (value > 0):
            self._proposal_covariance_reduction_factor = value
        else:
            raise TypeError("proposal_covariance_reduction_factor was not " +\
                "a positive number.")
    
    @property
    def restart_mode(self):
        """
        Property storing the kind of restart which should be performed if this
        Sampler's file already exists.
        """
        if not hasattr(self, '_restart_mode'):
             raise AttributeError("restart_mode referenced before it was set.")
        return self._restart_mode
    
    @restart_mode.setter
    def restart_mode(self, value):
        """
        Setter for the restart_mode property.
        
        value: if None, this Sampler will not support restarts
               if 'continue', the Sampler will continue exactly as it left off
               if 'reinitialize', the Sampler uses the previous 
        """
        allowed_modes = ['continue', 'update', 'reinitialize']
        if (value is None) or (value in allowed_modes):
            self._restart_mode = value
        else:
            raise ValueError("restart_mode was set to neither None, " +\
                "'continue', nor 'reinitialize'")
    
    @property
    def args(self):
        """
        Property storing extra positional arguments to pass on to the
        likelihood function.
        """
        if not hasattr(self, '_args'):
            raise AttributeError("args referenced before it was set.")
        return self._args
    
    @args.setter
    def args(self, value):
        """
        Setter for the extra positional arguments to pass on to the likelihood
        function.
        
        value: must be a sequence
        """
        if type(value) in sequence_types:
            self._args = [element for element in value]
        else:
            raise TypeError("args was set to a non-sequence.")
    
    @property
    def kwargs(self):
        """
        Property storing the keyword arguments to pass on to the likelihood
        function.
        """
        if not hasattr(self, '_kwargs'):
            raise AttributeError("kwargs referenced before it was set.")
        return self._kwargs
    
    @kwargs.setter
    def kwargs(self, value):
        """
        Setter for the keyword arguments to pass on to the likelihood.
        
        value: must be a dictionary with string keys
        """
        if isinstance(value, dict):
            self._kwargs = {key: value[key] for key in value}
        else:
            raise TypeError("kwargs was set to a non-dict.")
    
    @property
    def verbose(self):
        """
        Property storing a boolean switch which determines how much is printed
        out.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the verbose property.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise TypeError("verbose was set to a non-bool.")
    
    @property
    def steps_per_checkpoint(self):
        """
        Property storing the number of steps which are run between each
        checkpoint.
        """
        if not hasattr(self, '_steps_per_checkpoint'):
            raise AttributeError("steps_per_checkpoint referenced before " +\
                "it was set.")
        return self._steps_per_checkpoint
    
    @steps_per_checkpoint.setter
    def steps_per_checkpoint(self, value):
        """
        Setter for the number of steps which are run between each checkpoint.
        
        value: must be positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._steps_per_checkpoint = value
            else:
                raise ValueError("steps_per_checkpoint was set to a " +\
                    "non-positive integer.")
        else:
            raise TypeError("steps_per_checkpoint was set to a non-int.")
    
    @property
    def file_name(self):
        """
        Property storing the name of the hdf5 file in which the chain will be
        saved.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the name of the hdf5 file in which the chain will be saved.
        
        value: must be a string file name. If it already exists, a restart
               sequence will be initiated
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was not a string.")
    
    @property
    def chunk_index(self):
        """
        Property storing the index of the chunk this Sampler object is working
        on.
        """
        if not hasattr(self, '_chunk_index'):
            raise AttributeError("chunk_index referenced before it was" +\
                " computed.")
        return self._chunk_index
    
    @chunk_index.setter
    def chunk_index(self, value):
        """
        Setter for the chunk_index property.
        """
        if (type(value) in int_types) and (value >= 0):
            self._chunk_index = value
        else:
            raise TypeError("chunk_index was not a non-negative integer.")
    
    @staticmethod
    def clear_last_saved_chunk(file_name):
        """
        Clears the last saved chunk from the Sampler stored in the given file.
        This is useful if an error occurred in the last chunk
        
        file_name: string location of file storing the Sampler under
                   consideration
        """
        hdf5_file = h5py.File(file_name, 'r+')
        try:
            chunk_index_to_delete = hdf5_file.attrs['max_chunk_index']
            if chunk_index_to_delete == 0:
                raise ValueError("The index of the chunk to delete was 0, " +\
                    "so you might as well simply delete the hdf5 file " +\
                    "storing this Sampler object.")
            else:
                chunk_string_to_delete =\
                    'chunk{:d}'.format(chunk_index_to_delete)
                names_to_check = ['checkpoints', 'guess_distribution_sets',\
                    'jumping_distribution_sets', 'prior_distribution_sets']
                for name in names_to_check:
                    group = hdf5_file[name]
                    if chunk_string_to_delete in group:
                        del group[chunk_string_to_delete]
                hdf5_file.attrs['max_chunk_index'] = chunk_index_to_delete - 1
        except:
            hdf5_file.close()
            raise
        else:
            hdf5_file.close()
    
    def _setup_restart_continue(self, pos, lnprob, guess_distribution_set,\
        prior_distribution_set, jumping_distribution_set):
        """
        Sets up a restart with restart_mode == 'continue'.
        
        pos: the walkers' last saved positions in parameter space
        lnprob: lnprobability values of the walkers' last saved positions
        guess_distribution_set: the DistributionSet describing the initial
                                walker positions of the last saved chunk
        prior_distribution_set: the DistributionSet describing the priors in
                                the posterior explored by the last saved chunk
        jumping_distribution_set: the JumpingDistributionSet describing the
                                  proposal distributions of the MCMC in the
                                  last saved chunk
        """
        self._pos = pos
        self._lnprob = lnprob
        if self.prior_distribution_set is None:
            self.prior_distribution_set = prior_distribution_set
        elif (self.prior_distribution_set != prior_distribution_set):
            raise ValueError("prior_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
        self.guess_distribution_set = guess_distribution_set
        if self.jumping_distribution_set is None:
            self.jumping_distribution_set = jumping_distribution_set
        elif (self.jumping_distribution_set != jumping_distribution_set):
            raise ValueError("jumping_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
    
    def _setup_restart_update(self, pos, lnprob, guess_distribution_set,\
        prior_distribution_set, jumping_distribution_set,\
        last_saved_chunk_string):
        """
        Sets up a restarted run which is begun with an update to the
        JumpingDistributionSet but with no other update.
        
        pos: the walkers' last saved positions in parameter space
        lnprob: lnprobability values of the walkers' last saved positions
        guess_distribution_set: the DistributionSet describing the initial
                                walker positions of the last saved chunk
        prior_distribution_set: the DistributionSet describing the priors in
                                the posterior explored by the last saved chunk
        jumping_distribution_set: the JumpingDistributionSet describing the
                                  proposal distributions of the MCMC in the
                                  last saved chunk
        last_saved_chunk_string: string of form 'chunk{}'.format(chunk_index)
                                 where chunk index is the index of the last
                                 saved chunk
        """
        self.chunk_index = self.chunk_index + 1
        self.file.attrs['max_chunk_index'] = self.chunk_index
        new_chunk_string = 'chunk{0:d}'.format(self.chunk_index)
        self.file['checkpoints'].create_group(new_chunk_string)
        self._pos = pos
        self._lnprob = lnprob
        if self.prior_distribution_set is None:
            self.prior_distribution_set = prior_distribution_set
        elif (self.prior_distribution_set != prior_distribution_set):
            raise ValueError("prior_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
        if self.prior_distribution_set is not None:
            group = self.file['prior_distribution_sets']
            subgroup = group.create_group(new_chunk_string)
            self.prior_distribution_set.fill_hdf5_group(subgroup)
        self.guess_distribution_set = guess_distribution_set
        group = self.file['guess_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.guess_distribution_set.fill_hdf5_group(subgroup)
        if self.jumping_distribution_set is None:
            self.jumping_distribution_set =\
                self._generate_reinitialized_jumping_distribution_set(\
                jumping_distribution_set, last_saved_chunk_string)
        group = self.file['jumping_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.jumping_distribution_set.fill_hdf5_group(subgroup)
        self.checkpoint_index = 0
    
    def _setup_restart_reinitialize(self, guess_distribution_set,\
        prior_distribution_set, jumping_distribution_set,\
        last_saved_chunk_string):
        """
        Sets up a restart with restart_mode == 'reinitialize'.
        
        guess_distribution_set: the DistributionSet describing the initial
                                walker positions of the last saved chunk
        prior_distribution_set: the DistributionSet describing the priors in
                                the posterior explored by the last saved chunk
        jumping_distribution_set: the JumpingDistributionSet describing the
                                  proposal distributions of the MCMC in the
                                  last saved chunk
        last_saved_chunk_string: string of form 'chunk{}'.format(chunk_index)
                                 where chunk index is the index of the last
                                 saved chunk
        """
        self.chunk_index = self.chunk_index + 1
        self.file.attrs['max_chunk_index'] = self.chunk_index
        new_chunk_string = 'chunk{0:d}'.format(self.chunk_index)
        self.file['checkpoints'].create_group(new_chunk_string)
        if self.prior_distribution_set is None:
            self.prior_distribution_set = prior_distribution_set
        if prior_distribution_set != self.prior_distribution_set:
            print("prior_distribution_set is changing, so " +\
                  "the distribution being explored is being " +\
                  "changed discontinuously!")
        if self.prior_distribution_set is not None:
            group = self.file['prior_distribution_sets']
            subgroup = group.create_group(new_chunk_string)
            self.prior_distribution_set.fill_hdf5_group(\
                subgroup)
        if self.jumping_distribution_set is None:
            self.jumping_distribution_set =\
                self._generate_reinitialized_jumping_distribution_set(\
                jumping_distribution_set, last_saved_chunk_string)
        group = self.file['jumping_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.jumping_distribution_set.fill_hdf5_group(subgroup)
        if self.guess_distribution_set is None:
            self.guess_distribution_set =\
                self._generate_reinitialized_guess_distribution_set(\
                guess_distribution_set, last_saved_chunk_string)
        group = self.file['guess_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.guess_distribution_set.fill_hdf5_group(subgroup)
        self.checkpoint_index = 0
    
    def _generate_reinitialized_guess_distribution_set(self,\
        guess_distribution_set, last_saved_chunk_string, min_eigenvalue=1e-8):
        """
        Generates a new guess_distribution_set to use for a Sampler restarted
        with restart_mode=='reinitialize'.
        
        guess_distribution_set: the DistributionSet object describing how
                                walkers were initialized in last saved chunk.
        last_saved_chunk_string: string of the form
                                 'chunk{:d}'.format(chunk_index) where
                                 chunk_index is the index of the last saved
                                 chunk
        
        returns: a DistributionSet object to use to initialize walkers for the
                 next chunk of this sampler.
        """
        continuous_params = self.jumping_distribution_set.continuous_params
        continuous_parameter_indices =\
            np.array([self.parameters.index(par) for par in continuous_params])
        last_checkpoint_group =\
            self.file['checkpoints/{!s}'.format(last_saved_chunk_string)]
        last_checkpoint_group =\
            last_checkpoint_group['{:d}'.format(self.checkpoint_index-1)]
        last_checkpoint_loglikelihood =\
            last_checkpoint_group['lnprobability'].value
        walker_averaged_loglikelihood =\
            np.mean(last_checkpoint_loglikelihood, axis=-1)
        loglikelihood_cutoff = np.mean(walker_averaged_loglikelihood) -\
            (3 * np.std(walker_averaged_loglikelihood))
        likelihood_based_weights =\
            (walker_averaged_loglikelihood >= loglikelihood_cutoff).astype(int)
        likelihood_based_weights = (likelihood_based_weights[:,np.newaxis] *\
            np.ones(last_checkpoint_loglikelihood.shape)).flatten()
        last_checkpoint_chain = last_checkpoint_group['chain'].value
        last_checkpoint_chain_continuous =\
            last_checkpoint_chain[...,continuous_parameter_indices]
        flattened_shape = (-1, last_checkpoint_chain_continuous.shape[-1])
        last_checkpoint_chain_continuous =\
            np.reshape(last_checkpoint_chain_continuous, flattened_shape)
        transform_list =\
            guess_distribution_set.transform_set[continuous_params]
        last_checkpoint_chain_continuous =\
            transform_list(last_checkpoint_chain_continuous, axis=-1)
        last_checkpoint_continuous_mean =\
            np.sum(last_checkpoint_chain_continuous *\
            likelihood_based_weights[:,np.newaxis], axis=0) /\
            np.sum(likelihood_based_weights)
        last_checkpoint_continuous_covariance = np.cov(\
            last_checkpoint_chain_continuous, ddof=0, rowvar=False,\
            aweights=likelihood_based_weights)
        (eigenvalues, eigenvectors) =\
            la.eigh(last_checkpoint_continuous_covariance)
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        last_checkpoint_continuous_covariance =\
            np.dot(eigenvectors * eigenvalues[np.newaxis,:], eigenvectors.T)
        distribution = GaussianDistribution(last_checkpoint_continuous_mean,\
            last_checkpoint_continuous_covariance)
        new_guess_distribution_set = DistributionSet()
        new_guess_distribution_set.add_distribution(distribution,\
            continuous_params, transform_list)
        discrete_params = self.jumping_distribution_set.discrete_params
        for param in discrete_params:
            parameter_index = self.parameters.index(param)
            this_last_checkpoint_chain =\
                last_checkpoint_chain[...,parameter_index].flatten()
            transform = guess_distribution_set.transform_set[param]
            this_last_checkpoint_chain = transform(this_last_checkpoint_chain)
            distribution = CustomDiscreteDistribution(\
                *np.unique(this_last_checkpoint_chain, return_counts=True))
            new_guess_distribution_set.add_distribution(distribution,\
                param, transform)
        return new_guess_distribution_set
    
    def _generate_reinitialized_jumping_distribution_set(self,\
        jumping_distribution_set, last_saved_chunk_string,\
        min_eigenvalue=1e-8):
        """
        Generates a new jumping_distribution_set to use for a Sampler
        restarted with restart_mode=='reinitialize'.
        
        jumping_distribution_set: the JumpingDistributionSet object describing
                                  how walkers jumped through parameter space
                                  in the last saved chunk
        last_saved_chunk_string: string of the form
                                 'chunk{:d}'.format(chunk_index) where
                                 chunk_index is the index of the last saved
                                 chunk
        
        returns: a JumpingDistributionSet object to use to determine how
                 walkers travel through parameter space for the next chunk of
                 this sampler
        """
        continuous_params = jumping_distribution_set.continuous_params
        continuous_parameter_indices =\
            np.array([self.parameters.index(par) for par in continuous_params])
        last_checkpoint_group =\
            self.file['checkpoints/{!s}'.format(last_saved_chunk_string)]
        last_checkpoint_group =\
            last_checkpoint_group['{:d}'.format(self.checkpoint_index-1)]
        last_checkpoint_loglikelihood =\
            last_checkpoint_group['lnprobability'].value
        walker_averaged_loglikelihood =\
            np.mean(last_checkpoint_loglikelihood, axis=-1)
        loglikelihood_cutoff = np.mean(walker_averaged_loglikelihood) -\
            (3 * np.std(walker_averaged_loglikelihood))
        likelihood_based_weights =\
            (walker_averaged_loglikelihood >= loglikelihood_cutoff).astype(int)
        likelihood_based_weights = (likelihood_based_weights[:,np.newaxis] *\
            np.ones(last_checkpoint_loglikelihood.shape)).flatten()
        last_checkpoint_chain = last_checkpoint_group['chain'].value
        last_checkpoint_chain_continuous =\
            last_checkpoint_chain[...,continuous_parameter_indices]
        flattened_shape = (-1, last_checkpoint_chain_continuous.shape[-1])
        last_checkpoint_chain_continuous =\
            np.reshape(last_checkpoint_chain_continuous, flattened_shape)
        transform_list =\
            jumping_distribution_set.transform_set[continuous_params]
        last_checkpoint_chain_continuous =\
            transform_list(last_checkpoint_chain_continuous, axis=-1)
        last_checkpoint_continuous_covariance = np.cov(\
            last_checkpoint_chain_continuous, rowvar=False, ddof=0,\
            aweights=likelihood_based_weights)
        (eigenvalues, eigenvectors) =\
            la.eigh(last_checkpoint_continuous_covariance)
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        last_checkpoint_continuous_covariance =\
            np.dot(eigenvectors * eigenvalues[np.newaxis,:], eigenvectors.T)
        last_checkpoint_continuous_covariance /=\
            self.proposal_covariance_reduction_factor
        distribution =\
            GaussianJumpingDistribution(last_checkpoint_continuous_covariance)
        new_jumping_distribution_set = JumpingDistributionSet()
        new_jumping_distribution_set.add_distribution(distribution,\
            continuous_params, transform_list)
        new_jumping_distribution_set = new_jumping_distribution_set +\
            jumping_distribution_set.discrete_subset()
        return new_jumping_distribution_set
    
    def _load_distribution_sets(self, chunk_string):
        """
        Loads the three relevant distribution sets from the existing file at
        the file name of this Sampler.
        
        chunk_string: string of form 'chunk{}'.format(chunk_index) where
                      chunk_index the maximum chunk_index saved by this Sampler
                      in the past
        
        returns: tuple of form (guess_distribution_set, prior_distribution_set,
                 jumping_distribution_set)
        """
        guess_distribution_set_group_name =\
            'guess_distribution_sets/{!s}'.format(chunk_string)
        guess_distribution_set = DistributionSet.load_from_hdf5_group(\
            self.file[guess_distribution_set_group_name])
        prior_distribution_set_group_name =\
            'prior_distribution_sets/{!s}'.format(chunk_string)
        if prior_distribution_set_group_name in self.file:
            prior_distribution_set = DistributionSet.load_from_hdf5_group(\
                self.file[prior_distribution_set_group_name])
        else:
            prior_distribution_set = None
        jumping_distribution_set_group_name =\
            'jumping_distribution_sets/{!s}'.format(chunk_string)
        if jumping_distribution_set_group_name in self.file:
            jumping_distribution_set =\
                JumpingDistributionSet.load_from_hdf5_group(\
                self.file[jumping_distribution_set_group_name])
        elif self.use_ensemble_sampler:
            jumping_distribution_set = None
        else:
            raise ValueError("It seems that you are attempting to restart " +\
                "an EnsembleSampler with a MetropolisHastingsSampler.")
        return (guess_distribution_set, prior_distribution_set,\
            jumping_distribution_set)
    
    def _load_state(self):
        """
        Loads the most recent saved state in the hdf5 file at the heart of this
        Sampler.
        
        returns: tuple of form (pos, lnprob) where pos contains the current
                 locations of the walkers and lnprob contains the current
                 values of the walkers' lnprobability
        """
        group = self.file['state']
        pos = group['pos'].value
        lnprob = group['lnprob'].value
        subgroup = group['rstate']
        alg = subgroup.attrs['algorithm']
        keys = subgroup['keys'].value
        rstate_pos = subgroup.attrs['pos']
        has_gauss = subgroup.attrs['has_gauss']
        cached_gaussian = subgroup.attrs['cached_gaussian']
        self._rstate = (alg, keys, rstate_pos, has_gauss, cached_gaussian)
        return (pos, lnprob)
    
    def _setup_restarted_file(self):
        """
        Sets up this Sampler from the existing file at the file_name.
        """
        if self.restart_mode == None:
            raise ValueError("The file for this Sampler already exists but " +\
                "no restart_mode was given, please supply either " +\
                "restart_mode='continue' or restart_mode='reinitialize' in " +\
                "the initializer of this Sampler.")
        else:
            self._file = h5py.File(self.file_name, 'r+')
            self.chunk_index = self.file.attrs['max_chunk_index']
            chunk_string = 'chunk{0:d}'.format(self.chunk_index)
            (guess_distribution_set, prior_distribution_set,\
                jumping_distribution_set) =\
                self._load_distribution_sets(chunk_string)
            (pos, lnprob) = self._load_state()
            checkpoints_group_name = 'checkpoints/{!s}'.format(chunk_string)
            group = self.file[checkpoints_group_name]
            self.checkpoint_index = 0
            while '{}'.format(self.checkpoint_index) in group:
                self.checkpoint_index = self.checkpoint_index + 1
            if self.restart_mode == 'continue':
                self._setup_restart_continue(pos, lnprob,\
                    guess_distribution_set, prior_distribution_set,\
                    jumping_distribution_set)
            elif self.restart_mode == 'update':
                self._setup_restart_update(pos, lnprob,\
                    guess_distribution_set, prior_distribution_set,\
                    jumping_distribution_set, chunk_string)
            else:
                self._setup_restart_reinitialize(guess_distribution_set,\
                    prior_distribution_set, jumping_distribution_set,\
                    chunk_string)
    
    def _setup_new_file(self):
        """
        Writes the start of a new file at this Sampler's file_name. This occurs
        when a Sampler is started for the first time (i.e. this is not a
        restart).
        """
        self._file = h5py.File(self.file_name, 'w')
        (self.chunk_index, self.checkpoint_index) = (0, 0)
        self.file.attrs['max_chunk_index'] = self.chunk_index
        group = self.file.create_group('guess_distribution_sets')
        self.guess_distribution_set.fill_hdf5_group(\
            group.create_group('chunk0'))
        group = self.file.create_group('prior_distribution_sets')
        if self.has_priors:
            self.prior_distribution_set.fill_hdf5_group(\
                group.create_group('chunk0'))
        group = self.file.create_group('jumping_distribution_sets')
        if self.jumping_distribution_set is not None:
            self.jumping_distribution_set.fill_hdf5_group(\
                group.create_group('chunk0'))
        self.loglikelihood.fill_hdf5_group(\
            self.file.create_group('loglikelihood'))
        group = self.file.create_group('checkpoints')
        subgroup = group.create_group('chunk0')
        group = self.file.create_group('parameters')
        for (iparameter, parameter) in enumerate(self.parameters):
            group.attrs['{:d}'.format(iparameter)] = parameter
    
    def _setup(self):
        """
        Sets up this Sampler from the data (or lack thereof) in the existing or
        nonexisting file at the file name given to this Sampler.
        """
        restart = os.path.exists(self.file_name)
        if restart:
            self._setup_restarted_file()
        elif (self.guess_distribution_set is not None) and\
            ((self.jumping_distribution_set is not None) or\
            (self.use_ensemble_sampler)):
            self._setup_new_file()
        else:
            raise ValueError("If this is not a restart, " +\
                "guess_distribution_set and jumping_distribution_set " +\
                "must be given to Sampler initializer.")
    
    @property
    def started(self):
        """
        Property storing a boolean describing whether or not this sampler has
        been started (i.e. whether its file has been opened at least once).
        """
        if not hasattr(self, '_started'):
            self._started = False
        return self._started
    
    @property
    def file(self):
        """
        Property storing the h5py file in which all info from this sampler will
        be saved.
        """
        if not hasattr(self, '_file'):
            if self.started:
                self._file = h5py.File(self.file_name, 'r+')
            else:
                self._setup()
                self._started = True
        return self._file
    
    @property
    def jumping_distribution_set(self):
        """
        Property storing the JumpingDistributionSet object describing how
        walkers should jump from each point in the chain to the next point.
        """
        if not hasattr(self, '_jumping_distribution_set'):
            raise AttributeError("jumping_distribution_set referenced " +\
                "before it was set.")
        return self._jumping_distribution_set
    
    @jumping_distribution_set.setter
    def jumping_distribution_set(self, value):
        """
        Setter for the JumpingDistributionSet which describes how walkers
        should jump from each point in the chain to the next point.
        
        value: must be a JumpingDistributionSet object
        """
        if value is None:
            self._jumping_distribution_set = value
        elif isinstance(value, JumpingDistributionSet):
            if set(value.params) == set(self.parameters):
                self._jumping_distribution_set = value
            else:
                not_needed_parameters =\
                    set(value.params) - set(self.parameters)
                missing_parameters = set(self.parameters) - set(value.params)
                if not_needed_parameters and missing_parameters:
                    raise ValueError(("The given jumping_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({0!s}) and didn't describe some parameters which " +\
                        "were needed ({1!s}).").format(not_needed_parameters,\
                        missing_parameters))
                elif not_needed_parameters:
                    raise ValueError(("The given jumping_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({!s}).").format(not_needed_parameters))
                else:
                    raise ValueError(("The given jumping_distribution_set " +\
                        "is missing distributions for some parameters " +\
                        "({!s}).").format(missing_parameters))
        else:
            raise TypeError("jumping_distribution_set was set to an object " +\
                "which wasn't a JumpingDistributionSet object.")
    
    @property
    def guess_distribution_set(self):
        """
        Property storing the DistributionSet object which can be used to draw
        first positions of the walkers.
        """
        if not hasattr(self, '_guess_distribution_set'):
            raise AttributeError("guess_distribution_set referenced before " +\
                "it was set.")
        return self._guess_distribution_set
    
    @guess_distribution_set.setter
    def guess_distribution_set(self, value):
        """
        Setter for the DistributionSet object which can be used to draw first
        positions of the walkers.
        
        value: must be a DistributionSet object
        """
        if value is None:
            self._guess_distribution_set = value
        elif isinstance(value, DistributionSet):
            if set(value.params) == set(self.parameters):
                self._guess_distribution_set = value
            else:
                not_needed_parameters =\
                    set(value.params) - set(self.parameters)
                missing_parameters = set(self.parameters) - set(value.params)
                if not_needed_parameters and missing_parameters:
                    raise ValueError(("The given guess_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({0!s}) and didn't describe some parameters which " +\
                        "were needed ({1!s}).").format(not_needed_parameters,\
                        missing_parameters))
                elif not_needed_parameters:
                    raise ValueError(("The given guess_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({!s}).").format(not_needed_parameters))
                else:
                    raise ValueError(("The given guess_distribution_set " +\
                        "is missing distributions for some parameters " +\
                        "({!s}).").format(missing_parameters))
        else:
            raise TypeError("guess_distribution_set was set to something " +\
                "other than a DistributionSet object.")
    
    @property
    def has_priors(self):
        """
        Property storing True if this Sampler has priors and False if it does
        not.
        """
        if not hasattr(self, '_has_priors'):
            self._has_priors = (self.prior_distribution_set is not None)
        return self._has_priors
    
    @property
    def prior_distribution_set(self):
        """
        Property storing the DistributionSet object describing the prior
        distributions which should be accounted for in the posterior
        distribution.
        """
        if not hasattr(self, '_prior_distribution_set'):
            raise AttributeError("prior_distribution_set referenced before " +\
                "it was set.")
        return self._prior_distribution_set
    
    @prior_distribution_set.setter
    def prior_distribution_set(self, value):
        """
        Setter for the DistributionSet object describing the prior
        distributions which should be accounted for in the posterior
        distribution.
        
        value: must be a DistributionSet object
        """
        if value is None:
            self._prior_distribution_set = None
        elif isinstance(value, DistributionSet):
            if set(value.params) == set(self.parameters):
                self._prior_distribution_set = value
            if set(value.params) == set(self.parameters):
                self._prior_distribution_set = value
            else:
                not_needed_parameters =\
                    set(value.params) - set(self.parameters)
                missing_parameters = set(self.parameters) - set(value.params)
                if not_needed_parameters and missing_parameters:
                    raise ValueError(("The given prior_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({0!s}) and didn't describe some parameters which " +\
                        "were needed ({1!s}).").format(not_needed_parameters,\
                        missing_parameters))
                elif not_needed_parameters:
                    raise ValueError(("The given prior_distribution_set " +\
                        "described some parameters which aren't needed " +\
                        "({!s}).").format(not_needed_parameters))
                else:
                    raise ValueError(("The given prior_distribution_set " +\
                        "is missing distributions for some parameters " +\
                        "({!s}).").format(missing_parameters))
        else:
            raise TypeError("prior_distribution_set was set to something " +\
                "other than a DistributionSet object.")
    
    @property
    def nwalkers(self):
        """
        Property storing the integer number of independent walkers evolved by
        the sampler.
        """
        if not hasattr(self, '_nwalkers'):
            raise AttributeError("nwalkers referenced before it was set.")
        return self._nwalkers
    
    @nwalkers.setter
    def nwalkers(self, value):
        """
        Setter for the number of independent walkers evolved by the sampler.
        
        value: must be a positive integer
        """
        if type(value) in int_types:
            self._nwalkers = value
        else:
            raise TypeError("nwalkers was set to a non-int.")
    
    @property
    def loglikelihood(self):
        """
        Property storing the Loglikelihood object which should be explored.
        """
        if not hasattr(self, '_loglikelihood'):
            raise AttributeError("loglikelihood was referenced before it " +\
                "was set.")
        return self._loglikelihood
    
    @loglikelihood.setter
    def loglikelihood(self, value):
        """
        Setter for the Loglikelihood object which should be explored.
        
        value: a Loglikelihood object
        """
        if isinstance(value, Loglikelihood):
            self._loglikelihood = value
        else:
            raise TypeError("loglikelihood was not set to a " +\
                "Loglikelihood object.")
    
    @property
    def logprobability(self):
        """
        Property storing the callable which is actually passed on to the
        MetropolisHastingsSampler. Essentially, this logprobability function
        includes both the Loglikelihood and priors (if they exist).
        """
        if not hasattr(self, '_logprobability'):
            if self.has_priors:
                self._logprobability = (lambda pars, *args, **kwargs:\
                    (self.loglikelihood(pars, *args, **kwargs) +\
                    self.prior_distribution_set.log_value(\
                    dict(zip(self.parameters, pars)))))
            else:
                self._logprobability = self.loglikelihood
        return self._logprobability
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters needed by the SVDModel at the
        heart of this Loglikelihood.
        """
        return self.loglikelihood.num_parameters
    
    @property
    def parameters(self):
        """
        Property storing the parameters needed by the Model at the heart of
        this Loglikelihood.
        """
        return self.loglikelihood.parameters
    
    @property
    def sampler(self):
        """
        Property storing the MetropolisHastingsSampler (or EnsembleSampler)
        object at the core of this object.
        """
        if not hasattr(self, '_sampler'):
            if self.use_ensemble_sampler:
                self._sampler = EnsembleSampler(self.nwalkers,\
                    len(self.parameters), self.logprobability, args=self.args,\
                    kwargs=self.kwargs, threads=self.nthreads)
            else:
                self._sampler = MetropolisHastingsSampler(self.parameters,\
                    self.nwalkers, self.logprobability,\
                    self.jumping_distribution_set, nthreads=self.nthreads,\
                    args=self.args, kwargs=self.kwargs)
        return self._sampler
    
    @property
    def pos(self):
        """
        Property storing the current positions of all nwalkers of the walkers.
        """
        if not hasattr(self, '_pos'):
            self._pos = []
            iterations = 0
            while len(self._pos) < self.nwalkers:
                draw = self.guess_distribution_set.draw()
                if (self.prior_distribution_set is None) or\
                    np.isfinite(self.prior_distribution_set.log_value(draw)):
                    self._pos.append(\
                        [draw[param] for param in self.parameters])
                if (iterations >= 100 * self.nwalkers):
                    raise RuntimeError("100*nwalkers positions have been " +\
                        "drawn but not enough have had finite likelihood.")
                iterations += 1
            self._pos = np.array(self._pos)
        return self._pos
    
    @pos.setter
    def pos(self, value):
        """
        Setter for the current positions of this sampler's walkers.
        
        value: numpy.ndarray of shape (nwalkers, ndim)
        """
        if isinstance(value, np.ndarray) and (value.shape == self.pos.shape):
            self._pos = value
        else:
            raise ValueError("pos was not of the right size.")
    
    @property
    def lnprob(self):
        """
        Property storing the current values of the logprobability callable
        evaluated at the current walker positions.
        """
        if not hasattr(self, '_lnprob'):
            self._lnprob = None
        return self._lnprob
    
    @lnprob.setter
    def lnprob(self, value):
        """
        Setter for the values of the logprobability callable evaluated at the
        current walker positions.
        
        value: 1D numpy.ndarray of length nwalkers
        """
        if isinstance(value, np.ndarray):
            if value.shape == (self.nwalkers,):
                self._lnprob = value
            else:
                raise TypeError("lnprob doesn't have the expected shape, " +\
                    "which is (nwalkers,).")
        else:
            raise TypeError("lnprob should be an array but it isn't.")
    
    @property
    def rstate(self):
        """
        Property storing the current random state of the sampler.
        """
        if not hasattr(self, '_rstate'):
            self._rstate = np.random.get_state()
        return self._rstate
    
    @rstate.setter
    def rstate(self, value):
        """
        Setter for the current random state of the sampler.
        
        value: rstate property returned by self.sampler's run_mcmc function
        """
        self._rstate = value
    
    @property
    def checkpoint_index(self):
        """
        Property storing the index (starting at 0) of the current checkpoint.
        """
        if not hasattr(self, '_checkpoint_index'):
            raise AttributeError("checkpoint_index referenced before it " +\
                "was set.")
        return self._checkpoint_index
    
    @checkpoint_index.setter
    def checkpoint_index(self, value):
        """
        Setter for the index (starting at 0) of the current checkpoint.
        
        value: non-negative integer
        """
        if type(value) in int_types:
            self._checkpoint_index = value
        else:
            raise TypeError("checkpoint_index was set to a non-int.")
    
    def run_checkpoints(self, ncheckpoints, silence_error=False):
        """
        Runs this sampler for given number of checkpoints.
        
        ncheckpoints: positive integer
        silence_error: if True, if a KeyboardInterrupt is performed during the
                       run, then it is silenced and this function simply
                       returns. Default, False
        """
        if self.verbose:
            print("Starting checkpoint #{0} at {1!s}.".format(\
                self.checkpoint_index + 1, time.ctime()))
        checkpoint_index_start = self.checkpoint_index
        while self.checkpoint_index < (checkpoint_index_start + ncheckpoints):
            try:
                self.run_checkpoint()
            except KeyboardInterrupt:
                # TODO possibly check if some remnants were incompletely saved?
                if silence_error or (self.nthreads > 1):
                    break
                else:
                    raise
    
    def run_checkpoint(self):
        """
        Runs this sampler for a single checkpoint.
        """
        (self.pos, self.lnprob, self.rstate) = self.sampler.run_mcmc(self.pos,\
            self.steps_per_checkpoint, rstate0=self.rstate,\
            lnprob0=self.lnprob)
        self.update_file()
        self.sampler.reset()
        self.checkpoint_index = self.checkpoint_index + 1
        if self.verbose:
            print("Finished checkpoint #{0} at {1!s}.".format(\
                self.checkpoint_index, time.ctime()))
    
    def update_file(self):
        """
        Updates file in such a way that the most recently completed checkpoint
        is saved and the sampler can be restarted from its current state at any
        later time.
        """
        self.update_state()
        self.save_checkpoint()
        self.close()
    
    def update_state(self):
        """
        Updates the h5py file with information about the current state of the
        sampler so that a restart is possible at any later time.
        """
        if 'state' in self.file:
            group = self.file['state']
            del group['rstate'], group['pos'], group['lnprob']
        else:
            group = self.file.create_group('state')
        group.create_dataset('pos', data=self.pos)
        group.create_dataset('lnprob', data=self.lnprob)
        subgroup = group.create_group('rstate')
        subgroup.attrs['algorithm'] = self.rstate[0]
        subgroup.create_dataset('keys', data=self.rstate[1])
        subgroup.attrs['pos'] = self.rstate[2]
        subgroup.attrs['has_gauss'] = self.rstate[3]
        subgroup.attrs['cached_gaussian'] = self.rstate[4]
        
    def save_checkpoint(self):
        """
        Saves the data from the most recently completed checkpoint to disk.
        """
        group = self.file['checkpoints/chunk{0:d}'.format(self.chunk_index)]
        subgroup = group.create_group('{}'.format(self.checkpoint_index))
        subgroup.create_dataset('chain', data=self.sampler.chain)
        subgroup.create_dataset('lnprobability',\
            data=self.sampler.lnprobability)
        subgroup.create_dataset('acceptance_fraction',\
            data=self.sampler.acceptance_fraction)
        group.attrs['max_checkpoint_index'] = self.checkpoint_index
    
    def close(self):
        """
        Closes the h5py file object in which all information of this sampler is
        being saved (if it is open).
        """
        if hasattr(self, '_file'):
            self.file.close()
            del self._file
    

