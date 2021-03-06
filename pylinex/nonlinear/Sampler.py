"""
File: Sampler.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class which wraps around the
             MetropolisHastingsSampler defined in the distpy package.
"""
from __future__ import division
import os, time, h5py
import numpy as np
import numpy.linalg as la
from distpy import GaussianDistribution, CustomDiscreteDistribution,\
    DistributionSet, GaussianJumpingDistribution, JumpingDistributionSet,\
    MetropolisHastingsSampler
from ..util import bool_types, int_types, real_numerical_types,\
    sequence_types, create_hdf5_dataset, get_hdf5_value
from ..loglikelihood import Loglikelihood, load_loglikelihood_from_hdf5_group

try:
    from emcee import EnsembleSampler
except:
    have_emcee = False
    have_new_emcee = False
else:
    have_emcee = True
    try:
        from emcee import State as EmceeState
    except:
        have_new_emcee = False
    else:
        have_new_emcee = True

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
    def __init__(self, file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, steps_per_checkpoint=100, verbose=True,\
        restart_mode=None, num_threads=1, args=[], kwargs={},\
        use_ensemble_sampler=False, desired_acceptance_fraction=0.25):
        """
        Initializes a new sampler with the given file_name, loglikelihood,
        jumping distribution set, and guess distribution set.
        
        file_name: name of hdf5 file to save (if extant, a restart process is
                   triggered)
        num_walkers: number of MCMC chains being iterated together
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
                                by the last checkpoint.
        prior_distribution_set: optional extra DistributionSet object with
                                priors to include in the posterior to explore.
                                If None and this is a restart, priors are
                                copied from last run. if None and this is not a
                                restart, uses no priors.
        steps_per_checkpoint: integer number of steps taken between saving to
                              the hdf5 file. (if None, it is set to 100)
        verbose: if True, the time is printed after the completion of each
                          checkpoint (and before the first checkpoint)
        restart_mode: if None, no restarts are supported
                      if 'continue', runs old Sampler as if nothing stopped
                      if 'update', walkers are not moved but proposal
                                   covariance is updated based on walker
                                   positions/distributions
                      if 'trimmed_update', same as 'update' but walkers which
                                           stay more than 3 sigma from the mean
                                           lnprobability value will be ignored
                      if 'reinitialize', walkers position distribution and
                                         proposal distribution are reset using
                                         the walker positions/distributions
                      if 'trimmed_reinitialize', same as 'reinitialize' but
                                                 walkers which stay more than 3
                                                 sigma from the mean
                                                 lnprobability value will be
                                                 ignored
        num_threads: the number of threads to use in log likelihood
                     calculations for walkers. Default: 1, 1 is best unless
                     loglikelihood is very slow
        args: extra positional arguments to pass on to the likelihood
        kwargs: extra keyword arguments to pass on to the likelihood
        use_ensemble_sampler: if True, EnsembleSampler of emcee is used. This
                                       cannot be done if emcee is not installed
                              otherwise, MetropolisHastingsSampler from distpy
                                         is used
        desired_acceptance_fraction: only used if this is a restart,
                                     restart_mode in ['update', 'reinitialize']
                                     and jumping_distribution_set is None
        """
        self.use_ensemble_sampler = use_ensemble_sampler
        self.num_threads = num_threads
        self.restart_mode = restart_mode
        self.verbose = verbose
        self.num_walkers = num_walkers
        self.steps_per_checkpoint = steps_per_checkpoint
        self.loglikelihood = loglikelihood
        self.desired_acceptance_fraction = desired_acceptance_fraction
        self.jumping_distribution_set = jumping_distribution_set
        self.guess_distribution_set = guess_distribution_set
        self.prior_distribution_set = prior_distribution_set
        self.file_name = file_name
        self.args = args
        self.kwargs = kwargs
        self.file # loads in things for restart if necessary
        self.close()
    
    @property
    def num_threads(self):
        """
        Property storing the number of threads to use in calculating log
        likelihood values.
        """
        if not hasattr(self, '_num_threads'):
            raise AttributeError("num_threads referenced before it was set.")
        return self._num_threads
    
    @num_threads.setter
    def num_threads(self, value):
        """
        Setter for the number of threads to use in log likelihood calculations.
        
        value: a positive integer; 1 is best unless loglikelihood is very slow
        """
        if type(value) in int_types:
            if value > 0:
                self._num_threads = value
            else:
                raise ValueError("num_threads must be non-negative.")
        else:
            raise TypeError("num_threads was set to a non-int.")
    
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
    def desired_acceptance_fraction(self):
        """
        Property storing the desired acceptance fraction upon restarting. Only
        used if restart_mode in ['update', 'reinitialize'] and
        jumping_distribution_set given at initialization is None.
        """
        if not hasattr(self, '_desired_acceptance_fraction'):
            raise AttributeError("desired_acceptance_fraction was " +\
                "referenced before it was set.")
        return self._desired_acceptance_fraction
    
    @desired_acceptance_fraction.setter
    def desired_acceptance_fraction(self, value):
        """
        Setter for the desired_acceptance_fraction.
        
        value: must be a single number between 0 and 1, exclusive
        """
        if (type(value) in real_numerical_types) and (value > 0) and\
            (value < 1):
            self._desired_acceptance_fraction = value
        else:
            raise TypeError("desired_acceptance_fraction was not between 0 " +\
                "and 1, exclusive.")
    
    @property
    def proposal_covariance_reduction_factor(self):
        """
        Property storing the proposal_covariance_reduction_factor used if this
        is a restart, restart_mode in ['update', 'reinitialize'], and
        jumping_distribution_set given at initialization is None.
        """
        if not hasattr(self, '_proposal_covariance_reduction_factor'):
            self._proposal_covariance_reduction_factor =\
                (1 / (np.power(self.desired_acceptance_fraction,\
                (-2) / self.loglikelihood.num_parameters) - 1))
        return self._proposal_covariance_reduction_factor
    
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
        allowed_modes = ['continue', 'update', 'trimmed_update',\
            'reinitialize', 'trimmed_reinitialize', 'fisher_update']
        if (type(value) is type(None)) or (value in allowed_modes):
            if self.use_ensemble_sampler:
                if value != 'continue':
                    print("emcee's EnsembleSampler does not support " +\
                        "updates, so 'restart_mode' is being set to " +\
                        "'continue'.")
                self._restart_mode = 'continue'
            else:
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
    def file_split(original_file_name, chunk_index, first_output_file_name,\
        second_output_file_name):
        """
        Splits a file into two.
        
        original_file_name: the file location of original file
        chunk_index: if a negative integer between -num_chunks (exclusive) and,
                     -1 (inclusive), chunk_index is the number of chunks to cut
                     off the end of the original mcmc.hdf5 file. if a positive
                     integer between 1 (inclusive) and num_chunks (exclusive),
                     chunk_index is the number of chunks to cut off the start
                     of the original mcmc.hdf5 file.
        first_output_file_name: output file name containing earlier part of
                                chain
        second_output_file_name: output file name containing later part of
                                 chain
        """
        input_hdf5_file = h5py.File(original_file_name, 'r')
        try:
            num_chunks = 1 + input_hdf5_file.attrs['max_chunk_index']
            if chunk_index == 0:
                raise ValueError("The chunk_index given implies that the " +\
                    "second output file will contain all chunks, which " +\
                    "means there's no point calling this function.")
            elif chunk_index <= (-num_chunks):
                raise ValueError("chunk_index was less than or equal to " +\
                    "the negative of the number of chunks in the original " +\
                    "file.")
            elif chunk_index > num_chunks:
                raise ValueError("chunk_index is greater than implied " +\
                    "possible by the number of chunks in the file.")
            elif chunk_index == num_chunks:
                raise ValueError("The chunk_index given implies that the " +\
                    "first output file will contain all chunks, which " +\
                    "means there's no point calling this function.")
            else:
                if chunk_index < 0:
                    chunk_index = chunk_index + num_chunks
                input_prior_distribution_sets_group =\
                    input_hdf5_file['prior_distribution_sets']
                input_guess_distribution_sets_group =\
                    input_hdf5_file['guess_distribution_sets']
                input_jumping_distribution_sets_group =\
                    input_hdf5_file['jumping_distribution_sets']
                input_checkpoints_group = input_hdf5_file['checkpoints']
                first_output_hdf5_file = h5py.File(first_output_file_name, 'w')
                try:
                    first_output_hdf5_file.attrs['max_chunk_index'] =\
                        chunk_index - 1
                    input_hdf5_file.copy('loglikelihood',\
                        first_output_hdf5_file)
                    input_hdf5_file.copy('parameters', first_output_hdf5_file)
                    input_hdf5_file.copy('state', first_output_hdf5_file)
                    first_output_prior_distribution_sets_group =\
                        first_output_hdf5_file.create_group(\
                        'prior_distribution_sets')
                    for temp_chunk_index in range(chunk_index):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        if temp_chunk_string in\
                            input_prior_distribution_sets_group:
                            input_prior_distribution_sets_group.copy(\
                                 temp_chunk_string,\
                                 first_output_prior_distribution_sets_group)
                    first_output_guess_distribution_sets_group =\
                        first_output_hdf5_file.create_group(\
                        'guess_distribution_sets')
                    for temp_chunk_index in range(chunk_index):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        if temp_chunk_string in\
                            input_guess_distribution_sets_group:
                            input_guess_distribution_sets_group.copy(\
                                temp_chunk_string,\
                                first_output_guess_distribution_sets_group)
                    first_output_jumping_distribution_sets_group =\
                        first_output_hdf5_file.create_group(\
                        'jumping_distribution_sets')
                    for temp_chunk_index in range(chunk_index):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        if temp_chunk_string in\
                            input_jumping_distribution_sets_group:
                            input_jumping_distribution_sets_group.copy(\
                                temp_chunk_string,\
                                first_output_jumping_distribution_sets_group)
                    first_output_checkpoints_group =\
                        first_output_hdf5_file.create_group('checkpoints')
                    for temp_chunk_index in range(chunk_index):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        if temp_chunk_string in input_checkpoints_group:
                            input_checkpoints_group.copy(temp_chunk_string,\
                                first_output_checkpoints_group)
                except:
                    first_output_hdf5_file.close()
                    raise
                else:
                    first_output_hdf5_file.close()
                second_output_hdf5_file =\
                    h5py.File(second_output_file_name, 'w')
                try:
                    second_output_hdf5_file.attrs['max_chunk_index'] =\
                        num_chunks - chunk_index - 1
                    input_hdf5_file.copy('loglikelihood',\
                        second_output_hdf5_file)
                    input_hdf5_file.copy('parameters', second_output_hdf5_file)
                    input_hdf5_file.copy('state', second_output_hdf5_file)
                    second_output_prior_distribution_sets_group =\
                        second_output_hdf5_file.create_group(\
                        'prior_distribution_sets')
                    for temp_chunk_index in range(chunk_index, num_chunks):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        output_temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index - chunk_index)
                        if temp_chunk_string in\
                            input_prior_distribution_sets_group:
                            input_prior_distribution_sets_group.copy(\
                                temp_chunk_string,\
                                second_output_prior_distribution_sets_group,\
                                name=output_temp_chunk_string)
                    second_output_guess_distribution_sets_group =\
                        second_output_hdf5_file.create_group(\
                        'guess_distribution_sets')
                    for temp_chunk_index in range(chunk_index, num_chunks):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        output_temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index - chunk_index)
                        if temp_chunk_string in\
                            input_guess_distribution_sets_group:
                            input_guess_distribution_sets_group.copy(\
                                temp_chunk_string,\
                                second_output_guess_distribution_sets_group,\
                                name=output_temp_chunk_string)
                    second_output_jumping_distribution_sets_group =\
                        second_output_hdf5_file.create_group(\
                        'jumping_distribution_sets')
                    for temp_chunk_index in range(chunk_index, num_chunks):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        output_temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index - chunk_index)
                        if temp_chunk_string in\
                            input_jumping_distribution_sets_group:
                            input_jumping_distribution_sets_group.copy(\
                                temp_chunk_string,\
                                second_output_jumping_distribution_sets_group,\
                                name=output_temp_chunk_string)
                    second_output_checkpoints_group =\
                        second_output_hdf5_file.create_group('checkpoints')
                    for temp_chunk_index in range(chunk_index, num_chunks):
                        temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index)
                        output_temp_chunk_string =\
                            'chunk{:d}'.format(temp_chunk_index - chunk_index)
                        if temp_chunk_string in input_checkpoints_group:
                            input_checkpoints_group.copy(temp_chunk_string,\
                                second_output_checkpoints_group,\
                                name=output_temp_chunk_string)
                except:
                    second_output_hdf5_file.close()
                    raise
                else:
                    second_output_hdf5_file.close()
        except:
            input_hdf5_file.close()
            raise
        else:
            input_hdf5_file.close()
    
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
        if type(self.prior_distribution_set) is type(None):
            self.prior_distribution_set = prior_distribution_set
        elif (self.prior_distribution_set != prior_distribution_set):
            raise ValueError("prior_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
        self.guess_distribution_set = guess_distribution_set
        if type(self.jumping_distribution_set) is type(None):
            self.jumping_distribution_set = jumping_distribution_set
        elif (self.jumping_distribution_set != jumping_distribution_set):
            raise ValueError("jumping_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
    
    def _setup_restart_update(self, pos, lnprob, guess_distribution_set,\
        prior_distribution_set, jumping_distribution_set,\
        last_saved_chunk_string, trim_tails=False):
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
        trim_tails: if True, only walkers with logL values within 3sigma of the
                             mean logL value are retained for distribution
                             purposes
                    if False (default), all walkers are retained
        """
        self.chunk_index = self.chunk_index + 1
        self.file.attrs['max_chunk_index'] = self.chunk_index
        new_chunk_string = 'chunk{0:d}'.format(self.chunk_index)
        self._pos = pos
        self._lnprob = lnprob
        if type(self.prior_distribution_set) is type(None):
            self.prior_distribution_set = prior_distribution_set
        elif (self.prior_distribution_set != prior_distribution_set):
            raise ValueError("prior_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
        if type(self.prior_distribution_set) is not type(None):
            group = self.file['prior_distribution_sets']
            subgroup = group.create_group(new_chunk_string)
            self.prior_distribution_set.fill_hdf5_group(subgroup)
        self.guess_distribution_set = guess_distribution_set
        group = self.file['guess_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.guess_distribution_set.fill_hdf5_group(subgroup)
        if type(self.jumping_distribution_set) is type(None):
            self.jumping_distribution_set =\
                self._generate_reinitialized_jumping_distribution_set(\
                jumping_distribution_set, last_saved_chunk_string,\
                trim_tails=trim_tails)
        group = self.file['jumping_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.jumping_distribution_set.fill_hdf5_group(subgroup)
        self.file['checkpoints'].create_group(new_chunk_string)
        self.checkpoint_index = 0
    
    def _setup_restart_fisher_update(self, pos, lnprob,\
        guess_distribution_set, prior_distribution_set,\
        jumping_distribution_set, last_saved_chunk_string):
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
        self._pos = pos
        self._lnprob = lnprob
        if type(self.prior_distribution_set) is type(None):
            self.prior_distribution_set = prior_distribution_set
        elif (self.prior_distribution_set != prior_distribution_set):
            raise ValueError("prior_distribution_set changed since last " +\
                "run, so restart_mode can't be 'continue'.")
        if type(self.prior_distribution_set) is not type(None):
            group = self.file['prior_distribution_sets']
            subgroup = group.create_group(new_chunk_string)
            self.prior_distribution_set.fill_hdf5_group(subgroup)
        self.guess_distribution_set = guess_distribution_set
        group = self.file['guess_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.guess_distribution_set.fill_hdf5_group(subgroup)
        if type(self.jumping_distribution_set) is type(None):
            self.jumping_distribution_set =\
                self._generate_fisher_updated_jumping_distribution_set(\
                jumping_distribution_set, last_saved_chunk_string)
        group = self.file['jumping_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.jumping_distribution_set.fill_hdf5_group(subgroup)
        self.file['checkpoints'].create_group(new_chunk_string)
        self.checkpoint_index = 0
    
    def _setup_restart_reinitialize(self, guess_distribution_set,\
        prior_distribution_set, jumping_distribution_set,\
        last_saved_chunk_string, trim_tails=False):
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
        trim_tails: if True, only walkers with logL values within 3sigma of the
                             mean logL value are retained for distribution
                             purposes
                    if False (default), all walkers are retained
        """
        self.chunk_index = self.chunk_index + 1
        self.file.attrs['max_chunk_index'] = self.chunk_index
        new_chunk_string = 'chunk{0:d}'.format(self.chunk_index)
        if type(self.prior_distribution_set) is type(None):
            self.prior_distribution_set = prior_distribution_set
        if prior_distribution_set != self.prior_distribution_set:
            print("prior_distribution_set is changing, so " +\
                  "the distribution being explored is being " +\
                  "changed discontinuously!")
        if type(self.prior_distribution_set) is not type(None):
            group = self.file['prior_distribution_sets']
            subgroup = group.create_group(new_chunk_string)
            self.prior_distribution_set.fill_hdf5_group(\
                subgroup)
        if type(self.jumping_distribution_set) is type(None):
            self.jumping_distribution_set =\
                self._generate_reinitialized_jumping_distribution_set(\
                jumping_distribution_set, last_saved_chunk_string,\
                trim_tails=trim_tails)
        group = self.file['jumping_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.jumping_distribution_set.fill_hdf5_group(subgroup)
        if type(self.guess_distribution_set) is type(None):
            self.guess_distribution_set =\
                self._generate_reinitialized_guess_distribution_set(\
                guess_distribution_set, last_saved_chunk_string,\
                trim_tails=trim_tails)
        group = self.file['guess_distribution_sets']
        subgroup = group.create_group(new_chunk_string)
        self.guess_distribution_set.fill_hdf5_group(subgroup)
        self.file['checkpoints'].create_group(new_chunk_string)
        self.checkpoint_index = 0
    
    def get_chain_chunk(self, chunk_string, return_lnprobability=False):
        """
        Gets the chain (and possibly log probabilities) associated with the
        given chunk.
        
        chunk_string: the string describing the chunk to load
        return_lnprobability: if True, lnprobability returned alongside chain
        
        returns: (chain, lnprobability) if return_lnprobability else chain
                 where chain has shape (nwalkers, nsteps, nparameters) and
                 lnprobability has shape (nwalkers, nsteps)
        """
        group = self.file['checkpoints/{!s}'.format(chunk_string)]
        (chain, lnprobability) = ([], [])
        for checkpoint_index in range(group.attrs['max_checkpoint_index'] + 1):
            subgroup = group['{}'.format(checkpoint_index)]
            chain.append(get_hdf5_value(subgroup['chain']))
            lnprobability.append(get_hdf5_value(subgroup['lnprobability']))
        chain = np.concatenate(chain, axis=1)
        lnprobability = np.concatenate(lnprobability, axis=1)
        if return_lnprobability:
            return (chain, lnprobability)
        else:
            return chain
    
    def _generate_reinitialized_guess_distribution_set(self,\
        guess_distribution_set, last_saved_chunk_string, trim_tails=False,\
        min_eigenvalue=1e-8):
        """
        Generates a new guess_distribution_set to use for a Sampler restarted
        with restart_mode=='reinitialize'.
        
        guess_distribution_set: the DistributionSet object describing how
                                walkers were initialized in last saved chunk.
        last_saved_chunk_string: string of the form
                                 'chunk{:d}'.format(chunk_index) where
                                 chunk_index is the index of the last saved
                                 chunk
        trim_tails: if True, only walkers with logL values within 3sigma of the
                             mean logL value are retained for distribution
                             purposes
                    if False (default), all walkers are retained
        
        returns: a DistributionSet object to use to initialize walkers for the
                 next chunk of this sampler.
        """
        continuous_params = self.jumping_distribution_set.continuous_params
        continuous_parameter_indices =\
            np.array([self.parameters.index(par) for par in continuous_params])
        (last_chunk_chain, last_chunk_loglikelihood) = self.get_chain_chunk(\
            last_saved_chunk_string, return_lnprobability=True)
        walker_averaged_loglikelihood =\
            np.mean(last_chunk_loglikelihood, axis=-1)
        if trim_tails:
            loglikelihood_cutoff = np.mean(walker_averaged_loglikelihood) -\
                (3 * np.std(walker_averaged_loglikelihood))
            likelihood_based_weights = (walker_averaged_loglikelihood >=\
                loglikelihood_cutoff).astype(int)
        else:
            likelihood_based_weights =\
                np.ones_like(walker_averaged_loglikelihood)
        likelihood_based_weights = (likelihood_based_weights[:,np.newaxis] *\
            np.ones(last_chunk_loglikelihood.shape)).flatten()
        last_chunk_chain_continuous =\
            last_chunk_chain[...,continuous_parameter_indices]
        flattened_shape = (-1, last_chunk_chain_continuous.shape[-1])
        last_chunk_chain_continuous =\
            np.reshape(last_chunk_chain_continuous, flattened_shape)
        transform_list =\
            guess_distribution_set.transform_set[continuous_params]
        last_chunk_chain_continuous =\
            transform_list(last_chunk_chain_continuous, axis=-1)
        last_chunk_continuous_mean =\
            np.sum(last_chunk_chain_continuous *\
            likelihood_based_weights[:,np.newaxis], axis=0) /\
            np.sum(likelihood_based_weights)
        last_chunk_continuous_covariance = np.cov(\
            last_chunk_chain_continuous, ddof=0, rowvar=False,\
            aweights=likelihood_based_weights)
        if len(last_chunk_continuous_mean) == 1:
            last_chunk_continuous_covariance =\
                np.array([[float(last_chunk_continuous_covariance)]])
        (eigenvalues, eigenvectors) =\
            la.eigh(last_chunk_continuous_covariance)
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        last_chunk_continuous_covariance =\
            np.dot(eigenvectors * eigenvalues[np.newaxis,:], eigenvectors.T)
        distribution = GaussianDistribution(last_chunk_continuous_mean,\
            last_chunk_continuous_covariance)
        new_guess_distribution_set = DistributionSet()
        new_guess_distribution_set.add_distribution(distribution,\
            continuous_params, transform_list)
        discrete_params = self.jumping_distribution_set.discrete_params
        for param in discrete_params:
            parameter_index = self.parameters.index(param)
            this_last_chunk_chain =\
                last_chunk_chain[...,parameter_index].flatten()
            transform = guess_distribution_set.transform_set[param]
            this_last_chunk_chain = transform(this_last_chunk_chain)
            distribution = CustomDiscreteDistribution(\
                *np.unique(this_last_chunk_chain, return_counts=True))
            new_guess_distribution_set.add_distribution(distribution,\
                param, transform)
        return new_guess_distribution_set
    
    def _generate_reinitialized_jumping_distribution_set(self,\
        jumping_distribution_set, last_saved_chunk_string, trim_tails=False,\
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
        trim_tails: if True, only walkers with logL values within 3sigma of the
                             mean logL value are retained for distribution
                             purposes
                    if False (default), all walkers are retained
        
        returns: a JumpingDistributionSet object to use to determine how
                 walkers travel through parameter space for the next chunk of
                 this sampler
        """
        continuous_params = jumping_distribution_set.continuous_params
        continuous_parameter_indices =\
            np.array([self.parameters.index(par) for par in continuous_params])
        (last_chunk_chain, last_chunk_loglikelihood) = self.get_chain_chunk(\
            last_saved_chunk_string, return_lnprobability=True)
        walker_averaged_loglikelihood =\
            np.mean(last_chunk_loglikelihood, axis=-1)
        if trim_tails:
            loglikelihood_cutoff = np.mean(walker_averaged_loglikelihood) -\
                (3 * np.std(walker_averaged_loglikelihood))
            likelihood_based_weights = (walker_averaged_loglikelihood >=\
                loglikelihood_cutoff).astype(int)
        else:
            likelihood_based_weights =\
                np.ones_like(walker_averaged_loglikelihood)
        likelihood_based_weights = (likelihood_based_weights[:,np.newaxis] *\
            np.ones(last_chunk_loglikelihood.shape)).flatten()
        last_chunk_chain_continuous =\
            last_chunk_chain[...,continuous_parameter_indices]
        flattened_shape = (-1, last_chunk_chain_continuous.shape[-1])
        last_chunk_chain_continuous =\
            np.reshape(last_chunk_chain_continuous, flattened_shape)
        transform_list =\
            jumping_distribution_set.transform_set[continuous_params]
        last_chunk_chain_continuous =\
            transform_list(last_chunk_chain_continuous, axis=-1)
        last_chunk_continuous_covariance = np.cov(\
            last_chunk_chain_continuous, rowvar=False, ddof=0,\
            aweights=likelihood_based_weights)
        if last_chunk_continuous_covariance.shape == ():
            last_chunk_continuous_covariance =\
                np.array([[float(last_chunk_continuous_covariance)]])
        (eigenvalues, eigenvectors) =\
            la.eigh(last_chunk_continuous_covariance)
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
        last_chunk_continuous_covariance =\
            np.dot(eigenvectors * eigenvalues[np.newaxis,:], eigenvectors.T)
        last_chunk_continuous_covariance /=\
            self.proposal_covariance_reduction_factor
        distribution =\
            GaussianJumpingDistribution(last_chunk_continuous_covariance)
        new_jumping_distribution_set = JumpingDistributionSet()
        new_jumping_distribution_set.add_distribution(distribution,\
            continuous_params, transform_list)
        new_jumping_distribution_set = new_jumping_distribution_set +\
            jumping_distribution_set.discrete_subset()
        return new_jumping_distribution_set
    
    def _generate_fisher_updated_jumping_distribution_set(self,\
        jumping_distribution_set, last_saved_chunk_string,\
        max_standard_deviations=np.inf, larger_differences=1e-5,\
        smaller_differences=1e-6):
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
        max_standard_deviations: the maximum allowable 1-sigma deviations of
                                 each parameter. If this is a constant, it is
                                 assumed to apply to all parameters
        larger_differences: either single number or 1D array of numbers to use
                            as the numerical difference in parameters.
                            Default: 10^(-5). This is the amount by which the
                            parameters are shifted between evaluations of the
                            gradient. Only used if loglikelihood gradient is
                            not explicitly computable.
        smaller_differences: either single_number or 1D array of numbers to use
                             as the numerical difference in parameters.
                             Default: 10^(-6). This is the amount by which the
                             parameters are shifted during each approximation
                             of the gradient. Only used if loglikelihood
                             hessian is not explicitly computable
        
        returns: a JumpingDistributionSet object to use to determine how
                 walkers travel through parameter space for the next chunk of
                 this sampler
        """
        if jumping_distribution_set.discrete_params:
            raise TypeError("The fisher_update restart_mode can only be " +\
                "used when there are no discrete parameters in the " +\
                "likelihood.")
        else:
            parameters = jumping_distribution_set.params
            transform_list = jumping_distribution_set.transform_set[parameters]
            (last_chunk_chain, last_chunk_loglikelihood) =\
                self.get_chain_chunk(last_saved_chunk_string,\
                return_lnprobability=True)
            maximum_likelihood_index = np.unravel_index(\
                np.argmax(last_chunk_loglikelihood),\
                last_chunk_loglikelihood.shape)
            maximum_likelihood_parameters =\
                last_chunk_chain[maximum_likelihood_index]
            last_chunk_covariance =\
                self.loglikelihood.parameter_covariance_fisher_formalism(\
                maximum_likelihood_parameters, transform_list=transform_list,\
                max_standard_deviations=max_standard_deviations,\
                larger_differences=larger_differences,\
                smaller_differences=smaller_differences)
            last_chunk_covariance /=\
                self.proposal_covariance_reduction_factor
            distribution = GaussianJumpingDistribution(last_chunk_covariance)
            return JumpingDistributionSet([(distribution, parameters,\
                transform_list)])
    
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
        pos = get_hdf5_value(group['pos'])
        lnprob = get_hdf5_value(group['lnprob'])
        subgroup = group['rstate']
        alg = subgroup.attrs['algorithm']
        keys = get_hdf5_value(subgroup['keys'])
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
            loglikelihood_from_file =\
                load_loglikelihood_from_hdf5_group(self.file['loglikelihood'])
            if self.file.attrs['use_ensemble_sampler'] !=\
                self.use_ensemble_sampler:
                print("WARNING: The value of use_ensemble_sampler in" +\
                    "the existing file differs from the one given at " +\
                    "initialization. For consistency, the one from the " +\
                    "file is being used.")
                self.use_ensemble_sampler =\
                    self.file.attrs['use_ensemble_sampler']
            if type(self.loglikelihood) is type(None):
                self.loglikelihood = loglikelihood_from_file
            elif loglikelihood_from_file != self.loglikelihood:
                print("WARNING: The loglikelihood in the existing file is " +\
                    "not the same as the loglikelihood given at " +\
                    "initialization. The loglikelihood from the file is " +\
                    "the one being used because otherwise the posterior " +\
                    "has changed since before the restart.")
                self.loglikelihood = loglikelihood_from_file
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
                    jumping_distribution_set, chunk_string, trim_tails=False)
            elif self.restart_mode == 'trimmed_update':
                self._setup_restart_update(pos, lnprob,\
                    guess_distribution_set, prior_distribution_set,\
                    jumping_distribution_set, chunk_string, trim_tails=True)
            elif self.restart_mode == 'reinitialize':
                self._setup_restart_reinitialize(guess_distribution_set,\
                    prior_distribution_set, jumping_distribution_set,\
                    chunk_string, trim_tails=False)
            elif self.restart_mode == 'trimmed_reinitialize':
                self._setup_restart_reinitialize(guess_distribution_set,\
                    prior_distribution_set, jumping_distribution_set,\
                    chunk_string, trim_tails=True)
            else:
                # guaranteed here that self.restart_mode == 'fisher_update'
                self._setup_restart_fisher_update(pos, lnprob,\
                    guess_distribution_set, prior_distribution_set,\
                    jumping_distribution_set, chunk_string)
    
    def _setup_new_file(self):
        """
        Writes the start of a new file at this Sampler's file_name. This occurs
        when a Sampler is started for the first time (i.e. this is not a
        restart).
        """
        self._file = h5py.File(self.file_name, 'w')
        (self.chunk_index, self.checkpoint_index) = (0, 0)
        self.file.attrs['max_chunk_index'] = self.chunk_index
        self.file.attrs['use_ensemble_sampler'] = self.use_ensemble_sampler
        group = self.file.create_group('guess_distribution_sets')
        self.guess_distribution_set.fill_hdf5_group(\
            group.create_group('chunk0'))
        group = self.file.create_group('prior_distribution_sets')
        if self.has_priors:
            self.prior_distribution_set.fill_hdf5_group(\
                group.create_group('chunk0'))
        group = self.file.create_group('jumping_distribution_sets')
        if type(self.jumping_distribution_set) is not type(None):
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
        elif (type(self.guess_distribution_set) is not type(None)) and\
            ((type(self.jumping_distribution_set) is not type(None)) or\
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
        if type(value) is type(None):
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
        if type(value) is type(None):
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
            self._has_priors =\
                (type(self.prior_distribution_set) is not type(None))
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
        if type(value) is type(None):
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
    def num_walkers(self):
        """
        Property storing the integer number of independent walkers evolved by
        the sampler.
        """
        if not hasattr(self, '_num_walkers'):
            raise AttributeError("num_walkers referenced before it was set.")
        return self._num_walkers
    
    @num_walkers.setter
    def num_walkers(self, value):
        """
        Setter for the number of independent walkers evolved by the sampler.
        
        value: must be a positive integer
        """
        if type(value) in int_types:
            self._num_walkers = value
        else:
            raise TypeError("num_walkers was set to a non-int.")
    
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
        if (type(value) is type(None)) or isinstance(value, Loglikelihood):
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
                def logprobability(pars, *args, **kwargs):
                    prior_value = self.prior_distribution_set.log_value(\
                        dict(zip(self.parameters, pars)))
                    if np.isfinite(prior_value):
                        return (self.loglikelihood(pars, *args, **kwargs) +\
                            prior_value)
                    else:
                        return -np.inf
                self._logprobability = logprobability
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
                if have_emcee:
                    self._sampler = EnsembleSampler(self.num_walkers,\
                        len(self.parameters), self.logprobability,\
                        args=self.args, kwargs=self.kwargs,\
                        threads=self.num_threads)
                else:
                    raise ImportError("use_ensemble_sampler cannot be set " +\
                        "to True (EnsembleSampler cannot be used) if emcee " +\
                        "is not installed.")
            else:
                self._sampler = MetropolisHastingsSampler(self.parameters,\
                    self.num_walkers, self.logprobability,\
                    self.jumping_distribution_set,\
                    num_threads=self.num_threads, args=self.args,\
                    kwargs=self.kwargs)
        return self._sampler
    
    @property
    def pos(self):
        """
        Property storing the current positions of all num_walkers of the
        walkers.
        """
        if not hasattr(self, '_pos'):
            self._pos = []
            iterations = 0
            while len(self._pos) < self.num_walkers:
                draw = self.guess_distribution_set.draw()
                if (type(self.prior_distribution_set) is type(None)) or\
                    np.isfinite(self.prior_distribution_set.log_value(draw)):
                    self._pos.append(\
                        [draw[param] for param in self.parameters])
                if iterations > (100 * self.num_walkers):
                    raise RuntimeError(("100*num_walkers positions have " +\
                        "been drawn but not enough have had finite " +\
                        "likelihood. The last draw which failed was: " +\
                        "{}.").format(draw))
                iterations += 1
            self._pos = np.array(self._pos)
        return self._pos
    
    @pos.setter
    def pos(self, value):
        """
        Setter for the current positions of this sampler's walkers.
        
        value: numpy.ndarray of shape (num_walkers, ndim)
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
        
        value: 1D numpy.ndarray of length num_walkers
        """
        if isinstance(value, np.ndarray):
            if value.shape == (self.num_walkers,):
                self._lnprob = value
            else:
                raise TypeError("lnprob doesn't have the expected shape, " +\
                    "which is (num_walkers,).")
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
                if silence_error or (self.num_threads > 1):
                    break
                else:
                    raise
    
    def run_checkpoint(self):
        """
        Runs this sampler for a single checkpoint.
        """
        if self.use_ensemble_sampler:
            if have_new_emcee:
                starting_state = EmceeState(self.pos, log_prob=self.lnprob,\
                    random_state=self.rstate)
                ending_state = self.sampler.run_mcmc(starting_state,\
                    self.steps_per_checkpoint)
                self.pos = ending_state.coords
                self.lnprob = ending_state.log_prob
                self.rstate = ending_state.random_state
            else:
                (self.pos, self.lnprob, self.rstate) = self.sampler.run_mcmc(\
                    self.pos, self.steps_per_checkpoint, rstate0=self.rstate,\
                    lnprob0=self.lnprob)
        else:
            (self.pos, self.lnprob, self.rstate) = self.sampler.run_mcmc(\
                self.pos, self.steps_per_checkpoint,\
                initial_random_state=self.rstate, initial_lnprob=self.lnprob)
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
        create_hdf5_dataset(group, 'pos', data=self.pos)
        create_hdf5_dataset(group, 'lnprob', data=self.lnprob)
        subgroup = group.create_group('rstate')
        subgroup.attrs['algorithm'] = self.rstate[0]
        create_hdf5_dataset(subgroup, 'keys', data=self.rstate[1])
        subgroup.attrs['pos'] = self.rstate[2]
        subgroup.attrs['has_gauss'] = self.rstate[3]
        subgroup.attrs['cached_gaussian'] = self.rstate[4]
        
    def save_checkpoint(self):
        """
        Saves the data from the most recently completed checkpoint to disk.
        """
        group = self.file['checkpoints/chunk{0:d}'.format(self.chunk_index)]
        subgroup = group.create_group('{}'.format(self.checkpoint_index))
        create_hdf5_dataset(subgroup, 'chain', data=self.sampler.chain)
        create_hdf5_dataset(subgroup, 'lnprobability',\
            data=self.sampler.lnprobability)
        create_hdf5_dataset(subgroup, 'acceptance_fraction',\
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
    

