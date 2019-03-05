"""
File: pylinex/fitter/TruncationExtractor.py
Author: Keith Tauscher
Date: 1 Oct 2018

Description: Class which uses the rest of the module to perform an end-to-end
             extraction. The inputs of the class are data and error vectors,
             training set matrices and expanders.
"""
import os
import numpy as np
from distpy import DistributionSet, DiscreteUniformDistribution,\
    DistributionSet, DiscreteUniformDistribution, KroneckerDeltaDistribution,\
    JumpingDistributionSet, GridHopJumpingDistribution
from ..util import Savable, create_hdf5_dataset, int_types, sequence_types,\
    bool_types
from ..expander import Expander, NullExpander, ExpanderSet
from ..basis import TrainedBasis, BasisSum, effective_training_set_rank
from ..fitter import Fitter
from ..loglikelihood import LinearTruncationLoglikelihood
from .Sampler import Sampler
from .BurnRule import BurnRule
from .NLFitter import NLFitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TruncationExtractor(Savable):
    """
    Class which, given:
    
    1) 1D data array
    2) 1D error array
    3) names of different components of data
    4) training set matrices for each of component of the data
    5) (optional) Expander objects which transform curves from the space in
       which the training set is defined to the space in which the data is
       defined
    6) information criterion to use for balancing parameter number and
       goodness-of-fit
    
    
    extracts components of the data.
    """
    def __init__(self, data, error, names, training_sets, nterms_maxima,\
        file_name, information_criterion='deviance_information_criterion',\
        expanders=None, trust_ranks=False, verbose=True):
        """
        Initializes an Extractor object with the given data and error vectors,
        names, training sets, dimensions, compiled quantity, quantity to
        minimize, and expanders.
        
        data: 1D numpy.ndarray of observed values of some quantity
        error: 1D numpy.ndarray of error values on the observed data
        names: names of distinct bases to separate
        training_sets: training sets corresponding to given names of bases.
                       Must be 2D array where the first dimension represents
                       the number of the curve.
        nterms_maxima: the maximum number of terms for each basis
        file_name: string location of the file at which to place Sampler
        information_criterion: string name of the information criterion to
                               minimize to balance parameter number and
                               goodness-of-fit
        expanders: list of Expander objects which expand each of the basis sets
        trust_ranks: if True, all walkers are initialized on ranks
                     if False, they are initialized all over discrete space
        verbose: if True, messages should be printed to the screen
        """
        self.file_name = file_name
        self.data = data
        self.error = error
        self.names = names
        self.training_sets = training_sets
        self.nterms_maxima = nterms_maxima
        self.expanders = expanders
        self.information_criterion = information_criterion
        self.verbose = verbose
        self.trust_ranks = trust_ranks
    
    @property
    def trust_ranks(self):
        """
        Property storing a boolean which determines whether or not the ranks of
        the given training sets are to be trusted. If true, walkers are
        initialized at ranks. If False, walkers are initialized all over
        allowed parameter space.
        """
        if not hasattr(self, '_trust_ranks'):
            raise AttributeError("trust_ranks was referenced before it was " +\
                "set.")
        return self._trust_ranks
    
    @trust_ranks.setter
    def trust_ranks(self, value):
        """
        Setter for the boolean determining whether training set ranks should be
        used in initializing walkers.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._trust_ranks = value
        else:
            raise TypeError("trust_ranks was set to a non-bool.")
    
    @property
    def names(self):
        """
        Property storing the names of each subbasis.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("names was referenced before it was set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Setter for the names of the subbases.
        
        value: sequence of strings
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._names = [element for element in value]
            else:
                raise TypeError("Not all elements of names sequence were " +\
                    "strings.")
        else:
            raise TypeError("names was set to a non-sequence.")
    
    @property
    def nterms_maxima(self):
        """
        Property storing the maximum numbers of terms necessary for each basis.
        """
        if not hasattr(self, '_nterms_maxima'):
            raise AttributeError("nterms_maxima was referenced before it " +\
                "was set.")
        return self._nterms_maxima
    
    @nterms_maxima.setter
    def nterms_maxima(self, value):
        """
        Setter for the maximum numbers of terms for each basis
        
        value: sequence of numbers
        """
        if type(value) in sequence_types:
            if all([(type(element) in int_types) for element in value]):
                if all([(element > 1) for element in value]):
                    self._nterms_maxima =\
                        np.array([element for element in value])
                else:
                    raise ValueError("Not all maximum numbers of terms " +\
                        "were greater than 1.")
            else:
                raise TypeError("Not all maximum numbers of terms were " +\
                    "integers.")
        else:
            raise TypeError("nterms_maxima was set to a non-sequence.")
    
    @property
    def verbose(self):
        """
        Property storing a boolean switch determining which things are printed.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose was referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the verbose property which decided whether things are
        printed.
        
        value: must be a bool
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise TypeError("verbose was set to a non-bool.")
    
    @property
    def information_criterion(self):
        """
        Property storing string name of the information criterion to minimize.
        """
        if not hasattr(self, '_information_criterion'):
            raise AttributeError("information_criterion was referenced " +\
                                 "before it was set.")
        return self._information_criterion
    
    @information_criterion.setter
    def information_criterion(self, value):
        """
        Allows user to supply string name of the information criterion to
        minimize.
        """
        if isinstance(value, basestring):
            self._information_criterion = value
        else:
            raise TypeError("information_criterion was not a string.")
    
    @property
    def data(self):
        """
        Property storing the data from which pieces are to be extracted. Should
        be a 1D numpy array.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data property. It checks to ensure that value is a 1D
        numpy.ndarray or can be cast to one.
        """
        try:
            value = np.array(value)
        except:
            raise TypeError("data given to Extractor couldn't be cast as a " +\
                            "numpy.ndarray.")
        if value.ndim in [1, 2]:
            self._data = value
        else:
            raise ValueError("data must be 1D or 2D.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the data. A positive integer
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.data.shape[-1]
        return self._num_channels
    
    @property
    def error(self):
        """
        Property storing the error level in the data. This is used to define
        the dot product.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property.
        
        value: must be a 1D numpy.ndarray of positive values with the same
               length as the data.
        """
        try:
            value = np.array(value)
        except:
            raise TypeError("error could not be cast to a numpy.ndarray.")
        if value.shape == (self.num_channels,):
            self._error = value
        else:
            raise ValueError("error was set to a numpy.ndarray which " +\
                             "didn't have the expected shape (i.e. 1D with " +\
                             "length given by number of data channels.")
    
    @property
    def num_bases(self):
        """
        Property storing the number of sets of basis functions (also the same
        as the number of distinguished pieces of the data).
        """
        if not hasattr(self, '_num_bases'):
            self._num_bases = len(self.names)
        return self._num_bases
    
    @property
    def training_sets(self):
        """
        Property storing the training sets used in this extraction.
        
        returns a list of numpy.ndarrays
        """
        if not hasattr(self, '_training_sets'):
            raise AttributeError("training_sets was referenced before it " +\
                                 "was set.")
        return self._training_sets
    
    @training_sets.setter
    def training_sets(self, value):
        """
        Allows user to set training_sets with list of numpy arrays.
        
        value: sequence of numpy.ndarray objects storing training sets, which
               are 2D
        """
        if type(value) in sequence_types:
            num_training_sets = len(value)
            if num_training_sets == self.num_bases:
                if all([isinstance(ts, np.ndarray) for ts in value]):
                    if all([(ts.ndim == 2) for ts in value]):
                        self._training_sets = [ts for ts in value]
                    else:
                        raise ValueError("At least one of the training " +\
                                         "sets given to Extractor was not 2D.")
                else:
                    raise TypeError("At least one of the given training " +\
                                    "sets given to Extractor was not a " +\
                                    "numpy.ndarray.")
            else:
                raise ValueError(("The number of names given to Extractor " +\
                   "({0}) was not equal to the number of training sets " +\
                   "given ({1})").format(self.num_bases, num_training_sets))
        else:
            raise TypeError("training_sets of Extractor class was set to a " +\
                            "non-sequence.")
    
    @property
    def training_set_ranks(self):
        """
        Property storing the effective ranks of the training sets given. This
        is computed by fitting the training set with its own SVD modes and
        checking how many terms are necessary to fit down to the
        (expander-contracted) noise level.
        """
        if not hasattr(self, '_training_set_ranks'):
            self._training_set_ranks = {}
            for (name, expander, training_set) in\
                zip(self.names, self.expanders, self.training_sets):
                self._training_set_ranks[name] = effective_training_set_rank(\
                    training_set, expander.contract_error(self.error))
        return self._training_set_ranks
    
    @property
    def training_set_lengths(self):
        """
        Property storing the number of channels in each of the different
        training sets.
        """
        if not hasattr(self, '_training_set_lengths'):
            self._training_set_lengths =\
                [ts.shape[-1] for ts in self.training_sets]
        return self._training_set_lengths
    
    @property
    def total_number_of_combined_training_set_curves(self):
        """
        The number of combined training set curves which are given by the
        training sets of this Extractor.
        """
        if not hasattr(self, '_total_number_of_combined_training_set_curves'):
            self._total_number_of_combined_training_set_curves =\
                np.prod(self.training_set_lengths)
        return self._total_number_of_combined_training_set_curves
    
    @property
    def expanders(self):
        """
        Property storing the Expander objects connecting the training set
        spaces to the data space.
        
        returns: list of values which are either None or 2D numpy.ndarrays
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders was referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Allows user to set expanders.
        
        value: list of length self.num_bases. Each element is either None (only
               allowed if length of training set corresponding to element is
               num_channels) or an Expander object
        """
        if value is None:
            value = [NullExpander()] * self.num_bases
        if type(value) in sequence_types:
            num_expanders = len(value)
            if num_expanders == self.num_bases:
                for ibasis in range(self.num_bases):
                    expander = value[ibasis]
                    if isinstance(expander, Expander):
                        ts_len = self.training_set_lengths[ibasis]
                        if expander.is_compatible(ts_len, self.num_channels):
                            continue
                        else:
                            raise ValueError("At least one expander was " +\
                                             "not compatible with the " +\
                                             "given training set length " +\
                                             "and number of channels.")
                    else:
                        raise TypeError("Not all expanders are Expander " +\
                                        "objects.")
                self._expanders = value
            else:
                raise ValueError(("The number of expanders ({0}) given was " +\
                    "not equal to the number of names and training sets " +\
                    "({1}).").format(num_expanders, self.num_bases))
        else:
            raise TypeError("expanders was set to a non-sequence.")
    
    @property
    def basis_sum(self):
        """
        Property storing the Basis objects associated with all training sets.
        """
        if not hasattr(self, '_basis_sum'):
            bases = []
            for ibasis in range(self.num_bases):
                training_set = self.training_sets[ibasis]
                num_basis_vectors = self.nterms_maxima[ibasis]
                expander = self.expanders[ibasis]
                basis = TrainedBasis(training_set, num_basis_vectors,\
                    error=self.error, expander=expander)
                bases.append(basis)
            self._basis_sum = BasisSum(self.names, bases)
        return self._basis_sum
    
    @property
    def loglikelihood(self):
        """
        Property storing the TruncationLoglikelihood which will be explored to
        determine the number of terms to use of each basis.
        """
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = LinearTruncationLoglikelihood(\
                self.basis_sum, self.data, self.error,\
                information_criterion=self.information_criterion)
        return self._loglikelihood
    
    @property
    def file_name(self):
        """
        Property storing the location of the file at which to save the Sampler.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name was referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the location of the file at which to save the Sampler.
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name given was not a string.")
    
    @property
    def optimal_truncations(self):
        """
        Property storing the sequence of what have been determined to be
        optimal truncations.
        """
        if not hasattr(self, '_optimal_truncations'):
            if not os.path.exists(self.file_name):
                parameter_names =\
                    ['{!s}_nterms'.format(name) for name in self.names]
                jumping_distribution_set = JumpingDistributionSet()
                jumping_probability = 0.9
                jumping_distribution =\
                    GridHopJumpingDistribution(ndim=self.num_bases,\
                    jumping_probability=jumping_probability,\
                    minima=np.ones_like(self.nterms_maxima),\
                    maxima=self.nterms_maxima)
                jumping_distribution_set.add_distribution(\
                    jumping_distribution, parameter_names)
                guess_distribution_set = DistributionSet()
                for (name, nterms_maximum) in\
                    zip(self.names, self.nterms_maxima):
                    if self.trust_ranks:
                        guess_distribution = KroneckerDeltaDistribution(\
                            self.training_set_ranks[name], is_discrete=True)
                    else:
                        guess_distribution =\
                            DiscreteUniformDistribution(1, nterms_maximum)
                    guess_distribution_set.add_distribution(\
                        guess_distribution, '{!s}_nterms'.format(name))
                prior_distribution_set = DistributionSet()
                for (parameter_name, nterms_maximum) in\
                    zip(parameter_names, self.nterms_maxima):
                    prior_distribution =\
                        DiscreteUniformDistribution(1, nterms_maximum)
                    prior_distribution_set.add_distribution(\
                        prior_distribution, parameter_name)
                nwalkers = 100
                steps_per_checkpoint = 100
                num_checkpoints = 10
                sampler = Sampler(self.file_name, nwalkers,\
                    self.loglikelihood, verbose=self.verbose,\
                    jumping_distribution_set=jumping_distribution_set,\
                    guess_distribution_set=guess_distribution_set,\
                    prior_distribution_set=prior_distribution_set,\
                    steps_per_checkpoint=steps_per_checkpoint)
                sampler.run_checkpoints(num_checkpoints)
                sampler.close()
            burn_rule = BurnRule(min_checkpoints=1, desired_fraction=1)
            analyzer = NLFitter(self.file_name, burn_rule=burn_rule,\
                load_all_chunks=True)
            self._optimal_truncations =\
                analyzer.maximum_probability_parameters.astype(int)
        return self._optimal_truncations
    
    @property
    def truncated_basis_sum(self):
        """
        Property storing the basis sum with the "optimal" truncations.
        """
        if not hasattr(self, '_truncated_basis_sum'):
            self._truncated_basis_sum =\
                self.loglikelihood.truncated_basis_sum(\
                self.optimal_truncations)
        return self._truncated_basis_sum
    
    @property
    def optimal_fitter(self):
        """
        Property storing the Fitter object which minimizes the given
        information criterion.
        """
        if not hasattr(self, '_optimal_fitter'):
            self._optimal_fitter =\
                Fitter(self.truncated_basis_sum, self.data, error=self.error)
        return self._optimal_fitter
    
    @property
    def expander_set(self):
        """
        Property yielding an ExpanderSet object organizing the expanders here
        so that "true" curves for (e.g.) systematics can be found by using the
        data as well as a "true" curve for the signal.
        """
        if not hasattr(self, '_expander_set'):
            self._expander_set = ExpanderSet(self.data, self.error,\
                **{self.names[iname]: self.expanders[iname]\
                for iname in range(self.num_bases)})
        return self._expander_set
    
    def fill_hdf5_group(self, group, save_training_sets=False,\
        save_channel_estimates=False):
        """
        Fills the given hdf5 file group with data about this extraction,
        including the optimal fitter.
        
        group: the hdf5 file group into which to save data
        save_training_sets: bool determining whether to save training sets
        save_channel_estimates: save_channel_estimates argument to pass on to
                                Fitter.fill_hdf5_group
        """
        data_link = create_hdf5_dataset(group, 'data', data=self.data)
        error_link = create_hdf5_dataset(group, 'error', data=self.error)
        subgroup = group.create_group('names')
        for (iname, name) in enumerate(self.names):
            subgroup.attrs['{:d}'.format(iname)] = name
        subgroup = group.create_group('expanders')
        self.expander_set.fill_hdf5_group(subgroup)
        expander_links = [subgroup[name] for name in self.names]
        self.optimal_fitter.fill_hdf5_group(\
            group.create_group('optimal_fitter'), data_link=data_link,\
            error_link=error_link, expander_links=expander_links,\
            save_channel_estimates=save_channel_estimates)
        subgroup = group.create_group('ranks')
        for name in self.names:
            subgroup.attrs[name] = self.training_set_ranks[name]
        if save_training_sets:
            subgroup = group.create_group('training_sets')
            for ibasis in range(self.num_bases):
                create_hdf5_dataset(subgroup, self.names[ibasis],\
                    data=self.training_sets[ibasis])
    

