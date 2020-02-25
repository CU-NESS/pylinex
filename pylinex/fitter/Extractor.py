"""
File: pylinex/fitter/Extractor.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: Class which uses the rest of the module to perform an end-to-end
             extraction. The inputs of the class are data and error vectors,
             training set matrices and expanders.
"""
import numpy as np
from ..util import Savable, VariableGrid, create_hdf5_dataset, sequence_types,\
    int_types, bool_types, real_numerical_types
from ..quantity import QuantityFinder, FunctionQuantity, CompiledQuantity
from ..expander import Expander, NullExpander, ExpanderSet
from ..basis import TrainedBasis, BasisSum, effective_training_set_rank
from .MetaFitter import MetaFitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class Extractor(Savable, VariableGrid, QuantityFinder):
    """
    Class which, given:
    
    1) 1D data array
    2) 1D error array
    3) names of different components of data
    4) training set matrices for each of component of the data
    5) (optional) Expander objects which transform curves from the space in
       which the training set is defined to the space in which the data is
       defined
    6) dimensions over which the Extractor should analytically explore
    7) quantities of which to calculate grids
    8) the name of the quantity to minimize
    
    
    extracts components of the data.
    """
    def __init__(self, data, error, names, training_sets, dimensions,\
        compiled_quantity=CompiledQuantity('empty'),\
        quantity_to_minimize='bias_score', expanders=None,\
        num_curves_to_score=None, use_priors_in_fit=False,\
        prior_covariance_expansion_factor=1., prior_covariance_diagonal=False,\
        verbose=True):
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
        dimensions: the dimensions of the grid in which to search for the
                    chosen solution. Should be a list of dictionaries of arrays
                    where each element of the list is a dimension and the
                    arrays in each dictionary must be of equal length
        compiled_quantity: Quantity or CompiledQuantity to find at each point
                           in the grid described by dimensions
        quantity_to_minimize: the name of the Quantity object in the
                              CompiledQuantity to minimize to perform model
                              selection
        expanders: list of Expander objects which expand each of the basis sets
        num_curves_to_score: the approximate number of combined training set
                             curves to use when computing the bias_score
                             quantity
        use_priors_in_fit: if True, priors derived from training set will be
                                    used in fits
                           if False, no priors are used in fits
        prior_covariance_expansion_factor: factor by which prior covariance
                                           matrices should be expanded (if they
                                           are used), default 1
        prior_covariance_diagonal: boolean determining whether off-diagonal
                                   elements of the prior covariance are used or
                                   not, default False (meaning they are used).
                                   Setting this to true will weaken priors but
                                   should enhance numerical stability
        verbose: if True, messages should be printed to the screen
        """
        self.data = data
        self.error = error
        self.names = names
        self.num_curves_to_score = num_curves_to_score
        self.training_sets = training_sets
        self.expanders = expanders
        self.dimensions = dimensions
        if ('bias_score' in compiled_quantity) or\
            ((type(self.num_curves_to_score) is not type(None)) and\
            (self.num_curves_to_score == 0)):
            self.compiled_quantity = compiled_quantity
        else:
            self.compiled_quantity =\
                compiled_quantity + self.bias_score_quantity
        self.quantity_to_minimize = quantity_to_minimize
        self.prior_covariance_expansion_factor =\
            prior_covariance_expansion_factor
        self.prior_covariance_diagonal = prior_covariance_diagonal
        self.use_priors_in_fit = use_priors_in_fit
        self.verbose = verbose
    
    @property
    def use_priors_in_fit(self):
        """
        Property storing bool desribing whether priors will be used in fits.
        """
        if not hasattr(self, '_use_priors_in_fit'):
            raise AttributeError("use_priors_in_fit referenced before it " +\
                "was set.")
        return self._use_priors_in_fit
    
    @use_priors_in_fit.setter
    def use_priors_in_fit(self, value):
        """
        Setter for whether priors will be used in fits.
        
        value: if True, priors derived from training set will be used in fits
               if False, no priors are used in fits
        """
        if type(value) in bool_types:
            self._use_priors_in_fit = {name: value for name in self.names}
        elif isinstance(value, dict):
            if set([key for key in value.keys()]) <= set(self.names):
                if all([(type(value[key]) in bool_types) for key in value]):
                    self._use_priors_in_fit = value
                    for name in\
                        (set(self.names) - set([key for key in value.keys()])):
                        self._use_priors_in_fit[name] = False
                        print(("{!s} wasn't included in use_priors_in_fit " +\
                            "dict, so no priors will be used for it.").format(\
                            name))
                else:
                    raise TypeError("Not all values of the given " +\
                        "dictionary are bools.")
            else:
                raise ValueError("Not all names in the given dictionary " +\
                    "are names of this Extractor.")
        else:
            raise TypeError("If use_priors_in_fit is set to a non-bool, " +\
                "it should be set to a dictionary whose keys are the " +\
                "names of this Extractor and whose values are bools.")
        
    
    @property
    def num_curves_to_score(self):
        """
        Property storing the approximate number of curves for which to
        calculate the bias_score.
        """
        if not hasattr(self, '_num_curves_to_score'):
            raise AttributeError("num_curves_to_score referenced before it " +\
                "was set.")
        return self._num_curves_to_score
    
    @num_curves_to_score.setter
    def num_curves_to_score(self, value):
        """
        Setter for the maximum number of curves to score.
        
        value: if None, all combinations of training set curves are scored
               if value == 0, bias_score quantity is not calculated at all
               otherwise, value must be an integer number of curves to score
                          (+-1 block). If integer is greater than number of
                          curves, all curves are returned as if None was passed
                          as value
        """
        if type(value) is type(None):
            self._num_curves_to_score = None
        elif type(value) in int_types:
            if value >= 0:
                self._num_curves_to_score = value
            else:
                raise ValueError("Cannot score non-positive number of curves.")
        else:
            raise TypeError("num_curves_to_score was neither None nor of " +\
                "an integer type.")
    
    @property
    def bias_score_quantity(self):
        """
        Property storing the quantity which will calculate the training set
        based bias score.
        """
        if not hasattr(self, '_bias_score_quantity'):
            self._bias_score_quantity =\
                FunctionQuantity('bias_score', self.training_sets,\
                num_curves_to_score=self.num_curves_to_score)
        return self._bias_score_quantity
    
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
    def quantity_to_minimize(self):
        """
        Property storing string name of quantity to minimize.
        """
        if not hasattr(self, '_quantity_to_minimize'):
            raise AttributeError("quantity_to_minimize was referenced " +\
                                 "before it was set.")
        return self._quantity_to_minimize
    
    @quantity_to_minimize.setter
    def quantity_to_minimize(self, value):
        """
        Allows user to supply string name of the quantity to minimize.
        """
        if isinstance(value, basestring):
            if value in self.compiled_quantity:
                self._quantity_to_minimize = value
            else:
                raise ValueError("quantity_to_minimize was not in " +\
                                 "compiled_quantity.")
        else:
            raise TypeError("quantity_to_minimize was not a string.")
        

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
                        if (type(self.num_curves_to_score) is type(None)) or\
                            (self.num_curves_to_score == 0):
                            self._training_sets = [ts for ts in value]
                        else:
                            lengths = [len(ts) for ts in value]
                            perms =\
                                [np.random.permutation(l) for l in lengths]
                            self._training_sets = [value[its][perms[its]]\
                                for its in range(len(value))]
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
    def training_set_rank_indices(self):
        """
        Property storing a dictionary with tuples containing the dimension and
        rank index as values indexed by the associated subbasis name.
        """
        if not hasattr(self, '_training_set_rank_indices'):
            self._training_set_rank_indices = {}
            for name in self.names:
                rank = self.training_set_ranks[name]
                for (idimension, dimension) in self.dimensions:
                    if name in dimension:
                        if rank in dimension[name]:
                            rank_index =\
                                np.where(dimension[name] == rank)[0][0]
                        else:
                            print(("rank of {0!s} (1:d) not in its grid " +\
                                "dimension. Are you sure you're using " +\
                                "enough terms?").format(name, rank))
                            rank_index = None
                        self._training_set_rank_indices[name] =\
                            (idimension, rank_index)
                        break
                    else:
                        pass
        return self._training_set_rank_indices
    
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
        if type(value) is type(None):
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
                num_basis_vectors = self.maxima[self.names[ibasis]]
                expander = self.expanders[ibasis]
                basis = TrainedBasis(training_set, num_basis_vectors,\
                    error=self.error, expander=expander)
                bases.append(basis)
            self._basis_sum = BasisSum(self.names, bases)
        return self._basis_sum
    
    @property
    def prior_covariance_expansion_factor(self):
        """
        Property storing the factor by which the prior covariance matrix should
        be expanded (default 1).
        """
        if not hasattr(self, '_prior_covariance_expansion_factor'):
            raise AttributeError("prior_covariance_expansion_factor was " +\
                "referenced before it was set.")
        return self._prior_covariance_expansion_factor
    
    @prior_covariance_expansion_factor.setter
    def prior_covariance_expansion_factor(self, value):
        """
        Setter for the expansion factor of the prior covariance matrix
        
        value: positive number (usually greater than 1)
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._prior_covariance_expansion_factor = value
            else:
                raise ValueError("prior_covariance_expansion_factor was " +\
                    "set to a non-positive number.")
        else:
            raise TypeError("prior_covariance_expansion_factor was set to " +\
                "a non-number.")
    
    @property
    def prior_covariance_diagonal(self):
        """
        Property storing whether the prior covariance matrix should be taken to
        be diagonal or not (default False).
        """
        if not hasattr(self, '_prior_covariance_diagonal'):
            raise AttributeError("prior_covariance_diagonal was referenced " +\
                "before it was set.")
        return self._prior_covariance_diagonal
    
    @prior_covariance_diagonal.setter
    def prior_covariance_diagonal(self, value):
        """
        Setter for whether prior covariance used should be diagonal or not.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._prior_covariance_diagonal = value
        else:
            raise TypeError("prior_covariance_diagonal was set to a non-bool.")
    
    @property
    def priors(self):
        """
        Property storing a dictionary whose keys are names of bases with
        '_prior' appended and whose values are GaussianDistribution objects.
        """
        if not hasattr(self, '_priors'):
            self._priors = {}
            for name in self.names:
                if self.use_priors_in_fit[name]:
                    self.basis_sum[name].generate_gaussian_prior(\
                    covariance_expansion_factor=\
                    self.prior_covariance_expansion_factor,\
                    diagonal=self.prior_covariance_diagonal)
                    self._priors['{!s}_prior'.format(name)] =\
                        self.basis_sum[name].gaussian_prior
        return self._priors

    @property
    def meta_fitter(self):
        """
        Property storing the MetaFitter object doing the searching to find the
        right number of parameters.
        """
        if not hasattr(self, '_meta_fitter'):
            self._meta_fitter = MetaFitter(self.basis_sum, self.data,\
                self.error, self.compiled_quantity, self.quantity_to_minimize,\
                *self.dimensions, **self.priors)
        return self._meta_fitter
    
    @property
    def fitter(self):
        """
        Property storing the Fitter object which minimizes the Quantity object
        named quantity_to_minimize.
        """
        if not hasattr(self, '_fitter'):
            if self.data.ndim == 1:
                indices = self.meta_fitter.minimize_quantity(\
                    self.quantity_to_minimize)
                self._fitter = self.meta_fitter.fitter_from_indices(indices)
                if self.verbose:
                    print("Chose the following numbers of parameters: " +\
                        "{!s}".format(self._fitter.sizes))
            elif self.data.ndim > 1:
                self._fitter = np.ndarray(self.data.shape[:-1], dtype=object)
                for data_indices in np.ndindex(*self.data.shape[:-1]):
                    indices = self.meta_fitter.minimize_quantity(\
                        self.quantity_to_minimize, data_indices)
                    self._fitter[data_indices] =\
                        self.meta_fitter.fitter_from_indices(indices)
                    if self.verbose:
                        print("Chose the following numbers of parameters: " +\
                            "{!s}".format(self._fitter[data_indices]))
        return self._fitter
    
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
    
    def fill_hdf5_group(self, group, save_all_fitters=False,\
        save_training_sets=False, save_channel_estimates=False):
        """
        Fills the given hdf5 file group with data about this extraction,
        including the optimal fitter and the grids of statistics which were
        calculated.
        
        group: the hdf5 file group into which to save data
        save_all_fitters: bool determining whether to save all fitters in grid
        save_training_sets: bool determining whether to save training sets
        """
        data_link = create_hdf5_dataset(group, 'data', data=self.data)
        error_link = create_hdf5_dataset(group, 'error', data=self.error)
        subgroup = group.create_group('names')
        for name in self.names:
            subgroup.attrs[name] = name
        subgroup = group.create_group('expanders')
        self.expander_set.fill_hdf5_group(subgroup)
        expander_links = [subgroup[name] for name in self.names]
        self.meta_fitter.fill_hdf5_group(group.create_group('meta_fitter'),\
            save_all_fitters=save_all_fitters, data_link=data_link,\
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
    

