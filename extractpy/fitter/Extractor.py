"""
File: extractpy/fitter/Extractor.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: Class which uses the rest of the module to perform an end-to-end
             extraction. The inputs of the class are data and error vectors,
             training set matrices and expanders.
"""
import numpy as np
from ..util import Savable, VariableGrid, sequence_types, bool_types
from ..quantity import QuantityFinder
from ..expander import Expander, NullExpander
from ..basis import TrainedBasis, BasisSet
from .MetaFitter import MetaFitter

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
        compiled_quantity, quantity_to_minimize, expanders=None,\
        save_training_sets=False, verbose=True):
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
        save_training_sets: if True, when Extractor is written to hdf5 group,
                                     the training sets are saved in it.
                            otherwise, training sets are excluded
        verbose: if True, messages should be printed to the screen
        """
        self.data = data
        self.error = error
        self.names = names
        self.training_sets = training_sets
        self.expanders = expanders
        self.dimensions = dimensions
        self.compiled_quantity = compiled_quantity
        self.quantity_to_minimize = quantity_to_minimize
        self.save_training_sets = save_training_sets
        self.verbose = verbose
    
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
    def save_training_sets(self):
        """
        Property storing the boolean switch determining whether or not training
        sets are saved when this Extractor is saved.
        """
        if not hasattr(self, '_save_training_sets'):
            raise AttributeError("save_training_sets was referenced before " +\
                                 "it was set.")
        return self._save_training_sets
    
    @save_training_sets.setter
    def save_training_sets(self, value):
        """
        Sets the save_training_sets boolean switch.
        
        value: must be a bool
        """
        if type(value) in bool_types:
            self._save_training_sets = value
        else:
            raise TypeError("save_training_sets was set to a non-bool.")
    
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
        if isinstance(value, str):
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
        if value.ndim == 1:
            self._data = value
        else:
            raise ValueError("data must be 1D.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the data. A positive integer
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.data)
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
                             "didn't have the expected shape (i.e. data " +\
                             "shape).")
    
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
                    if all([(ts.ndim > 1) for ts in value]):
                        self._training_sets = value
                    else:
                        raise ValueError("At least one of the training " +\
                                         "sets given to Extractor was not 2D.")
                else:
                    raise TypeError("At least one of the given training " +\
                                    "sets given to Extractor was not a " +\
                                    "numpy.ndarray.")
            else:
                raise ValueError("The number of names given to Extractor " +\
                                 ("(%i) was not equal " % (self.num_bases,)) +\
                                 "to the number of training sets given " +\
                                 ("(%i)." % (num_training_sets,)))
        else:
            raise TypeError("training_sets of Extractor class was set to a " +\
                            "non-sequence.")
    
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
                for ibasis in xrange(self.num_bases):
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
                raise ValueError("The number of expanders " +\
                                 ("(%i) given was not " % (num_expanders,)) +\
                                 "equal to the number of names and " +\
                                 ("training sets (%i)." % (self.num_bases,)))
        else:
            raise TypeError("expanders was set to a non-sequence.")
    
    @property
    def basis_set(self):
        """
        Property storing the Basis objects associated with all training sets.
        """
        if not hasattr(self, '_basis_set'):
            bases = []
            for ibasis in xrange(self.num_bases):
                training_set = self.training_sets[ibasis]
                num_basis_vectors = self.maxima[self.names[ibasis]]
                expander = self.expanders[ibasis]
                basis = TrainedBasis(training_set, num_basis_vectors,\
                    error=self.error, expander=expander)
                bases.append(basis)
            self._basis_set = BasisSet(self.names, bases)
        return self._basis_set
    
    @property
    def priors(self):
        """
        Property storing a dictionary whose keys are names of bases with
        '_prior' appended and whose values are GaussianDistribution objects.
        """
        if not hasattr(self, '_priors'):
            self._priors = {}
            for name in self.names:
                self._priors[name + '_prior'] =\
                    self.basis_set[name].gaussian_prior
        return self._priors

    @property
    def meta_fitter(self):
        """
        Property storing the MetaFitter object doing the searching to find the
        right number of parameters.
        """
        if not hasattr(self, '_meta_fitter'):
            self._meta_fitter = MetaFitter(self.basis_set, self.data,\
                self.error, self.compiled_quantity, *self.dimensions,\
                **self.priors)
        return self._meta_fitter
    
    @property
    def fitter(self):
        """
        Property storing the Fitter object which minimizes the Quantity object
        named quantity_to_minimize.
        """
        if not hasattr(self, '_fitter'):
            self._fitter =\
                self.meta_fitter.minimize_quantity(self.quantity_to_minimize)
            if self.verbose:
                print "Chose the following numbers of parameters: %s" %\
                    (self._fitter.sizes,)
        return self._fitter
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this extraction,
        including the optimal fitter and the grids of statistics which were
        calculated.
        
        group: the hdf5 file group into which to save data
        """
        group.attrs['quantity_to_minimize'] = self.quantity_to_minimize
        self.fitter.fill_hdf5_group(group.create_group('optimal_fitter'))
        subgroup = group.create_group('dimensions')
        for (idimension, dimension) in enumerate(self.dimensions):
            subsubgroup = subgroup.create_group('dimension_%i' % (idimension,))
            for name in dimension:
                subsubgroup.create_dataset(name, data=dimension[name])
        subgroup = group.create_group('grids')
        for name in self.compiled_quantity.names:
            subgroup.create_dataset(name, data=self.meta_fitter[name])
        if self.save_training_sets:
            subgroup = group.create_group('training_sets')
            for ibasis in xrange(self.num_bases):
                subgroup.create_dataset(self.names[ibasis],\
                    data=self.training_sets[ibasis])
        
