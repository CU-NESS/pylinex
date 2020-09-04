"""
File: pylinex/nonlinear/LeastSquareFitCluster.py
Author: Keith Tauscher
Date: 7 May 2019

Description: File containing a class that runs many LeastSquareFitter objects
             with GaussianLoglikelihood objects which are identical except for
             data vectors that differ only by noise at the error level. The
             purpose of the class is to estimate the distribution of the
             parameters implied by the seed loglikelihood without trying to use
             local (derivative) information to estimate nonlocal properties, as
             is done in the Fisher matrix approach.
"""
import os, time, h5py
import numpy as np
from distpy import TransformList, GaussianDistribution,\
    DeterministicDistribution, KroneckerDeltaDistribution, DistributionSet
from ..util import int_types, bool_types, real_numerical_types, sequence_types
from ..loglikelihood import GaussianLoglikelihood
from .LeastSquareFitter import LeastSquareFitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class LeastSquareFitCluster(object):
    """
    Class that runs many LeastSquareFitter objects with GaussianLoglikelihood
    objects which are identical except for data vectors that differ only by
    noise at the error level. The purpose of the class is to estimate the
    distribution of the parameters implied by the seed loglikelihood without
    trying to use local (derivative) information to estimate nonlocal
    properties, as is done in the Fisher matrix approach.
    """
    def __init__(self, seed_loglikelihood, prior_set, prefix, num_fits,\
        transform_list=None, minimum_variances=0, save_all=False, **bounds):
        """
        Initializes a new LeastSquareFitCluster.
        
        seed_loglikelihood: GaussianLoglikelihood object containing the data,
                            error and model under concern. The model should be
                            given in transformed space (as defined by the given
                            TransformList)
        prior_set: must be a DistributionSet object whose parameters are the
                   same as the parameters of the Loglikelihood at the heart of
                   this fitter
        prefix: if None, no files are saved
                otherwise, string seeding file names where LeastSquareFitter
                           objects should be placed.
                           "{0!s}.{1:d}.hdf5".format(prefix, index) should be a
                           valid file location for index in range(num_fits) if
                           save_all is True. If save_all is False (default),
                           then the first fit (the one with no added noise) is
                           saved at "{!s}.hdf5".format(prefix)
        num_fits: integer number of LeastSquareFitter objects to create
        transform_list: TransformList (or something which can be cast to a
                        TransformList object) describing the space in which
                        the parameters of the loglikelihood exist
        minimum_variances: the minimum variances to include in covariance when
                           making an approximate Gaussian distribution
        save_all: if True, all LeastSquareFitter objects are saved if prefix is
                           not None.
                  if False (default), only the first fit is saved
        bounds: dictionary containing 2-tuples of (min, max) where min and max
                are either numbers or None indexed by parameter name.
                Parameters which are not in value are bounded by their own
                Model object's bounds property.
        """
        self.save_all = save_all
        self.seed_loglikelihood = seed_loglikelihood
        self.prior_set = prior_set
        self.prefix = prefix
        self.num_fits = num_fits
        self.minimum_variances = minimum_variances
        self.transform_list = transform_list
        self.bounds = bounds
    
    @property
    def minimum_variances(self):
        """
        Property storing the array of minimum variances to allow for the
        parameters (in the space of the loglikelihood).
        """
        if not hasattr(self, '_minimum_variances'):
            raise AttributeError("minimum_variances was referenced before " +\
                "it was set.")
        return self._minimum_variances
    
    @minimum_variances.setter
    def minimum_variances(self, value):
        """
        Setter for the array of minimum variances to allow for the parameters
        (in the space of the loglikelihood).
        
        value: either a single number or an array of length num_parameters
        """
        if type(value) in real_numerical_types:
            self._minimum_variances = value * np.ones(self.num_parameters)
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.num_parameters,):
                self._minimum_variances = value
            else:
                raise ValueError("minimum_variances was set to an array " +\
                    "that has an unexpected shape.")
        else:
            raise TypeError("minimum_variances was set to neither a number " +\
                "nor an array.")
    
    @staticmethod
    def load_from_first_file(file_name, num_fits, minimum_variances=0):
        """
        Loads a LeastSquareFitCluster from its first saved LeastSquareFitter.
        
        file_name: name of file where first LeastSquareFitter was saved
        num_fits: number of noise realizations to run
        minimum_variances: the minimum variances to include in covariance when
                           making an approximate Gaussian distribution
        
        returns: a LeastSquareFitCluster object 
        """
        prefix = file_name[:-len('.hdf5')]
        least_square_fitter = LeastSquareFitter(file_name=file_name)
        seed_loglikelihood = least_square_fitter.loglikelihood
        prior_set = least_square_fitter.prior_set
        transform_list = least_square_fitter.transform_list.inverse
        bounds = {name: least_square_fitter.bounds[iname]\
            for (iname, name) in enumerate(seed_loglikelihood.parameters)}
        return LeastSquareFitCluster(seed_loglikelihood, prior_set, prefix,\
            num_fits, transform_list=transform_list,\
            minimum_variances=minimum_variances, save_all=False, **bounds)
    
    @property
    def save_all(self):
        """
        Property storing a boolean determining whether all LeastSquareFitter
        objects created by this LeastSquareFitCluster should be saved or only
        the one with no added noise.
        """
        if not hasattr(self, '_save_all'):
            raise AttributeError("save_all was referenced before it was set.")
        return self._save_all
    
    @save_all.setter
    def save_all(self, value):
        """
        Setter for the save_all boolean property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._save_all = value
        else:
            raise TypeError("save_all was set to a non-bool.")
    
    @property
    def seed_loglikelihood(self):
        """
        A seed loglikelihood containing the "real" data. It will be used to
        create different likelihoods with different data vectors.
        """
        if not hasattr(self, '_seed_loglikelihood'):
            raise AttributeError("seed_loglikelihood was referenced before " +\
                "it was set.")
        return self._seed_loglikelihood
    
    @seed_loglikelihood.setter
    def seed_loglikelihood(self, value):
        """
        Sets the seed_loglikelihood.
        
        value: GaussianLoglikelihood object containing the data, error and
               model under concern.
        """
        if isinstance(value, GaussianLoglikelihood):
            self._seed_loglikelihood = value
        else:
            raise TypeError("The LeastSquareFitCluster class ")
    
    @property
    def parameters(self):
        """
        Property storing the list of parameter strings.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = self.seed_loglikelihood.parameters
        return self._parameters
    
    @property
    def num_parameters(self):
        """
        Property storing the number of parameters being solved for.
        """
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = len(self.parameters)
        return self._num_parameters
    
    @property
    def prior_set(self):
        """
        Property storing a DistributionSet object which allows for reasonable
        first guesses at the parameters' values to be drawn.
        """
        if not hasattr(self, '_prior_set'):
            raise AttributeError("prior_set referenced before it was set.")
        return self._prior_set
    
    @prior_set.setter
    def prior_set(self, value):
        """
        Setter for the prior_set distribution defining how to draw random first
        guesses at the parameters' values.
        
        value: must be a DistributionSet object whose parameters are the same
               as the parameters of the Loglikelihood at the heart of this
               fitter
        """
        if isinstance(value, DistributionSet):
            if set(value.params) == set(self.parameters):
                self._prior_set = value
            else:
                raise ValueError("The given prior_set described some " +\
                    "parameters which aren't needed or didn't describe " +\
                    "some parameters which were.")
        else:
            raise TypeError("prior_set was set to something other than a " +\
                "DistributionSet object.")
    
    @property
    def transform_list(self):
        """
        Property storing a TransformList object storing Transforms which define
        the space in which covariance estimates are returned.
        """
        if not hasattr(self, '_transform_list'):
            raise AttributeError("transform_list referenced before it was " +\
                "set.")
        return self._transform_list
    
    @transform_list.setter
    def transform_list(self, value):
        """
        Setter for the TransformList which defines the space in which
        covariance estimates are returned.
        
        value: must be a TransformList or something castable to a TransformList
               with a length given by the number of parameters in the vector to
               be optimized.
        """
        self._transform_list =\
            TransformList.cast(value, num_transforms=self.num_parameters)
    
    @property
    def prefix(self):
        """
        Property storing the prefix of the file names where the
        LeastSquareFitter objects used by this LeastSquareFitCluster should be
        placed.
        """
        if not hasattr(self, '_prefix'):
            raise AttributeError("prefix was referenced before it was set.")
        return self._prefix
    
    @prefix.setter
    def prefix(self, value):
        """
        Setter for the prefix of files where the LeastSquareFitter objects used
        by this LeastSquareFitCluster should be placed.
        
        value: if None, no fitters are saved
               otherwise, string seeding file names where LeastSquareFitter
                          objects should be placed.
                          "{0!s}.{1:d}.hdf5".format(prefix, index) should
                          be a valid file location for index in range(num_fits)
        """
        if (type(value) is type(None)) or isinstance(value, basestring):
            self._prefix = value
        else:
            raise TypeError("prefix was set to a non-string.")
    
    @property
    def num_fits(self):
        """
        Property storing the integer number of LeastSquareFitter objects to
        create.
        """
        if not hasattr(self, '_num_fits'):
            raise AttributeError("num_fits was referenced before it was set.")
        return self._num_fits
    
    @num_fits.setter
    def num_fits(self, value):
        """
        Setter for the number of LeastSquareFitter objects to create.
        
        value: positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._num_fits = value
            else:
                raise ValueError("num_fits was set to a non-positive integer.")
        else:
            raise TypeError("num_fits was set to a non-int.")
    
    @property
    def file_names(self):
        """
        Property storing the names of files which should be used as storage for
        the LeastSquareFitter objects this LeastSquareFitCluster creates.
        """
        if not hasattr(self, '_file_names'):
            if type(self.prefix) is type(None):
                self._file_names = [None] * self.num_fits
            elif self.save_all:
                zfill_width = int(np.ceil(np.log10(self.num_fits)))
                self._file_names = ['{0!s}.{1!s}.hdf5'.format(self.prefix,\
                    "{:d}".format(fit_index).zfill(zfill_width))\
                    for fit_index in range(self.num_fits)]
            else:
                file_name = '{!s}.hdf5'.format(self.prefix)
                self._file_names = [file_name] + ([None] * (self.num_fits - 1))
        return self._file_names
    
    @property
    def summary_file_name(self):
        """
        Property storing the name of the summary file.
        """
        if not hasattr(self, '_summary_file_name'):
            self._summary_file_name = '{!s}.summary.hdf5'.format(self.prefix)
        return self._summary_file_name
    
    @property
    def bounds(self):
        """
        Property storing the sequence of bounds for each parameter.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds was referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for the bounds property.
        
        value: a dictionary containing 2-tuples of (min, max) where min and max
               are either numbers or None indexed by parameter name. Parameters
               which are not in value are bounded by their own Model's bounds
               property.
        """
        if isinstance(value, dict):
            for name in value:
                if name not in self.parameters:
                    raise ValueError(("There was at least one key " +\
                        "({!s}) of the given bounds dictionary which " +\
                        "was not one of this fitter's parameters.").format(\
                        name))
            self._bounds = {}
            for name in self.parameters:
                if name in value:
                    self._bounds[name] = value[name]
                else:
                    self._bounds[name] =\
                        self.seed_loglikelihood.model.bounds[name]
        else:
            raise TypeError("bounds was set to a non-dictionary.")
    
    @property
    def noise_distribution(self):
        """
        Property storing the noise_distribution if error is 2D (otherwise the
        noise is directly sampled using numpy.random.normal). If error is 1D,
        this property should not be referenced as it is not needed.
        """
        if not hasattr(self, '_noise_distribution'):
            error = self.seed_loglikelihood.error
            if error.ndim == 2:
                self._noise_distribution = GaussianDistribution(\
                    np.zeros_like(self.seed_loglikelihood.data), error)
            else:
                raise RuntimeError("The noise_distribution property should " +\
                    "never be referenced if the error is 1D (i.e. if noise " +\
                    "is uncorrelated.")
        return self._noise_distribution
    
    def generate_loglikelihood(self, index):
        """
        Generates a new GaussianLoglikelihood with noise added in. If index is
        0, then the seed loglikelihood is returned.
        """
        if index == 0:
            return self.seed_loglikelihood
        else:
            data = self.seed_loglikelihood.data
            error = self.seed_loglikelihood.error
            if error.ndim == 1:
                noise_realization =\
                    np.random.normal(0, 1, size=error.shape) * error
            else:
                noise_realization = self.noise_distribution.draw()
            modified_data = self.seed_loglikelihood.data + noise_realization
            return self.seed_loglikelihood.change_data(modified_data)
    
    def run(self, iterations=1, attempt_threshold=100,\
        cutoff_loglikelihood=np.inf, verbose=False, doubly_verbose=False,\
        tolerance=None, **kwargs):
        """
        Creates the num_fits LeastSquareFitter objects which will be used to
        estimate the distribution of parameters implied by the seed
        loglikelihood.
        
        iterations: must be a positive integer
        attempt_threshold: number of times an iteration should recur before
                           failing
        cutoff_loglikelihood: if an iteration of this LeastSquareFitter
                              achieves a loglikelihood above this value, the
                              LeastSquareFitter is stopped early
                              default value is np.inf
        verbose: if True, a message is printed at the creation of each
                          LeastSquareFitter object.
        doubly_verbose: if True, verbose is passed on to each individual
                                 LeastSquareFitter
        tolerance: if not None (default None) and verbose is True, a message is
                   printed after each of the LeastSquareFitter objects that
                   used artificially modified data vectors specifying whether
                   the fit found a minimum that was within this tolerance (can
                   be array-like for multi-parameter fits)
        kwargs: Keyword arguments to pass on as options to
                scipy.optimize.minimize(method='SLSQP'). They can include:
                    ftol : float, precision goal for the loglikelihood in the
                           stopping criterion.
                    eps : float, step size used for numerical approximation of
                          the gradient.
                    disp : bool, set to True to print convergence messages.
                    maxiter : int, maximum number of iterations.
                run_if_iterations_exist: (default: True) determines if
                                         iterations should be run on the main
                                         LeastSquareFitter if some have already
                                         been run.
        """
        prior_set = self.prior_set
        if os.path.exists(self.summary_file_name):
            with h5py.File(self.summary_file_name, 'r') as hdf5_file:
                self._data_realizations = hdf5_file['data_realizations'][()]
                self._argmins = hdf5_file['argmins'][()]
                self._successes = hdf5_file['successes'][()]
        else:
            self._data_realizations = []
            self._argmins = []
            self._successes = []
            for (fit_index, file_name) in enumerate(self.file_names):
                while True:
                    loglikelihood = self.generate_loglikelihood(fit_index)
                    least_square_fitter = LeastSquareFitter(\
                        loglikelihood=loglikelihood, prior_set=prior_set,\
                        transform_list=self.transform_list.inverse,\
                        file_name=file_name, **self.bounds)
                    least_square_fitter.run(iterations=iterations,\
                        attempt_threshold=attempt_threshold,\
                        cutoff_loglikelihood=cutoff_loglikelihood,\
                        verbose=doubly_verbose, **kwargs)
                    is_successful = bool(np.max(least_square_fitter.successes))
                    self._data_realizations.append(loglikelihood.data)
                    self._argmins.append(least_square_fitter.argmin)
                    self._successes.append(is_successful)
                    if verbose:
                        print(("Finished least square fitter #{0:d} of " +\
                            "LeastSquareFitCluster at {1!s}, and it was " +\
                            "{2!s}successful.").format(1 + fit_index,\
                            time.ctime(), "" if is_successful else "not "))
                    if fit_index == 0:
                        prior_distribution = KroneckerDeltaDistribution(\
                            least_square_fitter.argmin)
                        prior_distribution_tuple =\
                            (prior_distribution, self.parameters, None)
                        prior_set = DistributionSet([prior_distribution_tuple])
                        iterations = 1
                        break
                    elif type(tolerance) is type(None):
                        break
                    else:
                        within_tolerance = np.any(np.abs(\
                            self._argmins[-1] - self._argmins[0]) < tolerance)
                        if verbose:
                            print(("Supplemental fit #{0:d} was{1!s} " +\
                                "within tolerance of {2}.").format(fit_index,\
                                "" if within_tolerance else " not", tolerance))
                        if within_tolerance:
                            if type(file_name) is not type(None):
                                os.remove(file_name)
                            continue
                        else:
                            break
            self._data_realizations = np.array(self._data_realizations)
            self._argmins = np.array(self._argmins)
            self._successes = np.array(self._successes)
            with h5py.File(self.summary_file_name, 'w') as hdf5_file:
                hdf5_file.create_dataset('data_realizations',\
                    data=self.data_realizations)
                hdf5_file.create_dataset('argmins', data=self.argmins)
                hdf5_file.create_dataset('successes', data=self.successes)
    
    @property
    def data_realizations(self):
        """
        Property storing the data realizations used by this cluster.
        """
        if not hasattr(self, '_data_realizations'):
            raise AttributeError("data_realizations was referenced before " +\
                "run was called.")
        return self._data_realizations
    
    @property
    def argmins(self):
        """
        Property storing the minima found by each of the LeastSquareFitter
        objects.
        """
        if not hasattr(self, '_argmins'):
            raise AttributeError("argmins was referenced before run was " +\
                "called.")
        return self._argmins
    
    @property
    def successes(self):
        """
        Property storing a boolean array describing whether each of the
        LeastSquareFitter objects created by this LeastSquareFitCluster
        successfully found a minimum or not (according to the tolerances used
        by default in scipy.optimize.minimize or as overridden by the kwargs
        first passed to the run method).
        """
        if not hasattr(self, '_successes'):
            raise AttributeError("successes was referenced before run was " +\
                "called.")
        return self._successes
    
    @property
    def approximate_gaussian_distribution(self):
        """
        Property storing the approximate GaussianDistribution describing the
        parameter distribution (in the space of the seed_loglikelihood's
        parameters).
        """
        if not hasattr(self, '_approximate_gaussian_distribution'):
            mean = np.mean(self.argmins, axis=0)
            if len(mean) == 1:
                covariance = np.var(self.argmins[:,0])
            else:
                covariance = np.cov(self.argmins, rowvar=False)
            covariance = covariance + np.diag(self.minimum_variances)
            self._approximate_gaussian_distribution =\
                GaussianDistribution(mean, covariance)
        return self._approximate_gaussian_distribution
    
    @property
    def approximate_gaussian_distribution_set(self):
        """
        Property storing the approximate Gaussian parameter distribution in a
        DistributionSet object, using the TransformList given at the
        initialization of this LeastSquareFitCluster object to define the
        Gaussian distribution in transformed space.
        """
        if not hasattr(self, '_approximate_gaussian_distribution_set'):
            self._approximate_gaussian_distribution_set =\
                DistributionSet([(self.approximate_gaussian_distribution,\
                self.parameters, self.transform_list)])
        return self._approximate_gaussian_distribution_set
    
    @property
    def sampled_distribution(self):
        """
        Property storing a DeterministicDistribution containing a sample of
        num_fits points found by the LeastSquareFitter objects created by this
        LeastSquareFitCluster.
        """
        if not hasattr(self, '_sampled_distribution'):
            self._sampled_distribution =\
                DeterministicDistribution(self.argmins)
        return self._sampled_distribution
    
    @property
    def sampled_distribution_set(self):
        """
        Property storing the parameter distribution in a DistributionSet
        object, using the TransformList given at the initialization of this
        LeastSquareFitCluster object to define the DeterministicDistribution
        sampled_distribution property in transformed space.
        """
        if not hasattr(self, '_sampled_distribution_set'):
            self._sampled_distribution_set =\
                DistributionSet([(self.sampled_distribution, self.parameters,\
                self.transform_list)])
        return self._sampled_distribution_set
    
    def triangle_plot(self, parameters=None, in_transformed_space=True,\
        figsize=(8, 8), fig=None, show=False, kwargs_1D={}, kwargs_2D={},\
        fontsize=28, nbins=100, plot_type='contour', plot_limits=None,\
        plot_reference_gaussian=True, contour_confidence_levels=0.95,\
        parameter_renamer=(lambda x: x), tick_label_format_string='{x:.3g}'):
        """
        Makes a triangle plot of the results of the LeastSquareFitter objects
        at the core of this LeastSquareFitCluster.
        
        parameters: sequence of string parameter names to include in the plot
        in_transformed_space: if True (default), parameters are plotted in
                                                 transformed space
        figsize: the size of the figure on which to put the triangle plot
        show: if True, matplotlib.pyplot.show is called before this function
                       returns
        kwargs_1D: keyword arguments to pass on to univariate_histogram
                   function
        kwargs_2D: keyword arguments to pass on to bivariate_histogram function
        fontsize: the size of the label fonts
        nbins: the number of bins for each sample
        plot_type: 'contourf', 'contour', or 'histogram'
        plot_limits: if not None, a dictionary whose keys are parameter names
                                  and whose values are 2-tuples of the form
                                  (low, high) representing the desired axis
                                  limits for each variable in untransformed
                                  space
                     if None (default), bins are used to decide plot limits
        plot_reference_gaussian: if True (default), a reference Gaussian is
                                                    plotted which was
                                                    approximated from the
                                                    sample which is used to
                                                    make this triangle plot
                                                    (only functions if
                                                    in_transformed_space is
                                                    True, where the Gaussian
                                                    approximation is made)
        contour_confidence_levels: the confidence level of the contour in the
                                   bivariate histograms. Only used if plot_type
                                   is 'contour' or 'contourf'. Can be single
                                   number or sequence of numbers
        tick_label_format_string: format string that can be called using
                                  tick_label_format_string.format(x=loc) where
                                  loc is the location of the tick in data
                                  coordinates
        
        returns: if show, nothing is returned as figure is plotted
                 otherwise, matplotlib.Figure is returned
        """
        if plot_reference_gaussian and in_transformed_space:
            reference_value_mean =\
                self.approximate_gaussian_distribution.internal_mean.A[0]
            reference_value_covariance =\
                self.approximate_gaussian_distribution.covariance.A
        else:
            if plot_reference_gaussian:
                print("plot_reference_gaussian was True but, this can only " +\
                    "be done if in_transformed_space is True, but it was " +\
                    "set to False.")
            reference_value_mean = None
            reference_value_covariance = None
        return_value = self.sampled_distribution_set.triangle_plot(\
            self.num_fits, parameters=parameters, plot_limits=plot_limits,\
            in_transformed_space=in_transformed_space, figsize=figsize,\
            fig=fig, show=show, kwargs_1D=kwargs_1D, kwargs_2D=kwargs_2D,\
            fontsize=fontsize, nbins=nbins, plot_type=plot_type,\
            reference_value_mean=reference_value_mean,\
            reference_value_covariance=reference_value_covariance,\
            contour_confidence_levels=contour_confidence_levels,\
            parameter_renamer=parameter_renamer,\
            tick_label_format_string=tick_label_format_string)
        self.sampled_distribution.reset()
        return return_value

