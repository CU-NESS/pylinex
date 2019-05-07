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
import os, time
import numpy as np
from distpy import cast_to_transform_list, GaussianDistribution,\
    DistributionSet
from ..util import int_types
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
        transform_list=None, **bounds):
        """
        Initializes a new LeastSquareFitCluster.
        
        seed_loglikelihood: GaussianLoglikelihood object containing the data,
                            error and model under concern.
        prior_set: must be a DistributionSet object whose parameters are the
                   same as the parameters of the Loglikelihood at the heart of
                   this fitter
        prefix: string seeding file names where LeastSquareFitter objects
                should be placed. "{0!s}.{1:d}.hdf5".format(prefix, index)
                should be a valid file location for index in range(num_fits)
        num_fits: integer number of LeastSquareFitter objects to create
        transform_list: TransformList (or something which can be cast to a
                        TransformList object) describing the space in which
                        the parameters of the loglikelihood exist
        bounds: dictionary containing 2-tuples of (min, max) where min and max
                are either numbers or None indexed by parameter name.
                Parameters which are not in value are bounded by their own
                Model object's bounds property.
        """
        self.seed_loglikelihood = seed_loglikelihood
        self.prior_set = prior_set
        self.prefix = prefix
        self.num_fits = num_fits
        self.transform_list = transform_list
        self.bounds = bounds
    
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
        self._transform_list = cast_to_transform_list(value,\
            num_transforms=len(self.parameters))
    
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
        
        value: string seeding file names where LeastSquareFitter objects should
               be placed. "{0!s}.{1:d}.hdf5".format(prefix, index) should
               be a valid file location for index in range(num_fits)
        """
        if isinstance(value, basestring):
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
            zfill_width = int(np.ceil(np.log10(self.num_fits)))
            self._file_names = ['{0!s}.{1!s}.hdf5'.format(self.prefix,\
                "{:d}".format(fit_index).zfill(zfill_width))\
                for fit_index in range(self.num_fits)]
        return self._file_names
    
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
        cutoff_loglikelihood=np.inf, verbose=False, **kwargs):
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
        kwargs: Keyword arguments to pass on as options to
                scipy.optimize.minimize(method='SLSQP'). They can include:
                    ftol : float, precision goal for the loglikelihood in the
                           stopping criterion.
                    eps : float, step size used for numerical approximation of
                          the gradient.
                    disp : bool, set to True to print convergence messages.
                    maxiter : int, maximum number of iterations.
        """
        for (fit_index, file_name) in enumerate(self.file_names):
            if verbose:
                print("Beginning least square fitter #{0:d} at {1!s}".format(\
                    1 + fit_index, time.ctime()))
            least_square_fitter = LeastSquareFitter(\
                loglikelihood=self.generate_loglikelihood(fit_index),\
                prior_set=self.prior_set, transform_list=None,\
                file_name=file_name, **self.bounds)
            least_square_fitter.run(iterations=iterations,\
                attempt_threshold=attempt_threshold,\
                cutoff_loglikelihood=cutoff_loglikelihood, **kwargs)
    
    @property
    def argmins(self):
        """
        Property storing the minima found by each of the LeastSquareFitter
        objects.
        """
        if not hasattr(self, '_argmins'):
            if all([os.path.exists(file_name)\
                for file_name in self.file_names]):
                self._argmins = [LeastSquareFitter(file_name=file_name).argmin\
                    for file_name in self.file_names]
                self._argmins = np.array(self._argmins)
            else:
                raise RuntimeError("argmins property cannot be accessed " +\
                    "until this LeastSquareFitCluster has created all of " +\
                    "the LeastSquareFitter objects it will create.")
        return self._argmins
    
    @property
    def approximate_gaussian_distribution(self):
        """
        Property storing the approximate GaussianDistribution describing the
        parameter distribution (in the space of the seed_loglikelihood's
        parameters).
        """
        if not hasattr(self, '_approximate_gaussian_distribution'):
            mean = np.mean(self.argmins, axis=0)
            covariance = np.cov(self.argmins, rowvar=False)
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

