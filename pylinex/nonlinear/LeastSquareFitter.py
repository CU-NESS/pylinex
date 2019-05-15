"""
File: pylinex/nonlinear/LeastSquareFitter.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: File containing class representing a least square fitter which
             uses gradient ascent to maximize the likelihood (if the gradient
             is computable; otherwise, other optimization algorithms are used).
"""
import os, h5py, time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from distpy import TransformList, cast_to_transform_list, DistributionSet
from ..util import create_hdf5_dataset, get_hdf5_value
from ..loglikelihood import Loglikelihood, GaussianLoglikelihood,\
    load_loglikelihood_from_hdf5_group

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class LeastSquareFitter(object):
    """
    Class representing a least square fitter which uses gradient ascent to
    maximize the likelihood (if the gradient is computable; otherwise, other
    optimization algorithms are used).
    """
    def __init__(self, loglikelihood=None, prior_set=None,\
        transform_list=None, file_name=None, **bounds):
        """
        Initializes a LeastSquareFitter with a Loglikelihood to maximize and a
        prior_set with which to initialize guesses.
        
        loglikelihood: the Loglikelihood object to maximize with this fitter.
                       Must be supplied unless file_name is not None and it
                       already exists.
        prior_set: a DistributionSet object with the same parameters as the
                   model in the loglikelihood describing how to draw reasonable
                   random guesses of their values. Must be supplied unless
                   file_name is not None and it already exists.
        transform_list: TransformList (or something which can be cast to a
                        TransformList object) describing how to find
                        transformed_argmin and covariance estimate in the
                        transform space
        file_name: if None, nothing in this object will be saved
                   if valid file path, this object is saved to this file
        bounds: extra bounds to apply in the form of keyword arguments of the
                form (min, max). Any parameters not included will be bounded by
                their Model's bounds parameter. If none is given, all Model's
                bounds are used
        """
        self.file_name = file_name
        if (type(self.file_name) is not type(None)) and\
            os.path.exists(self.file_name):
            self.load_setup_and_iterations()
        else:
            self.loglikelihood = loglikelihood
            self.prior_set = prior_set
            self.transform_list = transform_list
            self.bounds = bounds
            if type(self.file_name) is not type(None):
                self.save_setup()
    
    def load_setup_and_iterations(self):
        """
        Loads the setup of this fitter in an hdf5 file
        """
        hdf5_file = h5py.File(self.file_name, 'r')
        self.loglikelihood =\
            load_loglikelihood_from_hdf5_group(hdf5_file['loglikelihood'])
        self.prior_set =\
            DistributionSet.load_from_hdf5_group(hdf5_file['prior_set'])
        self.transform_list =\
            TransformList.load_from_hdf5_group(hdf5_file['transform_list'])
        group = hdf5_file['bounds']
        bounds = {}
        for name in self.parameters:
            subgroup = group[name]
            if 'lower' in subgroup.attrs:
                lower_bound = subgroup.attrs['lower']
            else:
                lower_bound = None
            if 'upper' in subgroup.attrs:
                upper_bound = subgroup.attrs['upper']
            else:
                upper_bound = None
            bounds[name] = (lower_bound, upper_bound)
        self.bounds = bounds
        group = hdf5_file['iterations']
        self._num_iterations = group.attrs['num_iterations']
        (successes, mins, argmins, transformed_argmins) = ([], [], [], [])
        covariance_estimates = []
        if isinstance(self.loglikelihood, GaussianLoglikelihood):
            reduced_chi_squared_statistics = []
        for iteration in range(self.num_iterations):
            subgroup = group['{:d}'.format(iteration)]
            successes.append(subgroup.attrs['success'])
            min_value = subgroup.attrs['min_value']
            mins.append(min_value)
            argmin = get_hdf5_value(subgroup['argmin'])
            argmins.append(argmin)
            transformed_argmins.append(self.transform_list.apply(argmin))
            if 'covariance_estimate' in subgroup:
                covariance_estimates.append(\
                    get_hdf5_value(subgroup['covariance_estimate']))
            else:
                covariance_estimates.append(None)
            reduced_chi_squared_statistics.append((2 * min_value) /\
                self.loglikelihood.degrees_of_freedom)
        self._successes = successes
        self._mins = mins
        self._argmins = argmins
        self._transformed_argmins = transformed_argmins
        self._covariance_estimates = covariance_estimates
        self._reduced_chi_squared_statistics = reduced_chi_squared_statistics
        try:
            hdf5_file.close()
        except:
            pass # for some reason, closing the file here sometimes causes an
                 # error. However, since the file is opened in read mode, this
                 # shouldn't cause corruption. Revisit this if an error related
                 # to hdf5 file references is seen.
    
    def save_setup(self):
        """
        Saves the setup of this fitter in an hdf5 file.
        """
        hdf5_file = h5py.File(self.file_name, 'w')
        self.loglikelihood.fill_hdf5_group(\
            hdf5_file.create_group('loglikelihood'))
        self.prior_set.fill_hdf5_group(hdf5_file.create_group('prior_set'))
        self.transform_list.fill_hdf5_group(\
            hdf5_file.create_group('transform_list'))
        group = hdf5_file.create_group('bounds')
        for (iparameter, parameter) in enumerate(self.parameters):
            subgroup = group.create_group(parameter)
            (lower_bound, upper_bound) = self.bounds[iparameter]
            if type(lower_bound) is not type(None):
                subgroup.attrs['lower'] = lower_bound
            if type(upper_bound) is not type(None):
                subgroup.attrs['upper'] = upper_bound
        group = hdf5_file.create_group('iterations')
        group.attrs['num_iterations'] = self.num_iterations
        hdf5_file.close()
    
    @property
    def num_iterations(self):
        """
        Property storing the index of the next iteration to save (which is the
        same as the number of iterations).
        """
        if not hasattr(self, '_num_iterations'):
            self._num_iterations = 0
        return self._num_iterations
    
    def increment_index(self):
        """
        Increments the index of the next iteration to save.
        """
        self._num_iterations = self.num_iterations + 1
    
    @property
    def file_name(self):
        """
        Property storing the filesystem location of the file in which to save
        this fitter, if it exists (this property is None in this case).
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name was referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Sets the file in which to save this fitter, if it exists.
        
        value: either None or the filesystem location at which to place an hdf5
               file
        """
        if (type(value) is type(None)) or isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was set to neither None nor a string.")
    
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
            num_transforms=self.loglikelihood.num_parameters)
    
    @property
    def loglikelihood(self):
        """
        Property storing the Loglikelihood being maximized by this fitter.
        """
        if not hasattr(self, '_loglikelihood'):
            raise AttributeError("loglikelihood was referenced before it " +\
                "was set.")
        return self._loglikelihood
    
    @loglikelihood.setter
    def loglikelihood(self, value):
        """
        Setter for the Loglikelihood object to maximize with this fitter.
        
        value: must be a Loglikelihood object
        """
        if isinstance(value, Loglikelihood):
            self._loglikelihood = value
        else:
            raise TypeError("loglikelihood was not set to a " +\
                "Loglikelihood object.")
    
    @property
    def parameters(self):
        """
        The names of the parameters of the model in the loglikelihood.
        """
        return self.loglikelihood.parameters
    
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
    def successes(self):
        """
        Property storing a list of booleans describing whether the
        LeastSquareFitter exited successfully on each iteration.
        """
        if not hasattr(self, '_successes'):
            self._successes = []
        return self._successes
    
    @property
    def num_successes(self):
        """
        Property storing the number of successful iterations this least square
        fitter has completed.
        """
        return sum(self.successes)
    
    @property
    def mins(self):
        """
        Property storing the minimum negative Loglikelihood values reached by
        each iteration of the fitter.
        """
        if not hasattr(self, '_mins'):
            self._mins = []
        return self._mins
    
    @property
    def argmins(self):
        """
        Property storing the parameters at which the mins were found.
        """
        if not hasattr(self, '_argmins'):
            self._argmins = []
        return self._argmins
    
    @property
    def reduced_chi_squared_statistics(self):
        """
        Property storing the reduced chi squared statistics associated with the
        points at which this LeastSquareFitter stops iterations.
        """
        if not hasattr(self, '_reduced_chi_squared_statistics'):
            self._reduced_chi_squared_statistics = []
        return self._reduced_chi_squared_statistics
    
    @property
    def transformed_argmins(self):
        """
        Property storing a list of transformed minima found so far by this
        LeastSquareFitter.
        """
        if not hasattr(self, '_transformed_argmins'):
            self._transformed_argmins = []
        return self._transformed_argmins
    
    @property
    def covariance_estimates(self):
        """
        Property storing the covariance estimates (in transformed space defined
        by transform_list property) at each result.
        """
        if not hasattr(self, '_covariance_estimates'):
            self._covariance_estimates = []
        return self._covariance_estimates
    
    @property
    def reconstructions(self):
        """
        Property storing the output of the model of the loglikelihood evaluated
        at the best known input value.
        """
        if not hasattr(self, '_reconstructions'):
            self._reconstructions = np.array([self.loglikelihood.model(argmin)\
                for argmin in self.argmins])
        return self._reconstructions
    
    @property
    def min(self):
        """
        Property storing the minimum negative Loglikelihood value found in all
        iterations of this fitter.
        """
        return np.min(self.mins)
    
    @property
    def best_fit_index(self):
        """
        Property storing the index of the maximum likelihood fit.
        """
        if self.mins:
            return np.argmin(self.mins)
        else:
            raise NotImplementedError("There is no best fit index because " +\
                "there have been no successful least square fits.")
    
    @property
    def best_successful_fit_index(self):
        """
        Property storing the index of the maximum likelihood fit when only
        least square fit successes are considered.
        """
        if self.successes:
            return np.argmin(np.where(self.successes, self.mins, np.inf))
        else:
            raise NotImplementedError("There is no best fit index because " +\
                "there have been no successful least square fits.")
    
    @property
    def argmin(self):
        """
        Property storing the parameter values of the point which was associated
        with the minimum negative Loglikelihood value found in all iterations
        of this fitter.
        """
        return self.argmins[self.best_fit_index]
    
    @property
    def successful_argmin(self):
        """
        Property storing the parameter values of the point which was associated
        with the minimum negative Loglikelihood value found in all iterations
        of this fitter which were successes.
        """
        return self.argmins[self.best_successful_fit_index]
    
    @property
    def reduced_chi_squared_statistic(self):
        """
        Property storing the single number reduced chi squared corresponding to
        the best fit result. Ideally, it should be near 1. If it is far from 1
        (the working definition of 'far' is determined by the number of degrees
        of freedom), then the fit is poor or the likelihood's error estimate is
        wrong.
        """
        return self.reduced_chi_squared_statistics[self.best_fit_index]
    
    @property
    def degrees_of_freedom(self):
        """
        Property storing the integer number of degrees of freedom of the
        likelihood explored by this distribution.
        """
        return self.loglikelihood.degrees_of_freedom
    
    @property
    def transformed_argmin(self):
        """
        Property storing the transformed parameter values at the minimum value
        found so far.
        """
        return self.transformed_argmins[self.best_fit_index]
    
    @property
    def covariance_estimate(self):
        """
        Property storing the covariance estimate from the iteration which led
        to the lowest endpoint. If None, something went wrong in inverting
        hessian.
        """
        return self.covariance_estimates[self.best_fit_index]
    
    @property
    def distribution_estimate(self):
        """
        Property which returns an estimate of the parameter distribution if a
        mean and covariance estimate has been found.
        """
        if type(self.covariance_estimate) is type(None):
            return None
        else:
            return GaussianDistribution(self.argmin, self.covariance_estimate)
    
    @property
    def reconstruction(self):
        """
        Property storing the maximum likelihood reconstruction of the modeled
        curve.
        """
        return self.reconstructions[self.best_fit_index]
    
    def generate_guess(self):
        """
        Uses the given prior_set to draw a reasonable random first guess at the
        input parameters which will be used as an input for sophisticated
        algorithms.
        """
        draw = self.prior_set.draw()
        return np.array([draw[parameter] for parameter in self.parameters])
    
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
            self._bounds = []
            for name in self.parameters:
                if name in value:
                    self._bounds.append(value[name])
                else:
                    self._bounds.append(self.loglikelihood.model.bounds[name])
        else:
            raise TypeError("bounds was set to a non-dictionary.")
    
    def iteration(self, attempt_threshold=1, **kwargs):
        """
        Runs an iteration of this fitter. This entails drawing a random first
        guess at the parameters and using standard algorithms to maximize the
        loglikelihood.
        
        attempt_threshold: the number of attempts to try drawing random first
                           guesses before giving up (the only reason multiple
                           attempts would be necessary is if loglikelihood
                           returns -np.inf in some circumstances) default: 1
        kwargs: Keyword arguments to pass on as options to
                scipy.optimize.minimize(method='SLSQP'). They can include:
                    ftol : float, precision goal for the loglikelihood in the
                           stopping criterion.
                    eps : float, step size used for numerical approximation of
                          the gradient.
                    disp : bool, set to True to print convergence messages.
                    maxiter : int, maximum number of iterations.
        """
        attempt = 0
        while True:
            guess = self.generate_guess()
            if np.isfinite(self.loglikelihood(guess)):
                break
            elif attempt >= attempt_threshold:
                raise RuntimeError(("The prior set given appears to be " +\
                    "insufficient because {:d} different attempts were " +\
                    "made to draw points with finite likelihood, but all " +\
                    "had 0 likelihood.").format(attempt_threshold))
            else:
                attempt += 1
        if self.loglikelihood.gradient_computable:
            optimize_result = minimize(self.loglikelihood, guess,\
                args=(True,), jac=self.loglikelihood.gradient, method='SLSQP',\
                bounds=self.bounds, options=kwargs)
        else:
            optimize_result = minimize(self.loglikelihood, guess,\
                args=(True,), method='SLSQP', bounds=self.bounds,\
                options=kwargs)
        if np.isnan(optimize_result.fun):
            raise ValueError("loglikelihood returned nan.")
        self.successes.append(optimize_result.success)
        self.mins.append(optimize_result.fun)
        argmin = optimize_result.x
        if isinstance(self.loglikelihood, GaussianLoglikelihood):
            self.reduced_chi_squared_statistics.append(\
                self.loglikelihood.reduced_chi_squared(argmin))
        self.argmins.append(argmin)
        self.transformed_argmins.append(self.transform_list.apply(argmin))
        if self.loglikelihood.hessian_computable:
            gradient = np.zeros(self.loglikelihood.num_parameters)
            try:
                hessian =\
                    self.loglikelihood.hessian(argmin, return_negative=True)
                hessian = self.transform_list.transform_hessian(\
                    hessian, gradient, argmin, first_axis=0)
                covariance_estimate = la.inv(hessian)
                covariance_estimate =\
                    (covariance_estimate + covariance_estimate.T) / 2.
            except:
                covariance_estimate = None
            self.covariance_estimates.append(covariance_estimate)
        else:
            self.covariance_estimates.append(None)
        self.save_iteration()
    
    def save_iteration(self):
        """
        Saves the last iteration of this LeastSquareFitter to the file 
        """
        if type(self.file_name) is not type(None):
            hdf5_file = h5py.File(self.file_name, 'r+')
            group = hdf5_file['iterations']
            subgroup = group.create_group('{:d}'.format(self.num_iterations))
            subgroup.attrs['success'] = self.successes[-1]
            subgroup.attrs['min_value'] = self.mins[-1]
            create_hdf5_dataset(subgroup, 'argmin', data=self.argmins[-1])
            if type(self.covariance_estimates[-1]) is not type(None):
                create_hdf5_dataset(subgroup, 'covariance_estimate',\
                    data=self.covariance_estimates[-1])
        self.increment_index()
        if type(self.file_name) is not type(None):
            group.attrs['num_iterations'] = self.num_iterations
            hdf5_file.close()
    
    def run(self, iterations=1, attempt_threshold=100,\
        cutoff_loglikelihood=np.inf, verbose=False,\
        run_if_iterations_exist=True, **kwargs):
        """
        Runs the given number of iterations of this fitter.
        
        iterations: must be a positive integer
        attempt_threshold: number of times an iteration should recur before
                           failing
        cutoff_loglikelihood: if an iteration of this LeastSquareFitter
                              achieves a loglikelihood above this value, the
                              LeastSquareFitter is stopped early
                              default value is np.inf
        verbose: if True, a message is printed at the beginning of each
                          iteration of this LeastSquareFitter
        run_if_iterations_exist: if False (default: True), iterations are only
                                                           run if they don't
                                                           already exist
        kwargs: Keyword arguments to pass on as options to
                scipy.optimize.minimize(method='SLSQP'). They can include:
                    ftol : float, precision goal for the loglikelihood in the
                           stopping criterion.
                    eps : float, step size used for numerical approximation of
                          the gradient.
                    disp : bool, set to True to print convergence messages.
                    maxiter : int, maximum number of iterations.
        """
        if (not run_if_iterations_exist) and\
            (self.num_iterations >= iterations):
            return
        for index in range(iterations):
            if verbose:
                print(("Starting iteration #{0:d} of LeastSquareFitter " +\
                    "at {1!s}.").format(1 + index, time.ctime()))
            if (len(self.argmins) > 0) and\
                (self.min < ((-1) * cutoff_loglikelihood)):
                continue
            self.iteration(attempt_threshold=attempt_threshold, **kwargs)
    
    def plot_reconstructions(self, parameter_indices=slice(None), model=None,\
        only_best=False, scale_factor=1, ax=None, channels=None, label=None,\
        xlabel=None, ylabel=None, title=None, fontsize=24, show=False,\
        **plot_kwargs):
        """
        Plots all reconstructions (or only the best one) found by this fitter.
        
        parameter_indices: indices of parameters to feed to model.
                           If None, all parameters are used
        model: model to run on given parameter sample
        only_best: if True, only the best fit reconstruction is plotted
        scale_factor: multiplicative factor with which to affect every
                      reconstruction, default 1.
        ax: Axes instance on which to make this plot, or None if a new one
            should be made
        channels: if None (default), channel index is shown on x-axis.
                  Otherwise, channels should be a 1D array to use as x-axis
        label: label to use in legend for curve(s)
        xlabel: string label to place on x-axis (defaults to parameter)
        ylabel: string label to place on y-axis (defaults to '# of occurrences'
                in 1D case and parameter2 in 2D case)
        title: string title to place on top of plot
        fontsize: size of font to use for labels, titles, etc.
        ax: the Axes instance on which to plot the histogram
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        hist_kwargs: extra keyword arguments to pass on to matplotlib's plot
                     function (e.g. 'color', 'label', etc.)
        
        returns: None if show is True, Axes instance with plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if self.num_successes != 0:
            if type(model) is type(None):
                model = self.loglikelihood.model
                parameter_indices = slice(None)
            if only_best:
                curve = model(self.argmin[parameter_indices])
                if type(channels) is type(None):
                    channels = np.arange(len(curve))
                ax.plot(channels, curve * scale_factor, label=label,\
                    **plot_kwargs)
            else:
                curves = np.array([model(argmin[parameter_indices])\
                    for (success, argmin) in zip(self.successes, self.argmins)\
                    if success])
                if type(channels) is type(None):
                    channels = np.arange(curves.shape[1])
                ax.plot(channels, curves[0] * scale_factor, label=label,\
                    **plot_kwargs)
                if curves.shape[0] > 1:
                    ax.plot(channels, curves[1:].T * scale_factor,\
                        **plot_kwargs)
            ax.set_xlim((channels[0], channels[-1]))
            if type(xlabel) is not type(None):
                ax.set_xlabel(xlabel, size=fontsize)
            if type(ylabel) is not type(None):
                ax.set_ylabel(ylabel, size=fontsize)
            if type(title) is not type(None):
                ax.set_title(title, size=fontsize)
            ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
                which='major')
            ax.tick_params(width=1.5, length=4.5, which='minor')
            if type(label) is not type(None):
                ax.legend(fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_parameter_histogram(self, parameter, parameter2=None,\
        use_transforms=True, xlabel=None, ylabel=None, title=None,\
        fontsize=24, ax=None, show=False, **hist_kwargs):
        """
        Plots a histogram of the parameter values on which iterations ended.
        This function is capable of making both 1D and 2D histograms.
        
        parameter: either the parameter to make a histogram of (in 1D case) or
                   parameter to plot on x-axis of histogram (in 2D case)
        parameter2: parameter to plot on y-axis of histogram (in 2D case).
                    If None, a 1D histogram of parameter is plotted
        use_transforms: if True (default), values of parameter(s) are
                        transformed before binning and plotting
        xlabel: string label to place on x-axis (defaults to parameter)
        ylabel: string label to place on y-axis (defaults to '# of occurrences'
                in 1D case and parameter2 in 2D case)
        title: string title to place on top of plot
        fontsize: size of font to use for labels, titles, etc.
        ax: the Axes instance on which to plot the histogram
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        hist_kwargs: extra keyword arguments to pass on to matplotlib's hist
                     (or hist2d) function (e.g. 'bins', 'label', etc.)
        
        returns: None if show is True, Axes instance with plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if use_transforms:
            to_sample = self.transformed_argmins
        else:
            to_sample = self.argmins
        to_sample = np.array(to_sample)[np.isfinite(np.array(self.mins)),:]
        parameter_index = self.loglikelihood.model.parameters.index(parameter)
        parameter_sample = to_sample[:,parameter_index]
        multidimensional = (type(parameter2) is not type(None))
        if multidimensional:
            parameter2_index =\
                self.loglikelihood.model.parameters.index(parameter2)
            parameter2_sample = to_sample[:,parameter2_index]
            ax.hist2d(parameter_sample, parameter2_sample, **hist_kwargs)
        else:
            ax.hist(parameter_sample, **hist_kwargs)
        if type(xlabel) is type(None):
            xlabel = parameter
        ax.set_xlabel(xlabel, size=fontsize)
        if type(ylabel) is type(None):
            if multidimensional:
                ylabel = parameter2
            else:
                ylabel = '# of occurrences'
        ax.set_ylabel(ylabel, size=fontsize)
        if type(title) is type(None):
            title = 'Least square fit parameter histogram'
        ax.set_title(title, size=fontsize)
        if 'label' in hist_kwargs:
            ax.legend(fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if show:
            pl.show()
        else:
            return ax
    
    def plot_loglikelihood_histogram(self, xlabel='$\ln{\mathcal{L}}$',\
        ylabel='# of occurrences', title='Log likelihood histogram',\
        fontsize=24, ax=None, show=False, **hist_kwargs):
        """
        Plots a histogram of the loglikelihood values found by this
        LeastSquareFitter.
        
        xlabel: string label to place on x-axis
        ylabel: string label to place on y-axis
        title: string title to place on top of plot
        fontsize: size of font to use for labels, titles, etc.
        ax: the Axes instance on which to plot the histogram
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        hist_kwargs: extra keyword arguments to pass on to matplotlib's hist
                     function (e.g. 'bins', 'label', etc.)
 
        returns: None if show is True, Axes instance containing plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        # '-' below necessary because, internally, -loglikelihood was minimized
        to_sample = -np.array(self.mins)
        to_sample = to_sample[np.isfinite(to_sample)]
        ax.hist(to_sample, **hist_kwargs)
        if type(xlabel) is not type(None):
            ax.set_xlabel(xlabel, size=fontsize)
        if type(ylabel) is not type(None):
            ax.set_ylabel(ylabel, size=fontsize)
        if type(title) is not type(None):
            ax.set_title(title, size=fontsize)
        if 'label' in hist_kwargs:
            ax.legend(fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if show:
            pl.show()
        else:
            return ax
        

