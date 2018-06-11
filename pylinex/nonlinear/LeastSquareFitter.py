"""
File: pylinex/nonlinear/LeastSquareFitter.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: File containing class representing a least square fitter which
             uses gradient ascent to maximize the likelihood (if the gradient
             is computable; otherwise, other optimization algorithms are used).
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from scipy.optimize import minimize
from distpy import cast_to_transform_list, DistributionSet
from ..loglikelihood import Loglikelihood, GaussianLoglikelihood

class LeastSquareFitter(object):
    """
    Class representing a least square fitter which uses gradient ascent to
    maximize the likelihood (if the gradient is computable; otherwise, other
    optimization algorithms are used).
    """
    def __init__(self, loglikelihood, prior_set, transform_list=None):
        """
        Initializes a LeastSquareFitter with a Loglikelihood to maximize and a
        prior_set with which to initialize guesses.
        
        loglikelihood: the Loglikelihood object to maximize with this fitter
        prior_set: a DistributionSet object with the same parameters as the
                   model in the loglikelihood describing how to draw reasonable
                   random guesses of their values
        
        """
        self.loglikelihood = loglikelihood
        self.prior_set = prior_set
        self.transform_list = transform_list
    
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
    def argmin(self):
        """
        Property storing the parameter values of the point which was associated
        with the minimum negative Loglikelihood value found in all iterations
        of this fitter.
        """
        return self.argmins[self.best_fit_index]
    
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
    
    def iteration(self, attempt_threshold=100):
        """
        Runs an iteration of this fitter. This entails drawing a random first
        guess at the parameters and using standard algorithms to maximize the
        loglikelihood.
        
        attempt_threshold: the number of attempts to try drawing random first
                           guesses before giving up (the only reason multiple
                           attempts would be necessary is if loglikelihood
                           returns -np.inf in some circumstances)
        """
        attempt = 0
        while True:
            guess = self.generate_guess()
            if np.isfinite(self.loglikelihood(guess)):
                break
            elif attempt >= attempt_threshold:
                raise RuntimeError(("The training set given appears to be " +\
                    "insufficient because {} different attempts were made " +\
                    "to draw points with finite likelihood.").format(\
                    attempt_threshold))
            else:
                attempt += 1
        if self.loglikelihood.gradient_computable:
            optimize_result = minimize(self.loglikelihood, guess,\
                args=(True,), jac=self.loglikelihood.gradient, method='BFGS')
        else:
            optimize_result = minimize(self.loglikelihood, guess,\
                args=(True,), method='Nelder-Mead')
        if np.isnan(optimize_result.fun):
            return
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
    
    def run(self, iterations=1):
        """
        Runs the given number of iterations of this fitter.
        
        iterations: must be a positive integer
        """
        for index in range(iterations):
            self.iteration()
    
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
        if self.num_successes == 0:
            raise RuntimeError("None of this LeastSquareFitter object's " +\
                "iterations were successful, so plotting the endpoints " +\
                "doesn't really make sense.")
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if model is None:
            model = self.loglikelihood.model
            parameter_indices = slice(None)
        if only_best:
            curve = model(self.argmin[parameter_indices])
            if channels is None:
                channels = np.arange(len(curve))
            ax.plot(channels, curve * scale_factor, label=label, **plot_kwargs)
        else:
            curves = np.array([model(argmin[parameter_indices])\
                for (success, argmin) in zip(self.successes, self.argmins)\
                if success])
            if channels is None:
                channels = np.arange(curve.shape[1])
            ax.plot(channels, curves[0] * scale_factor, label=label,\
                **plot_kwargs)
            if curves.shape[0] > 1:
                ax.plot(channels, curves[1:].T * scale_factor,\
                    **plot_kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, size=fontsize)
        if title is not None:
            ax.set_title(title, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if label is not None:
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
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if use_transforms:
            to_sample = self.transformed_argmins
        else:
            to_sample = self.argmins
        to_sample = np.array(to_sample)[np.isfinite(np.array(self.mins)),:]
        parameter_index = self.loglikelihood.model.parameters.index(parameter)
        parameter_sample = to_sample[:,parameter_index]
        multidimensional = (parameter2 is not None)
        if multidimensional:
            parameter2_index =\
                self.loglikelihood.model.parameters.index(parameter2)
            parameter2_sample = to_sample[:,parameter2_index]
            ax.hist2d(parameter_sample, parameter2_sample, **hist_kwargs)
        else:
            ax.hist(parameter_sample, **hist_kwargs)
        if xlabel is None:
            xlabel = parameter
        ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is None:
            if multidimensional:
                ylabel = parameter2
            else:
                ylabel = '# of occurrences'
        ax.set_ylabel(ylabel, size=fontsize)
        if title is None:
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
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        # '-' below necessary because, internally, -loglikelihood was minimized
        to_sample = -np.array(self.mins)
        to_sample = to_sample[np.isfinite(to_sample)]
        ax.hist(to_sample, **hist_kwargs)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, size=fontsize)
        if title is not None:
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
        

