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
from scipy.optimize import minimize
from distpy import cast_to_transform_list, DistributionSet
from ..loglikelihood import Loglikelihood

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
    def min(self):
        """
        Property storing the minimum negative Loglikelihood value found in all
        iterations of this fitter.
        """
        return np.min(self.mins)
    
    @property
    def argmin(self):
        """
        Property storing the parameter values of the point which was associated
        with the minimum negative Loglikelihood value found in all iterations
        of this fitter.
        """
        return self.argmins[np.argmin(self.mins)]
    
    @property
    def transformed_argmin(self):
        """
        Property storing the transformed parameter values at the minimum value
        found so far.
        """
        return self.transformed_argmins[np.argmin(self.mins)]
    
    @property
    def covariance_estimate(self):
        """
        Property storing the covariance estimate from the iteration which led
        to the lowest endpoint. If None, something went wrong in inverting
        hessian.
        """
        return self.covariance_estimates[np.argmin(self.mins)]
    
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
        self.mins.append(optimize_result.fun)
        argmin = optimize_result.x
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
    
    @property
    def reconstruction(self):
        """
        Property storing the output of the model of the loglikelihood evaluated
        at the best known input value.
        """
        if not hasattr(self, '_reconstruction'):
            self._reconstruction = self.loglikelihood.model(self.argmin)
        return self._reconstruction
        
    

