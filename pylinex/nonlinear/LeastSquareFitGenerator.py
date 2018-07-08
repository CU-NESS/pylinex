"""
File: pylinex/nonlinear/LeastSquareFitGenerator.py
Author: Keith Tauscher
Date: 6 Jul 2018

Description: File which contains class which generates LeastSquareFitter
             objects. Basically, the class is a wrapper around an error curve,
             a model, and a prior DistributionSet. Given data curves, it
             produces and runs LeastSquareFitter objects.
"""
import numpy as np
from distpy import DistributionSet
from ..util import sequence_types
from ..model import Model
from ..loglikelihood import GaussianLoglikelihood
from .LeastSquareFitter import LeastSquareFitter

class LeastSquareFitGenerator(object):
    """
    File which contains class which generates LeastSquareFitter objects.
    Basically, the class is a wrapper around an error curve and a model. Given
    data curves, it produces and runs LeastSquareFitter objects.
    """
    def __init__(self, error, model, prior_set):
        """
        Initializes a LeastSquareFitGenerator object with the given error
        array, Model object, and a prior DistributionSet object.
        
        error: 1D array of error on the curve being modeled
        model: Model object with which to fit each data curve
        prior_set: DistributionSet object which describes distribution of
                   parameters of model
        """
        self.error = error
        self.model = model
        self.prior_set = prior_set
    
    @property
    def error(self):
        """
        Property storing the error to use in the LeastSquareFitter objects
        which this object should generate.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Sets the error array this object will use to generate LeastSquareFitter
        objects.
        
        value: 1D numpy.ndarray
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._error = value
            else:
                raise ValueError("error was set to a non-1D array.")
        else:
            raise ValueError("error was set to a non-sequence.")
    
    @property
    def model(self):
        """
        Property storing the model which will be used to generate
        LeastSquareFitter objects.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the model which will be used to generate LeastSquareFitter
        objects.
        
        value: a Model object used to model the data given on a given run
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def prior_set(self):
        """
        Property storing the DistributionSet of priors with which to generate
        LeastSquareFitter objects.
        """
        if not hasattr(self, '_prior_set'):
            raise AttributeError("prior_set was referenced before it was set.")
        return self._prior_set
    
    @prior_set.setter
    def prior_set(self, value):
        """
        Setter for the DistributionSet of priors with which to generate
        LeastSquareFitter objects.
        
        value: a DistributionSet object which describes distribution(s) of
               model parameters
        """
        if isinstance(value, DistributionSet):
            distribution_params = set(value.params)
            model_params = set(self.model.parameters)
            if distribution_params == model_params:
                self._prior_set = value
            else:
                unnecessary_params = distribution_params - model_params
                missing_params = model_params - distribution_params
                if unnecessary_params and missing_params:
                    raise ValueError("There were some parameters described " +\
                        "by the given prior_set which were not parameters " +\
                        "of the model ({0!s}) and some parameters of the " +\
                        "model which were not described by the prior_set " +\
                        "({1!s}).".format(unnecessary_params, missing_params))
                elif unnecessary_params:
                    raise ValueError("Some parameters described by the " +\
                        "given prior_set were not parameters of the model " +\
                        "({!s}).".format(unnecessary_params))
                else:
                    raise ValueError("Some parameters of the model were " +\
                        "not described by the given prior_set ({!s}).".format(\
                        missing_params))
        else:
            raise TypeError("prior_set was set to a non-DistributionSet " +\
                "object.")
    
    def fit(self, data, iterations=1):
        """
        Generates a LeastSquareFitter to fit the given data and runs it for the
        given number of iterations.
        
        data: 1D vector of same length as the error array property
        iterations: the positive integer number of times to run the
                    LeastSquareFitter which this object/method generates
        
        returns: LeastSquareFitter object which has been run for the given
                 number of iterations
        """
        loglikelihood = GaussianLoglikelihood(data, self.error, self.model)
        least_square_fitter = LeastSquareFitter(loglikelihood, self.prior_set)
        least_square_fitter.run(iterations=iterations)
        return least_square_fitter

