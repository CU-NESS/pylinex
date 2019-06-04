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
from distpy import DistributionSet, TransformList
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
    def __init__(self, error, model, prior_set, transform_list=None, **bounds):
        """
        Initializes a LeastSquareFitGenerator object with the given error
        array, Model object, and a prior DistributionSet object.
        
        error: 1D array of error on the curve being modeled
        model: Model object with which to fit each data curve
        prior_set: DistributionSet object which describes distribution of
                   parameters of model
        transform_list: TransformList (or something which can be cast to a
                        TransformList object) describing how to find
                        transformed_argmin and covariance estimate in the
                        transform space. Note that this DOES NOT perform the
                        least square fit in transformed space
        **bounds: keyword arguments are interpreted as parameter bounds. The
                  keywords are the parameter names and the values should be
                  2-tuples of the form (min, max) where either can be None.
        """
        self.error = error
        self.model = model
        self.prior_set = prior_set
        self.transform_list = transform_list
        self.bounds = bounds
    
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
            TransformList.cast(value, num_transforms=self.model.num_parameters)
    
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
        
        value: 1D or 2D numpy.ndarray
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim in [1, 2]:
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
    
    @property
    def bounds(self):
        """
        Property storing the bounds dictionary which will be passed on to any
        LeastSquareFitter object's this object creates.
        """
        if not hasattr(self, '_bounds'):
            raise AttributeError("bounds was referenced before it was set.")
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        """
        Setter for the bounds dictionary.
        
        value: a dictionary to pass on to each LeastSquareFitter object this
               object creates
        """
        if isinstance(value, dict):
            self._bounds = value
        else:
            raise TypeError("bounds was set to a non-dict.")
    
    def fit(self, data, iterations=1, cutoff_loglikelihood=np.inf,\
        file_name=None):
        """
        Generates a LeastSquareFitter to fit the given data and runs it for the
        given number of iterations.
        
        data: 1D vector of same length as the error array property
        iterations: the positive integer number of times to run the
                    LeastSquareFitter which this object/method generates
        cutoff_loglikelihood: if an iteration of this LeastSquareFitter
                              achieves a loglikelihood above this value, the
                              LeastSquareFitter is stopped early
                              default value is np.inf
        file_name: if given (default None), this is a path to a file to which
                   the least square fitter to be saved (or from which it should
                   be loaded)
        
        returns: LeastSquareFitter object which has been run for the given
                 number of iterations
        """
        loglikelihood = GaussianLoglikelihood(data, self.error, self.model)
        least_square_fitter = LeastSquareFitter(loglikelihood=loglikelihood,\
            prior_set=self.prior_set, transform_list=self.transform_list,\
            file_name=file_name, **self.bounds)
        least_square_fitter.run(iterations=iterations,\
            cutoff_loglikelihood=cutoff_loglikelihood)
        return least_square_fitter

