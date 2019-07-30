"""
File: pylinex/nonlinear/InterpolatingLeastSquareFitter.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: File containing class which performs a least square fit by using
             a model based on interpolation between given training set curves.
             When initialized, the fitter can be called on multiple input
             curves. When given an input curve, the fitter takes the nearest
             ntrain training set curves to define a Delaunay mesh interpolator.
             Then it uses that interpolator, to maximize the loglikelihood.
"""
import numpy as np
from distpy import TransformList
from ..util import int_types, sequence_types
from ..expander import Expander, NullExpander
from ..model import InputInterpolatedModel
from ..loglikelihood import GaussianLoglikelihood
from .LeastSquareFitter import LeastSquareFitter

class InterpolatingLeastSquareFitter(object):
    """
    Class which performs a least square fit by using a model based on
    interpolation between given training set curves. When initialized, the
    fitter can be called on multiple input curves. When given an input curve,
    the fitter takes the nearest ntrain training set curves to define a
    Delaunay mesh interpolator. Then it uses that interpolator, to maximize the
    loglikelihood.
    """
    def __init__(self, error, parameter_names, training_inputs,\
        training_outputs, ntrain=None, should_compress=False,\
        transform_list=None, scale_to_cube=False, num_basis_vectors=None,\
        expander=None, interpolation_method='linear',\
        loglikelihood_callable=None):
        """
        Initializes this fitter with the error it should use in its likelihood,
        the names of its parameters and its training set inputs and outputs.
        
        error: the error to use to define the likelihood to maximize
        parameter_names: sequence of names of parameters (of length
                         input_dimension)
        training_inputs: 2D numpy.ndarray of shape
                         (num_elements, input_dimension)
        training_outputs: 2D numpy.ndarray of shape
                          (num_elements, output_dimension)
        ntrain: the number of training set elements to use to define the
                Delaunay mesh used to interpolate to any given input vector.
                ntrain must satisfy ntrain > input_dimension + 1
        should_compress: boolean determining whether the training_outputs
                         should be compressed by ignoring unimportant SVD
                         modes. If True, leads to some loss of information.
        transform_list: TransformList or sequence of input_dimension Transform
                        objects (or strings which can be cast to them) which
                        should be used in defining the interpolation space. If
                        None, no transforms are applied to any variables.
        scale_to_cube: boolean determining if the (transformed) training inputs
                       should be scaled to a cube before interpolation. This is
                       useful if different parameters have vastly different
                       extents.
        num_basis_vectors: number of basis vectors to be used in the case of
                           should_compress==True. If should_compress==False,
                           then this parameter is not required.
        expander: if the training output curves are in a different space than
                  the curves which will be fit to, an expander describing the
                  transformation from the former space to the latter can be
                  supplied. If None, the two spaces are identical.
        interpolation_method: either 'linear' or 'quadratic'
        loglikelihood_callable: a function which, when passed a
                                GaussianLoglikelihood made out of an
                                InputInterpolatedModel and a data and error
                                vector
        """
        self.error = error
        self.expander = expander
        self.compressed = should_compress
        self.training_inputs = training_inputs
        self.scale_to_cube = scale_to_cube
        self.parameters = parameter_names
        self.transform_list = transform_list
        self.num_basis_vectors = num_basis_vectors
        self.training_outputs = training_outputs
        self.ntrain = ntrain
        self.interpolation_method = interpolation_method
        self.loglikelihood_callable = loglikelihood_callable
    
    @property
    def interpolation_method(self):
        """
        Property storing whether this is a 'linear' or 'quadratic' interpolator
        """
        if not hasattr(self, '_interpolation_method'):
            raise AttributeError("interpolation_method referenced before " +\
                "it was set.")
        return self._interpolation_method
    
    @interpolation_method.setter
    def interpolation_method(self, value):
        """
        Setter for the interpolation method.
        
        value: either 'linear' or 'quadratic'
        """
        if value in ['linear', 'quadratic']:
            self._interpolation_method = value
        else:
            raise ValueError("interpolation_method was neither linear nor " +\
                "quadratic.")
    
    @property
    def ntrain(self):
        """
        Property storing the number of training set elements to use to define
        the Delaunay mesh used to interpolate to any given input vector.
        """
        if not hasattr(self, '_ntrain'):
            raise AttributeError("ntrain was referenced before it was set.")
        return self._ntrain
    
    @ntrain.setter
    def ntrain(self, value):
        """
        Setter for the number of training set points to include for any given
        input data vector.
        
        value: must be greater than input_dimension+1 to have enough points to
               define hyperplanes
        """
        if type(value) is type(None):
            self._ntrain = self.training_inputs.shape[0]
        elif type(value) in int_types:
            if value > self.input_dimension + 1:
                self._ntrain = value
            else:
                raise ValueError("ntrain was set to a non-positive integer.")
        else:
            raise TypeError("ntrain was sent to a non-int.")
    
    @property
    def error(self):
        """
        Property storing the error to use in the definition of the
        loglikelihood to maximize.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property used in the definition of the
        loglikelihood to maximize.
        
        value: must be a numpy.ndarray of the same shape as the data property
        """
        value = np.array(value)
        if value.ndim == 1:
            self._error = value
        else:
            raise ValueError("error given was not the same shape as the data.")
    
    @property
    def scale_to_cube(self):
        """
        Property storing the boolean determining if the (transformed) training
        inputs should be scaled to a cube before interpolation. This is useful
        if different parameters have vastly different extents.
        """
        if not hasattr(self, '_scale_to_cube'):
            raise AttributeError("scale_to_cube referenced before it was set.")
        return self._scale_to_cube
    
    @scale_to_cube.setter
    def scale_to_cube(self, value):
        """
        Setter for the scale_to_cube property which determines whether
        (transformed) training inputs should be scaled to cube before
        interpolation.
        
        value: either True or False
        """
        try:
            self._scale_to_cube = bool(value)
        except:
            raise TypeError("scale_to_cube couldn't be interpreted as a bool.")
    
    @property
    def training_inputs(self):
        """
        Property storing the inputs of the training set in a 2D numpy.ndarray
        """
        if not hasattr(self, '_training_inputs'):
            raise AttributeError("training_inputs referenced before it was " +\
                "set.")
        return self._training_inputs
    
    @training_inputs.setter
    def training_inputs(self, value):
        """
        Setter of the input points of the training set.
        
        value: must be a 2D numpy.ndarray of shape (nelements, input_dimension)
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[1] > 1:
                self._training_inputs = value
            else:
                raise ValueError("Since this object uses a Delaunay " +\
                    "triangulation which requires more than 1 dimension, " +\
                    "the input vectors must be at least 2D.")
        else:
            raise ValueError("training_inputs was set to an array which " +\
                "was not 2D.")
    
    @property
    def training_outputs(self):
        """
        Property storing the outputs of the training set in a 2D numpy.ndarray
        """
        if not hasattr(self, '_training_outputs'):
            raise AttributeError("training_outputs referenced before it " +\
                "was set.")
        return self._training_outputs
    
    @training_outputs.setter
    def training_outputs(self, value):
        """
        Setter of the output points of the training set.
        
        value: must be a 2D numpy.ndarray of shape
               (nelements, output_dimension)
        """
        value = np.array(value)
        if value.shape[0] == self.training_inputs.shape[0]:
            if value.ndim in [1, 2]:
                self._training_outputs = value
            else:
                raise ValueError("The outputs given in training_outputs " +\
                    "were not 1 dimensional arrays.")
        else:
            raise ValueError("The number of outputs in the given " +\
                "training_outputs was not the same as the number of inputs " +\
                "in the given training_inputs.")
    
    @property
    def parameters(self):
        """
        Property storing the list of string names of parameters described by
        the training set input points.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters referenced before it was set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the names of the input parameters.
        
        value: sequence of strings of length input_dimension
        """
        if type(value) in sequence_types:
            if len(value) == self.training_inputs.shape[1]:
                if all([isinstance(element, basestring) for element in value]):
                    self._parameters = [element for element in value]
                else:
                    raise TypeError("At least one of the given parameters " +\
                        "was not a string.")
            else:
                raise ValueError("The number of parameter names given was " +\
                    "not the same as the number of parameters given in " +\
                    "training_inputs.")
        else:
            raise TypeError("parameters was set to a non-sequence.")
    
    @property
    def compressed(self):
        """
        Boolean determining whether training set output points should be
        compressed before interpolation. This will reduce computation and
        memory usage but may lose information.
        """
        if not hasattr(self, '_compressed'):
            raise AttributeError("compressed was referenced before it was " +\
                "set.")
        return self._compressed
    
    @compressed.setter
    def compressed(self, value):
        """
        Setter for the compressed property which determines whether training
        set output points should be compressed before interpolation.
        
        value: either True or False
        """
        self._compressed = bool(value)
    
    @property
    def expander(self):
        """
        Property storing the expander which expands the training output curves
        to the space of the data which will be fit with this
        InterpolatingLeastSquareFitter.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Setter for the expander which expands the training output curves into
        the space of the data which will be fit with this fitter.
        
        value: must be an Expander object
        """
        if type(value) is type(None):
            self._expander = NullExpander()
        elif isinstance(value, Expander):
            self._expander = value
        else:
            raise TypeError("expander was set to a non-Expander object.")
    
    @property
    def input_dimension(self):
        """
        Property storing the integer dimension of the input points in the
        training set.
        """
        if not hasattr(self, '_input_dimension'):
            self._input_dimension = self.training_inputs.shape[1]
        return self._input_dimension
    
    @property
    def num_basis_vectors(self):
        """
        Property storing the number of basis vectors to use if compression is
        used. If should_compress is False, this property may return None.
        """
        if not hasattr(self, '_num_basis_vectors'):
            raise AttributeError("num_basis_vectors referenced before it " +\
                "was set.")
        return self._num_basis_vectors
    
    @num_basis_vectors.setter
    def num_basis_vectors(self, value):
        """
        Setter for the number of basis vectors to use in compression
        
        value: if None, not used (only allowed when should_compress is True)
               otherwise, must be a positive integer
        """
        if type(value) is type(None):
            self._num_basis_vectors = None
        elif type(value) in int_types:
            if value > 0:
                self._num_basis_vectors = value
            else:
                raise ValueError("num_basis_vectors was set to a " +\
                    "non-positive int.")
        else:
            raise TypeError("num_basis_vectors was set to neither None nor " +\
                "an int.")
    
    @property
    def transform_list(self):
        """
        Property storing the TransformList containing transforms with which to
        define the interpolated input space.
        """
        if not hasattr(self, '_transform_list'):
            raise AttributeError("transform_list referenced before it was " +\
                "set.")
        return self._transform_list
    
    @transform_list.setter
    def transform_list(self, value):
        """
        Setter for the TransformList with which to define the interpolated
        input space.
        
        value: if sequence of transforms, must have length input_dimension and
                                          every element must be castable to a
                                          transform
               if castable to a transform, that transform is assumed to apply
                                           to all parameters
        """
        self._transform_list =\
            TransformList.cast(value, num_transforms=len(self.parameters))
    
    @property
    def loglikelihood_callable(self):
        """
        Property storing the function to apply to loglikelihoods before passing
        them to LeastSquareFitter initializer when performing individual fits
        (or None if none exists).
        """
        if not hasattr(self, '_loglikelihood_callable'):
            raise AttributeError("loglikelihood_callable was referenced " +\
                "before it was set.")
        return self._loglikelihood_callable
    
    @loglikelihood_callable.setter
    def loglikelihood_callable(self, value):
        """
        Setter for the loglikelihood_callable.
        
        value: callable to call on loglikelihood before it is passed to a
               LeastSquareFitter. Can be None if no such function exists
        """
        if (type(value) is type(None)) or callable(value):
            self._loglikelihood_callable = value
        else:
            raise TypeError("loglikelihood_callable was neither None nor a " +\
                "callable.")
    
    def fit(self, data, iterations=1):
        """
        Fits the given data curve with an InputInterpolatedModel created using
        training set examples near the data.
        
        data: data curve to fit
        iterations: the number of iterations to run the least square fitter
        
        returns: a LeastSquareFitter object that has been run for the given
                 number of iterations.
        """
        try:
            if data.shape != self.error.shape:
                raise ValueError("data wasn't of the same shape as error.")
        except AttributeError:
            raise TypeError("data wasn't a numpy.ndarray object.")
        diffs = np.sum(((self.training_outputs - data[np.newaxis,:]) /\
            self.error) ** 2, axis=1)
        argsort = np.argsort(diffs)
        truncated_training_inputs =\
            self.training_inputs[argsort,:][:self.ntrain,:]
        truncated_training_outputs =\
            self.training_outputs[argsort,:][:self.ntrain,:]
        interpolated_model =\
            InputInterpolatedModel(self.parameters, truncated_training_inputs,\
            truncated_training_outputs, should_compress=self.compressed,\
            transform_list=self.transform_list,\
            scale_to_cube=self.scale_to_cube,\
            num_basis_vectors=self.num_basis_vectors, expander=self.expander,\
            error=self.error, interpolation_method=self.interpolation_method)
        loglikelihood =\
            GaussianLoglikelihood(data, self.error, interpolated_model)
        if type(self.loglikelihood_callable) is not type(None):
            loglikelihood = self.loglikelihood_callable(loglikelihood)
        least_square_fitter = LeastSquareFitter(loglikelihood,\
            interpolated_model.prior_set, transform_list=self.transform_list)
        least_square_fitter.run(iterations=iterations)
        return least_square_fitter
    
    def __call__(self, data, iterations=1):
        """
        Alias for the fit function.
        
        data: data curve to fit
        iterations: the number of iterations to run the least square fitter
        
        returns: a LeastSquareFitter object which have been run for the given
                 number of iterations
        """
        return self.fit(data, iterations=iterations)

