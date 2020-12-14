"""
File: pylinex/model/InputInterpolatedModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing class representing a model based on
             multidimensional linear interpolation on a Delaunay mesh.
"""
import numpy as np
from distpy import TransformList, DistributionSet
from ..util import int_types, sequence_types, create_hdf5_dataset,\
    get_hdf5_value
from ..interpolator import LinearInterpolator, QuadraticInterpolator,\
    DelaunayLinearInterpolator
from ..expander import Expander, NullExpander, load_expander_from_hdf5_group
from ..basis import TrainedBasis, effective_training_set_rank
from .Model import Model

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class InputInterpolatedModel(Model):
    """
    Class representing a model based on multidimensional linear interpolation
    on a Delaunay mesh.
    """
    def __init__(self, parameter_names, training_inputs, training_outputs,\
        should_compress=False, transform_list=None, scale_to_cube=False,\
        num_basis_vectors=None, expander=None, error=None,\
        interpolation_method='linear'):
        """
        Initializes a new model based on interpolation using a Delaunay mesh.
        If compression is used, the mean of the training outputs is not
        subtracted before SVD is taken.
        
        parameter_names: sequence of names of parameters (of length
                         input_dimension)
        training_inputs: 2D numpy.ndarray of shape
                         (num_elements, input_dimension)
        training_outputs: 2D numpy.ndarray of shape
                          (num_elements, output_dimension)
        should_compress: boolean determining whether the training_outputs
                         should be compressed by ignoring unimportant SVD
                         modes. If True, leads to some loss of information.
        transform_list: TransformList object which should be used in defining
                        the interpolation space. If None, no transforms are
                        applied to any variables.
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
        error: the error used to define the inner product when performing SVD
               (only used if should_compress is True)
        interpolation_method: either 'linear' or 'quadratic'
        """
        # the order of these things is very important
        self.expander = expander
        self.compressed = should_compress
        self.training_inputs = training_inputs
        self.scale_to_cube = scale_to_cube
        self.parameters = parameter_names
        self.transform_list = transform_list
        self.error = error
        self.num_basis_vectors = num_basis_vectors
        self.training_outputs = training_outputs
        self.interpolation_method = interpolation_method
        self.interpolator
    
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
        if value in ['delaunay_linear', 'linear', 'quadratic']:
            self._interpolation_method = value
        else:
            raise ValueError("interpolation_method was neither linear nor " +\
                "quadratic.")
    
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
    def transform_list(self):
        """
        Property storing the TransformList storing the Transform objects which
        define the interpolated input space.
        """
        if not hasattr(self, '_transform_list'):
            raise AttributeError("transform_list referenced before it was set.")
        return self._transform_list
    
    @transform_list.setter
    def transform_list(self, value):
        """
        Setter for the TransformList storing the Transform objects which define
        the interpolated input space.
        
        value: if TransformList, must have length input_dimension
               if sequence of transforms, must have length input_dimension and
                                          every element must be castable to a
                                          transform
               if castable to a transform, that transform is assumed to apply
                                           to all parameters
        """
        self._transform_list =\
            TransformList.cast(value, num_transforms=len(self.parameters))
    
    @property
    def error(self):
        """
        Property storing the 1D numpy.ndarray error which defines the inner
        product.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error which defines the inner product.
        
        value: 1D numpy.ndarray of same length as expanded training output
        """
        if type(value) is type(None):
            self._error = None
        else:
            value = np.array(value)
            if value.ndim == 1:
                self._error = value
            else:
                raise ValueError("error was set to an array which wasn't 1D.")
    
    @property
    def expander(self):
        """
        Property storing the Expander object which connects the space of
        training_outputs to the output space of this model.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Setter for the Expander object which connects the space of the
        training_outputs to the output space of this model.
        
        value: must be either None or an Expander object
        """
        if type(value) is type(None):
            self._expander = NullExpander()
        elif isinstance(value, Expander):
            self._expander = value
        else:
            raise TypeError("expander was set to a non-Expander object.")
    
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
    def ntrain(self):
        """
        Property storing the integer number of training set examples taken into
        account in this model.
        """
        if not hasattr(self, '_ntrain'):
            self._ntrain = self.training_inputs.shape[0]
        return self._ntrain
    
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
        if value.shape[0] == self.ntrain:
            if value.ndim in [1, 2]:
                self._training_outputs = value
                self.compressed = (self.compressed and (value.ndim == 2))
            else:
                raise ValueError("The outputs given in training_outputs " +\
                    "were not 1 dimensional arrays.")
        else:
            raise ValueError("The number of outputs in the given " +\
                "training_outputs was not the same as the number of inputs " +\
                "in the given training_inputs.")
        if self.compressed:
            if type(self.num_basis_vectors) is type(None):
                self.num_basis_vectors = effective_training_set_rank(\
                    self._training_outputs, self.error,\
                    mean_translation=False, method='abs',\
                    number_of_modes_to_consider=None, level=0.1)
            elif self.num_basis_vectors > self._training_outputs.shape[1]:
                raise ValueError("The given number of basis vectors " +\
                    "implies no compression! This doesn't make sense. You " +\
                    "might as well just set should_compress to False.")
            self._trained_basis = TrainedBasis(self._training_outputs,\
                self.num_basis_vectors, error=self.error,\
                expander=self.expander, mean_translation=False)
            self._training_outputs =\
                self.trained_basis.training_set_fit_coefficients
    
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
    def trained_basis(self):
        """
        Property storing the TrainedBasis object which computes the SVD modes
        used for compression.
        """
        if not hasattr(self, '_trained_basis'):
            raise AttributeError("trained_basis should be set " +\
                "automatically in __init__. If it was not, then it is " +\
                "either because no compression through SVD is used or " +\
                "because the outputs given were 1D.")
        return self._trained_basis
    
    @property
    def interpolator(self):
        """
        Property storing the Interpolator object which performs the actual
        interpolation at the heart of this model.
        """
        if not hasattr(self, '_interpolator'):
            if self.interpolation_method == 'delaunay_linear':
                self._interpolator =\
                    DelaunayLinearInterpolator(self.training_inputs,\
                    self.training_outputs, transform_list=self.transform_list,\
                    scale_to_cube=self.scale_to_cube)
            elif self.interpolation_method == 'linear':
                self._interpolator = LinearInterpolator(self.training_inputs,\
                    self.training_outputs, transform_list=self.transform_list,\
                    scale_to_cube=self.scale_to_cube)
            else: # self.interpolation_method == 'quadratic'
                self._interpolator = QuadraticInterpolator(\
                    self.training_inputs, self.training_outputs,\
                    transform_list=self.transform_list,\
                    scale_to_cube=self.scale_to_cube)
        return self._interpolator
    
    @property
    def prior_set(self):
        """
        Property storing the DistributionSet which is capable of drawing
        uniformly from interpolation space.
        """
        if not hasattr(self, '_prior_set'):
            self._prior_set = DistributionSet()
            self._prior_set.add_distribution(self.interpolator.prior,\
                self.parameters,\
                self.interpolator.combined_transform_list.transforms)
        return self._prior_set
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in outputs of this model
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.expander.expanded_space_size(\
                self.training_outputs.shape[-1])
        return self._num_channels
    
    def __call__(self, pars):
        """
        Evaluates the model at the given parameters by using the interpolator.
        
        pars: 1D numpy.ndarray of parameter values
        
        returns: 1D numpy.ndarray of model values of shape (num_channels,)
        """
        interpolated = self.interpolator(pars)
        if self.compressed:
            interpolated = np.dot(self.trained_basis.basis.T, interpolated)
        return self.expander(interpolated)
    
    @property
    def gradient_computable(self):
        """
        The gradient of the InputInterpolatedModel is computable because the
        gradient is estimated at each simplex of input points.
        """
        return True
    
    def gradient(self, pars):
        """
        Evaluates the gradient of this model at the given parameter values.
        
        pars: 1D numpy.ndarray of parameter values
        
        returns: 2D numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        interpolated = self.interpolator.gradient(pars).T
        if self.compressed:
            interpolated = np.dot(interpolated, self.trained_basis.basis)
        return self.expander(interpolated).T
    
    @property
    def hessian_computable(self):
        """
        At this point, the hessian of InputInterpolatedModel objects has not
        been implemented. It may be at some point in the future.
        """
        return True
    
    def hessian(self, pars):
        """
        Evaluates the hessian of this model at the given parameter values
        
        pars: 1D numpy.ndarray of parameter values
        
        returns: nothing yet because the hessian is not computable
        """
        interpolated = self.interpolator.hessian(pars)
        if self.compressed:
            #interpolated = np.einsum('ab,acd->bcd',\
            #    self.trained_basis.basis, interpolated)
            interpolated = np.sum(\
                self.trained_basis.basis[:,:,np.newaxis,np.newaxis] *\
                interpolated[:,np.newaxis,:,:], axis=0)
        interpolated = np.reshape(interpolated, (-1, self.num_parameters ** 2))
        interpolated = self.expander(interpolated)
        return np.reshape(interpolated, (-1,) + ((self.num_parameters,) * 2))
    
    def fill_hdf5_group(self, group, training_inputs_link=None,\
        training_outputs_link=None):
        """
        Fills the given hdf5 file group with information about this model. It
        isn't implemented yet, though.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'InputInterpolatedModel'
        group.attrs['parameter_names'] = self.parameters
        create_hdf5_dataset(group, 'training_inputs',\
            data=self.training_inputs, link=training_inputs_link)
        create_hdf5_dataset(group, 'training_outputs',\
            data=self.training_outputs, link=training_outputs_link)
        group.attrs['should_compress'] = self.compressed
        self.transform_list.fill_hdf5_group(\
            group.create_group('transform_list'))
        group.attrs['scale_to_cube'] = self.scale_to_cube
        if type(self.num_basis_vectors) is not type(None):
            group.attrs['num_basis_vectors'] = self.num_basis_vectors
        self.expander.fill_hdf5_group(group.create_group('expander'))
        if type(self.error) is not type(None):
            group.attrs['error'] = self.error
        group.attrs['interpolation_method'] = self.interpolation_method
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an InputInterpolatedModel from the given hdf5 group.
        
        group: hdf5 file group which has had an InputInterpolatedModel saved to
               it
        """
        parameter_names = [name for name in group.attrs['parameter_names']]
        training_inputs = get_hdf5_value(group['training_inputs'])
        training_outputs = get_hdf5_value(group['training_outputs'])
        should_compress = group.attrs['should_compress']
        transform_list =\
            TransformList.load_from_hdf5_group(group['transform_list'])
        scale_to_cube = group.attrs['scale_to_cube']
        if 'num_basis_vectors' in group.attrs:
            num_basis_vectors = group.attrs['num_basis_vectors']
        else:
            num_basis_vectors = None
        expander = load_expander_from_hdf5_group(group['expander'])
        if 'error' in group.attrs:
            error = group.attrs['error']
        else:
            error = None
        interpolation_method = group.attrs['interpolation_method']
        return InputInterpolatedModel(parameter_names, training_inputs,\
            training_outputs, should_compress=should_compress,\
            transform_list=transform_list, scale_to_cube=scale_to_cube,\
            num_basis_vectors=num_basis_vectors, expander=expander,\
            error=error, interpolation_method=interpolation_method)
    
    def __eq__(self, other):
        """
        Checks if other is essentially equivalent to this
        InputInterpolatedModel.
        
        other: the object to check for equality
        
        returns: False unless other is an InputInterpolatedModel. In that case,
                 it throws a NotImplementedError.
        """
        if not isinstance(other, InputInterpolatedModel):
            return False
        if self.scale_to_cube != other.scale_to_cube:
            return False
        if self.compressed != other.compressed:
            return False
        elif self.compressed and\
            (self.num_basis_vectors != other.num_basis_vectors):
            return False
        if self.interpolation_method != other.interpolation_method:
            return False
        if type(self.error) is type(None):
            if type(other.error) is not type(None):
                return False
        elif np.any(self.error != other.error):
            return False
        if self.transform_list != other.transform_list:
            return False
        if self.expander != other.expander:
            return False
        if self.parameters != other.parameters:
            return False
        if self.training_inputs.shape == other.training_inputs.shape:
            if np.any(self.training_inputs != other.training_inputs):
                return False
        else:
            return False
        if self.training_outputs.shape == other.training_outputs.shape:
            if np.any(self.training_outputs != other.training_outputs):
                return False
        else:
            return False
        return True
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of he underlying model, the bounds are
        passed through with no changes.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            minima = np.min(self.training_inputs, axis=0)
            maxima = np.max(self.training_inputs, axis=0)
            for (name, minimum, maximum) in\
                zip(self.parameters, minima, maxima):
                self._bounds[name] = (minimum, maximum)
        return self._bounds

