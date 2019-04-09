"""
File: pylinex/util/Interpolator.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing class which performs many-dimensional
             interpolation with the aid of a Delaunay mesh.
"""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from distpy import AffineTransform, cast_to_transform_list, TransformList,\
    UniformTriangulationDistribution
from ..util import sequence_types

shouldnt_instantiate_interpolator_error = NotImplementedError(\
    "Interpolator class should not be directly instantiated.")

class Interpolator(object):
    """
    Class which performs many-dimensional interpolation with the aid of a
    Delaunay mesh.
    """
    def __init__(self, inputs, outputs, transform_list=None,\
        scale_to_cube=False, save_memory=False, initialize_prior=False):
        """
        transform_list: TransformList to apply to the variables before passing
                        them to the Delaunay module in scipy
        save_memory: if True, slower computation will be used in interpolations
                              so that less RAM is used. This is really only
                              necessary on a memory-taxed machine or if the
                              output dimension is high (default False).
        """
        self.inputs = inputs
        self.transform_list = transform_list
        self._transform_inputs()
        self.scale_to_cube = scale_to_cube
        self.scale_inputs_to_cube()
        self.outputs = outputs
        self.save_memory = save_memory
        # computing self.prior initializes UniformPrior over (transformed and
        # scaled to cube) inputs and, in the process, the Delauanay mesh
        # underlying the interpolation scheme
        if initialize_prior:
            self.prior
    
    def _transform_inputs(self):
        """
        Transforms the current inputs of the interpolator (WARNING: Do not call
        this function twice! It will doubly transform the inputs).
        """
        self._inputs = self.transform_list.apply(self._inputs, axis=1)
    
    def scale_inputs_to_cube(self):
        """
        Scales the current inputs to the unit cube.
        """
        if self.scale_to_cube:
            ranges = np.ndarray((self.input_dimension, 3))
            ranges[:,0] = np.min(self._inputs, axis=0)
            ranges[:,1] = np.max(self._inputs, axis=0)
            ranges[:,2] = (ranges[:,1] - ranges[:,0])
            self._inputs = ((self._inputs - ranges[np.newaxis,:,0]) /\
                ranges[np.newaxis,:,2])
            self._ranges = ranges
    
    @property
    def save_memory(self):
        """
        Property storing a boolean switch between two modes in which the CPU is
        relatively memory-starved of process-starved.
        """
        if not hasattr(self, '_save_memory'):
            raise AttributeError("save_memory referenced before it was set.")
        return self._save_memory
    
    @save_memory.setter
    def save_memory(self, value):
        """
        Setter for the save_memory switch.
        
        value: either True or False
        """
        try:
            self._save_memory = bool(value)
        except:
            raise TypeError("save_memory couldn't be interpreted as a bool.")
    
    @property
    def delaunay(self):
        """
        Property storing the scipy.spatial.Delaunay object storing the Delaunay
        mesh at the heart of this Interpolator.
        """
        if not hasattr(self, '_delaunay'):
            self._delaunay = Delaunay(self.inputs)
        return self._delaunay
    
    @property
    def convex_hull_delaunay(self):
        """
        Property storing the scipy.spatial.ConvexHull object storing the
        ConvexHull.
        """
        if not hasattr(self, '_convex_hull_delaunay'):
            convex_hull = ConvexHull(self.inputs)
            convex_hull_vertices = convex_hull.points[convex_hull.vertices]
            print("convex_hull_vertices.shape={}".format(\
                convex_hull_vertices.shape))
            self._convex_hull_delaunay = Delaunay(convex_hull_vertices)
        return self._convex_hull_delaunay
    
    @property
    def inputs(self):
        """
        Property storing the inputs to the interpolator.
        """
        if not hasattr(self, '_inputs'):
            raise AttributeError("inputs was referenced before it was set.")
        return self._inputs
    
    @inputs.setter
    def inputs(self, value):
        """
        Setter for the inputs to the interpolator.
        
        value: a 2D numpy.ndarray of shape (ntrain, input_dimension)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 2:
                if value.shape[0] > value.shape[1]:
                    self._inputs = value
                else:
                    raise ValueError("Fewer training examples were given " +\
                        "than the dimension of the input!")
            else:
                raise TypeError("inputs was set to an array which wasn't 2D.")
        else:
            raise TypeError("inputs was set to a non-sequence.")
    
    @property
    def transform_list(self):
        """
        Property storing a TransformList object which applies to the parameters
        of the inputs to transform into interpolation space.
        """
        if not hasattr(self, '_transform_list'):
            raise AttributeError("transform_list referenced before it was set.")
        return self._transform_list
    
    @transform_list.setter
    def transform_list(self, value):
        """
        Setter for the TransformList object which applies to the parameters of
        the inputs to transform into interpolation space.
        
        value: sequence of Transform objects (or things which can be cast to
               Transform objects)
        """
        self._transform_list =\
            cast_to_transform_list(value, num_transforms=self.input_dimension)
    
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
    def scaling_transform_list(self):
        """
        Property storing a TransformList object filled with Transform objects
        which perform the scale to cube operation. If scale_to_cube is set to
        False, then this will be a TransformList object full of NullTransform
        objects.
        """
        if not hasattr(self, '_scaling_transform_list'):
            if self.scale_to_cube:
                transforms = []
                for itransform in range(len(self.transform_list)):
                    (minimum, maximum, breadth) = self.ranges[itransform]
                    scale_factor = (1. / breadth)
                    translation = (((-1.) * minimum) / breadth)
                    transforms.append(\
                        AffineTransform(scale_factor, translation))
                self._scaling_transform_list = TransformList(*transforms)
            else:
                self._scaling_transform_list =\
                    TransformList(*([None] * len(self.transform_list)))
        return self._scaling_transform_list
    
    @property
    def combined_transform_list(self):
        """
        Property storing the sequence of combined transforms which apply to
        each parameter. Essentially, each of these transforms is the scale to
        cube transform combined with the transform given in the initializer.
        """
        if not hasattr(self, '_combined_transform_list'):
            self._combined_transform_list =\
                self.transform_list * self.scaling_transform_list
        return self._combined_transform_list
    
    @property
    def ranges(self):
        """
        Property storing the minimum, maximum, and distance between the minimum
        and maximum for each input parameter when scaling to the cube.
        """
        if not hasattr(self, '_ranges'):
            raise AttributeError("ranges referenced before it was set.")
        return self._ranges
    
    @property
    def num_inputs(self):
        """
        Property storing the integer number of training set points.
        """
        if not hasattr(self, '_num_inputs'):
            self._num_inputs = self.inputs.shape[0]
        return self._num_inputs
    
    @property
    def input_dimension(self):
        """
        Property storing the input dimension of the input space of the
        interpolator.
        """
        if not hasattr(self, '_input_dimension'):
            self._input_dimension = self.inputs.shape[1]
        return self._input_dimension
    
    @property
    def outputs(self):
        """
        Property storing the training set outputs of this interpolator in a 2D
        numpy.ndarray of shape (num_inputs, output_dimension).
        """
        if not hasattr(self, '_outputs'):
            raise AttributeError("outputs referenced before it was set.")
        return self._outputs
    
    @outputs.setter
    def outputs(self, value):
        """
        Setter for the training set outputs of this interpolator.
        
        value: 2D numpy.ndarray of shape (num_inputs, output_dimension)
        """
        value = np.array(value)
        if value.ndim == 1:
            value = value[:,np.newaxis]
        if value.ndim == 2:
            if value.shape[0] == self.num_inputs:
                self._outputs = value
            else:
                raise ValueError("The number of outputs given was not the " +\
                    "same as the number of inputs given.")
        else:
            raise ValueError("outputs was set to something other than an " +\
                "array which was 2D.")
    
    @property
    def output_dimension(self):
        """
        Property storing the dimension of the output space of this
        interpolator.
        """
        if not hasattr(self, '_output_dimension'):
            self._output_dimension = self.outputs.shape[1]
        return self._output_dimension
    
    @property
    def prior(self):
        """
        Property storing a Distribution object capable of drawing uniformly
        from the convex hull of points, i.e. the points in this interpolator
        which aren't on the convex hull of the points of the delaunay.
        """
        if not hasattr(self, '_prior'):
            self._prior =\
                UniformTriangulationDistribution(self.convex_hull_delaunay)
        return self._prior
    
    def __call__(self, point):
        """
        Evaluates the interpolation approximation at the given parameters.
        
        point: 1D numpy.ndarray of parameters of shape (input_dimension,)
        
        returns: 1D numpy.ndarray of shape (output_dimension,)
        """
        return\
            self.value_gradient_and_hessian(point, transformed_space=False)[0]
    
    def gradient(self, point, transformed_space=False):
        """
        Evaluates the gradient of the interpolation approximation at the given
        parameters.
        
        point: 1D numpy.ndarray of parameters of shape (input_dimension,)
        transformed_space: if True, the gradient defined in the transformed
                                    space is returned.
                           otherwise, the gradient in the input space is
                                      returned (default: False)
        
        returns: 1D numpy.ndarray of shape (output_dimension, input_dimension)
        """
        return self.value_gradient_and_hessian(point,\
            transformed_space=transformed_space)[1]
    
    def hessian(self, point, transformed_space=False):
        """
        Computes the matrix of second partial derivatives of the interpolated
        quantity at the given point.
        
        point: 1D numpy.ndarray of parameters
        transformed_space: if True, the gradient defined in the transformed
                                    space is returned.
                           otherwise, the gradient in the input space is
                                      returned (default: False)
        
        returns: 3D numpy.ndarray of shape (num_quant, num_pars, num_pars) or
                 2D numpy.ndarray of shape (num_pars, num_pars)
        """
        return self.value_gradient_and_hessian(point,\
            transformed_space=transformed_space)[2]
    
    def value_gradient_and_hessian(self, point, transformed_space=False):
        """
        Calls the interpolator at the given parameter values.
        
        point: 1D numpy.ndarray of length input_dimension
        transformed_space: if True, the gradient defined in the transformed
                                    space is returned.
                           otherwise, the gradient in the input space is
                                      returned (default: False)
        
        returns: tuple of form (value, gradient, hessian) where value is 1D or
                 0D, gradient is 2D or 1D, and hessian is 3D or 2D (these
                 possibilities depend on whether output_dimension is 1 or not)
        """
        raise shouldnt_instantiate_interpolator_error

