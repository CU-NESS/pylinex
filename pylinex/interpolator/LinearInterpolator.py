"""
File: pylinex/util/LinearInterpolator.py
Author: Keith Tauscher
Date: 19 Apr 2019

Description: File containing class which performs many-dimensional linear
             interpolation. This is for use when Delaunay meshes of input space
             are computationally prohibitive (otherwise, you should use the
             DelaunayLinearInterpolator class).
"""
import numpy as np
import numpy.linalg as la
from .Interpolator import Interpolator

class LinearInterpolator(Interpolator):
    """
    Class which performs many-dimensional linear interpolation with the aid of
    a Delaunay mesh.
    """
    def value_gradient_and_hessian(self, point, transformed_space=False):
        """
        Computes both the interpolated value and the gradient at the given
        point.
        
        point: 1D numpy.ndarray of parameter values of shape (input_dimension,)
        transformed_space: if True, the gradient defined in the transformed
                                    space is returned.
                           otherwise, the gradient in the input space is
                                      returned (default: False)
        
        returns: tuple of numpy.ndarrays (value, gradient) where value has
                 shape (output_dimension,) and gradient has shape
                 (output_dimension, input_dimension)
        """
        original_point = point
        point = self.combined_transform_list.apply(point, axis=0)
        npoints = self.input_dimension + 1
        (points, values) = self.find_nearest_points(point, npoints)
        if self.save_memory:
            gradient =\
                np.ndarray((self.output_dimension, self.input_dimension))
            for ioutput in range(self.output_dimension):
                augmented_points = np.concatenate(\
                    (points, values[:,ioutput,np.newaxis]), axis=1)
                augmented_points -= augmented_points[:1,:]
                coefficients = la.svd(augmented_points)[2][-1] # last row of Vt
                gradient[ioutput] = (coefficients[:-1] / (-coefficients[-1]))
        else:
            augmented_points = np.zeros(\
                (self.output_dimension,) + ((self.input_dimension + 1,) * 2))
            augmented_points[:,:,:-1] += points[np.newaxis,:,:]
            augmented_points[:,:,-1] += values.T
            augmented_points -= augmented_points[:,:1,:]
            coefficients = la.svd(augmented_points)[2][:,-1,:]
            gradient = coefficients[:,:-1] / (-coefficients[:,-1:])
        value = values[-1] + np.dot(gradient, point - points[-1])
        hessian =\
            np.zeros((self.output_dimension,) + ((self.input_dimension,) * 2))
        if transformed_space:
            # no need to change hessian. It will be zero in transformed space
            gradient = self.scaling_transform_list.detransform_gradient(\
                gradient, original_point, axis=1)
        else:
            (gradient, hessian) =\
                self.combined_transform_list.detransform_derivatives(\
                (gradient, hessian), original_point, axis=1)
        if self.output_dimension == 1:
            value = value[0]
            gradient = gradient[0]
            hessian = hessian[0]
        return (value, gradient, hessian)

