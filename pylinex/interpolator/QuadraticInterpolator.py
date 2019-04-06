"""
File: pylinex/util/QuadraticInterpolator.py
Author: Keith Tauscher
Date: 21 Jan 2018

Description: File containing class which performs many-dimensional quadratic
             interpolation with the aid of a Delaunay mesh.
"""
import numpy as np
import numpy.linalg as la
from distpy import UniformTriangulationDistribution
from .Interpolator import Interpolator

class QuadraticInterpolator(Interpolator):
    """
    Class which performs many-dimensional quadratic interpolation with the aid
    of a Delaunay mesh.
    """
    def find_nearest_points(self, point):
        """
        Finds the nearest points
        
        point: 1D numpy.ndarray of parameters
        
        returns: tuple of form (points, values) where points is a 2D
                 numpy.ndarray of shape (npoints, input_dimension) and values
                 is a 2D numpy.ndarray of shape (npoints, output_dimension),
                 and npoints is (input_dimension+1)(input_dimension+2)/2
        """
        distances = np.sum((self.inputs - point[np.newaxis,:]) ** 2, axis=1)
        npoints =\
            ((self.input_dimension + 1) * (self.input_dimension + 2) // 2)
        points = np.argsort(distances)[:npoints]
        values = self.outputs[points,:]
        points = self.inputs[points,:]
        return (points, values)
    
    def value_gradient_and_hessian(self, point, transformed_space=False):
        """
        Computes the value, its gradient, and its hessian, of this
        interpolator.
        
        point: 1D numpy.ndarray of parameters
        transformed_space: if True, derivatives returned in transformed space
                           if False, derivatives returned in real space
        
        returns: tuple of (value, gradient, hessian)
        """
        original_point = point.copy()
        point = self.combined_transform_list.apply(point, axis=0)
        (points, values) = self.find_nearest_points(point)
        npoints =\
            (((self.input_dimension + 1) * (self.input_dimension + 2)) // 2)
        points = points - point[np.newaxis,:]
        matrix = points ** 2
        for index in range(self.input_dimension):
            matrix = np.concatenate((matrix,\
                points[:,index,np.newaxis] * points[:,index+1:]), axis=1)
        for index in range(self.input_dimension):
            matrix = np.concatenate((matrix, points[:,index:index+1]), axis=1)
        matrix = np.concatenate((matrix, np.ones((npoints, 1))), axis=1)
        try:
            matrix = la.inv(matrix)
        except KeyboardInterrupt:
            raise
        except:
            print('matrix={}'.format(matrix))
            raise
        if np.any(np.isnan(matrix)):
            raise ValueError("points were not well positioned (not quite " +\
                "sure why this happens). I have found that it happens less " +\
                "when there are more points given to the interpolator.")
        coefficients = np.dot(matrix, values)
        value = coefficients[-1,:]
        gradient = coefficients[-2:-(self.input_dimension+2):-1,:].T
        hessian_shape =\
            (self.output_dimension,) + ((self.input_dimension,) * 2)
        hessian = np.ndarray(hessian_shape)
        for index in range(self.input_dimension):
            min_index = self.input_dimension * (index + 1)
            min_index = min_index - ((index * (index + 1)) // 2)
            num_terms = self.input_dimension - 1 - index
            vector_slice = slice(min_index, min_index + num_terms)
            vector_piece = coefficients[vector_slice,:].T
            hessian[:,index,index] = (2 * coefficients[index,:])
            hessian[:,index,index+1:] = vector_piece
            hessian[:,index+1:,index] = vector_piece
        hessian[:,-1,-1] = (2 * coefficients[self.input_dimension-1,:])
        if transformed_space:
            (gradient, hessian) =\
                self.scaling_transform_list.detransform_derivatives(\
                (gradient, hessian), original_point, axis=1)
        else:
            (gradient, hessian) =\
                self.combined_transform_list.detransform_derivatives(\
                (gradient, hessian), original_point, axis=1)
        if self.output_dimension == 1:
            value = value[0]
            gradient = gradient[0]
            hessian = hessian[0]
        return (value, gradient, hessian)

