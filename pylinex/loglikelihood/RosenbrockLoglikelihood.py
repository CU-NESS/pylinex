"""
File: pylinex/loglikelihood/RosenbrockLoglikelihood.py
Author: Keith Tauscher
Date: 12 Dec 2018

Description: File containing a class representing a Likelihood which does not
             come from a model with parameters and a data vector, but instead
             comes from the Rosenbrock function of two variables.
"""
import numpy as np
from distpy import cast_to_transform_list
from ..util import real_numerical_types
from .Loglikelihood import Loglikelihood

class RosenbrockLoglikelihood(Loglikelihood):
    """
    Class representing a likelihood whose log value is proportional to the
    negative of the Rosenbrock function.
    """
    def __init__(self, xmin=1, scale_ratio=100, overall_scale=1):
        """
        Initializes a new Rosenbrock function based loglikelihood with the
        given parameters.
        
        xmin: the minimum when this is the 2D Rosenbrock function is (a,a^2)
        scale_ratio: the ratio of the coefficient of the (y-x^2)^2 term to the
                     coefficient of the (a-x)^2 term
        overall_scale: 
        """
        self.xmin = xmin
        self.scale_ratio = scale_ratio
        self.overall_scale = overall_scale
    
    @property
    def xmin(self):
        """
        Property storing the x coordinate of the if/when this likelihood has 2
        parameters.
        """
        if not hasattr(self, '_xmin'):
            raise AttributeError("xmin was referenced before it was set.")
        return self._xmin
    
    @xmin.setter
    def xmin(self, value):
        """
        Setter for the xmin property.
        
        value: a positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._xmin = value
            else:
                raise ValueError("xmin was set to a non-positive number.")
        else:
            raise TypeError("xmin was set to a non-number.")
    
    @property
    def overall_scale(self):
        """
        Property storing the scale ratio of the two terms of this loglikelihood
        """
        if not hasattr(self, '_overall_scale'):
            raise AttributeError("overall_scale was referenced before it " +\
                "was set.")
        return self._overall_scale
    
    @overall_scale.setter
    def overall_scale(self, value):
        """
        Setter for the overall_scale property.
        
        value: a positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._overall_scale = value
            else:
                raise ValueError("overall_scale was set to a non-positive " +\
                    "number.")
        else:
            raise TypeError("overall_scale was set to a non-number.")
    
    @property
    def scale_ratio(self):
        """
        Property storing the scale ratio of the two terms of this loglikelihood
        """
        if not hasattr(self, '_scale_ratio'):
            raise AttributeError("scale_ratio was referenced before it was " +\
                "set.")
        return self._scale_ratio
    
    @scale_ratio.setter
    def scale_ratio(self, value):
        """
        Setter for the scale_ratio property.
        
        value: a positive number
        """
        if type(value) in real_numerical_types:
            if value > 0:
                self._scale_ratio = value
            else:
                raise ValueError("scale_ratio was set to a non-positive " +\
                    "number.")
        else:
            raise TypeError("scale_ratio was set to a non-number.")
    
    @property
    def data(self):
        """
        Since the Rosenbrock likelihood is a simple function of the parameters
        instead of a modeled data vector, it has no data property.
        """
        raise AttributeError("The RosenbrockLoglikelihood class does not " +\
            "have a data property.")
    
    @property
    def num_channels(self):
        """
        Since the Rosenbrock likelihood is a simple function of the parameters
        instead of a modeled data vector, it has no num_channels property.
        """
        raise AttributeError("The RosenbrockLoglikelihood class does not " +\
            "have a num_channels property.")
    
    @property
    def degrees_of_freedom(self):
        """
        Since the Rosenbrock likelihood is a simple function of the parameters
        instead of a modeled data vector, it has no degrees_of_freedom
        property.
        """
        raise AttributeError("The RosenbrockLoglikelihood class does not " +\
            "have a degrees_of_freedom property.")
    
    @property
    def parameters(self):
        """
        Property storing the parameters of the likelihood, which are simply
        labeled a0,a1,...aN.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = ['a{:d}'.format(index) for index in range(2)]
        return self._parameters
    
    def __call__(self, parameters, return_negative=False):
        """
        Calls this loglikelihood at the given parameters.
        
        parameters: numpy.ndarray of parameter values
        return_negative: if True (default False), the value returned by this
                         function is multiplied by (-1)
        
        returns: a single number giving the loglikelihood up to a constant
        """
        (a0, a1) = parameters
        value = ((-1) * ((self.xmin - a0) ** 2)) -\
            (self.scale_ratio * ((a1 - (a0 ** 2)) ** 2))
        value = value * self.overall_scale
        if return_negative:
            return (-1) * value
        else:
            return value
    
    @property
    def gradient_computable(self):
        """
        Since this likelihood is formed from a simple function, the gradient
        is computable.
        """
        return True
    
    def gradient(self, parameters, return_negative=False):
        """
        Function which computes the gradient of the Rosenbrock loglikelihood
        at a given parameter vector.
        
        parameters: numpy.ndarray of parameter values
        return_negative: if True (default False), the value returned by this
                         function is multiplied by (-1)
        
        returns: numpy.ndarray of shape (loglikelihood.num_parameters,)
        """
        (a0, a1) = parameters
        tsa1ma02 = (2 * (self.scale_ratio * (a1 - (a0 ** 2))))
        x_part = ((-2) * ((a0 * (1 - tsa1ma02)) - self.xmin))
        y_part = ((-1) * tsa1ma02)
        gradient = np.stack([x_part, y_part], axis=-1)
        gradient = gradient * self.overall_scale
        if return_negative:
            return (-1) * gradient
        else:
            return gradient
    
    @property
    def hessian_computable(self):
        """
        Since this likelihood is formed from a simple function, the hessian
        is computable.
        """
        return True
    
    def hessian(self, parameters, return_negative=False):
        """
        Function which computes the hessian of the Rosenbrock loglikelihood
        at a given parameter vector.
        
        parameters: numpy.ndarray of parameter values
        return_negative: if True (default False), the value returned by this
                         function is multiplied by (-1)
        
        returns: numpy.ndarray of shape
                 (loglikelihood.num_parameters, loglikelihood.num_parameters)
        """
        (a0, a1) = parameters
        xy_part = (4 * self.scale_ratio * a0)
        yy_part = ((-2) * self.scale_ratio)
        xx_part = ((-2) * (1 - (2 * self.scale_ratio * a1) +\
            (6 * self.scale_ratio * (a0 ** 2))))
        hessian = np.stack([np.stack([xx_part, xy_part], axis=0),\
            np.stack([xy_part, yy_part], axis=0)], axis=0)
        hessian = hessian * self.overall_scale
        if return_negative:
            return (-1) * hessian
        else:
            return hessian
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with information about this Loglikelihood.
        
        group: the group to fill with information about this Loglikelihood
        """
        group.attrs['class'] = 'RosenbrockLoglikelihood'
        group.attrs['xmin'] = self.xmin
        group.attrs['scale_ratio'] = self.scale_ratio
        group.attrs['overall_scale'] = self.overall_scale
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Loglikelihood object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a Loglikelihood object
        
        returns: the Loglikelihood object loaded from the given hdf5 file group
        """
        xmin = group.attrs['xmin']
        scale_ratio = group.attrs['scale_ratio']
        overall_scale = group.attrs['overall_scale']
        return RosenbrockLoglikelihood(xmin=xmin, scale_ratio=scale_ratio,\
            overall_scale=overall_scale)
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: a Loglikelihood object to check for equality
        
        returns: True if other and self have the same properties
        """
        if isinstance(other, RosenbrockLoglikelihood):
            return np.allclose(\
                [self.xmin, self.scale_ratio, self.overall_scale],\
                [other.xmin, other.scale_ratio, other.overall_scale],\
                atol=0, rtol=1e-6)
        else:
            return False
    
    def auto_gradient(self, pars, return_negative=False, differences=1e-6,\
        transform_list=None):
        """
        Computes the gradient of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the gradient
        return_negative: if true, the negative of the gradient of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        differences: either single number or 1D array of numbers to use as the
                     numerical difference in parameter. Default: 10^(-6)
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: 1D numpy.ndarray of length num_parameters containing gradient
                 of loglikelihood value
        """
        transform_list = cast_to_transform_list(transform_list,\
            num_transforms=self.num_parameters)
        gradient = self.gradient(pars, return_negative=return_negative)
        return transform_list.transform_gradient(gradient, pars)
    
    def auto_hessian(self, pars, return_negative=False,\
        larger_differences=1e-5, smaller_differences=1e-6,\
        transform_list=None):
        """
        Computes the hessian of this Loglikelihood for minimization purposes.
        
        pars: value of the parameters at which to evaluate the hessian
        return_negative: if true, the negative of the hessian of the
                         loglikelihood is returned (this is useful for times
                         when the loglikelihood must be maximized since scipy
                         optimization functions only deal with minimization
        larger_differences: either single number or 1D array of numbers to use
                            as the numerical difference in parameters.
                            Default: 10^(-5). This is the amount by which the
                            parameters are shifted between evaluations of the
                            gradient. Only used if gradient is not explicitly
                            computable.
        smaller_differences: either single_number or 1D array of numbers to use
                             as the numerical difference in parameters.
                             Default: 10^(-6). This is the amount by which the
                             parameters are shifted during each approximation
                             of the gradient. Only used if hessian is not
                             explicitly computable
        transform_list: TransformList object (or something which can be cast to
                        one) defining the transforms to apply to the parameters
                        before computing the gradient. Default: None, parameter
                        space is not transformed
        
        returns: square 2D numpy.ndarray of side length num_parameters
                 containing hessian of loglikelihood value
        """
        transform_list = cast_to_transform_list(transform_list,\
            num_transforms=self.num_parameters)
        gradient = self.gradient(pars, return_negative=return_negative)
        gradient = transform_list.transform_gradient(gradient, pars)
        hessian = self.hessian(pars, return_negative=return_negative)
        return transform_list.transform_hessian(hessian, gradient, pars)

