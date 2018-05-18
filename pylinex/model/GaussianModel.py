"""
File: pylinex/model/GaussianModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class representing a model which is a simple
             Gaussian, with parameters
             ['gaussian_A', 'gaussian_mu', 'gaussian_sigma'].
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .LoadableModel import LoadableModel

class GaussianModel(LoadableModel):
    """
    Class representing a model which is a simple Gaussian, with parameters
    ['gaussian_A', 'gaussian_mu', 'gaussian_sigma'].
    """
    def __init__(self, x_values):
        """
        Initializes this GaussianModel by providing the x_values at which it
        should be evaluated to produce the output.
        
        x_values: 1D numpy.ndarray of values at which to evaluate the Gaussian
        """
        self.x_values = x_values
    
    @property
    def x_values(self):
        """
        Property storing the 1D numpy.ndarray of x_values at which the Gaussian
        of this model should be evaluated.
        """
        if not hasattr(self, '_x_values'):
            raise AttributeError("x_values referenced before it was set.")
        return self._x_values
    
    @x_values.setter
    def x_values(self, value):
        """
        Setter for the x_values at which the Gaussian of this model should be
        evaluated.
        
        value: must be a 1D numpy.ndarray of numbers
        """
        value = np.array(value)
        if value.ndim == 1:
            self._x_values = value
        else:
            raise ValueError("x_values was set to a non-array.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output which is the same
        as the number of x_values given.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.x_values)
        return self._num_channels
    
    @property
    def parameters(self):
        """
        Property storing the parameters describing this model:
        ['amplitude', 'center', 'standard_deviation']
        """
        return ['amplitude', 'center', 'standard_deviation']
    
    def __call__(self, pars):
        """
        Evaluates this model at the given parameters.
        
        pars: sequence yielding values of the three parameters
        
        returns: values of GaussianModel at its x_values in numpy.ndarray of
                 shape (num_channels,)
        """
        (amplitude, mean, stdv) = pars
        return (amplitude *\
            np.exp((((self.x_values - mean) / stdv) ** 2) / (-2.)))
    
    @property
    def gradient_computable(self):
        """
        Property returning whether the gradient is computable. Since this model
        is essentially hardcoded, the gradient is computable.
        """
        return True
    
    def gradient(self, pars):
        """
        Function computing the gradient of the GaussianModel at the given
        parameters.
        
        pars: 1D numpy.ndarray (of length 3) of parameters
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        (amplitude, mean, stdv) = pars
        value = np.exp((((self.x_values - mean) / stdv) ** 2) / (-2.))
        final = np.ndarray((self.num_channels, 3))
        final[:,0] = value
        weighted_deviance = (self.x_values - mean) / stdv
        mean_part = (amplitude * value * (weighted_deviance / stdv))
        final[:,1] = mean_part
        final[:,2] = (mean_part * weighted_deviance)
        return final
    
    @property
    def hessian_computable(self):
        """
        Property returning whether the hessian is computable. Since this model
        is essentially hardcoded, the hessian is computable.
        """
        return True
    
    def hessian(self, pars):
        """
        Function computing the hessian of the GaussianModel at the given
        parameters.
        
        pars: 1D numpy.ndarray (of length 3) of parameters
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        (amplitude, mean, stdv) = pars
        value = self(pars)
        final = np.ndarray((self.num_channels, 3, 3))
        weighted_deviances = ((self.x_values - mean) / stdv)
        weighted_deviances_squared = (weighted_deviances ** 2)
        variance = (stdv ** 2)
        value_over_variance = (value / variance)
        final[:,0,0] = 0
        final[:,1,1] = value_over_variance * (weighted_deviances_squared - 1)
        final[:,2,2] = (value_over_variance * weighted_deviances_squared *\
            (weighted_deviances_squared - 3))
        amp_mean_part = ((value * weighted_deviances) / (amplitude * stdv))
        final[:,0,1] = amp_mean_part
        final[:,1,0] = amp_mean_part
        amp_stdv_part =\
            ((value * weighted_deviances_squared) / (amplitude * stdv))
        final[:,0,2] = amp_stdv_part
        final[:,2,0] = amp_stdv_part
        mean_stdv_part =\
            (value * weighted_deviances * (weighted_deviances - 2)) / variance
        final[:,1,2] = mean_stdv_part
        final[:,2,1] = mean_stdv_part
        return final
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'GaussianModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this GaussianModel.
        
        other: object to check for equality
        
        returns: False unless other is a GaussianModel with the same x_values
        """
        return isinstance(other, GaussianModel) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        return GaussianModel(get_hdf5_value(group['x_values']))

