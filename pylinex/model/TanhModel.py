"""
File: pylinex/model/TanhModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class representing a model based on the tanh
             function, of the form A*tanh[(x-mu)/sigma]
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value
from .LoadableModel import LoadableModel

class TanhModel(LoadableModel):
    """
    Class representing a model based on the tanh function, of the form
    A*tanh[(x-mu)/sigma]
    """
    def __init__(self, x_values):
        """
        Initializes a new TanhModel by providing the x_values at which the tanh
        function at the heart of this model will be evaluated.
        
        x_values: 1D numpy.ndarray of real numbers
        """
        self.x_values = x_values
    
    @property
    def x_values(self):
        """
        Property storing the 1D numpy.ndarray of real numbers at which the tanh
        function at the heart of this model will be evaluated.
        """
        if not hasattr(self, '_x_values'):
            raise AttributeError("x_values referenced before it was set.")
        return self._x_values
    
    @x_values.setter
    def x_values(self, value):
        """
        Setter for the x_values at which the tanh function at the heart of this
        model will be evaluated.
        
        value: 1D numpy.ndarray of real numbers
        """
        self._x_values = value
    
    @property
    def num_channels(self):
        """
        Property storing the integer number of channels in the output of this
        model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.x_values)
        return self._num_channels
    
    @property
    def parameters(self):
        """
        Property storing the list of parameter names of this model. They are
        ['tanh_A', 'tanh_mu', 'tanh_sigma']
        """
        return ['tanh_A', 'tanh_mu', 'tanh_sigma']
    
    def __call__(self, pars):
        """
        Evaluates the tanh model at the given parameter values
        
        pars: 1D numpy.ndarray of 3 parameter values
        
        returns: 1D numpy.ndarray of output values of shape (num_channels)
        """
        (amplitude, mu, sigma) = pars
        return amplitude * np.tanh((self.x_values - mu) / sigma)
    
    @property
    def gradient_computable(self):
        """
        Property storing whether the gradient of this model is computable.
        Since this model is basically hardcoded, the gradient is computable.
        """
        return True
    
    def gradient(self, pars):
        """
        Evaluates the gradient of this TanhModel at the given parameters
        
        pars: 1D numpy.ndarray of 3 parameter values
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        (amplitude, mu, sigma) = pars
        weighted_deviance = (self.x_values - mu) / sigma
        tanh_deviance = np.tanh(weighted_deviance)
        final = np.ndarray((self.num_channels, 3))
        final[:,0] = tanh_deviance
        sech2_deviance = (1 - (tanh_deviance ** 2))
        mu_part = (amplitude * sech2_deviance) / (-sigma)
        final[:,1] = mu_part
        final[:,2] = (mu_part * weighted_deviance)
        return final
    
    @property
    def hessian_computable(self):
        """
        Property storing whether the hessian of this model is computable.
        Since this model is basically hardcoded, the hessian is computable.
        """
        return True
    
    def hessian(self, pars):
        """
        Evaluates the hessian of this TanhModel at the given parameters
        
        pars: 1D numpy.ndarray of 3 parameter values
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        (amplitude, mu, sigma) = pars
        final = np.ndarray((self.num_channels, 3, 3))
        weighted_deviance = ((self.x_values - mu) / sigma)
        tanh_deviance = np.tanh(weighted_deviance)
        sech2_deviance = (1 - (tanh_deviance ** 2))
        final[:,0,0] = 0
        amp_mu_part = (sech2_deviance / (-sigma))
        final[:,0,1] = amp_mu_part
        final[:,1,0] = amp_mu_part
        amp_sigma_part = amp_mu_part * weighted_deviance
        final[:,0,2] = amp_sigma_part
        final[:,2,0] = amp_sigma_part
        Asech2_sigma2 = (amplitude * sech2_deviance) / (sigma ** 2)
        mu_sigma_part =\
            (Asech2_sigma2 * (1 - (2 * weighted_deviance * tanh_deviance)))
        final[:,1,2] = mu_sigma_part
        final[:,2,1] = mu_sigma_part
        final[:,1,1] = (Asech2_sigma2 * (-2 * tanh_deviance))
        final[:,2,2] = (2 * weighted_deviance * Asech2_sigma2 *\
            (1 - (weighted_deviance * tanh_deviance)))
        return final
    
    def fill_hdf5_group(self, group):
        """
        Fills given hdf5 file group with information about this TanhModel.
        
        group: hdf5 file group to fill with information about this TanhModel
        """
        group.attrs['class'] = 'TanhModel'
        create_hdf5_dataset(group, 'x_values', data=self.x_values)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this TanhModel.
        
        other: object to check for equality
        
        returns: False unless other is a TanhModel with the same x_values
        """
        return isinstance(other, TanhModel) and\
            np.allclose(self.x_values, other.x_values, rtol=0, atol=1e-6)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        return TanhModel(get_hdf5_value(group['x_values']))
    
