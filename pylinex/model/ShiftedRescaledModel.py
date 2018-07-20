"""
File: pylinex/model/ShiftedRescaledModel.py
Author: Keith Tauscher
Date: 7 Jul 2018

Description: File containing abstract class representing a model which is a
             simple 1D function which is shifted in x and rescaled in both x
             and y.
"""
import numpy as np
from .LoadableModel import LoadableModel

cannot_instantiate_shifted_rescaled_model_error = NotImplementedError(\
    "ShiftedRescaledModel cannot be instantiated directly.")

class ShiftedRescaledModel(LoadableModel):
    """
    Abstract class representing a model which is a simple 1D function which is
    shifted in x and rescaled in both x and y.
    """
    @property
    def x_values(self):
        """
        Property storing the 1D numpy.ndarray of x_values at which the
        Lorentzian of this model should be evaluated.
        """
        if not hasattr(self, '_x_values'):
            raise AttributeError("x_values referenced before it was set.")
        return self._x_values
    
    @x_values.setter
    def x_values(self, value):
        """
        Setter for the x_values at which the Lorentzian of this model should be
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
        return ['amplitude', 'center', 'scale']
    
    def scaled_x_values(self, center, scale):
        """
        Scales the x_values of this LorentzianModel
        
        center: central x_value of scaled Lorentzian
        scale: the scale of the x values
               if scale==1, then same width as standard Lorentzian
        """
        return ((self.x_values - center) / scale)
    
    def __call__(self, pars):
        """
        Evaluates this model at the given parameters.
        
        pars: sequence yielding values of the three parameters
        
        returns: values of LorentzianModel at its x_values in numpy.ndarray of
                 shape (num_channels,)
        """
        (amplitude, center, scale) = pars
        return\
            amplitude * self.base_function(self.scaled_x_values(center, scale))
    
    @property
    def gradient_computable(self):
        """
        Property returning whether the gradient is computable. Since this model
        is essentially hardcoded, the gradient is computable.
        """
        return True
    
    def gradient(self, pars):
        """
        Function computing the gradient of the LorentzianModel at the given
        parameters.
        
        pars: 1D numpy.ndarray (of length 3) of parameters
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        (amplitude, center, scale) = pars
        scaled_x_values = self.scaled_x_values(center, scale)
        base_value = self.base_function(scaled_x_values)
        base_derivative_value = self.base_function_derivative(scaled_x_values)
        final = np.ndarray((self.num_channels, 3))
        final[:,0] = base_value
        final[:,1] = (((-1) * amplitude) * base_derivative_value) / scale
        final[:,2] = (final[:,1] * scaled_x_values)
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
        Function computing the hessian of the LorentzianModel at the given
        parameters.
        
        pars: 1D numpy.ndarray (of length 3) of parameters
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        (amplitude, center, scale) = pars
        squared_scale = (scale ** 2)
        amplitude_over_squared_scale = (amplitude / squared_scale)
        scaled_x_values = self.scaled_x_values(center, scale)
        base_value = self.base_function(scaled_x_values)
        base_derivative_value = self.base_function_derivative(scaled_x_values)
        base_second_derivative_value =\
            self.base_function_second_derivative(scaled_x_values)
        final = np.ndarray((self.num_channels, 3, 3))
        final[:,0,0] = 0
        final[:,1,1] =\
            amplitude_over_squared_scale * base_second_derivative_value
        final[:,0,1] = (((-1) * base_derivative_value) / scale)
        final[:,1,0] = final[:,0,1]
        final[:,0,2] = final[:,0,1] * scaled_x_values
        final[:,2,0] = final[:,0,2]
        fppzfpp = (base_derivative_value +\
            (scaled_x_values * base_second_derivative_value))
        final[:,1,2] = (amplitude_over_squared_scale * fppzfpp)
        final[:,2,1] = final[:,1,2]
        final[:,2,2] = (amplitude_over_squared_scale * scaled_x_values) *\
            (fppzfpp + base_derivative_value)
        return final
    
    def base_function(self, scaled_x_values):
        """
        This function calculates the base (standard) function at the given
        pre-scaled x_values.
        
        scaled_x_values: array of pre-scaled x_values
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    def base_function_derivative(self, scaled_x_values):
        """
        Calculates the derivative of the base (standard) function.
        
        scaled_x_values: array of pre-scaled x_values
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    def base_function_second_derivative(self, scaled_x_values):
        """
        Calculates the second derivative of the base (standard) function.
        
        scaled_x_values: array of pre-scaled x_values
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this LorentzianModel.
        
        other: object to check for equality
        
        returns: False unless other is a LorentzianModel with the same x_values
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        raise cannot_instantiate_shifted_rescaled_model_error
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model. They are taken from the
        submodels.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {'amplitude': (None, None), 'center': (None, None),\
                'scale': (0, None)}
        return self._bounds

