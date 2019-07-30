"""
File: pylinex/model/OutputInterpolatedModel.py
Author: Keith Tauscher
Date: 30 Jul 2019

Description: File containing a class representing a model which is an
             interpolated (in the output space) version of an input model.
"""
from __future__ import division
import numpy as np
from scipy.interpolate import make_interp_spline as make_spline
from ..util import int_types, sequence_types, create_hdf5_dataset
from .Model import Model

class OutputInterpolatedModel(Model):
    """
    Class representing a model which is an interpolation (in the output space)
    version of an input model.
    """
    def __init__(self, model, old_xs, new_xs, order=1):
        """
        Initializes a TransformedModel based around the given underlying model
        and the binner which will Bin it.
        
        model: a Model object
        old_xs: the x values at which underlying model returns values
        new_xs: the x values at which this model should return values
        order: order of spline to use in interpolation. Default is 1 (linear
               interpolation). odd positive integer, usually in {1,3,5}
        """
        self.model = model
        self.old_xs = old_xs
        self.new_xs = new_xs
        self.order = order
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output of this model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.new_xs)
        return self._num_channels
    
    @property
    def old_xs(self):
        """
        Property storing the x values at which underlying model returns
        values.
        """
        if not hasattr(self, '_old_xs'):
            raise AttributeError("old_xs was referenced before it was set.")
        return self._old_xs
    
    @old_xs.setter
    def old_xs(self, value):
        """
        Setter for the x values at which underlying model returns values.
        
        value: 1D array with length given by num_channels property of
               underlying model
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.model.num_channels,):
                self._old_xs = value
            else:
                raise ValueError("old_xs did not have the same length as " +\
                    "the outputs of the underlying model.")
        else:
            raise TypeError("old_xs was set to a non-array.")
    
    @property
    def new_xs(self):
        """
        Property storing the x values at which this model returns values.
        """
        if not hasattr(self, '_new_xs'):
            raise AttributeError("new_xs was referenced before it was set.")
        return self._new_xs
    
    @new_xs.setter
    def new_xs(self, value):
        """
        Setter for the x values at which this model returns values.
        
        value: 1D array
        """
        if type(value) in sequence_types:
            self._new_xs = np.array(value)
        else:
            raise TypeError("new_xs was set to a non-array.")
    
    @property
    def order(self):
        """
        Property storing the order of the spline interpolation used.
        """
        if not hasattr(self, '_order'):
            raise AttributeError("order was referenced before it was set.")
        return self._order
    
    @order.setter
    def order(self, value):
        """
        Setter for the order of the spline interpolation to use.
        
        value: odd positive integer, usually one of {1,3,5}
        """
        if type(value) in int_types:
            if value > 0:
                if (value % 2) == 0:
                    print("WARNING: order of spline interpolation is being " +\
                        "set to an even integer, which may produce strange " +\
                        "results. Is this definitely what you want?")
                self._order = value
            else:
                raise ValueError("order was set to a non-positive integer.")
        else:
            raise TypeError("order was set to a non-int.")
    
    @property
    def model(self):
        """
        Property storing the inner model (as a Model object) which is being
        interpolated (in output space).
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the inner model which is being interpolated (in output
        space).
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        return self.model.parameters
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return make_spline(self.old_xs, self.model(parameters),\
            k=self.order)(self.new_xs)
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'OutputInterpolatedModel'
        group.attrs['order'] = self.order
        self.model.fill_hdf5_group(group.create_group('model'))
        create_hdf5_dataset(group, 'old_xs', data=self.old_xs)
        create_hdf5_dataset(group, 'new_xs', data=self.new_xs)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this model, False otherwise
        """
        if isinstance(other, OutputInterpolatedModel):
            if self.model == other.model:
                if self.order == other.order:
                    if np.all(self.old_xs == other.old_xs):
                        if self.new_xs.shape == other.new_xs.shape:
                            return np.all(self.new_xs == other.new_xs)
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
    
    def quick_fit(self, data, error):
        """
        Performs a quick fit to the given data.
        
        data: curve to fit with the model
        error: noise level in the data
        
        returns: (parameter_mean, parameter_covariance)
        """
        raise NotImplementedError("quick_fit not implemented for " +\
            "OutputInterpolatedModel objects.")
    
    @property
    def bounds(self):
        """
        Property storing the natural bounds of the parameters of this model.
        Since this is just a rebranding of the underlying model, the bounds are
        passed through with no changes.
        """
        return self.model.bounds

