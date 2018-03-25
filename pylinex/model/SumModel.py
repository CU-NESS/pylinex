"""
File: pylinex/model/SumModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class representing a model which is the sum of
             an arbitrarily long list of submodels.
"""
import numpy as np
from .CompoundModel import CompoundModel

class SumModel(CompoundModel):
    """
    Class representing a model which is the sum of an arbitrarily long list of
    submodels.
    """
    def __init__(self, names, models):
        """
        Initializes a new SumModel through the providing of a list of submodels
        and their names.
        
        names: list of names of submodels
        models: sequence of Model objects
        """
        self.names = names
        self.models = models
    
    def __call__(self, parameters):
        """
        Evaluates this model at the given parameters by adding together each of
        the submodel outputs.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of model output values of shape (num_channels,)
        """
        partitions = self.partition_parameters(parameters)
        summands = [model(partition)\
            for (model, partition) in zip(self.models, partitions)]
        return np.sum(summands, axis=0)
    
    @property
    def gradient_computable(self):
        """
        Property storing boolean describing whether gradient of this model is
        computable. The gradient of this model is computable if and only if all
        submodel gradients are computable.
        """
        if not hasattr(self, '_gradient_computable'):
            self._gradient_computable = True
            for model in self.models:
                self._gradient_computable =\
                    (self._gradient_computable and model.gradient_computable)
        return self._gradient_computable
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        if not self.gradient_computable:
            raise NotImplementedError("gradient can't be computed for this " +\
                "Model because at least one of the Models it stores cannot " +\
                "compute its gradient.")
        partitions = self.partition_parameters(parameters)
        gradient_chunks = [model.gradient(partition)\
            for (model, partition) in zip(self.models, partitions)]
        return np.concatenate(gradient_chunks, axis=-1)
    
    @property
    def hessian_computable(self):
        """
        Property storing boolean describing whether hessian of this model is
        computable. The hessian of this model is computable if and only if all
        submodel hessians are computable.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable = True
            for model in self.models:
                self._hessian_computable =\
                    (self._hessian_computable and model.hessian_computable)
        return self._hessian_computable
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        if not self.hessian_computable:
            raise NotImplementedError("hessian can't be computed for this " +\
                "Model because at least one of the Models it stores cannot " +\
                "compute its Hessian.")
        partitions = self.partition_parameters(parameters)
        hessian_chunks = [model.hessian(partition)\
            for (model, partition) in zip(self.models, partitions)]
        num_channels = hessian_chunks[0].shape[0]
        final = np.zeros((num_channels,) + ((len(parameters),) * 2))
        start = 0
        for (model, partition) in zip(self.models, partitions):
            end = start + len(partition)
            final[:,start:end,start:end] = model.hessian(partition)
            start = end
        return final
    
    def fill_hdf5_group(self, group):
        """
        Fills an hdf5 file group with information about this SumModel.
        
        group: hdf5 file group to fill with information about this SumModel
        """
        group.attrs['class'] = 'SumModel'
        subgroup = group.create_group('models')
        for (iname, name) in enumerate(self.names):
            subsubgroup = subgroup.create_group('{}'.format(iname))
            subsubgroup.attrs['name'] = name
            self.models[iname].fill_hdf5_group(subsubgroup)
    
    def __eq__(self, other):
        """
        Checks for equality between this SumModel and other.
        
        other: object to check for equality
        
        returns: False unless other is a SumModel with the same names and
                 submodels
        """
        if not isinstance(other, SumModel):
            return False
        if self.names != other.names:
            return False
        return all([(smodel == omodel)\
            for (smodel, omodel) in zip(self.models, other.models)])

