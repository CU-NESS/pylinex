"""
File: pylinex/model/ProductModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing a class representing a model which is the product
             of an arbitrarily long list of submodels.
"""
import numpy as np
from .CompoundModel import CompoundModel

class ProductModel(CompoundModel):
    """
    Class representing a model which is the product of an arbitrarily long list
    of submodels.
    """
    def __init__(self, names, models):
        """
        Initializes a new ProductModel through the providing of a list of
        submodels and their names.
        
        names: list of names of submodels
        models: sequence of Model objects
        """
        self.names = names
        self.models = models        
    
    def __call__(self, parameters):
        """
        Evaluates this model at the given parameters by multiplying together
        each of the submodel outputs.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of model output values of shape (num_channels,)
        """
        partitions = self.partition_parameters(parameters)
        factors = [model(partition)\
            for (model, partition) in zip(self.models, partitions)]
        return np.prod(factors, axis=0)
    
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
        function_chunks = []
        gradient_chunks = []
        for (model, partition) in zip(self.models, partitions):
            function_chunks.append(model(partition))
            gradient_chunks.append(model.gradient(partition))
        for ichunk in range(len(self.models)):
            gradient_chunks[ichunk] *=\
                (np.prod(function_chunks[:ichunk], axis=0) *\
                np.prod(function_chunks[ichunk+1:], axis=0))[:,np.newaxis]
        return np.concatenate(gradient_chunks, axis=-1)
    
    @property
    def hessian_computable(self):
        """
        Property storing boolean describing whether hessian of this model is
        computable. The hessian of this model is computable if and only if all
        submodel gradients and hessians are computable.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable = True
            for model in self.models:
                self._hessian_computable =\
                    (self._hessian_computable and\
                    (model.gradient_computable and model.hessian_computable))
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
        function_chunks = []
        gradient_chunks = []
        hessian_chunks = []
        for (model, partition) in zip(self.models, partitions):
            function_chunks.append(model(partition))
            gradient_chunks.append(model.gradient(partition))
            hessian_chunks.append(model.hessian(partition))
        num_channels = function_chunks[0].shape[0]
        final = np.zeros((num_channels,) + ((len(parameters),) * 2))
        for ichunk1 in range(len(self.models)):
            slice1 = self.partition_slices[ichunk1]
            product_past_ichunk1 = np.prod(function_chunks[ichunk1+1:], axis=0)
            for ichunk2 in range(ichunk1):
                slice2 = self.partition_slices[ichunk2]
                hessian_piece = ((np.prod(function_chunks[:ichunk2], axis=0) *\
                    np.prod(function_chunks[ichunk2+1:ichunk1], axis=0)) *\
                    product_past_ichunk1)[:,np.newaxis,np.newaxis] *\
                    gradient_chunks[ichunk1][:,:,np.newaxis] *\
                    gradient_chunks[ichunk2][:,np.newaxis,:]
                final[:,slice1,slice2] = hessian_piece
                final[:,slice2,slice1] = np.swapaxes(hessian_piece, 1, 2)
            final[:,slice1,slice1] = (hessian_chunks[ichunk1] *\
                (np.prod(function_chunks[:ichunk1], axis=0) *\
                product_past_ichunk1)[:,np.newaxis,np.newaxis])
        return final
    
    def fill_hdf5_group(self, group):
        """
        Fills an hdf5 file group with information about this ProductModel.
        
        group: hdf5 file group to fill with information about this ProductModel
        """
        group.attrs['class'] = 'ProductModel'
        subgroup = group.create_group('models')
        for (iname, name) in enumerate(self.names):
            subsubgroup = subgroup.create_group('{}'.format(iname))
            subsubgroup.attrs['name'] = name
            self.models[iname].fill_hdf5_group(subsubgroup)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this ProductModel.
        
        other: object to check for equality
        
        returns: False unless other is a ProductModel with the same names and
                 submodels
        """
        if not isinstance(other, ProductModel):
            return False
        if self.names != other.names:
            return False
        return all([(smodel == omodel)\
            for (smodel, omodel) in zip(self.models, other.models)])

