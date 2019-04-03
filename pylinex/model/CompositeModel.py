"""
File: pylinex/model/CompositeModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing class representing a model consisting of submodels
             combined through the use of an Expression object.
"""
import numpy as np
from distpy import Expression
from .CompoundModel import CompoundModel

class CompositeModel(CompoundModel):
    """
    Class representing a model consisting of submodels combined through the use
    of an Expression object.
    """
    def __init__(self, expression, names, models, gradient_expressions=None,\
        hessian_expressions=None):
        """
        Initializes a CompositeModel with the given Expression object(s) and
        models.
        
        expression: the Expression object which, when evaluated with the
                    submodel outputs as inputs, yields the output of this model
        names: sequence of string names to give to the submodels
        models: the submodels of this CompositeModel
        gradient_expressions: sequence of expressions which, when evaluated
                              with the submodel outputs as inputs, yields the
                              gradient of the output of the model (without
                              taking into account the chain rule, which will be
                              done by this object)
        hessian_expressions: sequence of sequence of expressions which, when
                             evaluated with the submodel outputs as inputs,
                             yield the hessian of the output model (without
                             taking into account the chain rule, which will be
                             done by this object).
        """
        self.expression = expression
        self.names = names
        self.models = models
        self.gradient_expressions = gradient_expressions
        self.hessian_expressions = hessian_expressions
    
    @property
    def expression(self):
        """
        Property storing the Expression object describing the output of this
        model in terms of the outputs of the submodels.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression referenced before it was set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for the expression property describing the output of this model
        in terms of the outputs of the submodels.
        
        value: must be an Expression object
        """
        if isinstance(value, Expression):
            self._expression = value
        else:
            raise TypeError("expression was not an Expression object.")
    
    @property
    def gradient_expressions(self):
        """
        Property storing sequence of expressions which, when evaluated with the
        submodel outputs as inputs, yield the gradient of the output of the
        model (without taking into account the chain rule, which will be done
        by this object).
        """
        if not hasattr(self, '_gradient_expressions'):
            raise AttributeError("gradient_expressions referenced before " +\
                "it was set.")
        return self._gradient_expressions
    
    @gradient_expressions.setter
    def gradient_expressions(self, value):
        """
        Setter for the sequence of expressions which, when evaluated with the
        submodel outputs as inputs, yield the gradient of the output of the
        model (without taking into account the chain rule, which will be done
        by this object).
        
        value: list of expressions which yield gradients of shape
               (nchannel, num_parameters)
        """
        if type(value) is type(None):
            self._gradient_expressions = None
            return
        value = np.array(value)
        if value.shape == (len(self.names),):
            if all([isinstance(element, Expression) for element in value]):
                self._gradient_expressions = value
            else:
                raise TypeError("At least one of the given " +\
                    "gradient_expressions was not a string")
        else:
            raise ValueError("Number of gradient_expressions wasn't the " +\
                "same as the number of names.")
    
    @property
    def hessian_expressions(self):
        """
        Property storing sequence of sequence of expressions which, when
        evaluated with the submodel outputs as inputs, yield the hessian of
        the output of the model (without taking into account the chain rule,
        which will be done by this object).
        """
        if not hasattr(self, '_hessian_expressions'):
            raise AttributeError("hessian_expressions referenced before it " +\
                "was set.")
        return self._hessian_expressions
    
    @hessian_expressions.setter
    def hessian_expressions(self, value):
        """
        Setter for the sequence of sequence of expressions which, when
        evaluated with the submodel outputs as inputs, yield the hessian of
        the output of the model (without taking into account the chain rule,
        which will be done by this object).
        
        value: list of lists of expressions (shape: (len(names), len(names)))
        """
        if type(value) is type(None):
            self._hessian_expressions = None
            return
        value = np.array(value)
        if value.shape == ((len(self.names),) * 2):
            if all([isinstance(element, Expression)\
                for element in value.flatten()]):
                self._hessian_expressions = value
            else:
                raise TypeError("At least one of the given " +\
                    "hessian_expressions was not a string")
        else:
            raise ValueError("hessian_expressions wasn't of the expected " +\
                "shape.")
    
    def construct_arguments_from_partitions(self, partitions):
        """
        Evaluates the individual models to use in this CompositeModel.
        
        partitions: list of arrays of parameters to pass to submodels
        
        returns: the list of evaliated models
        """
        arguments = []
        for iname in range(self.expression.num_arguments):
            arguments.append(self.models[iname](partitions[iname]))
        return arguments
        
    
    def __call__(self, parameters):
        """
        Computes this model by calling all submodels and combining them with
        the expression at the core of this model. 
        
        parameters: array of parameter values
        
        returns: numpy.ndarray of shape (num_channels,)
        """
        partitions = self.partition_parameters(parameters)
        arguments = self.construct_arguments_from_partitions(partitions)
        return self.expression(*arguments)
    
    @property
    def gradient_computable(self):
        """
        Property storing boolean describing whether the gradient of this model
        is computable. This is only true if gradient expressions are given and
        the gradient of every submodel is computable.
        """
        if not hasattr(self, '_gradient_computable'):
            self._gradient_computable =\
                (type(self.gradient_expressions) is not type(None))
            for model in self.models:
                self._gradient_computable =\
                    (self._gradient_computable and model.gradient_computable)
        return self._gradient_computable
    
    def gradient(self, parameters):
        """
        Computes the gradient of this model.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        if not self.gradient_computable:
            raise NotImplementedError("gradient can't be computed for this " +\
                "Model either because expressions weren't given for the " +\
                "gradient or because the submodels have uncomputable " +\
                "gradients.")
        partitions = self.partition_parameters(parameters)
        arguments = self.construct_arguments_from_partitions(partitions)
        grad_parts = []
        for (iname, name) in enumerate(self.names):
            # line below represents the 'outer gradient' of the chain rule
            big_picture_part =\
                self.gradient_expressions[iname](*arguments)[:,np.newaxis]
            # line below represents the 'inner gradient' of the chain rule
            small_picture_part = self.models[iname].gradient(partitions[iname])
            # chain rule says gradient is product of inner and outer gradients
            grad_parts.append(big_picture_part * small_picture_part)
        return np.concatenate(grad_parts, axis=1)
    
    @property
    def hessian_computable(self):
        """
        Property storing boolean describing whether the hessian of this model
        is computable. This is only true if gradient and hessian expressions
        are given and the gradient and hessian of every submodel is computable.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable =\
                ((type(self.gradient_expressions) is not type(None)) and\
                (type(self.hessian_expressions) is not type(None)))
            for model in self.models:
                self._hessian_computable = (self._hessian_computable and\
                    (model.gradient_computable and model.hessian_computable))
        return self._hessian_computable
    
    def hessian(self, parameters):
        """
        Computes the hessian of this model.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        if not self.hessian_computable:
            raise NotImplementedError("hessian can't be computed for this " +\
                "Model either because expressions weren't given for the " +\
                "gradient/hessian or because the submodels have " +\
                "uncomputable hessians/gradients.")
        partitions = self.partition_parameters(parameters)
        arguments = self.construct_arguments_from_partitions(partitions)
        hess_parts = []
        for iname1 in range(len(self.names)):
            inner_hessian = self.models[iname1].hessian(partitions[iname1])
            outer_gradient = self.gradient_expressions[iname1](*arguments)
            inner_gradient1 = self.models[iname1].gradient(partitions[iname1])
            row = []
            for iname2 in range(len(self.names)):
                outer_hessian =\
                    self.hessian_expressions[iname1,iname2](*arguments)
                inner_gradient2 =\
                    self.models[iname2].gradient(partitions[iname2])
                hess_part = (inner_gradient1[:,:,np.newaxis] *\
                    (inner_gradient2[:,np.newaxis,:] *\
                    outer_hessian[:,np.newaxis,np.newaxis]))
                if iname1 == iname2:
                    hess_part += (outer_gradient[:,np.newaxis,np.newaxis] *\
                        inner_hessian)
                row.append(hess_part)
            row = np.concatenate(row, axis=2)
            hess_parts.append(row)
        return np.concatenate(hess_parts, axis=1)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'CompositeModel'
        self.expression.fill_hdf5_group(group.create_group('expression'))
        subgroup = group.create_group('models')
        for (iname, name) in enumerate(self.names):
            subsubgroup = subgroup.create_group('{}'.format(iname))
            subsubgroup.attrs['name'] = name
            self.models[iname].fill_hdf5_group(subsubgroup)
        if type(self.gradient_expressions) is not type(None):
            subgroup = group.create_group('gradient_expressions')
            for iname in range(len(self.names)):
                self.gradient_expressions[iname].fill_hdf5_group(\
                    subgroup.create_group('{}'.format(iname)))
        if type(self.hessian_expressions) is not type(None):
            subgroup = group.create_group('hessian_expressions')
            for iname1 in range(len(self.names)):
                for iname2 in range(iname1 + 1):
                    self.hessian_expressions[iname1,iname2].fill_hdf5_group(\
                        subgroup.create_group('{}_{}'.format(iname1, iname2)))
        
    def __eq__(self, other):
        """
        Checks whether other is the same as this CompositeModel.
        
        other: object to check for equality
        
        returns: False unless other is a CompositeModel with the same
                 expression(s), names, and submodels
        """
        if not isinstance(other, CompositeModel):
            return False
        if self.names != other.names:
            return False
        if any([(smodel != omodel)\
            for (smodel, omodel) in zip(self.models, other.models)]):
            return False
        if self.expression != other.expression:
            return False
        if type(self.gradient_expressions) is type(None):
            if type(other.gradient_expressions) is not type(None):
                return False
        else:
            try:
                if np.any(\
                    self.gradient_expressions != other.gradient_expressions):
                    return False
            except:
                return False
        if type(self.hessian_expressions) is type(None):
            if type(other.hessian_expressions) is not type(None):
                return False
        else:
            try:
                if np.any(\
                    self.hessian_expressions != other.hessian_expressions):
                    return False
            except:
                return False
        return True

