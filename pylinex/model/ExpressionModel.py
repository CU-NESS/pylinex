"""
File: pylinex/model/ExpressionModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing class representing a model which is described
             entirely by Expression object(s).
"""
import numpy as np
from ..util import Expression, sequence_types, Loadable
from .Model import Model

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ExpressionModel(Model, Loadable):
    """
    Class representing a model which is described entirely by Expression
    object(s).
    """
    def __init__(self, expression, parameters, gradient_expressions=None,\
        hessian_expressions=None):
        """
        Initializes this ExpressionModel from the given Expression(s) and
        parameter names.
        
        expression: Expression object which takes parameter values as inputs
                    and outputs this model's output
        parameters: list of parameters to input to expression
        gradient_expressions: sequence of Expression objects which collectively
                              yield gradient of this model
        hessian_expressions: sequence of sequences of Expression objects which
                             collectively yield hessian of this model
        """
        self.expression = expression
        self.parameters = parameters
        self.gradient_expressions = gradient_expressions
        self.hessian_expressions = hessian_expressions
    
    @property
    def expression(self):
        """
        Property storing the Expression object which takes parameters as inputs
        and produces the model output.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression referenced before it was set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for the Expression object at the core of this model.
        
        value: must be an Expression object
        """
        if isinstance(value, Expression):
            self._expression = value
        else:
            raise TypeError("expression was not an Expression object.")
    
    @property
    def parameters(self):
        """
        Property storing the list of string parameter names necessitated by
        this model.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters referenced before they were set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the list of string parameters necessitated by this model.
        
        value: sequence of string names (must have same length as the number of
               arguments needed by the expression)
        """
        if type(value) in sequence_types:
            if len(value) == self.expression.num_arguments:
                if all([isinstance(name, basestring) for name in value]):
                    if len(value) == len(set(value)):
                        self._parameters = [name for name in value]
                    else:
                        raise ValueError("There was at least one pair of " +\
                            "parameters with the same name.")
                else:
                    raise TypeError("At least one element of the " +\
                        "parameters sequence was not a string.")
            else:
                raise ValueError("parameters sequence given does not have " +\
                    "same length as expression requires.")
        else:
            raise TypeError("parameters was not a sequence.")
    
    @property
    def gradient_expressions(self):
        """
        Property storing the 1D numpy.ndarray of Expression objects which
        collectively yield the gradient.
        """
        if not hasattr(self, '_gradient_expressions'):
            raise AttributeError("gradient_expressions referenced before " +\
                "it was set.")
        return self._gradient_expressions
    
    @gradient_expressions.setter
    def gradient_expressions(self, value):
        """
        Setter for the sequence of Expression objects which collectively yield
        the gradient.
        
        value: list of expressions which each yield gradients of shape
               (num_channels,)
        """
        if value is None:
            self._gradient_expressions = None
            return
        value = np.array(value)
        if value.shape == (self.num_parameters,):
            if all([isinstance(element, Expression) for element in value]):
                self._gradient_expressions = value
            else:
                raise TypeError("At least one of the given " +\
                    "gradient_expressions was not a string")
        else:
            raise ValueError("Number of gradient_expressions wasn't the " +\
                "same as the number of parameters.")
    
    @property
    def hessian_expressions(self):
        """
        Property storing a 2D numpy.ndarray of Expression objects which
        collectively yield the hessian of this model.
        """
        if not hasattr(self, '_hessian_expressions'):
            raise AttributeError("hessian_expressions referenced before it " +\
                "was set.")
        return self._hessian_expressions
    
    @hessian_expressions.setter
    def hessian_expressions(self, value):
        """
        Setter for the sequence of sequences of Expression objects which
        collectively yield the hessian.
        
        value: list of lists of expressions which each yield hessians of shape
               (num_channels,)
        """
        if value is None:
            self._hessian_expressions = None
            return
        value = np.array(value)
        if value.shape == ((self.num_parameters,) * 2):
            if all([isinstance(element, Expression)\
                for element in value.flatten()]):
                self._hessian_expressions = value
            else:
                raise TypeError("At least one of the given " +\
                    "hessian_expressions was not an Expression object.")
        else:
            raise ValueError("hessian_expressions wasn't of the expected " +\
                "shape.")
    
    def __call__(self, parameters):
        """
        Evaluates this model at the given parameters by calling the Expression
        at the core of this model.
        
        parameters: 1D numpy.ndarray of same length as parameter names list
        
        returns: 1D numpy.ndarray of shape (num_channels,)
        """
        return self.expression(*parameters)
    
    @property
    def gradient_computable(self):
        """
        Property storing whether or not the gradient of this model is
        computable. This is True only if Expressions to determine the gradient
        are given.
        """
        if not hasattr(self, '_gradient_computable'):
            self._gradient_computable = (self.gradient_expressions is not None)
        return self._gradient_computable
    
    def gradient(self, parameters):
        """
        Function which evaluates the gradient of this model at the given
        parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of gradient values of shape
                 (num_channels, num_parameters)
        """
        if not self.gradient_computable:
            raise NotImplementedError("gradient can't be computed for this " +\
                "Model because expressions weren't given for the gradient.")
        return np.stack([expression(*parameters)\
            for expression in self.gradient_expressions], axis=1)
    
    @property
    def hessian_computable(self):
        """
        Property storing whether or not the hessian of this model is
        computable. This is True only if Expressions to determine the hessian
        are given.
        """
        if not hasattr(self, '_hessian_computable'):
            self._hessian_computable = (self.hessian_expressions is not None)
        return self._hessian_computable
    
    def hessian(self, parameters):
        """
        Function which evaluates the hessian of this model at the given
        parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: numpy.ndarray of hessian values of shape
                 (num_channels, num_parameters, num_parameters)
        """
        if not self.hessian_computable:
            raise NotImplementedError("hessian can't be computed for this " +\
                "Model because expressions weren't given for the hessian.")
        hessian00 = self.hessian_expressions[0,0](*parameters)
        hessian = np.ndarray((len(hessian00),) + ((self.num_parameters,) * 2))
        hessian[:,0,0] = hessian00
        for ipar1 in range(1, self.num_parameters):
            for ipar2 in range(ipar1 + 1):
                this_hessian =\
                    self.hessian_expressions[ipar1,ipar2](*parameters)
                hessian[:,ipar1,ipar2] = this_hessian
                hessian[:,ipar2,ipar1] = this_hessian
        return hessian
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        ExpressionModel.
        
        group: hdf5 file group to fill with information about this
               ExpressionModel
        """
        group.attrs['class'] = 'ExpressionModel'
        subgroup = group.create_group('parameters')
        for (iparameter, parameter) in enumerate(self.parameters):
            subgroup.attrs['{}'.format(iparameter)] = parameter
        self.expression.fill_hdf5_group(group.create_group('expression'))
        if self.gradient_expressions is not None:
            subgroup = group.create_group('gradient_expressions')
            for iparameter in range(self.num_parameters):
                self.gradient_expressions[iparameter].fill_hdf5_group(\
                    subgroup.create_group('{}'.format(iparameter)))
        if self.hessian_expressions is not None:
            subgroup = group.create_group('hessian_expressions')
            for ipar1 in range(self.num_parameters):
                for ipar2 in range(ipar1 + 1):
                    self.hessian_expressions[ipar1,ipar2].fill_hdf5_group(\
                        subgroup.create_group('{}_{}'.format(ipar1, ipar2)))
    
    def __eq__(self, other):
        """
        Checks if other is essentially equivalent to this ExpressionModel.
        
        other: object to check for equality
        
        returns: False unless other is an ExpressionModel with the same
                 expression(s) and parameters
        """
        if not isinstance(other, ExpressionModel):
            return False
        if self.parameters != other.parameters:
            return False
        if self.expression != other.expression:
            return False
        if self.gradient_expressions is None:
            if other.gradient_expressions is not None:
                return False
        else:
            try:
                if np.any(\
                    self.gradient_expressions != other.gradient_expressions):
                    return False
            except:
                return False
        if self.hessian_expressions is None:
            if other.hessian_expressions is not None:
                return False
        else:
            try:
                if np.any(\
                    self.hessian_expressions != other.hessian_expressions):
                    return False
            except:
                return False
        return True
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a model from the given group. The load_from_hdf5_group of a given
        subclass model should always be called.
        
        group: the hdf5 file group from which to load the Model
        
        returns: a Model of the Model subclass for which this is called
        """
        parameters = []
        subgroup = group['parameters']
        iparameter = 0
        while '{}'.format(iparameter) in subgroup.attrs:
            parameters.append(subgroup.attrs['{}'.format(iparameter)])
            iparameter += 1
        subgroup = group['expression']
        expression = Expression.load_from_hdf5_group(subgroup)
        if 'gradient_expressions' in group:
            subgroup = group['gradient_expressions']
            gradient_expressions = np.ndarray((len(parameters),), dtype=object)
            for ipar in range(len(parameters)):
                gradient_expressions[ipar] = Expression.load_from_hdf5_group(\
                    subgroup['{}'.format(ipar)])
        else:
            gradient_expressions = None
        if 'hessian_expressions' in group:
            subgroup = group['hessian_expressions']
            hessian_expressions =\
                np.ndarray((len(parameters),) * 2, dtype=object)
            for ipar1 in range(len(parameters)):
                for ipar2 in range(ipar1 + 1):
                    this_block = Expression.load_from_hdf5_group(\
                        subgroup['{}_{}'.format(ipar1, ipar2)])
                    hessian_expressions[ipar1,ipar2] = this_block
                    hessian_expressions[ipar2,ipar1] = this_block
        else:
            hessian_expressions = None
        return ExpressionModel(expression, parameters,\
            gradient_expressions=gradient_expressions,\
            hessian_expressions=hessian_expressions)

