"""
File: extractpy/quantity/FunctionQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a quantity that is calculated
             through the use of a member function of an object. It stores
             arguments and keyword arguments to pass onto the member function
             when called. When called, it must be given an object on which to
             call the function.
"""
from ..util import sequence_types
from .Quantity import Quantity

class FunctionQuantity(Quantity):
    """
    Class representing a quantity that is calculated through the use of a
    member function of an object. It stores the arguments and keyword arguments
    which must be passed on to the function when called. When called, it must
    be given an object on which to call the function.
    """
    def __init__(self, function_name, *function_args, **function_kwargs):
        """
        Initializes a new FunctionQuantity with the given function name and
        arguments.
        
        function_name: name of the function to call when this object is called
        *function_args: unpacked list of arguments to pass on to the function
        **function_kwargs: unpacked dict of keyword arguments to pass on to the
                           function
        """
        Quantity.__init__(self, function_name)
        self.function_args = function_args
        self.function_kwargs = function_kwargs
    
    @property
    def function_args(self):
        """
        Property storing the packed list of arguments to pass onto the function
        when this object is called.
        """
        if not hasattr(self, '_function_args'):
            raise AttributeError("function_args referenced before it was set.")
        return self._function_args
    
    @function_args.setter
    def function_args(self, value):
        """
        Setter for the function_args property.
        
        value: must be a sequence of some kind
        """
        if type(value) in sequence_types:
            self._function_args = list(value)
        else:
            raise TypeError("function_args was set to a non-sequence.")
    
    @property
    def function_kwargs(self):
        """
        Property storing the keyword arguments to pass on to the function when
        this object is called.
        """
        if not hasattr(self, '_function_kwargs'):
            raise AttributeError("function_kwargs referenced before it was " +\
                                 "set.")
        return self._function_kwargs
    
    @function_kwargs.setter
    def function_kwargs(self, value):
        """
        Setter for the function_kwargs property.
        
        value: must be a dict
        """
        if isinstance(value, dict):
            self._function_kwargs = value
        else:
            raise TypeError("function_kwargs was set to a non-dict.")
    
    def __call__(self, container, **kwargs):
        """
        Calls this FunctionQuantity.
        
        container: the object on which to call the function
        **kwargs: unpacked list of unused keyword arguments for compatibility
        
        returns: the return value of the function which underlies this Quantity
        """
        return eval("container." + self.name + "(*self.function_args, " +\
                    "**self.function_kwargs)")

