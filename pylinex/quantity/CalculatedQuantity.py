"""
File: pylinex/quantity/CalculatedQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a subclass of the Quantity class which performs
             functions on groups of quantities to produce a single quantity
             output.
"""
from types import FunctionType
from .Quantity import Quantity
from .CompiledQuantity import CompiledQuantity

class CalculatedQuantity(Quantity):
    """
    Class representing a quantity which is a function of (one or possibly
    multiple) other quantities. At its heart, it utilizes a CompiledQuantity to
    keep track of many quantities. Then, when it is called, the
    CompiledQuantity is called and its return values are passed onto the given
    function.
    """
    def __init__(self, name, function, compiled_quantity):
        """
        Initializes a new CalculatedQuantity with the given name, function, and
        quantities.
        
        name: string identifier of this quantity
        function: function to apply to values returned by compiled_quantity
        compiled_quantity: quantity containing all quantities necessary for
                           calling of given function. compiled_quantity can
                           either be a CompiledQuantity or a single Quantity
                           object (which will be made into a CompiledQuantity
                           object of one)
        """
        Quantity.__init__(self, name)
        self.function = function
        self.compiled_quantity = compiled_quantity
    
    @property
    def function(self):
        """
        Property storing the function which will be performed on the quantities
        underlying this object to calculate the return value.
        """
        if not hasattr(self, '_function'):
            raise AttributeError("function was referenced before it was set.")
        return self._function
    
    @function.setter
    def function(self, value):
        """
        Setter for the function property.
        
        value: must have type FunctionType
        """
        if type(value) is FunctionType:
            self._function = value
        else:
            raise TypeError("function property was not of FunctionType.")
    
    @property
    def num_arguments(self):
        """
        Property storing the number of arguments accepted by the function
        underlying this object.
        """
        if not hasattr(self, '_num_arguments'):
            self._num_arguments = self.function.__code__.co_argcount
        return self._num_arguments
    
    @property
    def compiled_quantity(self):
        """
        Property storing the CompiledQuantity object containing the Quantity
        objects necessary for the calculation of the return value.
        """
        if not hasattr(self, '_compiled_quantity'):
            raise AttributeError("compiled_quantity was referenced before " +\
                                 "it was set.")
        return self._compiled_quantity
    
    @compiled_quantity.setter
    def compiled_quantity(self, value):
        """
        Setter for the compiled_quantity property.
        
        value: if value if a CompiledQuantity, it is used directly
               if value is another Quantity, it is made into a CompiledQuantity
                                             of one
        """
        if isinstance(value, Quantity) and\
            (not isinstance(value, CompiledQuantity)):
            value = CompiledQuantity('sole', value)
        if isinstance(value, CompiledQuantity):
            if value.num_quantities == self.num_arguments:
                self._compiled_quantity = value
            else:
                raise ValueError("The number of arguments of the function " +\
                                 "underlying this object was not the same " +\
                                 "as the number of quantities in the " +\
                                 "CompiledQuantity underlying this object.")
        else:
            raise TypeError("compiled_quantity was not a CompiledQuantity " +\
                            "object.")
    
    def __call__(self, *args, **kwargs):
        """
        Calls this quantity by calculating the values associated with the
        compiled_quantity property and plugging them into the function
        property.
        
        args: arguments to pass on to compiled_quantity
        kwargs: keyword arguments to pass on to compiled_quantity
        """
        return self.function(*self.compiled_quantity(*args, **kwargs))

