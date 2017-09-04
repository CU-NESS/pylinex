"""
File: extractpy/quantity/ConstantQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a subclass of the Quantity class representing a
             quantity which always has the same value, given to the
             ConstantQuantity upon initialization.
"""
from ..util import numerical_types, sequence_types
from .Quantity import Quantity

class ConstantQuantity(Quantity):
    """
    Class representing a Quantity which always has the same constant value.
    That value is given at initialization.
    """
    def __init__(self, constant, name=None):
        """
        Initializes a new ConstantQuantity with the given constant value and
        string identifier.
        
        constant: the constant value to return when this object is called
        name: a string identifier for this Quantity (if None is given, then it
              is given by the string version of the constant value of this
              ConstantQuantity)
        """
        self.constant = constant
        if name is None:
            name = "%s" % (self.constant,)
        Quantity.__init__(self, name)
    
    @property
    def constant(self):
        """
        Property storing the constant value of this Quantity.
        """
        if not hasattr(self, '_constant'):
            raise AttributeError("constant referenced before it was set.")
        return self._constant
    
    @constant.setter
    def constant(self, value):
        """
        Setter for the return value of this Quantity.
        
        value: any number or sequence
        """
        if type(value) in (numerical_types + sequence_types):
            self._constant = value
        else:
            raise TypeError("The constant value of a ConstantQuantity must " +\
                            "be of some sort of numerical type or of a " +\
                            "sequence type.")
    
    def __call__(self, *args, **kwargs):
        """
        Calls this quantity. This function should be called with no arguments!
        The args and kwargs in the function signature are there only for
        compatibility.
        
        returns: the constant value of this ConstantQuantity
        """
        return self.constant

