"""
File: extractpy/quantity/Quantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: An abstract class representing a quantity which is to be
             calculated after it is defined and can be calculated many times
             with many different arguments. The main feature of a quantity is
             that it can be called (with or without) arguments to retrieve its
             value.
"""
class Quantity(object):
    """
    A class which represents any quantity that can be defined and then
    evaluated at a later time by calling it. This is merely an abstract
    superclass of other Quantity classes like ConstantQuantity,
    AttributeQuantity, FunctioQuantity, CompiledQuantity, and
    CalculatedQuantity.
    """
    def __init__(self, name):
        """
        Initializes the new quantity by setting its name.
        
        name: a string identifying this quantity
        """
        self.name = name
    
    @property
    def name(self):
        """
        Property storing the string name of this quantity.
        """
        if not hasattr(self, '_name'):
            self._name = None
        return self._name
    
    @name.setter
    def name(self, value):
        """
        Setter for the name property.
        
        value: must be a string
        """
        if isinstance(value, str):
            self._name = value
        else:
            raise TypeError("name property of Quantity must be a string.")
    
    def __call__(self):
        """
        Every Quantity object can be called. This class, however, is not meant
        to be directly instantiated.
        """
        raise NotImplementedError("Quantity class should not be directly " +\
                                  "instantiated because it doesn't know " +\
                                  "how to be called or what its name is.")

