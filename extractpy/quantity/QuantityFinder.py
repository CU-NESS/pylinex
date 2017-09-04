"""
File: extractpy/quantity/QuantityFinder.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class which stores a CompiledQuantity filled
             with Quantities it will find.
"""
from .Quantity import Quantity
from .CompiledQuantity import CompiledQuantity

class QuantityFinder(object):
    """
    Class representing an object which stores a CompiledQuantity filled with
    Quantities it will find.
    """
    @property
    def compiled_quantity(self):
        """
        Property storing the CompiledQuantity object storing all of the
        different quantities that this object should make grids of.
        """
        if not hasattr(self, '_compiled_quantity'):
            raise AttributeError("compiled_quantity property referenced " +\
                                 "before it was set somehow.")
        return self._compiled_quantity
    
    @compiled_quantity.setter
    def compiled_quantity(self, value):
        """
        Setter for the compiled_quantity property.
        
        value: if value is a CompiledQuantity, it is used directly
               if value is a different Quantity, it is made into a
                                                 CompiledQuantity of one for
                                                 the purposes of calculation
        """
        if isinstance(value, CompiledQuantity):
            self._compiled_quantity = value
        elif isinstance(value, Quantity):
            self._compiled_quantity = CompiledQuantity(value.name, value)
        else:
            raise TypeError("compiled_quantity must be a CompiledQuantity " +\
                            "object or a single Quantity object.")
    
    @property
    def num_quantities(self):
        """
        Property storing the number of Quantity objects in the CompiledQuantity
        underlying this object.
        """
        if not hasattr(self, '_num_quantities'):
            self._num_quantities = self.compiled_quantity.num_quantities
        return self._num_quantities

