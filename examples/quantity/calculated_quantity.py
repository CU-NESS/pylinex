"""
File: examples/quantity/calculated_quantity.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how CalculatedQuantity class is used. Basically,
             the CalculatedQuantity class uses an underlying Quantity object
             and a function to pass it to.
"""
from pylinex import AttributeQuantity, CalculatedQuantity

class A(object):
    @property
    def desired(self):
        """
        Returns the first "taxi cab number", 1729.
        """
        return 1729

quantity =\
    CalculatedQuantity('AddOne', lambda x: x+1, AttributeQuantity('desired'))
a = A()

if quantity(a) != 1730:
    raise AssertionError("CalculatedQuantity gave incorrect answer somehow.")
