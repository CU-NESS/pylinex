"""
File: examples/quantity/AttributeQuantity.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how the AttributeQuantity class. Basically, it
             defers the calling of an attribute/property until desired but
             still encodes what will be called.
"""
from pylinex import AttributeQuantity

class A(object):
    @property
    def desired(self):
        """
        Returns the first "taxi cab number", 1729.
        """
        return 1729

a = A()
quantity = AttributeQuantity('desired')

if quantity(a) != 1729:
    raise AssertionError("For some reason, quantity did not successfully " +\
                         "find attribute.")

