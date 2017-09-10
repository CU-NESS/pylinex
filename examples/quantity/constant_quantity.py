"""
File: examples/quantity/constant_quantity.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the ConstantQuantity class which
             returns a predefined constant no matter what object it is passed.
"""
from pylinex import ConstantQuantity

class A(object):
    pass

a = A()
quantity = ConstantQuantity(3)
if quantity(a) != 3:
    raise AssertionError("ConstantQuantity didn't return the constant with " +\
                         "which it was associated.")
