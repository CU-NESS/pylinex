"""
File: examples/quantity/compiled_quantity.py
Author: Keith Tauscher
Date: 10 Sep 2017

Description: Example showing how to use the CompiledQuantity which is really
             just a bundle of Quantity objects into a single Quantity.
"""
from pylinex import AttributeQuantity, CompiledQuantity

class A(object):
    @property
    def b(self):
        return 127
    @property
    def c(self):
        return 255

a = A()
(quantity_b, quantity_c) = (AttributeQuantity('b'), AttributeQuantity('c'))
compiled_quantity = CompiledQuantity('mixed', quantity_b, quantity_c)
if compiled_quantity(a) != [127, 255]:
    raise AssertionError("CompiledQuantity test failed.")
