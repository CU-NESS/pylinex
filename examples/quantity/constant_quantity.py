from pylinex import ConstantQuantity

class A(object):
    pass

a = A()
quantity = ConstantQuantity(3)
if quantity(a) != 3:
    raise AssertionError("ConstantQuantity didn't return the constant with " +\
                         "which it was associated.")
