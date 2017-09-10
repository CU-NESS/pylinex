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
