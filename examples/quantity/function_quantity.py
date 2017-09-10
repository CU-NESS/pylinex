from pylinex import FunctionQuantity

class A(object):
    def foo(self, *args, **kwargs):
        return_val = 11
        for arg in args:
            return_val += arg
        for key in kwargs:
            return_val += kwargs[key]
        return return_val

specific_args = (3,)
specific_kwargs = {'kwarg': 4}
quantity = FunctionQuantity('foo', *specific_args, **specific_kwargs)

a = A()
if quantity(a) != 18:
    raise AssertionError("FunctionQuantity test failed.")
