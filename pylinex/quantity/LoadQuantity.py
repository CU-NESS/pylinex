"""
File: pylinex/quantity/LoadQuantity.py
Author: Keith Tauscher
Date: 13 Feb 2018

Description: File containing function which loads Quantity objects from hdf5
             file groups.
"""
from .ConstantQuantity import ConstantQuantity
from .AttributeQuantity import AttributeQuantity
from .FunctionQuantity import FunctionQuantity
from .CompiledQuantity import CompiledQuantity

def load_quantity_from_hdf5_group(group):
    """
    Loads Quantity object from the given hdf5 file group.
    
    group: hdf5 file group from which to load Quantity object
    """
    try:
        class_name = group.attrs['class']
    except:
        raise TypeError("The given hdf5 file group does not seem to " +\
            "contain a Quantity.")
    try:
        cls = eval(class_name)
    except:
        raise TypeError("Class name was not consistent with a Quantity " +\
            "object existing in the given hdf5 file group.")
    return cls.load_from_hdf5_group(group)

