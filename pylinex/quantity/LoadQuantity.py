"""
File: pylinex/quantity/LoadQuantity.py
Author: Keith Tauscher
Date: 22 Sep 2017

Description: File containing function which loads Quantity objects from hdf5
             file groups.
"""
from .ConstantQuantity import ConstantQuantity
from .AttributeQuantity import AttributeQuantity
from .FunctionQuantity import FunctionQuantity
from .CompiledQuantity import CompiledQuantity

def load_quantity_from_hdf5_group(self, group):
    """
    Loads Quantity object from the given hdf5 file group.
    
    group: hdf5 file group from which to load Quantity object
    """
    try:
        name = group.attrs['name']
        class_name = group.attrs['class']
    except:
        raise ValueError("name and class weren't in hdf5 group from which " +\
            "to load Quantity.")
    if class_name == 'ConstantQuantity':
        constant = group.attrs['constant']
        return ConstantQuantity(constant, name=name)
    elif class_name == 'AttributeQuantity':
        attribute_name = group.attrs['attribute_name']
        return AttributeQuantity(attribute_name, name=name)
    elif class_name == 'FunctionQuantity':
        args = []
        iarg = 0
        subgroup = group['args']
        while 'arg_{}'.format(iarg) in subgroup.attrs:
            args.append(subgroup.attrs['arg_{}'.format(iarg)])
            iarg += 1
        subgroup = group['kwargs']
        for key in subgroup.attrs:
            kwargs[key] = subgroup.attrs[key]
        return FunctionQuantity(name, *args, **kwargs)
    elif class_name == 'CompiledQuantity':
        iquantity = 0
        quantities = []
        while 'quantity_{}'.format(iquantity) in group:
            subgroup = group['quantity_{}'.format(iquantity)]
            quantities.append(load_quantity_from_hdf5_group(subgroup))
            iquantity += 1
        return CompiledQuantity(name, *quantities)
    else:
        raise ValueError("Class of Quantity was not recognized.")
    

