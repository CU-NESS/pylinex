"""
File: pylinex/expander/LoadExpander.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing function which load Expander objects from hdf5
             file group.
"""
from .NullExpander import NullExpander
from .PadExpander import PadExpander
from .RepeatExpander import RepeatExpander
from .ModulationExpander import ModulationExpander
from .MatrixExpander import MatrixExpander
from .CompositeExpander import CompositeExpander
from .ShapedExpander import ShapedExpander

def load_expander_from_hdf5_group(group):
    """
    Loads an Expander object from the given hdf5 file group.
    
    group: hdf5 file group from which to load data with which to recreate an
           Expander object
    
    returns: Expander object of appropriate type
    """
    try:
        class_name = group.attrs['class']
    except:
        raise ValueError("The given hdf5 file does not appear to contain " +\
                         "an Expander.")
    if class_name == 'NullExpander':
        return NullExpander()
    elif class_name == 'PadExpander':
        pads_before = group.attrs['pads_before']
        pads_after = group.attrs['pads_after']
        pad_value = group.attrs['pad_value']
        return PadExpander(pads_before, pads_after, pad_value)
    elif class_name == 'RepeatExpander':
        nrepeats = group.attrs['nrepeats']
        return RepeatExpander(nrepeats)
    elif class_name == 'ModulationExpander':
        modulating_factors = group['modulating_factors'].value
        return ModulationExpander(modulating_factors)
    elif class_name == 'MatrixExpander':
        matrix = group['matrix'].value
        return MatrixExpander(matrix)
    elif class_name == 'CompositeExpander':
        expanders = []
        iexpander = 0
        while ('expander_{}'.format(iexpander)) in group:
            subgroup = group['expander_{}'.format(iexpander)]
            expanders.append(load_expander_from_hdf5_group(subgroup))
            iexpander += 1
        return CompositeExpander(*expanders)
    elif class_name == 'ShapedExpander':
        input_shape = tuple(group.attrs['input_shape'])
        output_shape = tuple(group.attrs['output_shape'])
        expander = load_expander_from_hdf5_group(group['expander'])
        return ShapedExpander(expander, input_shape, output_shape)
    else:
        raise ValueError("The given hdf5 group does not appear to contain " +\
                         "an Expander")

