"""
"""
from ..util import get_hdf5_value
from .ExpanderSet import ExpanderSet
from .LoadExpander import load_expander_from_hdf5_group

def load_expander_set_from_hdf5_group(group):
    data = get_hdf5_value(group['__data__'])
    error = get_hdf5_value(group['__error__'])
    expanders = {}
    for name in group:
        if name not in ['__data__', '__error__']:
            expanders[name] = load_expander_from_hdf5_group(group[name])
    return ExpanderSet(data, error, **expanders)


