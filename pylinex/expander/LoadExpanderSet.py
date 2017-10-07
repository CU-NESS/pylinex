"""
"""
from .ExpanderSet import ExpanderSet
from .LoadExpander import load_expander_from_hdf5_group

def load_expander_set_from_hdf5_group(group):
    data = group['__data__'].value
    error = group['__error__'].value
    expanders = {}
    for name in group:
        if name not in ['__data__', '__error__']:
            expanders[name] = load_expander_from_hdf5_group(group[name])
    return ExpanderSet(data, error, **expanders)


