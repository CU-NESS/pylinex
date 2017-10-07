"""
"""
import numpy as np
from ..util import Savable
from .Expander import Expander

class ExpanderSet(Savable):
    """
    """
    def __init__(self, data, error, **expanders):
        """
        """
        self.error = error
        self.data = data
        self.expanders = expanders
    
    @property
    def error(self):
        """
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        """
        value = np.array(value)
        if (value.ndim == 1) and (len(value) > 1):
            self._error = value
        else:
            raise ValueError("error was either not 1D or had length 1.")
    
    @property
    def expanders(self):
        """
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        """
        if isinstance(value, dict):
            new_expanders = {}
            new_expected_channels = {}
            for name in value:
                if isinstance(value[name], Expander):
                    try:
                        new_expected_channels[name] =\
                            value[name].original_space_size(self.num_channels)
                    except:
                        raise ValueError(("Expander object given for key, " +\
                            "'{0}', was not compatible with the given " +\
                            "number of channels ({1}).").format(name,\
                            self.num_channels))
                    else:
                        new_expanders[name] = value[name]
                else:
                    raise TypeError(("The value of the expanders " +\
                        "dictionary with key, '{0}', was not an Expander " +\
                        "object. It was a {1!s}.").format(name,\
                        type(value[name])))
            self._expanders = new_expanders
            self._expected_channels = new_expected_channels
        else:
            raise TypeError(("expanders was set to a {!s}. It should be a " +\
                "dict.").format(type(value)))
    
    @property
    def expected_channels(self):
        """
        """
        if not hasattr(self, '_expected_channels'):
            raise AttributeError("expected_channels was referenced before " +\
                "it was set. This is strange as it should be set when the " +\
                "expanders dictionary was set.")
        return self._expected_channels
    
    @property
    def data(self):
        """
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        """
        value = np.array(value)
        if (value.ndim in [1, 2]) and (value.shape[-1] == self.num_channels):
            self._data = value
        else:
            raise ValueError("data must be 1D or 2D to be acted upon with " +\
                "expanders.")
    
    @property
    def num_channels(self):
        """
        """
        return self.error.shape[-1]
    
    def marginalize(self, name, true_curve):
        """
        """
        if name in self.expanders:
            true_curve = np.array(true_curve)
            if true_curve.shape[-1] == self.expected_channels[name]:
                new_data = self.data - self.expanders[name](true_curve)
                return ExpanderSet(new_data, self.error,\
                    **{key: self[key] for key in self if key != name})
            else:
                raise ValueError(("The length of the given true curve " +\
                    "({0}) was not the expected value ({1}).").format(\
                    true_curve.shape[-1], self.expected_channels[name]))
        else:
            raise KeyError(("The key to marginalize over ({}) was not " +\
                "in this ExpanderSet.").format(name))
    
    def channels_affected(self, name):
        """
        """
        if name in self.expanders:
            expander = self.expanders[name]
            original_space_size =\
                expander.original_space_size(self.num_channels)
            return expander.channels_affected(original_space_size)
        else:
            raise KeyError(("The key to check for affected channels ({}) " +\
                "was not in this ExpanderSet.").format(name))
    
    def channels_unaffected(self, name):
        """
        """
        channels = np.arange(self.num_channels)
        is_affected = np.isin(channels, self.channels_affected(name))
        is_unaffected = np.logical_not(is_affected)
        unaffected_indices = is_unaffected.nonzero()[0]
        return channels[unaffected_indices]
    
    @property
    def separable(self):
        """
        """
        if not hasattr(self, '_separable'):
            affected = np.zeros(self.num_channels, dtype=bool)
            channels = np.arange(self.num_channels)
            self._separable = True
            for name in self.expanders:
                newly_affected =\
                    np.isin(channels, self.channels_affected(name))
                if np.any(np.logical_and(affected, newly_affected)):
                    self._separable = False
                    break
                else:
                    affected = np.logical_or(affected, newly_affected)
        return self._separable
    
    def separate(self):
        """
        """
        if not self.separable:
            raise RuntimeError("Separation can't be performed on this " +\
                "ExpanderSet because the separable property is False.")
        curves = {}
        channels = np.arange(self.num_channels)
        for name in self.expanders:
            dummy_data = self.data.copy()
            dummy_data[...,self.channels_unaffected(name)] = 0.
            inferred = self.expanders[name].invert(dummy_data, self.error)
            curves[name] = inferred
        residual = self.data.copy()
        for name in self.expanders:
            residual = residual - self.expanders[name](curves[name])
        return (curves, residual)
    
    def reset_data(self, new_data):
        """
        """
        return ExpanderSet(new_data, self.error, **self.expanders)

    def __contains__(self, key):
        """
        """
        return (key in self.expanders)
    
    def __getitem__(self, key):
        """
        """
        try:
            return self.expanders[key]
        except KeyError:
            raise KeyError(("{} was not a valid key for this " +\
                "ExpanderSet.").format(key))
    
    def __iter__(self):
        """
        """
        return iter(self.expanders.keys())
    
    def values(self):
        """
        """
        return iter(self.expanders.values())
    
    def fill_hdf5_group(self, group):
        """
        """
        group.create_dataset('__data__', data=self.data)
        group.create_dataset('__error__', data=self.error)
        for name in self.expanders:
            self.expanders[name].fill_hdf5_group(group.create_group(name))

