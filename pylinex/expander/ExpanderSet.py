"""
File: pylinex/expander/ExpanderSet.py
Author: Keith Tauscher
Date: 8 Oct 2017

Description: File containing a container for many Expander objects associated
             with different names.
"""
import numpy as np
from ..util import Savable, Loadable, create_hdf5_dataset, get_hdf5_value
from .Expander import Expander
from .LoadExpander import load_expander_from_hdf5_group

class ExpanderSet(Savable, Loadable):
    """
    Container class for many Expander objects associated with string names.
    """
    def __init__(self, data, error, **expanders):
        """
        Initializes a new ExpanderSet with the given contents.
        
        data: the data from which to infer "true" curves
        error: the error from which to define inner product
        expanders: dictionary of expanders to store in this object
        """
        self.error = error
        self.data = data
        self.expanders = expanders
    
    @property
    def error(self):
        """
        Property storing the error with which to define the inner product.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error property
        
        value: must be a 1D numpy.ndarray
        """
        value = np.array(value)
        if (value.ndim == 1) and (len(value) > 1):
            self._error = value
        else:
            raise ValueError("error was either not 1D or had length 1.")
    
    @property
    def expanders(self):
        """
        Property storing the dictionary of Expanders at the heart of this
        ExpanderSet.
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Setter for the expanders dictionary property.
        
        value: a dictionary filled with compatible Expander objects (meaning
               they can all be summed together given appropriately sized input
               bases)
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
        Property storing a dictionary whose keys are the names of this
        ExpanderSet and whose values are the input number of channels expected
        for those names.
        """
        if not hasattr(self, '_expected_channels'):
            raise AttributeError("expected_channels was referenced before " +\
                "it was set. This is strange as it should be set when the " +\
                "expanders dictionary was set.")
        return self._expected_channels
    
    @property
    def data(self):
        """
        Property storing a numpy.ndarray of data corresponding to the size of
        the error.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data of this ExpanderSet
        
        value: numpy.ndarray of same shape as error
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
        Property storing the number of channels in the data/error.
        """
        return self.error.shape[-1]
    
    def marginalize(self, name, true_curve):
        """
        "Marginalizes" this ExpanderSet over the given name by subtracting the
        expanded version of the given true curve.
        
        name: string name of component over which to marginalize
        true_curve: the true value (i.e. the one associated with the data) of
                    the data component associated with name
        
        returns: ExpanderSet object with the same expanders except the one
                 associated with name is removed and the data has true_curve
                 subtracted
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
        Finds the channels affected by the given component.
        
        name: string name of the desired component
        
        returns: numpy.ndarray of channels affected by the Expander associated
                 with the given name
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
        Finds the channels unaffected by the given component.
        
        name: string name of the desired component
        
        returns: numpy.ndarray of channels affected by the Expander associated
                 with the given name
        """
        channels = np.arange(self.num_channels)
        is_affected = np.isin(channels, self.channels_affected(name))
        is_unaffected = np.logical_not(is_affected)
        unaffected_indices = is_unaffected.nonzero()[0]
        return channels[unaffected_indices]
    
    @property
    def separable(self):
        """
        Property storing a boolean describing whether or not it is possible in
        principle to completely separate the different components (i.e. the
        channels they affect do not overlap).
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
        Separates the different components of the data.
        
        returns: if this ExpanderSet is separable, returns tuple whose first
                 element is a dictionary whose keys are names of components and
                 whose values are the corresponding separated curves and whose
                 second element is the residual after separation.
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
        Sets the data of thie ExpanderSet to a different array.
        
        new_data: numpy.ndarray to set self.data to
        """
        return ExpanderSet(new_data, self.error, **self.expanders)

    def __contains__(self, key):
        """
        Checks whether there is an Expander in this ExpanderSet with
        identifying string given by key.
        
        key: name to check
        
        returns: True if there is an Expander object associated with key
                 False otherwise
        """
        return (key in self.expanders)
    
    def __getitem__(self, key):
        """
        Gets the Expander associated with the given key.
        
        key: name associated with desired Expander object
        
        returns: Expander object associated with key
        """
        try:
            return self.expanders[key]
        except KeyError:
            raise KeyError(("{} was not a valid key for this " +\
                "ExpanderSet.").format(key))
    
    def __iter__(self):
        """
        Returns an iterator over the names associated with this ExpandeSet.
        """
        return iter(self.expanders.keys())
    
    def values(self):
        """
        Returns an iterator over the Expanders corresponding to the names
        associated with this ExpanderSet.
        """
        return iter(self.expanders.values())
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None):
        """
        Fills an hdf5 group with information about this ExpanderSet.
        
        group: the hdf5 group to fill with information
        data_link: link to existing data dataset, if it exists (see
                   create_hdf5_dataset docs for info about accepted formats)
        error_link: link to existing error dataset, if it exists (see
                   create_hdf5_dataset docs for info about accepted formats)
        """
        data_link = create_hdf5_dataset(group, '__data__', data=self.data,\
            link=data_link)
        error_link = create_hdf5_dataset(group, '__error__', data=self.error,\
            link=error_link)
        for name in self.expanders:
            self.expanders[name].fill_hdf5_group(group.create_group(name))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an ExpanderSet object from the given hdf5 file group.
        
        group: hdf5 file group which ExpanderSet was saved in using one of
               fill_hdf5_group or save methods
        
        returns: ExpanderSet object loaded from the given hdf5 file group
        """
        data = get_hdf5_value(group['__data__'])
        error = get_hdf5_value(group['__error__'])
        expanders = {}
        for name in group:
            if name not in ['__data__', '__error__']:
                expanders[name] = load_expander_from_hdf5_group(group[name])
        return ExpanderSet(data, error, **expanders)

