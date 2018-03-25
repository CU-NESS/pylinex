"""
File: pylinex/quantity/FunctionQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a quantity that is calculated
             through the use of a member function of an object. It stores
             arguments and keyword arguments to pass onto the member function
             when called. When called, it must be given an object on which to
             call the function.
"""
from ..util import sequence_types, Savable, Loadable, create_hdf5_dataset,\
    get_hdf5_value
from .Quantity import Quantity

class FunctionQuantity(Quantity, Savable, Loadable):
    """
    Class representing a quantity that is calculated through the use of a
    member function of an object. It stores the arguments and keyword arguments
    which must be passed on to the function when called. When called, it must
    be given an object on which to call the function.
    """
    def __init__(self, function_name, *function_args, **function_kwargs):
        """
        Initializes a new FunctionQuantity with the given function name and
        arguments.
        
        function_name: name of the function to call when this object is called
        *function_args: unpacked list of arguments to pass on to the function
        **function_kwargs: unpacked dict of keyword arguments to pass on to the
                           function
        """
        Quantity.__init__(self, function_name)
        self.function_args = function_args
        self.function_kwargs = function_kwargs
    
    @property
    def function_args(self):
        """
        Property storing the packed list of arguments to pass onto the function
        when this object is called.
        """
        if not hasattr(self, '_function_args'):
            raise AttributeError("function_args referenced before it was set.")
        return self._function_args
    
    @function_args.setter
    def function_args(self, value):
        """
        Setter for the function_args property.
        
        value: must be a sequence of some kind
        """
        if type(value) in sequence_types:
            self._function_args = list(value)
        else:
            raise TypeError("function_args was set to a non-sequence.")
    
    @property
    def function_kwargs(self):
        """
        Property storing the keyword arguments to pass on to the function when
        this object is called.
        """
        if not hasattr(self, '_function_kwargs'):
            raise AttributeError("function_kwargs referenced before it was " +\
                                 "set.")
        return self._function_kwargs
    
    @function_kwargs.setter
    def function_kwargs(self, value):
        """
        Setter for the function_kwargs property.
        
        value: must be a dict
        """
        if isinstance(value, dict):
            self._function_kwargs = value
        else:
            raise TypeError("function_kwargs was set to a non-dict.")
    
    def __call__(self, container, **kwargs):
        """
        Calls this FunctionQuantity.
        
        container: the object on which to call the function
        **kwargs: unpacked list of unused keyword arguments for compatibility
        
        returns: the return value of the function which underlies this Quantity
        """
        return eval("container." + self.name + "(*self.function_args, " +\
                    "**self.function_kwargs)")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with data about this FunctionQuantity.
        
        group: hdf5 file group to fill with data about this FunctionQuantity
        """
        group.attrs['class'] = 'FunctionQuantity'
        group.attrs['name'] = self.name
        subgroup = group.create_group('args')
        for iarg in range(len(self.function_args)):
            create_hdf5_dataset(subgroup, 'arg_{}'.format(iarg),\
                data=self.function_args[iarg])
        subgroup = group.create_group('kwargs')
        for key in self.function_kwargs:
            create_hdf5_dataset(subgroup, key, data=self.function_kwargs[key])
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a FunctionQuantity from the given hdf5 group.
        
        group: hdf5 file group from which to load a FunctionQuantity
        
        returns: FunctionQuantity loaded from given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'FunctionQuantity'
        except:
            raise TypeError("This hdf5 file group does not seem to contain " +\
                "a FunctionQuantity.")
        name = group.attrs['name']
        args = []
        iarg = 0
        subgroup = group['args']
        while 'arg_{}'.format(iarg) in subgroup:
            args.append(get_hdf5_value(subgroup['arg_{}'.format(iarg)]))
            iarg += 1
        subgroup = group['kwargs']
        kwargs = {}
        for key in subgroup:
            kwargs[key] = get_hdf5_value(subgroup[key])
        return FunctionQuantity(name, *args, **kwargs)

