"""
File: pylinex/util/VariableGrid.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing an object which can store
             dimensions of a grid of arbitrary dimension, D1, embedded in a
             space of arbitrary dimension, D2 (where D2>=D1). Each individual
             dimension of the smaller grid can traverse any finite path through
             the larger space. Each individual dimension of the larger space
             has a name and all of them must have values in each point of the
             grid on the smaller space.
"""
import numpy as np
from distpy import sequence_types
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class VariableGrid(object):
    """
    Class representing an object which can store dimensions of a grid of
    arbitrary dimension, D1, embedded in a space of arbitrary dimension, D2
    (where D2>=D1). Each individual dimension of the smaller grid can traverse
    any finite path through the larger space. Each individual dimension of the
    larger space has a name and all of them must have values in each point of
    the grid on the smaller space.
    """
    @property
    def names(self):
        """
        Property storing the string names of each basis.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("names was referenced before it was set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Allows user to set names property.
        
        value: must be sequence of strings
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._names = value
            else:
                raise TypeError("Not every element of names was a string.")
        else:
            raise TypeError("names was set to a non-sequence.")
    
    @property
    def dimensions(self):
        """
        Property storing the dimensions of the grids to calculate. It takes the
        form of a list of lists of dictionaries containing subsets.
        """
        if not hasattr(self, '_dimensions'):
            raise AttributeError("dimensions was referenced before it was " +\
                                 "set somehow.")
        return self._dimensions
    
    @dimensions.setter
    def dimensions(self, value):
        """
        Setter for the dimensions property.
        
        value: must be a list (whose length is the number of dimensions of the
               desired grid) whose values are dictionaries (which have, as
               keys, names of the dimensions of the larger space and have, as
               values, numpy.ndarray objects giving dimension's range; note
               that all ranges of each dictionary must be the same length)
        """
        type_error = TypeError("dimensions should be a list of " +\
                               "dictionaries of arrays.")
        if type(value) in sequence_types:
            self._shape = []
            self._dimensions_by_name = {name: None for name in self.names}
            self._maxima = {name: -np.inf for name in self.names}
            self._minima = {name: np.inf for name in self.names}
            for (idimension, dimension) in enumerate(value):
                if isinstance(dimension, dict):
                    variable_range_lengths = []
                    for name in dimension:
                        previous_idimension = self._dimensions_by_name[name]
                        if type(previous_idimension) is type(None):
                            self._dimensions_by_name[name] = idimension
                        else:
                            raise KeyError(("Variable, {0!s}, was given in " +\
                                "both dimension #{1} and dimension " +\
                                "#{2}.").format(name, previous_idimension,\
                                idimension))
                        variable_range = dimension[name]
                        if type(variable_range) in sequence_types:
                            variable_range = np.array(variable_range)
                            if variable_range.ndim != 1:
                                raise type_error
                            variable_range_lengths.append(len(variable_range))
                            variable_range_minimum = np.min(variable_range)
                            variable_range_maximum = np.max(variable_range)
                            if variable_range_minimum < self._minima[name]:
                                self._minima[name] = variable_range_minimum
                            if variable_range_maximum > self._maxima[name]:
                                self._maxima[name] = variable_range_maximum
                        else:
                            raise type_error
                    variable_range_lengths = np.array(variable_range_lengths)
                    if np.all(variable_range_lengths ==\
                        variable_range_lengths[0]):
                        self._shape.append(variable_range_lengths[0])
                    else:
                        raise ValueError("Not all arrays in a given " +\
                                         "dimension dictionary were the " +\
                                         "same length.")
                else:
                    raise type_error
            for name in self._dimensions_by_name:
                if type(self._dimensions_by_name[name]) is type(None):
                    raise KeyError("The grid didn't use the variable, '" +\
                                   name + "'.")
        else:
            raise type_error
        self._shape = tuple(self._shape)
        self._dimensions = value
    
    @property
    def dimensions_by_name(self):
        """
        Property storing a dictionary whose keys are the names of variables and
        whose values are the indices of the dimension where the name is varied.
        """
        if not hasattr(self, '_dimensions_by_name'):
            raise AttributeError("dimensions_by_name was referenced before " +\
                                 "it was set. This shouldn't happen " +\
                                 "because it should be set automatically " +\
                                 "when dimensions is set.")
        return self._dimensions_by_name
    
    @property
    def shape(self):
        """
        Property storing the shape of the grids to calculate.
        """
        if not hasattr(self, '_shape'):
            raise AttributeError("shape was referenced before it was set. " +\
                                 "This shouldn't happen because the shape " +\
                                 "should be set automatically when " +\
                                 "dimensions are set.")
        return self._shape
    
    @property
    def ndim(self):
        """
        Property storing the number of dimensions in this grid.
        """
        if not hasattr(self, '_ndim'):
            self._ndim = len(self.shape)
        return self._ndim
    
    def point_from_indices(self, indices):
        """
        Retrieves a given point in the full space from the indices of the grid.
        
        indices: sequence whose values are the indices of each grid dimension
        
        returns: dictionary whose keys are the names of the variables and the
                 values are the indices (0<=indices[i]<shape[i] for all
                 0<=i<ndim) of the grid dimensions
        """
        point = {}
        for idimension, dimension in enumerate(self.dimensions):
            index = indices[idimension]
            for name in dimension:
                point[name] = dimension[name][index]
        return point
    
    @property
    def minima(self):
        """
        Property storing the minimum of each dimension in a dictionary indexed
        by variable name.
        """
        if not hasattr(self, '_minima'):
            raise AttributeError("minima was referenced before it was set. " +\
                                 "This shouldn't happen because minima " +\
                                 "should be set automatically when " +\
                                 "dimensions are set.")
        return self._maxima
    
    @property
    def maxima(self):
        """
        Property storing the maximum of each dimension in a dictionary indexed
        by variable name.
        """
        if not hasattr(self, '_maxima'):
            raise AttributeError("maxima was referenced before it was set. " +\
                                 "This shouldn't happen because maxima " +\
                                 "should be set automatically when " +\
                                 "dimensions are set.")
        return self._maxima

