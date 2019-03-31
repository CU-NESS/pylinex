"""
File: pylinex/expander/ExpanderList.py
Author: Keith Tauscher
Date: 29 Mar 2019

Description: File containing a class representing a list of Expander objects.
"""
from __future__ import division
from ..util import int_types, sequence_types, Savable, Loadable
from .Expander import Expander
from .NullExpander import NullExpander
from .CompositeExpander import CompositeExpander
from .LoadExpander import load_expander_from_hdf5_group
from .ExpanderSet import ExpanderSet

class ExpanderList(Savable, Loadable):
    """
    Class representing a sequence of Expander objects.
    """
    def __init__(self, *expanders):
        """
        Initializes a new ExpanderList.
        
        expanders: Expander objects or None's
        """
        self.expanders = expanders
    
    @property
    def expanders(self):
        """
        Property storing list of Expander objects at the heart of this object.
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Setter for the Expander objects at the heart of this object.
        
        value: sequence of Expander objects or objects which can be cast to
               Expander objects
        """
        if type(value) in sequence_types:
            if all([((element is None) or isinstance(element, Expander))\
                for element in value]):
                self._expanders =\
                    [(NullExpander() if (element is None) else element)\
                    for element in value]
            else:
                raise ValueError("Not all elements of the expanders " +\
                    "sequence could be cast to Expander objects.")
        else:
            raise TypeError("expanders was set to a non-sequence.")
    
    @property
    def num_expanders(self):
        """
        Property storing the number of expanders in this ExpanderList.
        """
        if not hasattr(self, '_num_expanders'):
            self._num_expanders = len(self.expanders)
        return self._num_expanders
    
    def __len__(self):
        """
        Finds the length of this ExpanderList.
        
        returns: number of Expander objects in this ExpanderList
        """
        return self.num_expanders
    
    def apply(self, unexpanded, should_sum=False):
        """
        Expands the given point from the unexpanded space to the expanded
        space.
        
        unexpanded: sequence of unexpanded curves
        should_sum: if True, results are summed together, default: False
        
        returns: the expanded version of the curves in unexpanded
        """
        if len(unexpanded) == self.num_expanders:
            if should_sum:
                result = None
                for (expander, curves) in zip(self.expanders, unexpanded):
                    if result is None:
                        result = expander(curves)
                    else:
                        result = result + expander(curves)
                return result
            else:
                return [expander(curves)\
                    for (expander, curves) in zip(self.expanders, unexpanded)]
        else:
            raise ValueError("The sequence of unexpanded curves given to " +\
                "ExpanderList did not have the correct length (it should " +\
                "be the same as the number of Expanders).")
    
    def __call__(self, unexpanded, should_sum=False):
        """
        Expands the given point from the unexpanded space to the expanded
        space.
        
        unexpanded: sequence of unexpanded curves
        should_sum: if True, results are summed together, default: False
        
        returns: the expanded version of the curves in unexpanded
        """
        return self.apply(unexpanded, should_sum=should_sum)
    
    def contract_errors(self, error):
        """
        Contracts the given expanded error into a sequence of errors in the
        original spaces of each expander.
        
        error: 1D array in the expanded space
        
        returns: sequence of 1D arrays in the original spaces
        """
        return [expander.contract_error(error) for expander in self.expanders]
    
    def copy(self):
        """
        Returns a deep copy of this ExpanderList.
        """
        return ExpanderList(*[expander.copy() for expander in self.expanders])
    
    def invert(self, data, error, return_residual=False):
        """
        Inverts the data into original space curves. If this is not possible
        because Expanders are coupled, a ValueError is thrown.
        
        data: expanded space data from which to infer original space curves
        error: error on expanded space data
        return_residual: if True, residual of inversion is returned as well
        
        returns: sequence of original space curves or tuple whose first element
                 is that sequence and whose second and last element is the
                 residual of the inverted fit
        """
        expander_dict = {'a{:d}'.format(iexpander): expander\
            for (iexpander, expander) in enumerate(self.expanders)}
        expander_set = ExpanderSet(data, error, **expander_dict)
        (curves, residual) = expander_set.separate()
        curves = [curves['a{:d}'.format(index)]\
            for index in range(self.num_expanders)]
        return ((curves, residual) if return_residual else curves)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this
        ExpanderList.
        
        group: hdf5 file group to fill with information about this
               ExpanderList
        """
        for (iexpander, expander) in enumerate(self.expanders):
            subgroup = group.create_group('expander_{}'.format(iexpander))
            expander.fill_hdf5_group(subgroup)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a ExpanderList object from the given hdf5 file group.
        
        group: the hdf5 file group from which to load a ExpanderList
        
        returns: a ExpanderList object derived from the give hdf5 file group
        """
        expanders = []
        while 'expander_{}'.format(len(expanders)) in group:
            subgroup = group['expander_{}'.format(len(expanders))]
            expanders.append(load_expander_from_hdf5_group(subgroup))
        return ExpanderList(*expanders)
    
    def __iter__(self):
        """
        Returns an iterator over this ExpanderList. Since ExpanderList
        objects are their own iterators, this method returns self after
        resetting the internal iteration variable.
        """
        self._iteration = 0
        return self
    
    def __next__(self):
        """
        Finds the next element in the iteration over this ExpanderList.
        
        returns: next Expander object in this ExpanderList
        """
        return self.next()
    
    def next(self):
        """
        Finds the next element in the iteration over this ExpanderList.
        
        returns: next Expander object in this ExpanderList
        """
        if self._iteration == self.num_expanders:
            del self._iteration
            raise StopIteration
        to_return = self.expanders[self._iteration]
        self._iteration = self._iteration + 1
        return to_return
    
    def append(self, expander):
        """
        Appends the given Expander object (or object castable as a expander)
        to this ExpanderList.
        
        expander: must be an Expander or None
        """
        if expander is None:
            self.expanders.append(NullExpander())
        elif isinstance(expander, Expander):
            self.expanders.append(expander)
        else:
            raise TypeError("Given expander was neither an Expander " +\
                "object nor an object which could be cast to an Expander " +\
                "object.")
        if hasattr(self, '_num_expanders'):
            delattr(self, '_num_expanders')
    
    def extend(self, expander_list):
        """
        Extends this ExpanderList by concatenating the Expander objects
        stored in this ExpanderList with the ones stored in expander_list.
        """
        if isinstance(expander_list, ExpanderList):
            self.expanders.extend(expander_list.expanders)
        elif type(expander_list) in sequence_types:
            expander_list = ExpanderList(*expander_list)
            self.expanders.extend(expander_list.expanders)
        else:
            raise TypeError("Can only extend ExpanderList objects with " +\
                "other ExpanderList objects or by sequences which can be " +\
                "used to initialize an ExpanderList object.")
        if hasattr(self, '_num_expanders'):
            delattr(self, '_num_expanders')
    
    def __add__(self, other):
        """
        "Adds" this ExpanderList to other by returning a new ExpanderList
        object with the Expanders in both objects.
        
        other: either a Expander (or something castable as a Expander) or a
               ExpanderList
        
        returns: an ExpanderList composed of all of the Expanders in this
                 ExpanderList as well as the Expander(s) in other
        """
        if isinstance(other, ExpanderList):
            return ExpanderList(*(self.expanders + other.expanders))
        elif other is None:
            return ExpanderList(*(self.expanders + [NullExpander()]))
        elif isinstance(other, Expander):
            return ExpanderList(*(self.expanders + [other]))
        else:
            raise TypeError("The only things which can be added to an " +\
                "ExpanderList is another ExpanderList or an object which " +\
                "can be cast to an Expander.")
    
    def __iadd__(self, other):
        """
        "Adds" other to this ExpanderList to other by appending/extending its
        expanders to this object. Note that this does not create a new
        ExpanderList.
        
        other: either an Expander (or something castable as a Expander) or a
               ExpanderList
        """
        if isinstance(other, ExpanderList):
            self.extend(other)
        elif (other is None) or isinstance(other, Expander):
            self.append(other)
        else:
            raise TypeError("The only things which can be added to a " +\
                "ExpanderList is another ExpanderList or an object which " +\
                "can be cast to an Expander.")
        return self
    
    def __mul__(self, other):
        """
        "Multiplies" other by this ExpanderList by forming a new ExpanderList
        of composite expanders with this ExpanderList's Expanders forming
        the inner expanders and other's Expanders forming the outer
        expanders.
        
        other: must be a ExpanderList object with the same number of Expander
               objects
        
        returns: ExpanderList of combined expanders with this ExpanderList
                 holding the inner expander and other holding outer expanders
        """
        if not isinstance(other, ExpanderList):
            raise TypeError("ExpanderList objects can only be multiplied " +\
                "by other ExpanderList objects.")
        if self.num_expanders != other.num_expanders:
            raise ValueError("ExpanderList objects can only be multiplied " +\
                "by ExpanderList objects of the same length.")
        expanders = []
        for (inner, outer) in zip(self.expanders, other.expanders):
            if isinstance(inner, NullExpander):
                expanders.append(outer)
            elif isinstance(outer, NullExpander):
                expanders.append(inner)
            else:
                expanders.append(CompositeExpander(inner, outer))
        return ExpanderList(*expanders)
    
    def __getitem__(self, index):
        """
        Gets a specific element or set of elements of the Expanders sequence.
        
        index: the index of the element(s) to retrieve. Can be an integer, a
               slice, or a sequence of integers.
        
        returns: an Expander object or an ExpanderList object
        """
        if type(index) in int_types:
            return self.expanders[index]
        elif isinstance(index, slice):
            return ExpanderList(*self.expanders[index])
        elif type(index) in sequence_types:
            if all([type(element) in int_types for element in index]):
                return ExpanderList(*[self.expanders[element]\
                    for element in index])
            else:
                raise TypeError("Not all elements of sequence index were " +\
                    "integers.")
        else:
            raise TypeError("index type not recognized.")
    
    def __eq__(self, other):
        """
        Checks if other is a ExpanderList with the same Expanders as this
        one.
        
        other: object to check for equality
        
        returns: True iff other is ExpanderList with same Expanders
        """
        if isinstance(other, ExpanderList):
            if len(self) == len(other):
                for (s_expander, o_expander) in zip(self, other):
                    if s_expander != o_expander:
                        return False
                return True
            else:
                return False
        else:
            return False
    
    def __ne__(self, other):
        """
        Ensures that (a!=b) == (not (a==b)).
        
        other: object to check for inequality
        
        returns: False iff other is ExpanderList with same Expanders
        """
        return (not self.__eq__(other))

