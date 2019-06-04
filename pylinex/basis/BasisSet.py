"""
File: pylinex/basis/BasisSet.py
Author: Keith Tauscher
Date: 30 Oct 2017

Description: File containing a class which is a constainer for Basis objects.
             It doesn't necessarily put any restrictions on the Basis objects
             which can be held.
"""
from ..util import int_types, sequence_types
from .Basis import Basis
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class BasisSet(object):
    """
    Container for Basis objects.
    """
    def __init__(self, names, bases):
        """
        Initializes a new BasisSet with the given names and bases.
        
        names: list of names of subbases
        bases: list of Basis objects corresponding to the given names
        """
        if isinstance(names, list) and isinstance(bases, list):
            if len(names) == len(bases):
                self.names = names
                self.component_bases = bases
            else:
                raise ValueError("Lengths of names and bases are not equal.")
        elif isinstance(names, basestring) and isinstance(bases, Basis):
            self.names = [names]
            self.component_bases = [bases]
        else:
            raise TypeError("names and bases given to BasisSet were not " +\
                            "lists.")
    
    def copy(self):
        """
        Finds and returns a deep copy of this BasisSet.
        
        returns: new BasisSet object with names and bases copied from this one
        """
        return BasisSet([name for name in self.names],\
            [basis.copy() for basis in self.component_bases])
    
    @property
    def component_bases(self):
        """
        Property storing the list of Basis objects which underly this BasisSet.
        """
        if not hasattr(self, '_component_bases'):
            raise AttributeError("For some reason, component_bases haven't " +\
                                 "been set yet.")
        return self._component_bases
    
    @component_bases.setter
    def component_bases(self, value):
        """
        Allows for the component_bases property to be set in the initializer.
        
        value: a list of Basis objects
        """
        if type(value) in sequence_types:
            if all([isinstance(basis, Basis) for basis in value]):
                self._component_bases = [basis for basis in value]
            else:
                raise ValueError("Not all elements of the component_bases " +\
                                 "list were Basis objects.")
        else:
            raise TypeError("component_bases was not a list.")
    
    @property
    def names(self):
        """
        Property storing the names associated with the sets of basis vectors
        underlying this BasisSet.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("For some reason, names has not yet been " +\
                "set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Allows for the setting of names corresponding to each component basis.
        
        value: a sequence of strings of the same length as the component_bases
               property
        """
        if len(value) == len(set(value)):
            self._names = value
        else:
            raise ValueError("Not all names given to BasisSet were unique.")
    
    @property
    def num_bases(self):
        """
        Property storing the integer number of Basis objects contained in this
        BasisSet.
        """
        if not hasattr(self, '_num_bases'):
            self._num_bases = len(self.names)
        return self._num_bases
    
    @property
    def slices_by_name(self):
        """
        Property storing a dictionary connecting the name of each Basis
        underlying this BasisSet to the slice describing the indices of its
        basis vectors.
        """
        if not hasattr(self, '_slices_by_name'):
            rindex = 0
            self._slices_by_name = {None: slice(None)}
            for name in self.names:
                next_rindex = rindex + len(self[name])
                self._slices_by_name[name] = slice(rindex, next_rindex)
                rindex = next_rindex
        return self._slices_by_name
    
    @property
    def parameter_names(self):
        """
        Gets names to associate with the coefficients of this BasisSet. It is a
        list of strings of length num_basis_vectors.
        """
        if not hasattr(self, '_parameter_names'):
            self._parameter_names = sum([['{0!s}_a{1:d}'.format(name, index)\
                for index in range(len(self[name]))] for name in self.names],\
                [])
        return self._parameter_names
    
    @property
    def num_basis_vectors(self):
        """
        Property storing the number of basis functions stored in this BasisSet
        """
        if not hasattr(self, '_num_basis_vectors'):
            self._num_basis_vectors =\
                sum([len(self[name]) for name in self.names])
        return self._num_basis_vectors
    
    def __getitem__(self, key):
        """
        Allows for the usage of square-bracket indexing notation for getting
        sub-bases or truncations of this basis.
        
        key: if key is None, this object is returned
             if key is a string, it is assumed to be the name of a Basis
                                 to return
             if key is a dict, then the keys of the dict should be the names of
                               the component Basis objects underlying this
                               BasisSet and the values are slices with which to
                               take subsets of the Basis objects.
        
        returns: if key is None or a dict, returns a BasisSet object
                 otherwise, returns Basis object
        """
        if isinstance(key, basestring):
            return self.component_bases[self.names.index(key)]
        elif isinstance(key, dict):
            return self.basis_subsets(**key)
        elif type(key) is type(None):
            return self
        elif type(key) in int_types:
            return self.component_bases[key]
        else:
            raise KeyError("key not recognized.")
    
    def __delitem__(self, key):
        """
        Deletes the basis associated with the given string key (i.e. name).
        
        key: string which is one of the names of the bases in this BasisSet
        """
        if key in self.names:
            name_index = self.names.index(key)
            self.names.pop(name_index)
            self.component_bases.pop(name_index)
            del self.sizes[key]
            delattr(self, '_num_bases')
            delattr(self, '_slices_by_name')
            delattr(self, '_num_basis_vectors')
            delattr(self, '_parameter_names')
        else:
            raise ValueError(("There is no basis with key '{!s}' in this " +\
                "BasisSet.").format(key))
    
    @property
    def sizes(self):
        """
        Property storing a dictionary with basis names as keys and the number
        of basis vectors in that basis set as values.
        """
        if not hasattr(self, '_sizes'):
            self._sizes = {name: len(self[name]) for name in self.names}
        return self._sizes
    
    def component_basis_subsets(self, **subsets):
        """
        Finds a subset of each basis in this BasisSet.
        
        **subsets: subsets to take for each name for which basis vectors should
                   be truncated
        
        returns: list of new component bases after subsetting
        """
        if not all([key in self.names for key in subsets]):
            raise ValueError("Cannot subset a basis which is not in the " +\
                             "BasisSet.")
        new_component_bases = []
        for name in self.names:
            if name in subsets:
                new_component_bases.append(self[name][:subsets[name]])
            else:
                new_component_bases.append(self[name])
        return new_component_bases
    
    def basis_subsets(self, **subsets):
        """
        Creates a new BasisSet object with the given subsetted component bases.
        
        **subsets: subsets to take for each name for which basis vectors should
                   be truncated
        
        returns: BasisSet object with subsetted component bases
        """
        return BasisSet(self.names, self.component_basis_subsets(**subsets))
    
    def __iter__(self):
        """
        Returns an iterator over the sets of basis functions in this BasisSet.
        This object acts as its own iterator. This method simply resets private
        attribute values to restart the internal iterator. The return value of
        the iteration is a name string, not a Basis object.
        """
        self._index_of_basis_to_return = 0
        return self
    
    def next(self):
        """
        Since this BasisSet is its own iterator, it must have a next() method
        which returns the next basis. In this case, the "private"
        _index_of_basis_to_return property is used to store the index of the
        name to return.
        """
        if self._index_of_basis_to_return == len(self.names):
            delattr(self, '_index_of_basis_to_return')
            raise StopIteration
        self._index_of_basis_to_return += 1
        return self.names[self._index_of_basis_to_return - 1]
    
    def __next__(self):
        """
        Alias for next included for Python 2/3 compatibility.
        """
        return self.next()
    
    def __contains__(self, key):
        """
        Checks to see if the given key string describes a basis in this
        BasisSet.
        
        key: string name for which to check
        
        returns: True if key is in names. False otherwise
        """
        if isinstance(key, basestring):
            return (key in self.names)
        else:
            raise TypeError("Only strings can be checked for through the " +\
                "__contains__ method.")
    
    def fill_hdf5_group(self, group, basis_links=None, expander_links=None):
        """
        Fills the given hdf5 file group with data about this basis set.
        
        group: the hdf5 file group to fill
        """
        if type(basis_links) is type(None):
            basis_links = [None for name in self.names]
        if type(expander_links) is type(None):
            expander_links = [None for name in self.names]
        for (iname, name) in enumerate(self.names):
            basis_link = basis_links[iname]
            expander_link = expander_links[iname]
            subgroup = group.create_group('basis_{}'.format(iname))
            self[name].fill_hdf5_group(subgroup, basis_link=basis_link,\
                expander_link=expander_link)
            subgroup.attrs['name'] = name
    
    @staticmethod
    def load_names_and_bases_from_hdf5_group(group):
        """
        Loads a BasisSet object from an hdf5 file group.
        
        group: the hdf5 file group from which to load data about the BasisSet
        
        returns: (names, bases) where names is a list of strings and bases is a
                 list of Basis objects
        """
        names = []
        component_bases = []
        iname = 0
        while ('basis_{}'.format(iname)) in group:
            subgroup = group['basis_{}'.format(iname)]
            names.append(subgroup.attrs['name'])
            component_bases.append(Basis.load_from_hdf5_group(subgroup))
            iname += 1
        return (names, component_bases)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BasisSet from the given hdf5 group.
        
        group: the hdf5 group from which to load BasisSet
        
        returns: BasisSet object stored in the given hdf5 group
        """
        return BasisSet(*BasisSet.load_names_and_bases_from_hdf5_group(group))
    
    def __add__(self, other):
        """
        Allows for the addition of two BasisSets. It basically concatenates the
        component bases of the two BasisSets into one.
        
        other: BasisSet object with which to combine this one
        
        returns: BasisSet object containing the basis vectors from both
                 BasisSet objects being added
        """
        if isinstance(other, BasisSet):
            return BasisSet(self.names + other.names,\
                self.component_bases + other.component_bases)
        else:
            raise TypeError("Can only add together two BasisSet objects.")
    
    def __call__(self, parameters, expanded=True):
        """
        Finds the values of the bases in this BasisSet associated with the
        given parameters.
        
        parameters: 1D array of parameters corresponding to the combination of
                    all component bases.
        
        returns: list of outcomes of all bases
        """
        return [self[name](parameters[...,self.slices_by_name[name]],\
            expanded=expanded) for name in self.names]
    
    def __eq__(self, other):
        """
        Checks to see if other is a BasisSet containing the same component
        bases under the same names.
        
        other: object with which to check for equality
        
        returns: True if other is a BasisSet containing the same bases under
                 the same names. False otherwise
        """
        if type(self) == type(other):
            if self.num_bases:
                if self.names == other.names:
                    return (self.component_bases == other.component_bases)
                else:
                    return False
            else:
                return False
        else:
            return False
    
    def __ne__(self, other):
        """
        Checks whether other is a functionally different BasisSet than this
        one. This function enforces (self != other) == (not (self == other)).
        
        other: object with which to check for inequality
        
        returns: the opposite of __eq__ called with same arguments
        """
        return (not self.__eq__(other))

