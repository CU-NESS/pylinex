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
    def name_dict(self):
        """
        Dictionary connecting names to the internal index with which they are
        associated.
        """
        if not hasattr(self, '_name_dict'):
            self._name_dict =\
                {self.names[iname]: iname for iname in range(len(self.names))}
        return self._name_dict
    
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
                self._slices_by_name[name] =\
                    slice(rindex, rindex + self[name].num_basis_vectors)
                rindex = rindex + self[name].num_basis_vectors
        return self._slices_by_name
    
    @property
    def num_basis_vectors(self):
        """
        Property storing the number of basis functions stored in this BasisSet
        """
        if not hasattr(self, '_num_basis_vectors'):
            self._num_basis_vectors =\
                sum([self[name].num_basis_vectors for name in self.names])
        return self._num_basis_vectors
    
    def __getitem__(self, key):
        """
        Allows for the usage of square-bracket indexing notation for getting 
        
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
            return self.component_bases[self.name_dict[key]]
        elif isinstance(key, dict):
            return self.basis_subsets(**key)
        elif key is None:
            return self
        elif type(key) in int_types:
            return self.component_bases[key]
        else:
            raise KeyError("key not recognized.")
    
    @property
    def sizes(self):
        """
        Property storing a dictionary with basis names as keys and the number
        of basis vectors in that basis set as values.
        """
        if not hasattr(self, '_sizes'):
            self._sizes =\
                {name: self[name].num_basis_vectors for name in self.names}
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
        attribute values to restart the internal iterator.
        """
        self._index_of_basis_to_return = 0
        return self
    
    def next(self):
        """
        Since this BasisSet is its own iterator, it must have a next() method
        which returns the next basis. In this case, the "private"
        _index_of_basis_to_return property is used to store the index of the
        name of the next basis to return.
        """
        if self._index_of_basis_to_return == len(self.names):
            raise StopIteration
        self._index_of_basis_to_return += 1
        return self[self.names[self._index_of_basis_to_return - 1]]
    
    def __next__(self):
        """
        Alias for next included for Python 2/3 compatibility.
        """
        return self.next()
    
    def fill_hdf5_group(self, group, basis_links=None, expander_links=None):
        """
        Fills the given hdf5 file group with data about this basis set.
        
        group: the hdf5 file group to fill
        """
        if basis_links is None:
            basis_links = [None for name in self.names]
        if expander_links is None:
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
    

