"""
File: pylinex/quantity/CompiledQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a list of Quantities to be
             evaluated with the same (or overlapping) arguments. When it is
             called, each underlying Quantity is called.
"""
from ..util import int_types, sequence_types, Savable, Loadable
from .Quantity import Quantity
from .AttributeQuantity import AttributeQuantity
from .ConstantQuantity import ConstantQuantity
from .FunctionQuantity import FunctionQuantity
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class CompiledQuantity(Quantity, Savable, Loadable):
    """
    Class representing a list of Quantities to be evaluated with the same (or
    overlapping) arguments. When it is called, each underlying Quantity is
    called.
    """
    def __init__(self, name, *quantities):
        """
        Initialized a new CompiledQuantity made of the given quantities.
        
        name: string identifier of this Quantity
        *quantities: unpacked list of quantities to call when this object is
                     called
        """
        Quantity.__init__(self, name)
        self.quantities = quantities
    
    @property
    def quantities(self):
        """
        Property storing the individual quantities underlying this
        CompiledQuantity.
        """
        if not hasattr(self, '_quantities'):
            raise AttributeError("quantities was referenced before it was " +\
                                 "set.")
        return self._quantities
    
    @quantities.setter
    def quantities(self, value):
        """
        Setter for the quantities property.
        
        value: must be a sequence of Quantity objects
        """
        if type(value) in sequence_types:
            if all([isinstance(quantity, Quantity) for quantity in value]):
                self._quantities = value
            else:
                raise TypeError("Not all elements of the sequence to " +\
                                 "which quantities was set were Quantity " +\
                                 "objects.")
        else:
            raise TypeError("quantities was set to a non-sequence.")
    
    @property
    def num_quantities(self):
        """
        Property storing the number of quantities compiled in this object.
        """
        if not hasattr(self, '_num_quantities'):
            self._num_quantities = len(self.quantities)
        return self._num_quantities
    
    @property
    def names(self):
        """
        Property storing the list of names of the quantities underlying this
        object.
        """
        if not hasattr(self, '_names'):
            self._names = [quantity.name for quantity in self.quantities]
        return self._names
    
    @property
    def index_dict(self):
        """
        Property storing a dictionary that connects the names of quantities to
        their indices in the list of Quantity objects underlying this object.
        """
        if not hasattr(self, '_index_dict'):
            self._index_dict = {}
            for (iquantity, quantity) in enumerate(self.quantities):
                self._index_dict[quantity.name] = iquantity
        return self._index_dict
    
    @property
    def can_index_by_string(self):
        """
        Property storing a Boolean describing whether this Quantity can be
        indexed by string. This essentially checks whether the names of the
        quantities underlying this object are unique.
        """
        if not hasattr(self, '_can_index_by_string'):
            self._can_index_by_string =\
                (self.num_quantities == len(set(self.names)))
        return self._can_index_by_string
    
    def __getitem__(self, index):
        """
        Gets the quantity associated with the index.
        
        index: if index is a string, it is assumed to be the name of one of the
                                     Quantity objects underlying this object.
                                     (index can only be a string if the
                                     can_index_by_string property is True.)
               if index is an int, it is taken to be the index of one of the
                                   Quantity objects underlying this one.
        
        returns: the Quantity object described by the given index
        """
        if type(index) in int_types:
            return self.quantities[index]
        elif isinstance(index, basestring):
            if self.can_index_by_string:
                return self.quantities[self.index_dict[index]]
            else:
                raise TypeError("CompiledQuantity can only be indexed by " +\
                                "string if the names of the quantities " +\
                                "underlying it are unique.")
        else:
            raise IndexError("CompiledQuantity can only be indexed by an " +\
                             "integer index or a string quantity name.")
    
    def __add__(self, other):
        """
        Appends other to this CompiledQuantity.
        
        other: CompiledQuantity (or some other Quantity)
        
        returns: if other is another CompiledQuantity, names quantity lists of
                                                       both CompiledQuantity
                                                       objects are combined
                 otherwise, other must be a Quantity object. It will be added
                            to the quantity list of this CompiledQuantity
                            (whose name won't change)
        """
        if isinstance(other, CompiledQuantity):
            new_name = '{0!s}+{1!s}'.format(self.name, other.name)
            new_quantities = self.quantities + other.quantities
        elif isinstance(other, Quantity):
            new_name = self.name
            new_quantities = [quantity for quantity in self.quantities]
            new_quantities.append(other)
        else:
            raise TypeError("Only Quantity objects can be added to " +\
                "compiled quantities.")
        return CompiledQuantity(new_name, *new_quantities)
    
    def __call__(self, *args, **kwargs):
        """
        Finds the values of all of the Quantity objects underlying this obejct.
        
        args: list of arguments to pass on to the constituent Quantity objects
        kwargs: list of keyword arguments to pass on to the constituent
                Quantity objects
        
        returns: list containing the values of all of the Quantity objects
                 underlying this one
        """
        return [quantity(*args, **kwargs) for quantity in self.quantities]
    
    def __contains__(self, key):
        """
        Checks if a quantity with the given name exists in this
        CompiledQuantity.
        
        key: string name of Quantity to check for
        
        returns: True if there exists at least one Quantity named key
        """
        return any([(quantity.name == key) for quantity in self.quantities])
    
    def fill_hdf5_group(self, group, exclude=[]):
        """
        Fills given hdf5 file group with data about this CompiledQuantity.
        
        group: hdf5 file group to fill with data about this CompiledQuantity
        """
        iquantity = 0
        group.attrs['name'] = self.name
        group.attrs['class'] = 'CompiledQuantity'
        for quantity in self.quantities:
            excluded = (quantity.name in exclude)
            savable = isinstance(quantity, Savable)
            if (not excluded) and savable:
                subgroup = group.create_group('quantity_{}'.format(iquantity))
                if isinstance(quantity, Savable):
                    quantity.fill_hdf5_group(subgroup)
                else:
                    raise TypeError("This CompiledQuantity cannot be saved " +\
                        "because it contains Quantity objects which cannot " +\
                        "be saved.")
                iquantity += 1
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a CompiledQuantity from the given hdf5 group.
        
        group: hdf5 file group from which to load a CompiledQuantity
        
        returns: CompiledQuantity loaded from given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'CompiledQuantity'
        except:
            raise TypeError("This hdf5 file group does not seem to contain " +\
                "a CompiledQuantity.")
        name = group.attrs['name']
        iquantity = 0
        quantities = []
        while 'quantity_{}'.format(iquantity) in group:
            subgroup = group['quantity_{}'.format(iquantity)]
            try:
                class_name = subgroup.attrs['class']
                cls = eval(class_name)
            except:
                raise TypeError("One of the quantities in this " +\
                    "CompiledQuantity could not be loaded; its class was " +\
                    "not recognized.")
            quantities.append(cls.load_from_hdf5_group(subgroup))
            iquantity += 1
        return CompiledQuantity(name, *quantities)

