"""
File: pylinex/quantity/AttributeQuantity.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing a class representing a quantity which is an
             attribute of an object (which may or may not exist at the time of
             initialization).
"""
from ..util import Savable
from .Quantity import Quantity

class AttributeQuantity(Quantity, Savable):
    """
    Class representing a quantity which is an attribute of an object. That
    object may or may not exist at the time of the initialization of this
    object.
    """
    def __init__(self, attribute_name, name=None):
        """
        Initializes a new AttributeQuantity with the given identifiers.
        
        attribute_name: name of attribute to retrieve when this Quantity is
                        called. Must be a string.
        name: if None, name is taken to be identical to attribute_name
              otherwise, can be any string
        """
        self.attribute_name = attribute_name
        if name is None:
            name = self.attribute_name
        Quantity.__init__(self, name)
    
    def __call__(self, container, **kwargs):
        """
        Gets the attribute this quantity corresponds to from the given
        container.
        
        container: the object from which to retrieve this attribute
        kwargs: dictionary of other keyword arguments (they go unused)
        """
        return getattr(container, self.name)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this Quantity.
        
        group: hdf5 file group to fill with information about this Quantity
        """
        group.attrs['name'] = self.name
        group.attrs['class'] = 'AttributeQuantity'
        group.attrs['attribute_name'] = self.attribute_name
        
