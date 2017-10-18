"""
File: pylinex/util/h5py_extensions.py
Author: Keith Tauscher
Date: 8 Oct 2017

Description: File containing functions which extend the ability to link to
             extant datasets. Using the functions here, slices of extant
             datasets can be stored indirectly (and, thus, efficiently).
"""
from .TypeCategories import sequence_types
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
try:
    import h5py
except:
    # allows this to be imported even by users who do not have h5py
    pass

class HDF5Link(object):
    """
    An object representing a link to an extant h5py.Database object or
    h5py.Group object. It also allows for slices of data to be saved.
    """
    def __init__(self, link, slices=None):
        """
        Initializes a new DatasetLink with the given parameters.
        
        link: either extant h5py.Dataset or string absolute path from within
              file (only used if hard_link is not given)
        slices: if None, data is not sliced
                otherwise, slices should be a tuple of slices desrcibing the
                           slicing to apply to the data when reading/writing
        """
        is_string = isinstance(link, basestring)
        is_dataset = isinstance(link, h5py.Dataset)
        is_group = isinstance(link, h5py.Group)
        if is_string:
            self.soft_link = link
            self.hard_link = None
        elif is_dataset or is_group:
            self.hard_link = link
            self.soft_link = link.name
        else:
            raise TypeError("link must be a string, h5py.Group object or " +\
                "h5py.Dataset object.")
        if isinstance(slices, slice):
            self.slices = (slices,)
        else:
            self.slices = slices
        

def create_hdf5_dataset(group, name, data=None, link=None):
    """
    Creates a new hdf5 dataset, copies a hard link to an extant dataset, or
    creates a new region reference which essentially links to slices of an
    extant dataset. If both a link and data are given, the link is preferred.

    The 3 methods of calling are:
    
    1) create_hdf5_dataset(group, name, data=data)
        -- creates a new Dataset at group[name] storing data
    2) create_hdf5_dataset(group, name, link=link)
        -- creates a new hard or soft link to the extant h5py.Dataset (or
           h5py.Group) referenced to by link
    3) create_hdf5_dataset(group, name, data=data, link=link)
        -- if link is None, same as method 1. otherwise, same as method 2.
    
    group: the h5py.Group object in which to create this dataset/reference/link
    name: the name inside the group of the dataset/reference/link to create
    data: if None, link (to an extant dataset) must be provided
          otherwise, data must be a numpy.ndarray object
    link: HDF5Link to extant dataset (or arg or list of args with which to
          create one)
    
    returns h5py.Dataset object if one is newly created here
            HDF5Link object otherwise
    """
    if link is None:
        if data is None:
            raise ValueError("No data or link to data was given!")
        else:
            return group.create_dataset(name, data=data)
    elif not isinstance(link, HDF5Link):
        if type(link) in sequence_types:
            link = HDF5Link(*link)
        else:
            link = HDF5Link(link)
    if link.slices is None:
        if link.hard_link is None:
            group[name] = h5py.SoftLink(link.soft_link)
        else:
            group[name] = link.hard_link
    else:
        subgroup = group.create_group(name)
        subgroup.attrs['__isregionref__'] = True
        for (islice, link_slice) in enumerate(link.slices):
            subsubgroup = subgroup.create_group('dimension_{}'.format(islice))
            for attribute in ['start', 'stop', 'step']:
                attribute_value = getattr(link_slice, attribute)
                if attribute_value is not None:
                    subsubgroup.attrs[attribute] = attribute_value
        subgroup.attrs['__refpath__'] = link.soft_link
    return link

def get_hdf5_value(obj):
    """
    Gets the numpy.ndarray data stored in obj, whether directly or indirectly.
    
    obj: either a h5py.Dataset or a region reference created with the
         create_hdf5_dataset function defined here.
    
    returns: the numpy.ndarray stored, whether directly or indirectly, in the
             given object
    """
    try:
        return obj.value
    except:
        if ('__isregionref__' in obj.attrs) and obj.attrs['__isregionref__']:
            refpath = obj.attrs['__refpath__']
            slices = []
            idimension = 0
            while 'dimension_{}'.format(idimension) in obj:
                group = obj['dimension_{}'.format(idimension)]
                if 'start' in group.attrs:
                    start = group.attrs['start']
                else:
                    start = None
                if 'stop' in group.attrs:
                    stop = group.attrs['stop']
                else:
                    stop = None
                if 'step' in group.attrs:
                    step = group.attrs['step']
                else:
                    step = None
                slices.append(slice(start, stop, step))
                idimension += 1
            return obj.file[refpath][tuple(slices)]
        else:
            raise

