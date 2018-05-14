"""
File: pylinex/model/LoadableModel.py
Author: Keith Tauscher
Date: 13 May 2018

Description: File containing class representing a model which can be loaded via
             a class method. More concretely, if there is a XModel subclass of
             LoadableModel, then it must implement a
             load_from_hdf5_group(group) static method. This allows for a
             XModel.load(hdf5_file_name) to load an XModel from the hdf5 file
             at the given location.
"""
from ..util import Loadable
from .Model import Model

class LoadableModel(Model, Loadable):
    """
    Class representing a model which can be loaded via a class method. More
    concretely, if there is a XModel subclass of LoadableModel, then it must
    implement a load_from_hdf5_group(group) static method. This allows for a
    XModel.load(hdf5_file_name) to load an XModel from the hdf5 file at the
    given location.
    """
    pass

