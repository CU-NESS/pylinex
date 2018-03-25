"""
File: pylinex/hdf5/Loading.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing functions which load objects from hdf5 files. They
             require the use of h5py, a common Python library.
"""
from ..quantity import load_quantity_from_hdf5_group
from ..expander import load_expander_from_hdf5_group
from ..model import load_model_from_hdf5_group
from ..loglikelihood import load_loglikelihood_from_hdf5_group
try:
    import h5py
except:
    have_h5py = False
else:
    have_h5py = True

def ensure_h5py_installed():
    """
    Called whenever a load_XXXX_from_hdf5_file function is called. It ensures
    that h5py is installed. Otherwise, it throws an error.
    """
    if not have_h5py:
        raise NotImplementedError("hdf5 files can't be loaded because h5py " +\
            "isn't installed. Install h5py and retry to continue.")

def load_quantity_from_hdf5_file(file_name):
    """
    Loads a Quantity object from the given hdf5 file.
    
    group: hdf5 file from which to load data with which to recreate a Quantity
           object
    
    returns: Quantity object of appropriate type
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    quantity = load_quantity_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return quantity

def load_expander_from_hdf5_file(file_name):
    """
    Loads an Expander object from the given hdf5 file.
    
    group: hdf5 file from which to load data with which to recreate an Expander
           object
    
    returns: Expander object of appropriate type
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    expander = load_expander_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return expander

def load_model_from_hdf5_file(file_name):
    """
    Loads a Model object from an hdf5 file.
    
    file_name: name of the hdf5 file from which to load data about the Model
    
    returns: Model object loaded from the given hdf5 file
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    model = load_model_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return model

def load_loglikelihood_from_hdf5_file(file_name):
    """
    Loads a Loglikelihood object from an hdf5 file.
    
    file_name: name of the hdf5 file from which to load data about the
               Loglikelihood
    
    returns: Loglikelihood object loaded from the given hdf5 file
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    loglikelihood = load_loglikelihood_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return model

