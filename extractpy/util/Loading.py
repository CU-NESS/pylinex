"""
File: extractpy/util/Loading.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing functions which load objects from hdf5 files. They
             require the use of h5py, a common Python library.
"""
from ..expander import load_expander_from_hdf5_group
from ..basis import load_basis_from_hdf5_group, load_basis_set_from_hdf5_group
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
                                  "isn't installed. Install h5py and retry " +\
                                  "to continue.")


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

def load_basis_from_hdf5_file(file_name):
    """
    Allows for Basis objects to be read from hdf5 files.
    
    file_name: name of the hdf5 file from which to load the basis
    
    returns: Basis object stored in the hdf5 file
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    basis = load_basis_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return basis

def load_basis_set_from_hdf5_file(file_name):
    """
    Loads a BasisSet object from an hdf5 file.
    
    file_name: name of the hdf5 file from which to load data about the BasisSet
    
    returns: BasisSet object loaded from the given hdf5 file
    """
    ensure_h5py_installed()
    hdf5_file = h5py.File(file_name, 'r')
    basis_set = load_basis_set_from_hdf5_group(hdf5_file)
    hdf5_file.close()
    return basis_set

