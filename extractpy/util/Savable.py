"""
File: extractpy/util/Savable.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class representing any object which can be saved
             to an hdf5 file. Any subclass must implement a
             fill_hdf5_group(self, group) function.
"""
try:
    import h5py
except:
    have_h5py = False
else:
    have_h5py = True

class Savable(object):
    """
    Class representing any object which can be saved to an hdf5 file. This
    class cannot be directly instantiated.
    """
    def fill_hdf5_group(self, group):
        """
        Fills hdf5 file group with data about this object. This function must
        be implemented by any subclasses of this function.
        
        group: hdf5 file group with which to fill with data about this object
        """
        raise NotImplementedError("Savable class cannot be directly " +\
                                  "instantiated. fill_hdf5_group function " +\
                                  "must be re-implemented.")
    
    def save(self, file_name):
        """
        Saves data about this object in an hdf5 file at the given file name.
        
        file_name: path to where hdf5 file with data about this object should
                   be saved
        """
        if have_h5py:
            hdf5_file = h5py.File(file_name, 'w')
            self.fill_hdf5_group(hdf5_file)
            hdf5_file.close()
        else:
            raise NotImplementedError("Saving cannot be performed if h5py " +\
                                      "is not installed. Install h5py and " +\
                                      "retry to continue.")

