"""
File: pylinex/loglikelihood/LoadLoglikelihood.py
Author: Keith Tauscher
Date: 25 Feb 2018

Description: File containing a function which loads an arbitrary Loglikelihood
             object from a group of an open hdf5 file.
"""
from .RosenbrockLoglikelihood import RosenbrockLoglikelihood
from .GaussianLoglikelihood import GaussianLoglikelihood
from .PoissonLoglikelihood import PoissonLoglikelihood
from .GammaLoglikelihood import GammaLoglikelihood
from .LinearTruncationLoglikelihood import LinearTruncationLoglikelihood

def load_loglikelihood_from_hdf5_group(group):
    """
    Loads a loglikelihood from the given hdf5 group.
    
    group: the hdf5 file group from which to load the Loglikelihood
    
    returns: Loglikelihood object of the correct type
    """
    try:
        class_name = group.attrs['class']
        cls = eval(class_name)
    except KeyError:
        raise ValueError("This group doesn't appear to contain a " +\
            "Loglikelihood.")
    except NameError:
        raise ValueError("The class name of this loglikelihood is not known!")
    return cls.load_from_hdf5_group(group)

