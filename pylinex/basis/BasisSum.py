"""
File: pylinex/basis/BasisSum.py
Author: Keith Tauscher
Date: 30 Oct 2017

Description: File containing a set of Basis objects representing different
             things but combined into a single Basis. This allows for
             conceptual differences between different basis vectors to be
             retained.
"""
import numpy as np
import matplotlib.pyplot as pl
from .Basis import Basis
from .BasisSet import BasisSet
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class BasisSum(BasisSet, Basis):
    """
    Subclass of Basis class which is also a container of Basis classes. Allows
    for different basis vectors to be given distinct meanings.
    """
    def __init__(self, names, bases):
        """
        Initializes a new BasisSet with the given names and bases.
        
        names: list of names of subbases
        bases: list of Basis objects corresponding to the given names
        """
        BasisSet.__init__(self, names, bases)
        try:
            self.basis = np.concatenate(\
                [basis.expanded_basis for basis in self.component_bases],\
                axis=0)
            self.translation = np.sum([basis.expanded_translation\
                for basis in self.component_bases], axis=0)
        except KeyboardInterrupt:
            raise
        except:
            raise ValueError("The shapes of the given bases were not " +\
                "compatible. They can be put into a BasisSet but not a " +\
                "BasisSum.")
        else:
            self.expander = None
    
    def copy(self):
        """
        Finds and returns a deep copy of this BasisSum.
        
        returns: new BasisSum object with names and bases copied from this one
        """
        return BasisSum([name for name in self.names],\
            [basis.copy() for basis in self.component_bases])
    
    def basis_dot_products(self, error=None):
        """
        Finds the dot products between the Basis objects underlying this
        BasisSet.
        
        error: error to use in assessing overlap
        
        returns: 2D square numpy.ndarray of dot products
        """
        dot_products = np.ndarray((len(self.names),) * 2)
        for ibasis1 in range(len(self.component_bases)):
            basis1 = self.component_bases[ibasis1]
            for ibasis2 in range(ibasis1, len(self.component_bases)):
                basis2 = self.component_bases[ibasis2]
                dot_product = basis1.dot(basis2, error=error)
                dot_products[ibasis1,ibasis2] = dot_product
                dot_products[ibasis2,ibasis1] = dot_product
        return dot_products
    
    def basis_subsets(self, **subsets):
        """
        Creates a new BasisSum object with the given subsetted component bases.
        
        **subsets: subsets to take for each name for which basis vectors should
                   be truncated
        
        returns: BasisSum object with subsetted component bases
        """
        return BasisSum(self.names, self.component_basis_subsets(**subsets))
    
    def __add__(self, other):
        """
        Allows for the addition of two BasisSums. It basically concatenates the
        component bases of the two BasisSums into one.
        
        other: BasisSum object with which to combine this one
        
        returns: BasisSum object containing the basis vectors from both
                 BasisSum objects being added
        """
        if isinstance(other, BasisSum):
            return BasisSum(self.names + other.names,\
                self.component_bases + other.component_bases)
        else:
            raise TypeError("Can only add together two BasisSum objects.")
    
    def fill_hdf5_group(self, group, *args, **kwargs):
        """
        Fills given hdf5 file group with information about this BasisSum.
        
        group: the hdf5 group to fill with information about this BasisSum
        args: arguments to pass on to BasisSet.fill_hdf5_group
        kwargs: keyword arguments to pass on to BasisSet.fill_hdf5_group
        """
        BasisSet.fill_hdf5_group(self, group, *args, **kwargs)

    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BasisSum from the given hdf5 group.
        
        group: the hdf5 group from which to load BasisSum
        
        returns: BasisSum object stored in the given hdf5 group
        """
        return BasisSum(*BasisSet.load_names_and_bases_from_hdf5_group(group))
    
    def __call__(self, parameters):
        """
        Finds the sum of all component bases when the parameters are as given
        
        parameters: 1D array of parameters concatenated across different basis
                    sets.
        
        return: sum of the results of all the expanded bases in this BasisSum
        """
        return Basis.__call__(self, parameters)

