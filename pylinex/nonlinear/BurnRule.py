"""
File: pylinex/nonlinear/BurnRule.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: File containing a class representing a rule to help determine
             which checkpoints of a chain to include in the "burn-in" phase,
             i.e. which checkpoints should be excluded from the chain because
             they are likely to skew the distribution sampled.
"""
import numpy as np
from ..util import Savable, Loadable, bool_types, int_types, numerical_types

class BurnRule(Savable, Loadable):
    """
    Class representing a rule to help determine which checkpoints of a chain to
    include in the "burn-in" phase, i.e. which checkpoints should be excluded
    from the chain because they are likely to skew the distribution sampled.
    
    This class is callable, such that, if burn_rule is a BurnRule object, then
    burn_rule(100) returns a 1D numpy array of the checkpoints which should be
    included in the final output.
    """
    def __init__(self, min_checkpoints=1, desired_fraction=0.5, thin=None,\
        burn_end=False):
        """
        Initializes a new BurnRule object with the given arguments.
        
        min_checkpoints: the minimum number of checkpoints to include in the
                         output when this BurnRule is called.
        desired_fraction: number between 0 and 1 (inclusive). the desired
                          fraction of the available chain to include in the
                          final output. If the desired_fraction would yield
                          fewer than min_checkpoints checkpoints, then
                          min_checkpoints are returned
        thin: either None (default, corresponding to 1) or a positive integer
              representing the stride with which to read the chain
        burn_end: if True, the end of the chain is burned off instead of the
                           beginning
        """
        self.min_checkpoints = min_checkpoints
        self.desired_fraction = desired_fraction
        self.thin = thin
        self.burn_end = burn_end
    
    @property
    def min_checkpoints(self):
        """
        Property storing the integer minimum number of checkpoints which can be
        included in the final output when this BurnRule object is called.
        """
        if not hasattr(self, '_min_checkpoints'):
            raise AttributeError("min_checkpoints was referenced before it " +\
                "was set.")
        return self._min_checkpoints
    
    @min_checkpoints.setter
    def min_checkpoints(self, value):
        """
        Setter for the minimum number of checkpoints to include in the output
        when this BurnRule object is called.
        
        value: must be a positive integer
        """
        if (type(value) in int_types) and (value > 0):
            self._min_checkpoints = value
        else:
            raise TypeError("min_checkpoints was set to something other " +\
                "than a positive integer.")
    
    @property
    def desired_fraction(self):
        """
        Property storing the fraction (between 0 and 1) of the chain which
        should be returned in the limit of an infinite chain.
        """
        if not hasattr(self, '_desired_fraction'):
            raise AttributeError("desired_fractions was referenced before " +\
                "it was set.")
        return self._desired_fraction
    
    @desired_fraction.setter
    def desired_fraction(self, value):
        """
        Setter for the fraction of the chain which should be returned in the
        limit of an infinite chain.
        
        value: must satisfy 0<=value<=1
        """
        if type(value) in numerical_types:
            if (value >= 0) and (value <= 1):
                self._desired_fraction = value
            else:
                raise ValueError(\
                    "desired_fraction is not between 0 and 1 (inclusive).")
        else:
            raise TypeError("desired_fraction doesn't seem to be a number.")
    
    @property
    def thin(self):
        """
        Property storing the thinning factor used when reading the chain.
        """
        if not hasattr(self, '_thin'):
            raise AttributeError("thin was referenced before it was set.")
        return self._thin
    
    @thin.setter
    def thin(self, value):
        """
        Setter for the thinning factor.
        
        value: either None (default, corresponding to 1) or a positive integer
        """
        if type(value) is type(None):
            value = 1
        if type(value) in int_types:
            self._thin = value
        else:
            raise TypeError("thin was set to a non-integer.")
    
    @property
    def burn_end(self):
        """
        If True, the chain should be burned at the end instead of the
        beginning. This is an unusual but sometimes useful feature.
        """
        if not hasattr(self, '_burn_end'):
            raise AttributeError("burn_end was referenced before it was set.")
        return self._burn_end
    
    @burn_end.setter
    def burn_end(self, value):
        """
        Setter for the burn_end property, which determines whether the chain
        should be burned at the end instead of the beginning.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._burn_end = value
        else:
            raise TypeError("burn_end was set to a non-bool.")
    
    def fill_hdf5_group(self, group):
        """
        Stores information about this BurnRule in the given hdf5 file group.
        
        group: hdf5 file group to fill with information about this BurnRule
        """
        group.attrs['class'] = 'BurnRule'
        group.attrs['min_checkpoints'] = self.min_checkpoints
        group.attrs['desired_fraction'] = self.desired_fraction
        group.attrs['thin'] = self.thin
        group.attrs['burn_end'] = self.burn_end
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BurnRule from the given hdf5 file group.
        
        group: hdf5 file group on which fill_hdf5_group was called when this
               BurnRule was saved
        
        returns: a BurnRule object loaded from the given hdf5 file group
        """
        if ('class' in group.attrs) and (group.attrs['class'] == 'BurnRule'):
            min_checkpoints = group.attrs['min_checkpoints']
            desired_fraction = group.attrs['desired_fraction']
            thin = group.attrs['thin']
            burn_end = group.attrs['burn_end']
            return BurnRule(min_checkpoints=min_checkpoints,\
                desired_fraction=desired_fraction, thin=thin,\
                burn_end=burn_end)
        else:
            raise ValueError("group doesn't appear to point to a BurnRule " +\
                "object.")
    
    def __call__(self, num_checkpoints):
        """
        Applies this BurnRule to a situation where there are num_checkpoints
        checkpoints.
        
        num_checkpoints: the integer number of checkpoints in the chain to
                         which to apply this BurnRule
        
        returns: 1D numpy.ndarray storing the checkpoints which are not burned
                 off the chain
        """
        to_include_by_min = self.min_checkpoints
        to_include_by_fraction =\
            int(round(num_checkpoints * self.desired_fraction))
        to_include = max(to_include_by_min, to_include_by_fraction)
        if self.burn_end:
            return np.arange(to_include)
        else:
            return np.arange(num_checkpoints - to_include, num_checkpoints)
    

