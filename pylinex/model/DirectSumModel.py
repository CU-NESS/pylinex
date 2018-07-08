"""
File: pylinex/model/DirectSumModel.py
Author: Keith Tauscher
Date: 30 Jun 2018

Description: File containing a class representing a special SumModel: one whose
             submodels do not overlap in channel space.
"""
import numpy as np
import scipy.linalg as scila
import matplotlib.pyplot as pl
from ..expander import ExpanderSet
from .BasisModel import BasisModel
from .TruncatedBasisHyperModel import TruncatedBasisHyperModel
from .ExpandedModel import ExpandedModel
from .SumModel import SumModel

class DirectSumModel(SumModel):
    """
    Class representing a special SumModel: one whose submodels do not overlap
    in channel space.
    """
    def __init__(self, names, models):
        """
        Initializes a new DirectSumModel with the given names and submodels.
        
        names: sequence of string names of submodels
        models: sequence of models corresponding to given names which are
                either ExpandedModel objects or BasisModel objects which
                include an Expander
        """
        self.names = names
        self.models = models
        self.expanders
    
    @property
    def expanders(self):
        """
        Property storing the expanders of the models of this DirectSumModel.
        """
        if not hasattr(self, '_expanders'):
            expanders = []
            for model in self.models:
                if isinstance(model, BasisModel) or\
                    isinstance(model, ExpandedModel) or\
                    isinstance(model, TruncatedBasisHyperModel):
                    expanders.append(model.expander)
                else:
                    raise TypeError("At least one model was neither a " +\
                        "BasisModel object nor an ExpandedModel object.")
            self._expanders = expanders
        return self._expanders
    
    def quick_fit(self, data, error=None):
        """
        Performs a quick fit to the given data with the error.
        
        data: 1D vector in output space of all expanders
        error: non-negative 1D vector of errors on each data point
        
        returns: (mean, covariance) where mean and covariance are those of the
                 parameter distribution
        """
        if error is None:
            error = np.ones_like(data)
        expander_dict = {name: expander\
            for (name, expander) in zip(self.names, self.expanders)}
        if not ExpanderSet(data, error, **expander_dict).separable:
            raise ValueError("The length of this data implies that the " +\
                "model is not separable (i.e. that the individual models " +\
                "cannot be used easily to find subfits which can then be " +\
                "combined).")
        fits = [model.quick_fit(data, error=error) for model in self.models]
        means = [fit[0] for fit in fits]
        covariances = [fit[1] for fit in fits]
        mean = np.concatenate(means)
        covariance = scila.block_diag(*covariances)
        return (mean, covariance)
    
    def fill_hdf5_group(self, group):
        """
        Fills an hdf5 file group with information about this SumModel.
        
        group: hdf5 file group to fill with information about this SumModel
        """
        group.attrs['class'] = 'DirectSumModel'
        subgroup = group.create_group('models')
        for (iname, name) in enumerate(self.names):
            subsubgroup = subgroup.create_group('{:d}'.format(iname))
            subsubgroup.attrs['name'] = name
            self.models[iname].fill_hdf5_group(subsubgroup)
    
    def __eq__(self, other):
        """
        Checks for equality between this DirectSumModel and other.
        
        other: object to check for equality
        
        returns: False unless other is a DirectSumModel with the same names and
                 submodels
        """
        if not isinstance(other, DirectSumModel):
            return False
        if self.names != other.names:
            return False
        return all([(smodel == omodel)\
            for (smodel, omodel) in zip(self.models, other.models)])

