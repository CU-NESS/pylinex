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
from distpy import GaussianDistribution
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
        if not ExpanderSet(np.zeros(self.num_channels),\
            np.ones(self.num_channels),\
            **dict(zip(self.names, self.expanders))).separable:
            raise ValueError("The expanders of the given model were not " +\
                "separable, so you might as well use the SumModel class " +\
                "instead of the DirectSumModel class.")
    
    @property
    def expanders(self):
        """
        Property storing the expanders of the models of this DirectSumModel.
        """
        if not hasattr(self, '_expanders'):
            expanders = []
            for model in self.models:
                try:
                    expanders.append(model.expander)
                except:
                    raise TypeError(("At least one model (type: {!s}) did " +\
                        "not have an expander property.").format(type(model)))
            self._expanders = expanders
        return self._expanders
    
    def quick_fit(self, data, error, quick_fit_parameters=[], prior=None):
        """
        Performs a quick fit to the given data with the error.
        
        data: 1D vector in output space of all expanders
        error: non-negative 1D vector of errors on each data point
        quick_fit_parameters: quick_fit_parameters to use for underlying models
                              if necessary
        
        returns: (mean, covariance) where mean and covariance are those of the
                 parameter distribution
        """
        if type(error) is type(None):
            error = np.ones_like(data)
        if len(quick_fit_parameters) != self.num_quick_fit_parameters:
            raise ValueError("quick_fit_parameters length was not equal to " +\
                "the number of quick_fit_parameters of this model.")
        if type(prior) is type(None):
            priors = [None] * self.num_models
        elif isinstance(prior, GaussianDistribution):
            priors = []
            pars_used = 0
            for (imodel, model) in enumerate(self.models):
                priors.append(prior.marginalize(\
                    slice(pars_used, pars_used + model.num_parameters)))
                pars_used += model.num_parameters
        else:
            raise TypeError("prior must either be None or a " +\
                "GaussianDistribution object.")
        fits = []
        pars_used = 0
        for (imodel, model) in enumerate(self.models):
            these_quick_fit_parameters = quick_fit_parameters[\
                pars_used:pars_used+model.num_quick_fit_parameters]
            fits.append(model.quick_fit(data, error,\
                quick_fit_parameters=these_quick_fit_parameters,\
                prior=priors[imodel]))
            pars_used = pars_used + model.num_quick_fit_parameters
        means = [fit[0] for fit in fits]
        covariances = [fit[1] for fit in fits]
        mean = np.concatenate(means)
        covariance = scila.block_diag(*covariances)
        return (mean, covariance)
    
    @property
    def quick_fit_parameters(self):
        """
        Property storing the quick_fit parameters
        """
        if not hasattr(self, '_quick_fit_parameters'):
            self._quick_fit_parameters = []
            for (iname, name) in enumerate(self.names):
                self._quick_fit_parameters = self._quick_fit_parameters +\
                    ['{0!s}_{1!s}'.format(name, parameter)\
                    for parameter in self.models[iname].quick_fit_parameters]
        return self._quick_fit_parameters
    
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

