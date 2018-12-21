"""
File: pylinex/loglikelihood/NonlinearTruncationLoglikelihood.py
Author: Keith Tauscher
Date: 29 Sep 2018

Description: File containing a class which represents a DIC-like loglikelihood
             which uses the number of coefficients to use in each of a number
             of bases as the parameters of the likelihood.
"""
import numpy as np
from ..util import create_hdf5_dataset, real_numerical_types, sequence_types
from ..basis import Basis, BasisSet
from ..fitter import Fitter
from ..model import TruncatedBasisHyperModel, CompositeModel
from .GaussianLoglikelihood import GaussianLoglikelihood

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class NonlinearTruncationLoglikelihood(GaussianLoglikelihood):
    """
    Class which represents a DIC-like loglikelihood which uses the number of
    coefficients to use in each of a number of bases as the parameters of the
    likelihood.
    """
    def __init__(self, basis_set, data, error, expression,\
        parameter_penalty=2, default_num_terms=None):
        """
        Initializes a new TruncationLoglikelihood with the given basis_sum,
        data, and error.
        
        basis_set: BasisSet objects containing basis with the largest number
                   of basis vectors allowed for each component
        data: 1D data vector to fit
        error: 1D vector of noise level estimates for data
        expression: Expression object which forms full model from submodels.
                    The ith submodel (with i starting at 0) should be
                    represented by {i} in the expression string
        parameter_penalty: the logL parameter penalty for adding a parameter in
                           any given model. Should be a non-negative constant.
                           It defaults to 2, which is the penalty used for the
                           Deviance Information Criterion (DIC)
        """
        self.basis_set = basis_set
        self.data = data
        self.error = error
        self.parameter_penalty = parameter_penalty
        names = basis_set.names
        if type(default_num_terms) in sequence_types:
            if len(default_num_terms) == len(names):
                default_num_terms = list(default_num_terms)
            else:
                raise ValueError("The default_num_terms sequence was not " +\
                    "of the same length as the list of names in the given " +\
                    "basis_set.")
        else:
            default_num_terms = [default_num_terms] * len(names)
        models =\
            [TruncatedBasisHyperModel(basis_set[name], default_num_terms=term)\
            for (name, term) in zip(names, default_num_terms)]
        self.model = CompositeModel(expression, names, models)
    
    @property
    def basis_set(self):
        """
        Property storing the BasisSet object 
        """
        if not hasattr(self, '_basis_set'):
            raise AttributeError("basis_set was referenced before it was set.")
        return self._basis_set
    
    @basis_set.setter
    def basis_set(self, value):
        """
        Setter for the basis_set object.
        
        value: a BasisSet object
        """
        if isinstance(value, BasisSet):
            self._basis_set = value
        else:
            raise TypeError("basis_set was set to a non-BasisSet object.")
    
    @property
    def parameter_penalty(self):
        """
        Property storing the penalty imposed on the log-likelihood when an
        extra parameter is included in any given model.
        """
        if not hasattr(self, '_parameter_penalty'):
            raise AttributeError("parameter_penalty was referenced before " +\
                "it was set.")
        return self._parameter_penalty
    
    @parameter_penalty.setter
    def parameter_penalty(self, value):
        """
        Setter for the penalty assessed when an extra parameter is included in
        any given model.
        
        value: a non-negative number
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._parameter_penalty = value
            else:
                raise ValueError("parameter_penalty was set to a negative " +\
                    "number.")
        else:
            raise TypeError("parameter_penalty was set to a non-number.")

