"""
File: extractpy/fitter/__init__.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: Imports classes from the extractpy.fitter module:
             
             Fitter: Performs weighted least square fit with priors with a
                     single BasisSet object and one or many data vectors.
             
             MetaFitter: Performs many weighted least square fits with priors
                         by taking subbases of a single BasisSet.
             
             Extractor: Finds a model for and separates each component of a
                        data vector using training sets.
"""
from .Fitter import Fitter
from .MetaFitter import MetaFitter
from .Extractor import Extractor

