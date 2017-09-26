"""
File: pylinex/fitter/__init__.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: Imports classes from the pylinex.fitter module:
             
             TrainingSetIterator: Some problems involve iterating over many
                                  different training sets at once. This object
                                  does that task in chunks since training sets
                                  are often large and numerous enough that
                                  there are billions of total combined training
                                  set curves. This object is flexible enough
                                  that not all of them need to be returned
                                  (it's up to the user)
             
             Fitter: Performs weighted least square fit with priors with a
                     single BasisSet object and one or many data vectors.
             
             MetaFitter: Performs many weighted least square fits with priors
                         by taking subbases of a single BasisSet.
             
             Extractor: Finds a model for and separates each component of a
                        data vector using training sets.
             
"""
from pylinex.fitter.TrainingSetIterator import TrainingSetIterator
from pylinex.fitter.Fitter import Fitter
from pylinex.fitter.MetaFitter import MetaFitter
from pylinex.fitter.Extractor import Extractor

