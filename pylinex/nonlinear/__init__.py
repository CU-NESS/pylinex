"""
File: pylinex/nonlinear/__init__.py
Author: Keith Tauscher
Date: 14 Jan 2018

Description: The pylinex.nonlinear module contains classes which perform fits
             using models from the pylinex.model module.
"""
from pylinex.nonlinear.LeastSquareFitter import LeastSquareFitter
from pylinex.nonlinear.InterpolatingLeastSquareFitter\
    import InterpolatingLeastSquareFitter
from pylinex.nonlinear.LeastSquareFitGenerator import LeastSquareFitGenerator
from pylinex.nonlinear.Sampler import Sampler
from pylinex.nonlinear.BurnRule import BurnRule
from pylinex.nonlinear.NLFitter import NLFitter
from pylinex.nonlinear.TruncationExtractor import TruncationExtractor
from pylinex.nonlinear.RankDecider import RankDecider
