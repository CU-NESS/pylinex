"""
File: examples/basis/basis_sum_equality.py
Author: Keith Tauscher
Date: 4 Jun 2019

Description: Quick example script showing that the __eq__ method works for both
             the BasisSet and BasisSum class, even though they act somewhat
             differently.
"""
import numpy as np
from pylinex import Basis, BasisSet, BasisSum

x_values = np.linspace(-1, 1, 101)
names = ['lower', 'higher']
bases = [Basis(x_values[np.newaxis,:] ** np.arange(2)[:,np.newaxis]),\
    Basis(x_values[np.newaxis,:] ** np.arange(2, 4)[:,np.newaxis])]

basis_set = BasisSet(names, bases)
basis_sum = BasisSum(names, bases)

assert(basis_set != basis_sum)
assert(basis_set == basis_set)
assert(basis_sum == basis_sum)

