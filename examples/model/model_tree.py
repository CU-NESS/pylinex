"""
File: examples/model/model_tree.py
Author: Keith Tauscher
Date: 4 Oct 2020

Description: Example script showing the features and uses of the ModelTree
             class.
"""
import numpy as np
from pylinex import Basis, BasisModel, SumModel, ProductModel, ModelTree

num_channels = 101
xs = np.linspace(-1, 1, num_channels)

gain_model = BasisModel(Basis([xs ** 1]))
signal_model = BasisModel(Basis([xs ** 2]))
nuisance_model = BasisModel(Basis([xs ** 3]))
ideal_model = SumModel(['signal', 'nuisance'], [signal_model, nuisance_model])
multiplicative_model =\
    ProductModel(['gain', 'ideal'], [gain_model, ideal_model])
offset_model = BasisModel(Basis([xs ** 0]))

model = SumModel(['multiplicative', 'offset'],\
    [multiplicative_model, offset_model])

model_tree = ModelTree(model)
assert(model_tree.leaves ==\
    [gain_model, signal_model, nuisance_model, offset_model])
assert(model_tree.name_chains == [['multiplicative', 'gain'],\
    ['multiplicative', 'ideal', 'signal'],\
    ['multiplicative', 'ideal', 'nuisance'], ['offset']])
assert(model_tree.modulator_leaf_lists == [[1,2],[0],[0],[]])
assert(model_tree.modulators[0] == ideal_model)
assert(np.all(model_tree.modulators[3]([]) == np.ones(101)))
true_gain = gain_model([np.random.normal()])
true_signal = signal_model([np.random.normal()])
true_nuisance = nuisance_model([np.random.normal()])
true_offset = offset_model([np.random.normal()])
assert(np.all(model_tree.evaluate_from_leaves([true_gain, true_signal,\
    true_nuisance, true_offset]) == (true_offset + (true_gain *\
    (true_signal + true_nuisance)))))


a_model = BasisModel(Basis([xs ** 0]))
b_model = BasisModel(Basis([xs ** 1]))
c_model = BasisModel(Basis([xs ** 2]))
ab_model = ProductModel(['a', 'b'], [a_model, b_model])
abc_model = ProductModel(['ab', 'c'], [ab_model, c_model])
try:
    model_tree = ModelTree(abc_model)
except ValueError as error:
    pass
else:
    raise AssertionError("This model tree composed of a ProductModel with " +\
        "a ProductModel as one of its submodels should have caused an " +\
        "error to be thrown.")

