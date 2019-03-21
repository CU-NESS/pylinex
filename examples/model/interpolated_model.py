"""
File: examples/model/interpolated_model.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing how to initialize and use the InterpolatedModel
             class.
"""
from __future__ import division
import numpy as np
from distpy import GaussianDistribution, DistributionSet
from pylinex import GaussianModel, InterpolatedModel

interpolation_method = 'quadratic'

np.random.seed(0)

nchannels = 20
channels = np.arange(nchannels)

train_ndraw = 100
test_ndraw = 10

nbasis_vectors = 5

true_model = GaussianModel(channels)

loose_prior_set = DistributionSet()
tight_prior_set = DistributionSet()
loose_gaussian = GaussianDistribution(-100, 1000)
tight_gaussian = GaussianDistribution(-100, 100)
loose_prior_set.add_distribution(loose_gaussian, 'amplitude')
tight_prior_set.add_distribution(tight_gaussian, 'amplitude')
loose_gaussian = GaussianDistribution(nchannels // 2, 10)
tight_gaussian = GaussianDistribution(nchannels // 2, 1)
loose_prior_set.add_distribution(loose_gaussian, 'center')
tight_prior_set.add_distribution(tight_gaussian, 'center')
loose_gaussian = GaussianDistribution(10, 4)
tight_gaussian = GaussianDistribution(10, 1)
loose_prior_set.add_distribution(loose_gaussian, 'scale')
tight_prior_set.add_distribution(tight_gaussian, 'scale')

draw1 = loose_prior_set.draw(train_ndraw)
inputs1 = np.stack([draw1[par] for par in true_model.parameters], axis=1)
true_outputs1 = np.array([true_model(inp) for inp in inputs1])

draw2 = tight_prior_set.draw(test_ndraw)
inputs2 = np.stack([draw2[par] for par in true_model.parameters], axis=1)
true_outputs2 = np.array([true_model(inp) for inp in inputs2])


transform_list = [None, None, 'log10']
interpolated_model = InterpolatedModel(true_model.parameters, inputs1,\
    true_outputs1, should_compress=True, transform_list=transform_list,\
    scale_to_cube=True, num_basis_vectors=nbasis_vectors, expander=None,\
    error=None, interpolation_method=interpolation_method)

interpolated_outputs2 = np.array([interpolated_model(inp) for inp in inputs2])

#print("true_outputs={}".format(true_outputs2))
#print("interpolated_outputs={}".format(interpolated_outputs2))
#print("absolute_error={}".format(true_outputs2-interpolated_outputs2))
#print("fractional_error={}".format(\
#    (true_outputs2-interpolated_outputs2)/true_outputs2))

