"""
File: examples/loglikelihood/likelihood_distribution_harmonizer.py
Author: Keith Tauscher
Date: 1 Jul 2018

Description: File containing example script showing the basic use case of the
             LikelihoodDistributionHarmonizer class.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, DistributionSet
from pylinex import PadExpander, ExpandedModel, ConstantModel, SumModel,\
    DirectSumModel, BasisModel, Basis, GaussianLoglikelihood,\
    LikelihoodDistributionHarmonizer

fontsize = 24

smaller_num_channels = 40
num_models = 5
larger_num_channels = num_models * smaller_num_channels
channels = np.arange(larger_num_channels)
noise_level = 1

names = [chr(ord('A') + index) for index in range(num_models)]
expanders = [PadExpander('{:d}*'.format(index),\
    '{:d}*'.format(num_models - index - 1)) for index in range(num_models)]
models = [ExpandedModel(ConstantModel(smaller_num_channels), expander)\
    for expander in expanders]
direct_sum_model = DirectSumModel(names, models)

linear_model = BasisModel(Basis(channels[np.newaxis,:]))

true_linear_term = 0.125
linear_term_stdv = 0.03125
true_parameters =\
    np.concatenate([(np.arange(num_models) + 1) ** 2, [true_linear_term]])
true_curve = np.concatenate([(parameter * np.ones(smaller_num_channels))\
    for parameter in true_parameters[:-1]], axis=0) +\
    (channels * true_parameters[-1])
error = np.ones_like(true_curve) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = true_curve + noise

full_sum_model =\
    SumModel(['constant', 'linear'], [direct_sum_model, linear_model])
gaussian_loglikelihood = GaussianLoglikelihood(data, error, full_sum_model)
incomplete_guess_distribution_set = DistributionSet([(\
    GaussianDistribution(true_linear_term, linear_term_stdv ** 2),\
    'linear_a0', None)])
unknown_name = 'constant'
marginal_draws = 100
conditional_draws = 10

joint_distribution_set = LikelihoodDistributionHarmonizer(\
    incomplete_guess_distribution_set, gaussian_loglikelihood, unknown_name,\
    marginal_draws, conditional_draws=conditional_draws).joint_distribution_set

ndraw = marginal_draws *\
    (1 if (type(conditional_draws) is type(None)) else conditional_draws)

curve_sample = full_sum_model.curve_sample(joint_distribution_set, ndraw)

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.plot(channels, curve_sample.T, color='r', alpha=0.01)
ax.plot(channels, data, color='k')

pl.show()

