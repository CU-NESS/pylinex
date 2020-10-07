"""
File: examples/model/multi_conditional_fit_model.py
Author: Keith Tauscher
Date: 5 Oct 2020

Description: Example script showing a use of the MultiConditionalFitModel
             class.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
from distpy import GaussianDistribution, DistributionSet
from pylinex import Basis, LegendreBasis, FourierBasis, BasisModel,\
    GaussianModel, LorentzianModel, SumModel, ProductModel,\
    MultiConditionalFitModel

fontsize = 24
triangle_plot_fontsize = 8

seed = 0
np.random.seed(seed)

include_noise = True
include_nuisance_prior = False
include_offset_prior = False

num_channels = 1001
xs = np.linspace(-1, 1, num_channels)

num_nuisance_terms = 5
num_offset_terms = 5

gain_model = LorentzianModel(xs)
nuisance_model =\
    BasisModel(LegendreBasis(num_channels, num_nuisance_terms - 1))
signal_model = GaussianModel(xs)
ideal_model = SumModel(['signal', 'nuisance'], [signal_model, nuisance_model])
multiplicative_model =\
    ProductModel(['gain', 'ideal'], [gain_model, ideal_model])
#offset_model = BasisModel(FourierBasis(num_channels, 30)[1+num_nuisance_terms:1+num_nuisance_terms+num_offset_terms])
offset_model = BasisModel(FourierBasis(num_channels, 30)[1:1+num_offset_terms])
full_model = SumModel(['multiplicative', 'offset'],\
    [multiplicative_model, offset_model])

true_gain_parameters = np.array([1, 0.2, 2])
true_signal_parameters = np.array([-1, 0, 0.3])
true_nuisance_parameters = np.random.normal(0, 10, size=num_nuisance_terms)
true_offset_parameters = np.random.normal(0, 10, size=num_offset_terms)

true_gain = gain_model(true_gain_parameters)
true_signal = signal_model(true_signal_parameters)
true_nuisance = nuisance_model(true_nuisance_parameters)
true_offset = offset_model(true_offset_parameters)

noiseless_data = (true_gain * (true_signal + true_nuisance)) + true_offset

error = np.ones_like(xs)
if include_noise:
    true_noise = np.random.normal(0, 1, size=num_channels) * error
else:
    true_noise = np.zeros(num_channels)

data = noiseless_data + true_noise
unknown_name_chains = [['multiplicative', 'ideal', 'nuisance'], ['offset']]

if include_nuisance_prior:
    nuisance_prior_covariance =\
        0.9 * np.ones((num_nuisance_terms, num_nuisance_terms))
    for index in range(num_nuisance_terms):
        nuisance_prior_covariance[index,index] = 1
    nuisance_prior = GaussianDistribution(true_nuisance_parameters,\
        nuisance_prior_covariance)
else:
    nuisance_prior = None

if include_offset_prior:
    offset_prior_covariance =\
        0.9 * np.ones((num_offset_terms, num_offset_terms))
    for index in range(num_offset_terms):
        offset_prior_covariance[index,index] = 1
    offset_prior = GaussianDistribution(true_offset_parameters,\
        offset_prior_covariance)
else:
    offset_prior = None

priors = [nuisance_prior, offset_prior]

model = MultiConditionalFitModel(full_model, data, error, unknown_name_chains,\
    priors=priors)

(recreation, conditional_mean, conditional_covariance,\
    log_prior_at_conditional_mean) = model(np.concatenate(\
    [true_gain_parameters, true_signal_parameters]),\
    return_conditional_mean=True, return_conditional_covariance=True,\
    return_log_prior_at_conditional_mean=True)

input_unknown_parameters =\
    np.concatenate([true_nuisance_parameters, true_offset_parameters])

conditional_distribution =\
    GaussianDistribution(conditional_mean, conditional_covariance)
unknown_parameters = sum([[parameter for parameter in full_model.parameters\
    if '_'.join(name_chain) in parameter]\
    for name_chain in unknown_name_chains], [])
conditional_distribution_set =\
    DistributionSet([(conditional_distribution, unknown_parameters)])
num_samples = int(1e6)
fig = conditional_distribution_set.triangle_plot(num_samples,\
    reference_value_mean=input_unknown_parameters, nbins=200,\
    fontsize=triangle_plot_fontsize, contour_confidence_levels=[0.68, 0.95],\
    parameter_renamer=(lambda parameter: '_'.join(parameter.split('_')[-2:])))
conditional_distribution_set.triangle_plot(num_samples, fig=fig,\
    reference_value_mean=input_unknown_parameters, nbins=200,\
    fontsize=triangle_plot_fontsize, plot_type='histogram',\
    parameter_renamer=(lambda parameter: '_'.join(parameter.split('_')[-2:])))

if not (include_nuisance_prior or include_offset_prior):
    assert(np.isclose(log_prior_at_conditional_mean, 0))

fig = pl.figure(figsize=(12,9))

ax = fig.add_subplot(211)
ax.scatter(xs, data, color='k', label='with noise')
ax.scatter(xs, noiseless_data, color='C0', label='without noise')
ax.plot(xs, recreation, color='C2', label='recreation')
ax.set_xlim((xs[0], xs[-1]))
ax.set_xlabel('$x$', size=fontsize)
ax.set_ylabel('$y$', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.legend(fontsize=fontsize)

ax = fig.add_subplot(212)
ax.scatter(xs, data - recreation, color='k', label='with noise')
ax.scatter(xs, noiseless_data - recreation, color='C0', label='without noise')
ax.set_xlim((xs[0], xs[-1]))
ax.set_xlabel('$x$', size=fontsize)
ax.set_ylabel('$\delta y$', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.legend(fontsize=fontsize)

fig.subplots_adjust(left=0.11, right=0.95, bottom=0.10, top=0.97, hspace=0.53)

pl.show()

