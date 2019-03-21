"""
File: examples/nonlinear/truncation_extractor.py
Author: Keith Tauscher
Date: 20 Mar 2019

Description: Example showing the use of the TruncationExtractor specialty
             class, which runs an MCMC over the discrete space of truncations
             of bases of a linear model.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution
from pylinex import RepeatExpander, PadExpander, Basis, BasisModel,\
    TransformedModel, GaussianModel, TruncationExtractor

seed = 0
np.random.seed(seed)

num_regions = 4

verbose = True
integration_time_in_hrs = 100
num_foreground_terms = 3

num_frequencies = 500
frequencies = np.linspace(50, 100, num_frequencies)
signal_model = GaussianModel(frequencies)
normalization_frequency = np.sqrt(frequencies[0] * frequencies[-1])
log_normed_frequencies = np.log(frequencies / normalization_frequency)
log_frequency_powers = np.arange(num_foreground_terms)
foreground_basis =\
    log_normed_frequencies[np.newaxis,:] ** log_frequency_powers[:,np.newaxis]
foreground_model = TransformedModel(BasisModel(Basis(foreground_basis)), 'exp')

signal_name = 'signal'
foreground_names =\
    ['foreground_reg_{:d}'.format(region) for region in range(num_regions)]
names = [signal_name] + foreground_names

signal_distribution_set = DistributionSet()
signal_distribution_set.add_distribution(UniformDistribution(-1, -0.5),\
    'amplitude')
signal_distribution_set.add_distribution(UniformDistribution(60, 90), 'center')
signal_distribution_set.add_distribution(UniformDistribution(10, 30), 'scale')
signal_ndraw = 10000
signal_training_set =\
    signal_model.curve_sample(signal_distribution_set, signal_ndraw)

foreground_distribution_set = DistributionSet()
foreground_distribution_set.add_distribution(\
    UniformDistribution(1e3, 3e3), 'a0', 'exp')
if num_foreground_terms > 1:
    foreground_distribution_set.add_distribution(\
        UniformDistribution(-2.4, -2.6), 'a1')
if num_foreground_terms > 2:
    foreground_distribution_set.add_distribution(\
        UniformDistribution(-1e-1, 1e-1), 'a2')
if num_foreground_terms > 3:
    foreground_distribution_set.add_distribution(\
        UniformDistribution(-1e-2, 1e-2), 'a3')
foreground_ndraw = 10000
foreground_training_sets = [foreground_model.curve_sample(\
    foreground_distribution_set, foreground_ndraw)\
    for region in range(num_regions)]
training_sets = [signal_training_set] + foreground_training_sets

signal_nterms_maximum = 40
foreground_nterms_maximum = 40
foreground_nterms_maxima = [foreground_nterms_maximum] * num_regions
nterms_maxima = [signal_nterms_maximum] + foreground_nterms_maxima

extractor_file_name = 'TEST_TRUNCATION_EXTRACTOR.hdf5'
sampler_file_name = 'TEST_TRUNCATION_SAMPLER.hdf5'
trust_ranks = False
information_criterion = 'deviance_information_criterion'

signal_expander = RepeatExpander(num_regions)
foreground_expanders = [PadExpander('{:d}*'.format(region),\
    '{:d}*'.format(num_regions - region - 1)) for region in range(num_regions)]
expanders = [signal_expander] + foreground_expanders

foreground_parameter_vectors = foreground_distribution_set.draw(num_regions)
foreground_parameter_vectors = np.stack(\
    [foreground_parameter_vectors['a{:d}'.format(index)]\
    for index in range(num_foreground_terms)], axis=1)
true_foreground = np.concatenate([foreground_model(parameter_vector)\
    for parameter_vector in foreground_parameter_vectors])
true_signal =\
    np.concatenate([signal_model(np.array([-0.8, 80, 18]))] * num_regions)
data = true_foreground + true_signal
channel_width_in_MHz = (frequencies[1] - frequencies[0])
error = data / (6e4 * np.sqrt(channel_width_in_MHz * integration_time_in_hrs))
true_noise = np.random.normal(0, 1, size=error.shape) * error
data = data + true_noise

try:
    extractor = TruncationExtractor(data, error, names, training_sets,\
        nterms_maxima, sampler_file_name,\
        information_criterion=information_criterion, expanders=expanders,\
        trust_ranks=trust_ranks, verbose=verbose)
    truncations = extractor.optimal_truncations
    truncations = {'{!s}_nterms'.format(name): truncation\
        for (name, truncation) in zip(names, truncations)}
    print("truncations={}".format(truncations))
    optimal_fitter = extractor.optimal_fitter
    (channel_bias, channel_error) =\
        (optimal_fitter.channel_bias, optimal_fitter.channel_error)
    channels = np.arange(num_regions * num_frequencies)
    pl.scatter(channels, channel_bias, color='k')
    pl.fill_between(channels, -error, error, color='r', alpha=0.2)
    pl.fill_between(channels, -channel_error, channel_error, color='r', alpha=0.5)
except:
    if os.path.exists(sampler_file_name):
        os.remove(sampler_file_name)
    raise
else:
    os.remove(sampler_file_name)

pl.show()

