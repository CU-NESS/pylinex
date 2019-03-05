"""
File: examples/loglikelihood/nested_likelihood_distribution_harmonizer.py
Author: Keith Tauscher
Date: 21 Jul 2018

Description: Example showing how to use the LikelihoodDistributionHarmonizer
             class on a likelihood whose model is a nested combination of
             SumModel and ProductModel objects.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, DistributionSet
from pylinex import Basis, BasisModel, SumModel, ProductModel,\
    GaussianLoglikelihood, LikelihoodDistributionHarmonizer

fontsize = 24

ndraw = 1000
num_x_values = 1000
noise_level = 1e-2

x_values = np.linspace(-1, 1, num_x_values)

model_A = BasisModel(Basis((x_values ** 0)[np.newaxis,:]))
model_B = BasisModel(Basis((x_values ** 1)[np.newaxis,:]))
model_C = BasisModel(Basis((1 + (x_values ** 2))[np.newaxis,:]))

model = SumModel(['A', 'B'], [model_A, model_B])
model = ProductModel(['AB', 'C'], [model, model_C])

input_B = 0.2
noiseless_data = model(np.array([1, input_B, 1]))
error = np.ones_like(noiseless_data) * noise_level
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise

gaussian_loglikelihood = GaussianLoglikelihood(data, error, model)

guess_distribution_set_A = DistributionSet(\
    [(GaussianDistribution(1, noise_level ** 2), 'AB_A_a0', None)])
guess_distribution_set_C = DistributionSet(\
    [(GaussianDistribution(1, noise_level ** 2), 'C_a0', None)])
incomplete_guess_distribution_set =\
    guess_distribution_set_A + guess_distribution_set_C
unknown_name_chain = ['AB', 'B']

likelihood_distribution_harmonizer = LikelihoodDistributionHarmonizer(\
    incomplete_guess_distribution_set, gaussian_loglikelihood,\
    unknown_name_chain, ndraw)

full_distribution_set =\
    likelihood_distribution_harmonizer.full_distribution_set
draw = full_distribution_set.draw(ndraw)

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.hist(draw['AB_B_a0'], color='k', histtype='stepfilled', density=True,\
    label='harmonized')
ylim = ax.get_ylim()
ax.plot([input_B] * 2, ylim, color='r', label='input')
ax.set_ylim(ylim)
ax.set_xlabel('Parameter value', size=fontsize)
ax.set_ylabel('PDF', size=fontsize)
ax.set_title('Parameters found through likelihood distribution ' +\
    'harmonization', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')
full_distribution_set.reset()

curve_sample = model.curve_sample(full_distribution_set, ndraw)

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.plot(x_values, curve_sample.T, alpha=0.01, color='k')
ax.plot(x_values, data, color='r')
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('y', size=fontsize)
ax.set_title('Curves found through likelihood distribution harmonization',\
    size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

