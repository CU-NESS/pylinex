"""
File: examples/nonlinear/other_sampler.py
Author: Keith Tauscher
Date: 24 Mar 2018

Description: Example showing another example of using the Sampler class.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import GaussianDistribution, GaussianJumpingDistribution,\
    JumpingDistributionSet, DistributionSet
from pylinex import GaussianLoglikelihood, BasisModel, Basis, Sampler,\
    BurnRule, NLFitter

file_name = 'TEMPORARY_TEST_DELETE_THIS_IF_YOU_SEE_IT.hdf5'
cmap = 'bone'
fontsize = 32
num_walkers = 50
iterations = 1000
steps_per_checkpoint = 100
bins_per_side = 20
num_threads = 1 # to test multithreading, set this to > 1

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution = GaussianJumpingDistribution(5.76 * np.identity(2))
parameters = ['a0', 'a1']
jumping_distribution_set.add_distribution(jumping_distribution, parameters)

guess_distribution_set = DistributionSet()
guess_distribution_set.add_distribution(GaussianDistribution(0, 100), 'a0')
guess_distribution_set.add_distribution(GaussianDistribution(0, 100), 'a1')

model = BasisModel(Basis(np.identity(2)))
data = np.zeros(2)
error = np.ones(2)

loglikelihood = GaussianLoglikelihood(data, error, model)

if not os.path.exists(file_name):
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set, guess_distribution_set,\
        prior_distribution_set=guess_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, num_threads=num_threads)
    sampler.run_checkpoints(iterations // steps_per_checkpoint,\
        silence_error=True)
    sampler.close()

burn_rule = BurnRule(min_checkpoints=5, desired_fraction=0.5)
fitter = NLFitter(file_name, burn_rule)
flattened_chain = np.reshape(fitter.chain, (-1, 2))

x_chain = flattened_chain[:,0]
y_chain = flattened_chain[:,1]

approximate_posterior = fitter.approximate_gaussian_posterior._data[0][0]
approximate_posterior_mean = approximate_posterior.mean
approximate_posterior_covariance = approximate_posterior.variance

true_mean = data
true_covariance = np.diag(error ** 2)

print('true_mean={}'.format(true_mean))
print('approximate_posterior_mean={}'.format(approximate_posterior_mean))
print('true_covariance={}'.format(true_covariance))
print('approximate_posterior_covariance={}'.format(\
    approximate_posterior_covariance))



pl.figure()
pl.plot(x_chain.T)
pl.title('x', size=fontsize)
pl.figure()
pl.plot(y_chain.T)
pl.title('y', size=fontsize)
pl.figure()
pl.hist2d(x_chain, y_chain, bins=bins_per_side, cmap=cmap)
xlim = pl.xlim()
ylim = pl.ylim()
pl.figure()
max_arg = np.max(np.abs([xlim[0], xlim[1], ylim[0], ylim[1]]))
xs = np.linspace(-max_arg, max_arg, 101)
ys = np.linspace(-max_arg, max_arg, 101)
(xs, ys) = np.meshgrid(xs, ys)
stationary_distribution = GaussianDistribution(data, np.diag(error ** 2))
flat_xs = xs.flatten()
flat_ys = ys.flatten()
parameter_arrays = np.stack((flat_xs, flat_ys), axis=1)
log_zs = np.array([stationary_distribution.log_value(parameter_array)\
    for parameter_array in parameter_arrays])
log_zs = np.reshape(log_zs, xs.shape)
argmax = np.argmax(log_zs)
extent = ([-max_arg, max_arg] * 2)
pl.imshow(np.exp(log_zs), cmap=cmap, extent=extent)
pl.show()

os.remove(file_name)

