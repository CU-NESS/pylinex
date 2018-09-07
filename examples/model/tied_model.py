"""
File: examples/model/tied_model.py
Author: Keith Tauscher
Date: 6 Sep 2018

Description: Example script showing how to use the TiedModel class in the
             context of a LeastSquareFitter object.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import DistributionSet, UniformDistribution
from pylinex import TiedModel, GaussianModel, GaussianLoglikelihood,\
    LeastSquareFitter

fontsize = 24
xs = np.linspace(-1, 1, 1000)
noise_level = 0.1
num_iterations = 10
error = np.ones_like(xs) * noise_level

model = GaussianModel(xs)

tied_model =\
    TiedModel(['a', 'b'], [model, model], 'shared', 'amplitude', 'scale')

parameters = ['shared_amplitude', 'shared_scale', 'a_center', 'b_center']
assert(parameters == tied_model.parameters)

parameter_vector = np.array([2, 0.15, -0.75, 0.75])

print("unprocessed_parameters={}".format(parameter_vector))
print("processed_parameters={}".format(\
    tied_model.form_parameters(parameter_vector)))

true_nonrandom_vector = tied_model(parameter_vector)
true_nonrandom_gradient = tied_model.gradient(parameter_vector)
data_vector =\
    true_nonrandom_vector + (np.random.normal(0, 1, size=error.shape) * error)

loglikelihood = GaussianLoglikelihood(data_vector, error, tied_model)
prior_set = DistributionSet()
prior_set.add_distribution(UniformDistribution(0.9, 1.1), 'shared_amplitude')
prior_set.add_distribution(UniformDistribution(0.1, 0.2), 'shared_scale')
prior_set.add_distribution(UniformDistribution(-1, -0.2), 'a_center')
prior_set.add_distribution(UniformDistribution(0.2, 1), 'b_center')
least_square_fitter = LeastSquareFitter(loglikelihood, prior_set)
least_square_fitter.run(iterations=num_iterations)
argmin = least_square_fitter.successful_argmin
fit_vector = tied_model(argmin)

fig = pl.figure(figsize=(12, 9))
ax = fig.add_subplot(111)
ax.scatter(xs, data_vector, color='k', alpha=0.25, label='True+Noise')
ax.plot(xs, true_nonrandom_gradient[:,0], color='b',\
    label='shared_amplitude derivative')
ax.plot(xs, true_nonrandom_gradient[:,1], color='g',\
    label='shared_scale derivative')
ax.plot(xs, true_nonrandom_gradient[:,2], color='c',\
    label='a_center derivative')
ax.plot(xs, true_nonrandom_gradient[:,3], color='y',\
    label='b_center derivative')
ax.plot(xs, true_nonrandom_vector, color='k',\
    label='True: {!s}'.format(parameter_vector))
ax.plot(xs, fit_vector, color='r', label='Fit: {!s}'.format(argmin))
ax.legend(fontsize=fontsize)
ax.set_xlabel('x', size=fontsize)
ax.set_xlim((xs[0], xs[-1]))
ax.set_ylabel('y', size=fontsize)
ax.set_title('TiedModel test', size=fontsize)
ax.tick_params(labelsize=fontsize, width=2.5, length=7.5, which='major')
ax.tick_params(labelsize=fontsize, width=1.5, length=4.5, which='minor')

pl.show()

