"""
File: examples/model/conditional_fit_model.py
Author: Keith Tauscher
Date: 31 May 2019

Description: Example script showing how objects of the ConditionalFitModel
             class relate to the model given at initialization.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import ChiSquaredDistribution
from pylinex import LegendreBasis, SumModel, ProductModel, LorentzianModel,\
    TruncatedBasisHyperModel, GaussianModel, ConditionalFitModel

fontsize = 24
first_seed = None
second_seed = None

half_num_channels = 50
num_channels = (2 * half_num_channels) + 1
x_values = np.linspace(-1, 1, num_channels)

gaussian_model = GaussianModel(x_values)
lorentzian_model = LorentzianModel(x_values)
max_num_basis_vectors = 10
num_basis_vectors = 5
basis = LegendreBasis(num_channels, max_num_basis_vectors - 1)
basis_model = TruncatedBasisHyperModel(basis)
sum_model = SumModel(['basis', 'lorentzian'], [basis_model, lorentzian_model])

gaussian_parameters = np.array([1, 0.2, 1])
lorentzian_parameters = np.array([15, 0, 0.5])
np.random.seed(first_seed)
basis_parameters =\
    np.concatenate([np.random.normal(0, 100, size=num_basis_vectors),\
    np.zeros(max_num_basis_vectors - num_basis_vectors), [num_basis_vectors]])

noise_level = 1
error = np.ones((num_channels,)) * noise_level
np.random.seed(second_seed)
noise = np.random.normal(0, 1, size=error.shape) * error

full_model = ProductModel(['gaussian', 'sum'], [gaussian_model, sum_model])
full_parameters = np.concatenate([gaussian_parameters, basis_parameters,\
    lorentzian_parameters])

data = full_model(full_parameters) + noise
unknown_name_chain = ['sum', 'basis']

model = ConditionalFitModel(full_model, data, error, unknown_name_chain)
parameters =\
    np.concatenate([gaussian_parameters, lorentzian_parameters,\
    [num_basis_vectors]])
print("model.parameters={}".format(model.parameters))

degrees_of_freedom = num_channels - num_basis_vectors
chi_squared = np.sum(np.power((data - model(parameters)) / error, 2)) /\
    degrees_of_freedom
chi_squared_distribution =\
    ChiSquaredDistribution(degrees_of_freedom, reduced=True)
probability_level = 0.95
threshold =\
    chi_squared_distribution.left_confidence_interval(probability_level)[1]

try:
    assert(chi_squared < threshold)
except:
    print(("Chi squared, {0:.4g}, was above threshold for probability " +\
        "level {1:.2f}, {2:.4g}.").format(chi_squared, probability_level,\
        threshold))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(211)
ax.scatter(x_values, data, color='k', label='full model + noise')
ax.plot(x_values, model(parameters), color='r', label='ConditionalFitModel')
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('Total y', size=fontsize)
ax.set_title('ConditionalFitModel test')
ax.legend(fontsize=fontsize)
ax = fig.add_subplot(212)
ax.scatter(x_values, data - model(parameters), color='k',\
    label='$\chi^2={:.4f}$'.format(chi_squared))
ax.legend(fontsize=fontsize)
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('Residual y', size=fontsize)

pl.show()

