"""
File: examples/model/conditional_fit_model.py
Author: Keith Tauscher
Date: 4 Oct 2020

Description: Example script showing how objects of the
             SingleConditionalFitModel class relate to the model given at
             initialization.
"""
import numpy as np
import matplotlib.pyplot as pl
from distpy import ChiSquaredDistribution, GaussianDistribution,\
    DistributionSet
from pylinex import LegendreBasis, SumModel, ProductModel, LorentzianModel,\
    BasisModel, GaussianModel, SingleConditionalFitModel

fontsize = 24
triangle_plot_fontsize = 10
include_noise = True
first_seed = None
second_seed = None

half_num_channels = 50
num_channels = (2 * half_num_channels) + 1
x_values = np.linspace(-1, 1, num_channels)

gaussian_model = GaussianModel(x_values)
lorentzian_model = LorentzianModel(x_values)
num_basis_vectors = 10
basis = LegendreBasis(num_channels, num_basis_vectors - 1)
basis_model = BasisModel(basis)
sum_model = SumModel(['basis', 'lorentzian'], [basis_model, lorentzian_model])

gaussian_parameters = np.array([1, 0.2, 1])
lorentzian_parameters = np.array([15, 0, 0.5])
np.random.seed(first_seed)
basis_parameters = np.random.normal(0, 100, size=num_basis_vectors)

noise_level = 1
error = np.ones((num_channels,)) * noise_level
np.random.seed(second_seed)
if include_noise:
    noise = np.random.normal(0, 1, size=error.shape) * error
else:
    noise = np.zeros_like(error)

full_model = ProductModel(['gaussian', 'sum'], [gaussian_model, sum_model])
full_parameters = np.concatenate([gaussian_parameters, basis_parameters,\
    lorentzian_parameters])

data = full_model(full_parameters) + noise
unknown_name_chain = ['sum', 'basis']

model = SingleConditionalFitModel(full_model, data, error, unknown_name_chain)
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

(recreation, conditional_mean, conditional_covariance) = model(parameters,\
    return_conditional_mean=True, return_conditional_covariance=True)
conditional_distribution =\
    GaussianDistribution(conditional_mean, conditional_covariance)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(211)
ax.scatter(x_values, data, color='k', label='full model + noise')
ax.plot(x_values, recreation, color='r', label='SingleConditionalFitModel')
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('Total y', size=fontsize)
ax.set_title('SingleConditionalFitModel test')
ax.legend(fontsize=fontsize)
ax = fig.add_subplot(212)
ax.scatter(x_values, data - model(parameters), color='k',\
    label='$\chi^2={:.4f}$'.format(chi_squared))
ax.legend(fontsize=fontsize)
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('Residual y', size=fontsize)

num_samples = int(1e6)
marginalized_parameters = [parameter for parameter in full_model.parameters\
    if parameter not in model.parameters]
conditional_distribution_set =\
    DistributionSet([(conditional_distribution, marginalized_parameters)])

fig = conditional_distribution_set.triangle_plot(num_samples,\
    reference_value_mean=basis_parameters, nbins=200,\
    fontsize=triangle_plot_fontsize,\
    contour_confidence_levels=[0.68, 0.95, 0.997], plot_type='contour',\
    parameter_renamer=(lambda parameter: '_'.join(parameter.split('_')[-2:])))
conditional_distribution_set.triangle_plot(num_samples, fig=fig,\
    reference_value_mean=basis_parameters, nbins=200,\
    fontsize=triangle_plot_fontsize, plot_type='histogram',\
    parameter_renamer=(lambda parameter: '_'.join(parameter.split('_')[-2:])))

pl.show()

