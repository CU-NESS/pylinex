from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import GammaDistribution, TruncatedGaussianDistribution,\
    TruncatedGaussianJumpingDistribution, DistributionSet,\
    JumpingDistributionSet
from pylinex import ConstantModel, GammaLoglikelihood, Sampler, BurnRule,\
    NLFitter

true_value = 1
num_averaged = 10
num_points = 10
total_number = num_averaged * num_points

model = ConstantModel(num_points)
true_parameters = np.ones(1) * true_value
true_variance = (true_value ** 2) / num_averaged
mean = model(true_parameters)
#data = np.ones(num_points) * true_value
data =\
    GammaDistribution(num_averaged, true_value / num_averaged).draw(num_points)
loglikelihood = GammaLoglikelihood(data, model, num_averaged)

file_name = 'TESTINGGAMMALOGLIKELIHOODDELETEIFYOUSEETHIS.TEMP'
nwalkers = 100
steps_per_checkpoint = 100
num_checkpoints = 100
restart_mode = None
guess_distribution_set = DistributionSet([(TruncatedGaussianDistribution(\
    true_value, true_variance / 1000, low=0), 'a', None)])
jumping_distribution_set = JumpingDistributionSet([\
    (TruncatedGaussianJumpingDistribution(true_variance, low=0), 'a', None)])
try:
    loglikelihood.save(file_name)
    assert(loglikelihood == GammaLoglikelihood.load(file_name))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

sampler = Sampler(file_name, nwalkers, loglikelihood,\
    jumping_distribution_set=jumping_distribution_set,\
    guess_distribution_set=guess_distribution_set,\
    steps_per_checkpoint=steps_per_checkpoint, restart_mode=restart_mode)
sampler.run_checkpoints(num_checkpoints, silence_error=True)

burn_rule = BurnRule(desired_fraction=0.5)
fitter = NLFitter(file_name, burn_rule)

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax = fitter.plot_univariate_histogram('a', ax=ax, apply_transforms=True,\
    fontsize=28, matplotlib_function='fill_between', show_intervals=False,\
    bins=100, xlabel='$a$', ylabel='PDF', title='Gamma loglikelihood test')

x_values = true_value +\
    (np.linspace(-1, 1, 1000) * (4 * np.sqrt(true_variance)))
x_values = x_values[x_values > 0]
true_distribution = GammaDistribution(total_number, true_value / total_number)
pdf_values = np.exp(true_distribution.log_value(x_values))
pdf_values = pdf_values / np.max(pdf_values)
ax.plot(x_values, pdf_values, color='r', linewidth=2, linestyle='--')

os.remove(file_name)
pl.show()

