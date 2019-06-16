"""
File: examples/nonlinear/nonlinear_truncation_sampling.py
Author: Keith Tauscher
Date: 17 Dec 2018

Description: Example script showing simple case of sampling a
             NonlinearTruncationLoglikelihood object.
"""
from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import Expression, DiscreteUniformDistribution,\
    InfiniteUniformDistribution, DistributionSet, GaussianJumpingDistribution,\
    GridHopJumpingDistribution, JumpingDistributionSet
from pylinex import FourierBasis, BasisSet, Fitter, GaussianLoglikelihood,\
    NonlinearTruncationLoglikelihood, LikelihoodDistributionHarmonizer,\
    Sampler, BurnRule, NLFitter, load_loglikelihood_from_hdf5_file

seed = 0
np.random.seed(seed)

fontsize = 24
ndraw = int(1e3)

num_channels = 1 + int(1e2)
noise_level = 1
(odd_amplitude, even_amplitude) = (100, 3)
x_values = np.linspace(-np.pi, np.pi, num_channels)
names = ['even', 'odd']
ndim = len(names)
jumping_probability = 0.05
nterms_minima = np.array([1, 1])
error = np.ones(num_channels) * noise_level
base_basis = FourierBasis(num_channels, 10, error=error)
bases = [base_basis.even_subbasis, base_basis.odd_subbasis]
nterms_maxima = np.array([basis.num_basis_vectors for basis in bases])
basis_set = BasisSet(names, bases)
odd_part = np.sin(x_values) #(np.sin(x_values) * (1 - ((x_values / np.pi) ** 2)))
even_part = np.ones_like(x_values) #((x_values / np.pi) ** 2)
true_curve = ((odd_amplitude * odd_part) + (even_amplitude * even_part))
noise = np.random.normal(0, 1, size=error.shape) * error
data = noise + true_curve

expression = Expression('{0}+{1}', num_arguments=2)

loglikelihood = NonlinearTruncationLoglikelihood(basis_set, data, error,\
    expression, parameter_penalty=1)

file_name = 'TESTINGNONLINEARTRUNCATIONLOGLIKELIHOODDELETETHIS.hdf5'
try:
    loglikelihood.save(file_name)
    loaded_loglikelihood = load_loglikelihood_from_hdf5_file(file_name)
    assert(loglikelihood == loaded_loglikelihood)
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

even_model = loglikelihood.models[loglikelihood.basis_set.names.index('even')]
even_loglikelihood =\
    GaussianLoglikelihood(even_amplitude * even_part, error, even_model)
even_likelihood_distribution_harmonizer = LikelihoodDistributionHarmonizer(\
    None, even_loglikelihood, [], ndraw, quick_fit_parameters=[len(bases[0])])
even_distribution_set =\
    even_likelihood_distribution_harmonizer.joint_distribution_set
even_distribution_set.modify_parameter_names(\
    lambda name: 'even_{!s}'.format(name))
odd_model = loglikelihood.models[loglikelihood.basis_set.names.index('odd')]
odd_loglikelihood =\
    GaussianLoglikelihood(odd_amplitude * odd_part, error, odd_model)
odd_likelihood_distribution_harmonizer = LikelihoodDistributionHarmonizer(\
    None, odd_loglikelihood, [], ndraw, quick_fit_parameters=[len(bases[1])])
odd_distribution_set =\
    odd_likelihood_distribution_harmonizer.joint_distribution_set
odd_distribution_set.modify_parameter_names(\
    lambda name: 'odd_{!s}'.format(name))
guess_distribution_set = even_distribution_set + odd_distribution_set
print("guess_distribution_set.draw()={}".format(guess_distribution_set.draw()))
guess_distribution_set.reset()

prior_distribution_set = DistributionSet()
prior_distribution_set.add_distribution(DiscreteUniformDistribution(\
    even_model.min_terms, even_model.max_terms), 'even_nterms')
prior_distribution_set.add_distribution(DiscreteUniformDistribution(\
    odd_model.min_terms, odd_model.max_terms), 'odd_nterms')
prior_distribution_set.add_distribution(InfiniteUniformDistribution(\
    guess_distribution_set.numparams - 2), [parameter\
    for parameter in guess_distribution_set.params\
    if 'nterms' not in parameter])

jumping_distribution_set = JumpingDistributionSet()
jumping_distribution_set.add_distribution(GridHopJumpingDistribution(\
    ndim=ndim, jumping_probability=jumping_probability, minima=nterms_minima,\
    maxima=nterms_maxima), ['even_nterms', 'odd_nterms'])
jumping_distribution_set.add_distribution(GaussianJumpingDistribution(\
    np.identity(guess_distribution_set.numparams - 2)), [parameter\
    for parameter in guess_distribution_set.params\
    if 'nterms' not in parameter])

file_name = 'TEST_NONLINEAR_TRUNCATION_SAMPLER.hdf5'
num_walkers = 64
steps_per_checkpoint = 100
num_checkpoints = 25
verbose = True
restart_mode = None
desired_acceptance_fraction = 0.25

try:
    sampler = Sampler(file_name, num_walkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
        restart_mode=restart_mode,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(num_checkpoints, silence_error=True)
    sampler.close()
    burn_rule = BurnRule(min_checkpoints=1, desired_fraction=1)
    fitter = NLFitter(file_name, burn_rule=burn_rule)
    fitter.plot_diagnostics()
    fitter.triangle_plot(parameters=['even_nterms', 'odd_nterms'],\
        figsize=(12, 12), fontsize=28, nbins=np.mean(nterms_maxima),\
        plot_type='histogram')
    fitter.plot_chain(parameters='^.*_nterms', figsize=(12, 12), show=False)
    print("mcmc_chosen={}".format({parameter: value\
        for (parameter, value) in zip(loglikelihood.parameters,\
        fitter.maximum_probability_parameters.astype(int))\
        if 'nterms' in parameter}))
except:
    if os.path.exists(file_name):
        os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.show()

