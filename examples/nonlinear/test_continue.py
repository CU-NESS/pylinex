from __future__ import division
import os
import numpy as np
from distpy import GaussianDistribution, DistributionSet,\
    GaussianJumpingDistribution, JumpingDistributionSet
from pylinex import Basis, BasisModel, GaussianLoglikelihood, Sampler, NLFitter

seed = 0
file_name1 = 'TESTING_SAMPLER_CONTINUE1.hdf5'
file_name2 = 'TESTING_SAMPLER_CONTINUE2.hdf5'
nwalkers = 10
num_channels = 10
data = np.zeros(num_channels)
error = np.ones(num_channels)
covariance = np.diag(error ** 2)
model = BasisModel(Basis(np.identity(num_channels)))
loglikelihood = GaussianLoglikelihood(data, error, model)
desired_acceptance_fraction = 0.25
jumping_distribution = GaussianJumpingDistribution(covariance *\
    (np.power(desired_acceptance_fraction, (-2) / num_channels) - 1))
jumping_distribution_set =\
    JumpingDistributionSet([(jumping_distribution, model.parameters, None)])
guess_distribution = GaussianDistribution(data, covariance)
guess_distribution_set =\
    DistributionSet([(guess_distribution, model.parameters, None)])
prior_distribution_set = None
steps_per_checkpoint = 10
verbose = True
nthreads = 1
args = []
kwargs = {}
use_ensemble_sampler = False
half_ncheckpoints = 50

try:
    np.random.seed(seed)
    sampler = Sampler(file_name1, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
        restart_mode=None, nthreads=nthreads, args=args, kwargs=kwargs,\
        use_ensemble_sampler=use_ensemble_sampler,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(2 * half_ncheckpoints)
    sampler.close()
    
    np.random.seed(seed)
    sampler = Sampler(file_name2, nwalkers, loglikelihood,\
        jumping_distribution_set=jumping_distribution_set,\
        guess_distribution_set=guess_distribution_set,\
        prior_distribution_set=prior_distribution_set,\
        steps_per_checkpoint=steps_per_checkpoint, verbose=verbose,\
        restart_mode=None, nthreads=nthreads, args=args, kwargs=kwargs,\
        use_ensemble_sampler=use_ensemble_sampler,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(half_ncheckpoints)
    sampler.close()
    sampler = Sampler(file_name2, nwalkers, loglikelihood,\
        jumping_distribution_set=None, guess_distribution_set=None,\
        prior_distribution_set=None, steps_per_checkpoint=steps_per_checkpoint,\
        verbose=verbose, restart_mode='continue', nthreads=nthreads, args=args,\
        kwargs=kwargs, use_ensemble_sampler=use_ensemble_sampler,\
        desired_acceptance_fraction=desired_acceptance_fraction)
    sampler.run_checkpoints(half_ncheckpoints)
    sampler.close()

    fitters = [NLFitter(file_name1), NLFitter(file_name2)]
    chains = [fitter.chain for fitter in fitters]
    print('chains_equal={}'.format(np.all(chains[0] == chains[1])))
    for fitter in fitters:
        fitter.close()
except KeyboardInterrupt:
    if os.path.exists(file_name1):
        os.remove(file_name1)
    if os.path.exists(file_name2):
        os.remove(file_name2)
    raise
else:
    os.remove(file_name1)
    os.remove(file_name2)

