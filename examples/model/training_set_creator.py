import os
import numpy as np
import matplotlib.pyplot as pl
from distpy import UniformDistribution, GaussianDistribution, DistributionSet
from pylinex import GaussianModel, TrainingSetCreator

x_values = np.linspace(-1, 1, 100)
model = GaussianModel(x_values)

amplitude_distribution = GaussianDistribution(10, 9)
center_distribution = GaussianDistribution(0, 0.01)
scale_deviation_distribution = UniformDistribution(-2, 0)
distribution_tuples = [(amplitude_distribution, 'amplitude', None),\
    (center_distribution, 'center', None),\
    (scale_distribution, 'scale', 'log10')]
prior_set = DistributionSet(distribution_tuples=distribution_tuples)

num_curves = 10
file_name = 'TESTING_TRAINING_SET_CREATOR_DELETE_THIS.hdf5'
seed = 0
verbose = True

try:
    training_set_creator = TrainingSetCreator(model, prior_set, num_curves,\
        file_name, seed=seed, verbose=verbose)
    training_set_creator.generate()
    training_set = training_set_creator.get_training_set()
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

pl.plot(x_values, training_set.T, alpha=0.3)

pl.show()

