"""
Example script showing that the GaussianLoglikelihood class can be initialized
with a 1D array of standard deviations or a SparseSquareBlockDiagonalMatrix
object of covariances. They are equivalent (to near bit-level precision) when
they represent the same covariance matrix.

**File**: $PYLINEX/examples/loglikelihood/gaussian_loglikelihood.py  
**Author**: Keith Tauscher  
**Date**: 1 Jun 2021
"""
import numpy as np
from distpy import SparseSquareBlockDiagonalMatrix, UniformDistribution
from pylinex import GaussianLoglikelihood, ConstantModel

np.random.seed(0)

num_channels = 1000

error_distribution = UniformDistribution(1, 2)
error = error_distribution.draw(num_channels)

blocked =\
    SparseSquareBlockDiagonalMatrix((error ** 2)[:,np.newaxis,np.newaxis])

data = np.random.normal(0, 1, size=num_channels) * error

model = ConstantModel(num_channels)

stdv_loglikelihood = GaussianLoglikelihood(data, error, model)
blocked_loglikelihood = GaussianLoglikelihood(data, blocked, model)

parameters = np.random.normal(0, 10, size=(100,1))
for parameter in parameters:
    assert(np.isclose(stdv_loglikelihood(parameter),\
        blocked_loglikelihood(parameter), rtol=0, atol=1e-10))

