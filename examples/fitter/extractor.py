import numpy as np
import numpy.random as rand
from distpy import GaussianDistribution
from pylinex import AttributeQuantity, Extractor

ntrain = 1000
nvar = 20
xs = np.linspace(-1, 1, 100)
mean_vector = np.zeros(nvar)
stdvs = 1. / (1 + np.arange(nvar))
covariance_matrix = np.diag(np.power(stdvs, 2))
coefficient_distribution = GaussianDistribution(mean_vector, covariance_matrix)
training_set_coefficients = coefficient_distribution.draw(ntrain)
training_set = np.array([np.polyval(coeff[-1::-1], xs)\
                                       for coeff in training_set_coefficients])
quantity = AttributeQuantity('BPIC')

error = np.ones_like(xs) * 1e-3
data = rand.normal(0, 1, error.shape) * error

extractor = Extractor(data, error, ['signal'], [training_set],\
    [{'signal': np.arange(1, 21)}], quantity, 'BPIC')

extractor.fitter.plot_subbasis_fit(name='signal', show=True)

