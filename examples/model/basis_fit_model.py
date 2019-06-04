"""
File: examples/model/basis_fit_model.py
Author: Keith Tauscher
Date: 4 Jun 2019

Description: Example script showing a simple usage of the BasisFitModel class
             with polynomial fits.
"""
import numpy as np
import matplotlib.pyplot as pl
from pylinex import Basis, BasisFitModel

fontsize = 20

num_polynomial_terms = 5
half_num_channels = 50
num_channels = (2 * half_num_channels) + 1
x_values = np.linspace(-1, 1, num_channels)

basis = Basis(x_values[np.newaxis,:] **\
    np.arange(num_polynomial_terms)[:,np.newaxis])

noiseless_data = basis(np.random.normal(0, 10, size=num_polynomial_terms))
noise_level = 1
error = np.ones(num_channels) * noise_level * np.exp(-np.abs(x_values))
noise = np.random.normal(0, 1, size=error.shape) * error
data = noiseless_data + noise

model = BasisFitModel(basis, data, error=error)

expected = np.polyval(np.polyfit(x_values, data, num_polynomial_terms - 1,\
    w=1/error), x_values)
actual = model(np.array([num_polynomial_terms]))

fig = pl.figure(figsize=(12,9))
ax = fig.add_subplot(111)
ax.scatter(x_values, data, color='k', label='data')
ax.plot(x_values, expected, color='C0', label='expected')
ax.plot(x_values, actual, color='C1', label='actual')
ax.plot(x_values, actual - expected, color='C2', label='difference')
ax.legend(fontsize=fontsize)
ax.set_xlabel('x', size=fontsize)
ax.set_ylabel('y', size=fontsize)
ax.set_title('BasisFitModel test', size=fontsize)

pl.show()

