import numpy as np
from pylinex import PolynomialBasis

x_values = np.linspace(-1, 1, 100)
basis = PolynomialBasis(x_values, 10)
basis.plot(x_values=x_values, show=True)

