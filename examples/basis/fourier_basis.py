import numpy as np
from pylinex import FourierBasis

x_values = np.linspace(-1, 1, 100)
basis = FourierBasis(100, 5)
basis.plot(x_values=x_values, show=True)
