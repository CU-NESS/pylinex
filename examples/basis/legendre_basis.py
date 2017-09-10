import numpy as np
from pylinex import LegendreBasis

x_values = np.linspace(-1, 1, 100)
basis = LegendreBasis(100, 5)
basis.plot(x_values=x_values, show=True)

