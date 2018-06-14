import numpy as np

def autocorrelation(curve):
    """
    Computes the correlation of the given curve with itself. Values of 1 (-1)
    indicate perfect (anti-)correlation.
    
    curve: 1D numpy.ndarray of values to correlate
    """
    correlation = np.correlate(curve, curve, mode='full')[-len(curve):] /\
        (len(curve) - np.arange(len(curve)))
    return correlation / correlation[0]

