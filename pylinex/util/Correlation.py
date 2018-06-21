import numpy as np

def autocorrelation(curves):
    """
    Computes the correlation of the given curve with itself. Values of 1 (-1)
    indicate perfect (anti-)correlation.
    
    curves: 1D or 2D numpy.ndarray of values to correlate. if 2D, then each
            element of curves, curves[i], will be used to independently measure
            the correlation and then the average will be returned
    
    returns: (correlation, noise_level) where both are 1D arrays of length
             curves.shape[-1]. noise_level represents the level expected when
             the input curve follows a zero-mean normal distribution
    """
    if curves.ndim == 1:
        curves = curves[np.newaxis,:]
    degrees_of_freedom = (curves.shape[-1] - np.arange(curves.shape[-1]))
    correlation =\
        np.array([(np.correlate(curve, curve, mode='full')[-len(curve):] /\
        degrees_of_freedom) for curve in curves])
    correlation = np.mean(correlation, axis=0)
    correlation = correlation / correlation[0]
    noise_level = np.power(curves.shape[0] * degrees_of_freedom, -0.5)
    return (correlation, noise_level)

