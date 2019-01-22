"""
File: pylinex/util/Correlation.py
Author: Keith Tauscher
Date: 12 Sep 2018

Description: File defining functions which calculate the autocorrelation of a
             curve (or the average autocorrelation of many curves) and the
             chi-squared and psi-squared statistics associated with it.
"""
from __future__ import division
import numpy as np

def autocorrelation(curves, error=None, normalize_by_chi_squared=False):
    """
    Computes the correlation of the given curve with itself. If
    normalize_by_chi_squared=True, values of 1 (-1) indicate perfect
    (anti-)correlation.
    
    curves: 1D or 2D numpy.ndarray of values to correlate. if 2D, then each
            element of curves, curves[i], will be used to independently measure
            the correlation and then the average will be returned
    error: the vector of noise level estimates, if applicable. If None, it is
           assumed that all points are equally important
    normalize_by_chi_squared: if True, the array is scaled so that the 0
                                       correlation is 1. This changes the
                                       effective noise level as well (and
                                       correlates the noise levels of different
                                       channels).
                              if False, then the noise level assumes that
                                        curves is to be interpreted as if it
                                        was random noise with no correlations
                                        and a standard deviation of 1
    
    returns: (correlation, noise_level) where both are 1D arrays of length
             curves.shape[-1]. noise_level represents the level expected when
             the input curve follows a zero-mean normal distribution
    """
    if curves.ndim == 1:
        curves = curves[np.newaxis,:]
    assert(curves.ndim == 2)
    if error is None:
        normalized_curves = curves
    else:
        normalized_curves = curves / error[np.newaxis,:]
    degrees_of_freedom = (curves.shape[-1] - np.arange(curves.shape[-1]))
    correlation =\
        np.array([(np.correlate(curve, curve, mode='full')[-len(curve):] /\
        degrees_of_freedom) for curve in normalized_curves])
    correlation = np.mean(correlation, axis=0)
    if normalize_by_chi_squared:
        correlation = correlation / correlation[0]
    noise_level = np.power(degrees_of_freedom * curves.shape[0], -0.5)
    return (correlation, noise_level)

def psi_squared(curves, error=None, normalize_by_chi_squared=False,\
    return_null_hypothesis_error=False, minimum_correlation_spacing=1):
    """
    Calculates the psi_squared statistic.
    
    curves: 1D or 2D numpy.ndarray of values to correlate. if 2D, then each
            element of curves, curves[i], will be used to independently measure
            the correlation and then the average will be returned
    error: the vector of noise level estimates, if applicable. If None, it is
           assumed that all points are equally important
    normalize_by_chi_squared: if True (not default), psi_squared is normed by
                                                     (chi_squared ** 2)
    return_null_hypothesis_error: if True, expected error on psi_squared (if
                                           curves are noise-like) is returned
                                           as well
                                  (Must be False if curves is not 1D or
                                  minimum_correlation_spacing is not 1)
    minimum_correlation_spacing: the minimum correlation spacing to consider in
                                 the calculation, default: 1
    
    returns: single number, mean-square nonzero correlation
    """
    (correlation, correlation_noise_level) = autocorrelation(curves,\
        error=error, normalize_by_chi_squared=normalize_by_chi_squared)
    normalized_correlation =\
        (correlation / correlation_noise_level)[minimum_correlation_spacing:]
    return_value = np.mean(np.power(normalized_correlation, 2))
    if return_null_hypothesis_error:
        if curves.ndim == 1:
            if minimum_correlation_spacing == 1:
                return_value = (return_value, np.sqrt(14 / curves.size))
            else:
                raise NotImplementedError("I am not yet sure how " +\
                    "psi_squared's variance should be expected to change " +\
                    "if the minimum correlation spacing is not set to 1.")
        else:
            raise NotImplementedError("I am not yet sure how psi_squared's " +\
                "variance should be expected to change if multiple curves " +\
                "are used.")
    return return_value

def chi_squared(curves, error=None, return_null_hypothesis_error=False,\
    num_parameters=0):
    """
    Calculates the standard reduced chi-squared statistic using the given
    curves, which are assumed normalized (unless error is not None).
    
    curves: 1D or 2D numpy.ndarray of values to correlate. if 2D, then each
            element of curves, curves[i], will be used to independently measure
            chi_squared and then the average will be returned
    error: if None, curves are assumed to be normalized
           otherwise, curves have error divided out
    return_null_hypothesis_error: if True, expected error on chi_squared (if
                                           curves are noise-like) is returned
                                           as well
    num_parameters: number of modes of noise assumed to be removed, default: 0
    
    returns: single number, mean-square data value
    """
    if error is None:
        normed_curves = curves
    elif curves.ndim == 2:
        normed_curves = curves / error[np.newaxis,:]
    else:
        normed_curves = curves / error
    return_value = np.mean(np.power(normed_curves, 2))
    return_value /= (1 - (num_parameters / curves.size))
    if return_null_hypothesis_error:
        return_value =\
            (return_value, np.sqrt(2 / (curves.size - num_parameters)))
    return return_value

