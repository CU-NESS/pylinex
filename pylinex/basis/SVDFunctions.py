"""
File: pylinex/basis/SVDFunctions.py
Author: Neil Bassett
Date: 3 May 2021

Description: File containing functions used to perform Singular Value
             Decomposition (SVD) on training sets to obtain basis vectors
             used for the TrainedBasis or PartiallyTrainedBasis classes.
"""
import numpy as np
import numpy.linalg as la

def weighted_SVD(matrix, error=None, full_matrices=False):
    """
    Finds the most important modes of the given matrix given the weightings
    given by the error.

    matrix a horizontal rectangular matrix
    error weighting applied to the dimension corresponding to the rows
    """
    if type(error) is type(None):
        error = np.ones(matrix.shape[0])
    expanded_error = error[:,np.newaxis]
    to_svd = matrix / expanded_error
    (SVD_U, SVD_S, SVD_V_transpose) =\
        la.svd(to_svd, full_matrices=full_matrices)
    SVD_U = SVD_U * expanded_error
    return SVD_U, SVD_S, SVD_V_transpose.T

def weighted_SVD_basis(curves, error=None, Neigen=None):
    """
    Finds a basis using weighted SVD performed on the given curves.

    curves: 2D numpy.ndarray of curves with which to define the basis
    error: if None, no weights are used
           otherwise, it should be a 1D numpy.ndarray of the same length as the
                      curves (Nchannel)
    Neigen: number of basis vectors to return

    returns: 2D numpy.ndarray of shape (Neigen, Nchannel)
    """
    if type(error) is type(None):
        error = np.ones(curves.shape[-1])
    if type(Neigen) is type(None):
        Neigen = curves.shape[-1]
    SVD_U, SVD_S, SVD_V = weighted_SVD(curves.T, error=error)
    return (SVD_U.T[:Neigen], SVD_S, SVD_V.T[:Neigen])
