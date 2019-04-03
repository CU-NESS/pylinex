"""
File: pylinex/basis/EffectiveRank.py
Author: Keith Tauscher
Date: 17 Oct 2017

Description: File containing function which, given a training set of curves and
             a corresponding noise level, determines the effective rank of the
             training set, which is the number of modes to fit within the error
             (see the docstring for effective_training_set_rank for details on
             what that can mean).
"""
import numpy as np
from .TrainedBasis import TrainedBasis

def effective_training_set_rank(training_set, noise_level, method='abs',\
    number_of_modes_to_consider=None, use_min_noise_level=False, level=1.,\
    suppress_runtime_error=False):
    """
    Finds the number of modes which are needed to fit the given training set to
    the given noise level.
    
    training_set: 2D numpy.ndarray of shape (ncurves, nchannels)
    noise_level: 1D numpy.ndarray of shape (nchannels,)
    method: if 'rms', RMS of normalized bias (bias/error) must be less than
                      level for all curves for rank to be returned
            if 'abs', normalized bias (bias/error) must be less than level for
                      all curves and all channels
    number_of_modes_to_consider: if int, maximum number of modes to compute.
                                         Should be much larger than the
                                         expected rank. If it is not larger
                                         than the rank, this will throw a
                                         RuntimeError.
                                 if None, exhaustive search is performed by
                                          internally setting
                                          number_of_modes_to_consider as the
                                          minimum of ncurves and nchannels
    use_min_noise_level: if True, minimum of noise level used for every channel
                         otherwise, noise level's changes with different data
                                    channels are accounted for
    level: multiple of the noise level to consider
    suppress_runtime_error: if True, if no considered rank satisfies constraint
                                     defined by the arguments to this function,
                                     number_of_modes_to_consider is returned
                            if False, if no considered rank satisfies
                                      constraint defined by the arguments to
                                      this function, a RuntimeError is raised.
                                      This is the default behavior.
    
    returns: integer number of modes necessary to fit every curve in the
             training set to within noise_level
    """
    if type(number_of_modes_to_consider) is type(None):
        number_of_modes_to_consider = np.min(training_set.shape)
    svd_basis = TrainedBasis(training_set, number_of_modes_to_consider,\
        error=noise_level)
    level2 = (level ** 2)
    for rank in range(1, number_of_modes_to_consider + 1):
        importance_weighted_basis =\
            svd_basis.basis[:rank].T * svd_basis.importances[np.newaxis,:rank]
        fit = np.dot(importance_weighted_basis,\
            svd_basis.training_set_space_singular_vectors[:rank]).T
        if use_min_noise_level:
            normalized_bias = (fit - training_set) / np.min(noise_level)
        else:
            normalized_bias = (fit - training_set) / noise_level[np.newaxis,:]
        if method.lower() == 'rms':
            mean_squared_normalized_bias =\
                np.mean(np.power(normalized_bias, 2), axis=1)
            if np.all(mean_squared_normalized_bias < level2):
                return rank
        elif method.lower() == 'abs':
            if np.all(normalized_bias < level):
                return rank
        else:
            raise ValueError("method not recognized. Must be 'rms' or 'abs'.")
    if suppress_runtime_error:
        return number_of_modes_to_consider
    else:
        raise RuntimeError("The rank of the given training set was larger " +\
            "than the number of modes considered.")

