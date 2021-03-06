"""
File: pylinex/basis/TrainingSetPlot.py
Author: Keith Tauscher
Date: 22 Feb 2019

Description: File containing a function which plots a three panel figure
             summarizing a given training set.
"""
import numpy as np
import matplotlib.pyplot as pl
from .TrainedBasis import TrainedBasis

def plot_training_set_with_modes(training_set, num_modes, error=None,\
    mean_translation=False, x_values=None, curve_slice=slice(None),\
    subtract_mean=False, alpha=1., fontsize=24, xlabel='',\
    extra_ylabel_string='', title='', figsize=(12,20), show=False):
    """
    Plots a three panel figure summarizing the given training set. The top
    panel shows the training set itself. The middle panel shows the basis
    coming from the training set (with the given number of modes) assuming the
    given error curve. The bottom panel shows the residuals when the basis
    shown in the second panel is used to fit the training set.
    
    training_set: 2D numpy.ndarray of shape (ncurves, nchannels) containing set
                  of training curves
    num_modes: the number of eigenmodes to use in the basis to plot
    error: the error distribution expected in data for which the training set
           will be used to fit
    mean_translation: if True (default False), the mean of the training set is
                      subtracted before taking SVD.
    x_values: np.ndarray of x values with which to plot training set, basis,
              and residuals. If None, set to np.arange(training_set.shape[1])
    curve_slice: slice to apply to the first axis of training_set and residuals
                 when they are plotted in the top and bottom panels
    subtract_mean: if True (default: False), mean of training set is subtracted
                                             in top panel
    alpha: opacity of curves plotted in training set and residuals panels
    fontsize: size of fonts for labels and title
    xlabel: string label describing x_values
    extra_ylabel_string: string to add to end of ylabel of each panel (usually
                         a space and a units string)
    title: title to put on top of Figure
    figsize: size of figure on which to plot 3 panels
    show: if True, matplotlib.pyplot.show() is called before this function
          returns
    
    returns: if show is False, returns Figure object, otherwise None
    """
    if type(x_values) is type(None):
        x_values = np.arange(training_set.shape[1])
    xlim = (x_values[0], x_values[-1])
    basis = TrainedBasis(training_set, num_modes, error=error,\
        mean_translation=mean_translation)
    residuals = training_set - basis(basis.training_set_fit_coefficients)
    if mean_translation:
        residuals = residuals - np.mean(training_set, axis=0)[np.newaxis,:]
    fig = pl.figure(figsize=figsize)
    ax = fig.add_subplot(311)
    ax.plot(x_values, (training_set[curve_slice,:] -\
        (np.mean(training_set[curve_slice,:], axis=0, keepdims=True)\
        if subtract_mean else 0)).T, alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_title(title, size=fontsize)
    ax.set_ylabel('Training set{0!s}{1!s}'.format(\
        ' - mean' if subtract_mean else '', extra_ylabel_string),\
        size=fontsize)
    ax.tick_params(labelsize=fontsize, length=7.5, width=2.5, which='major',\
        labelbottom=False, direction='inout')
    ax.tick_params(labelsize=fontsize, length=4.5, width=1.5, which='minor',\
        labelbottom=False, direction='inout')
    ax = fig.add_subplot(312)
    ax.plot(x_values, basis.basis.T)
    ax.set_xlim(xlim)
    ax.set_ylabel('Modes{!s}'.format(extra_ylabel_string), size=fontsize)
    ax.tick_params(labelsize=fontsize, length=7.5, width=2.5, which='major',\
        top=True, labelbottom=False, direction='inout')
    ax.tick_params(labelsize=fontsize, length=4.5, width=1.5, which='minor',\
        top=True, labelbottom=False, direction='inout')
    ax = fig.add_subplot(313)
    ax.plot(x_values, residuals[curve_slice,:].T, alpha=alpha)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel('Residuals{!s}'.format(extra_ylabel_string), size=fontsize)
    ax.tick_params(labelsize=fontsize, length=7.5, width=2.5, which='major',\
        top=True, direction='inout')
    ax.tick_params(labelsize=fontsize, length=4.5, width=1.5, which='minor',\
        top=True, direction='inout')
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.95, hspace=0)
    if show:
        pl.show()
    else:
        return fig

