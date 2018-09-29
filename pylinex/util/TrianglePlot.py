"""
File: pylinex/util/TrianglePlot.py
Author: Keith Tauscher
Date: 18 Aug 2018

Description: File containing functions which plot univariate histograms,
             bivariate histograms, and triangle plots (which are really just
             combinations of the previous two types).
"""
from __future__ import division
import numpy as np
import scipy.linalg as scila
import matplotlib.pyplot as pl
from matplotlib.ticker import StrMethodFormatter
from distpy import real_numerical_types, sequence_types

def univariate_histogram(sample, reference_value=None, bins=None,\
    matplotlib_function='fill_between', show_intervals=False, xlabel='',\
    ylabel='', title='', fontsize=28, ax=None, show=False, **kwargs):
    """
    Plots a 1D histogram of the given sample.
    
    sample: the 1D sample of which to take a histogram
    reference_value: a point at which to plot a dashed reference line
    bins: bins to pass to numpy.histogram: default, None
    matplotlib_function: either 'fill_between', 'bar', or 'plot'
    show_intervals: if True, 95% confidence intervals are plotted
    xlabel: the string to use in labeling x axis
    ylabel: the string to use in labeling y axis
    title: title string with which to top plot
    fontsize: the size of the tick label font
    ax: if None, new Figure and Axes are created
        otherwise, this Axes object is plotted on
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    kwargs: keyword arguments to pass on to matplotlib.Axes.plot or
            matplotlib.Axes.fill_between
    
    returns: None if show is True, otherwise Axes instance with plot
    """
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    (nums, bins) = np.histogram(sample, bins=bins)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    num_bins = len(bin_centers)
    ylim = (0, 1.1 * np.max(nums))
    if 'color' in kwargs:
        color = kwargs['color']
        del kwargs['color']
    else:
        # 95% interval color
        color = 'C0'
    cumulative = np.cumsum(nums)
    cumulative = cumulative / cumulative[-1]
    cumulative_is_less_than_025 = np.argmax(cumulative > 0.025)
    cumulative_is_more_than_975 = np.argmax(cumulative > 0.975) + 1
    interval_95p =\
        (cumulative_is_less_than_025, cumulative_is_more_than_975 + 1)
    if matplotlib_function in ['bar', 'plot']:
        if matplotlib_function == 'bar':
            ax.bar(bin_centers, nums,\
                width=(bins[-1] - bins[0]) / num_bins, color=color, **kwargs)
        else:
            ax.plot(bin_centers, nums, color=color, **kwargs)
        if show_intervals:
            ax.plot([bins[interval_95p[0]]]*2, ylim, color='r', linestyle='--')
            ax.plot([bins[interval_95p[1]]]*2, ylim, color='r', linestyle='--')
    elif matplotlib_function == 'fill_between':
        if show_intervals:
            ax.plot(bin_centers, nums, color='k', linewidth=1)
            half_bins = np.linspace(bins[0], bins[-1], (2 * len(bins)) - 1)
            interpolated_nums = np.interp(half_bins, bin_centers, nums)
            ax.fill_between(\
                half_bins[2*interval_95p[0]:2*interval_95p[1]],\
                np.zeros((2 * (interval_95p[1] - interval_95p[0]),)),\
                interpolated_nums[2*interval_95p[0]:2*interval_95p[1]],\
                color=color)
            ax.fill_between(bin_centers, nums,\
                np.ones_like(nums) * 1.5 * np.max(nums), color='w')
        else:
            ax.fill_between(bin_centers, np.zeros_like(nums), nums,\
                color=color, **kwargs)
    else:
        raise ValueError("matplotlib_function not recognized.")
    ax.set_ylim(ylim)
    if reference_value is not None:
        ax.plot([reference_value] * 2, ylim, color='r', linewidth=1,\
            linestyle='--')
        ax.set_ylim(ylim)
    ax.set_xlim((bins[0], bins[-1]))
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel(ylabel, size=fontsize)
    ax.set_title(title, size=fontsize)
    ax.tick_params(width=2, length=6, labelsize=fontsize)
    if show:
        pl.show()
    else:
        return ax

def confidence_contour_2D(xsample, ysample, nums=None,\
    confidence_contours=0.95, hist_kwargs={}):
    """
    Finds the posterior distribution levels which represent the boundaries of
    confidence intervals of the given confidence level(s).
    
    xsample: xsample to contour
    ysample: ysample to contour
    nums: if histogram has already been created, nums can be passed here
    confidence_contours: confidence level as a number between 0 and 1 or a 1D
                         array of such numbers, default: 0.95
    hist_kwargs: only used if nums is None, contains keyword arguments to pass
                 to histogram2d function
    
    returns: 1D array of confidence contours
    """
    if nums is None:
        (nums, xedges, yedges) =\
            np.histogram2d(xsample, ysample, **hist_kwargs)
    nums = np.sort(nums.flatten())
    cdf_values = np.cumsum(nums)
    cdf_values = (cdf_values / cdf_values[-1])
    confidence_levels = 1 - cdf_values
    if type(confidence_contours) in real_numerical_types:
        confidence_contours = [confidence_contours]
    if type(confidence_contours) in sequence_types:
        confidence_contours = np.sort(confidence_contours)
        return np.where(np.all(confidence_levels[np.newaxis,:] <=\
            confidence_contours[:,np.newaxis], axis=-1), nums[0],\
            np.interp(confidence_contours, confidence_levels[-1::-1],\
            nums[-1::-1]))
    else:
        raise TypeError("confidence_contours was set to neither a single " +\
            "number or a 1D sequence of numbers.")

def bivariate_histogram(xsample, ysample, reference_value_mean=None,\
    reference_value_covariance=None, bins=None, matplotlib_function='imshow',\
    xlabel='', ylabel='', title='', fontsize=28, ax=None, show=False,\
    contour_confidence_levels=0.95, **kwargs):
    """
    Plots a 2D histogram of the given joint sample.
    
    xsample: the sample to use for the x coordinates
    ysample: the sample to use for the y coordinates
    reference_value_mean: points to plot a dashed reference line for axes
    reference_value_covariance: if not None, used (along with
                                reference_value_mean) to plot reference ellipse
    bins: bins to pass to numpy.histogram2d, default: None
    matplotlib_function: function to use in plotting. One of ['imshow',
                         'contour', 'contourf']. default: 'imshow'
    xlabel: the string to use in labeling x axis
    ylabel: the string to use in labeling y axis
    title: title with which to top plot
    fontsize: the size of the tick label font (and other fonts)
    ax: if None, new Figure and Axes are created
        otherwise, this Axes object is plotted on
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    contour_confidence_levels: the confidence level of the contour in the
                               bivariate histograms. Only used if
                               matplotlib_function is 'contour' or 'contourf'.
                               Can be single number or sequence of numbers
    kwargs: keyword arguments to pass on to matplotlib.Axes.imshow (any but
            'origin', 'extent', or 'aspect') or matplotlib.Axes.contour or
            matplotlib.Axes.contourf (any)
    
    returns: None if show is True, otherwise Axes instance with plot
    """
    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)
    (nums, xbins, ybins) = np.histogram2d(xsample, ysample, bins=bins)
    xlim = (xbins[0], xbins[-1])
    ylim = (ybins[0], ybins[-1])
    xbin_centers = (xbins[1:] + xbins[:-1]) / 2
    ybin_centers = (ybins[1:] + ybins[:-1]) / 2
    if matplotlib_function == 'imshow':
        ax.imshow(nums.T, origin='lower',\
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto',\
            **kwargs)
    else:
        pdf_max = np.max(nums)
        contour_levels = confidence_contour_2D(xsample, ysample, nums=nums,\
            confidence_contours=contour_confidence_levels)
        contour_levels = np.sort(contour_levels)
        if matplotlib_function == 'contour':
            ax.contour(xbin_centers, ybin_centers, nums.T, contour_levels,\
                **kwargs)
        elif matplotlib_function == 'contourf':
            contour_levels = np.concatenate([contour_levels, [pdf_max]])
            ax.contourf(xbin_centers, ybin_centers, nums.T, contour_levels,\
                **kwargs)
        else:
            raise ValueError("matplotlib_function not recognized.")
    if reference_value_mean is not None:
        if reference_value_mean[0] is not None:
            ax.plot([reference_value_mean[0]] * 2, ylim, color='r',\
                linewidth=1, linestyle='--')
        if reference_value_mean[1] is not None:
            ax.plot(xlim, [reference_value_mean[1]] * 2, color='r',\
                linewidth=1, linestyle='--')
        if (reference_value_mean[0] is not None) and\
            (reference_value_mean[1] is not None) and\
            (reference_value_covariance is not None):
            reference_value_mean = np.array(reference_value_mean)
            sqrt_covariance_matrix = scila.sqrtm(reference_value_covariance)
            angles = np.linspace(0, 2 * np.pi, num=1000, endpoint=False)
            circle_points = np.array([np.cos(angles), np.sin(angles)])
            ellipse_points = reference_value_mean[:,np.newaxis] +\
                np.dot(sqrt_covariance_matrix, circle_points)
            (ellipse_xs, ellipse_ys) = ellipse_points
            ax.fill(ellipse_xs, ellipse_ys, edgecolor='g', linewidth=1,\
                fill=(matplotlib_function=='contourf'), linestyle='--',\
                color='r')
    ax.tick_params(width=2, length=6, labelsize=fontsize)
    ax.set_xlabel(xlabel, size=fontsize)
    ax.set_ylabel(ylabel, size=fontsize)
    ax.set_title(title, size=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show:
        pl.show()
    else:
        return ax

def triangle_plot(samples, labels, figsize=(8, 8), show=False,\
    kwargs_1D={}, kwargs_2D={}, fontsize=28, nbins=100,\
    plot_type='contour', reference_value_mean=None,\
    reference_value_covariance=None, contour_confidence_levels=0.95):
    """
    Makes a triangle plot out of N samples corresponding to (possibly
    correlated) random variables
    
    samples: tuple of N 1D samples of the same length or an array of shape
             (N,m) where m is a single integer
    labels: the labels to use for each sample
    figsize: the size of the figure on which to put the triangle plot
    show: if True, matplotlib.pyplot.show is called before this function
                   returns
    kwargs_1D: keyword arguments to pass on to univariate_histogram function
    kwargs_2D: keyword arguments to pass on to bivariate_histogram function
    fontsize: the size of the label fonts
    nbins: the number of bins for each sample
    plot_type: 'contourf', 'contour', or 'histogram'
    reference_value_mean: reference values to place on plots, if there are any
    reference_value_covariance: if not None, used (along with
                                reference_value_mean) to plot reference
                                ellipses in each bivariate histogram
    contour_confidence_levels: the confidence level of the contour in the
                               bivariate histograms. Only used if plot_type is
                               'contour' or 'contourf'. Can be single number or
                               sequence of numbers
    """
    fig = pl.figure(figsize=figsize)
    samples = np.array(samples)
    num_samples = samples.shape[0]
    if plot_type == 'contour':
        matplotlib_function_1D = 'plot'
        matplotlib_function_2D = 'contour'
        full_kwargs_1D = {}
        full_kwargs_2D = {'cmap': 'Dark2', 'linewidth': 2}
    elif plot_type == 'contourf':
        matplotlib_function_1D = 'fill_between'
        matplotlib_function_2D = 'contourf'
        full_kwargs_1D = {}
        full_kwargs_2D = {'colors':\
            ['C{:d}'.format(index) for index in [0, 2, 4, 6, 1, 3, 5, 7]]}
    elif plot_type == 'histogram':
        matplotlib_function_1D = 'bar'
        matplotlib_function_2D = 'imshow'
        full_kwargs_1D = {}
        full_kwargs_2D = {}
    else:
        raise ValueError("plot_type not recognized.")
    full_kwargs_1D.update(kwargs_1D)
    full_kwargs_2D.update(kwargs_2D)
    ticks = []
    bins = []
    for (isample, sample) in enumerate(samples):
        min_to_include = np.min(sample)
        max_to_include = np.max(sample)
        if (reference_value_mean is not None) and\
            (reference_value_mean[isample] is not None):
            min_to_include =\
                min(min_to_include, reference_value_mean[isample])
            max_to_include =\
                max(max_to_include, reference_value_mean[isample])
        middle = (max_to_include + min_to_include) / 2
        width = max_to_include - min_to_include
        bins.append(np.linspace(min_to_include - (width / 10),\
            max_to_include + (width / 10), nbins + 1))
        ticks.append(np.linspace(middle - (width / 2.5),\
            middle + (width / 2.5), 3))
    tick_label_formatter = StrMethodFormatter('{x:.3g}')
    for (column, column_sample) in enumerate(samples):
        column_label = labels[column]
        if reference_value_mean is None:
            reference_value_x = None
        else:
            reference_value_x = reference_value_mean[column]
        for (row, row_sample) in enumerate(samples):
            if row < column:
                continue
            row_label = labels[row]
            plot_number = ((num_samples * row) + column + 1)
            ax = fig.add_subplot(num_samples, num_samples, plot_number)
            if row == column:
                univariate_histogram(column_sample,\
                    reference_value=reference_value_x,\
                    bins=bins[column],\
                    matplotlib_function=matplotlib_function_1D,\
                    show_intervals=False, xlabel='', ylabel='', title='',\
                    fontsize=fontsize, ax=ax, show=False, **full_kwargs_1D)
            else:
                if reference_value_mean is None:
                    reference_value_y = None
                else:
                    reference_value_y = reference_value_mean[row]
                reference_value_submean =\
                    (reference_value_x, reference_value_y)
                if reference_value_covariance is None:
                    reference_value_subcovariance = None
                else:
                    indices = np.array([column, row])
                    reference_value_subcovariance =\
                        reference_value_covariance[indices,:][:,indices]
                bivariate_histogram(column_sample, row_sample,\
                    reference_value_mean=reference_value_submean,\
                    reference_value_covariance=reference_value_subcovariance,\
                    bins=(bins[column], bins[row]),\
                    matplotlib_function=matplotlib_function_2D, xlabel='',\
                    ylabel='', title='', fontsize=fontsize, ax=ax,\
                    show=False,\
                    contour_confidence_levels=contour_confidence_levels,\
                    **full_kwargs_2D)
            ax.set_xticks(ticks[column])
            if row != column:
                ax.set_yticks(ticks[row])
            ax.xaxis.set_major_formatter(tick_label_formatter)
            ax.yaxis.set_major_formatter(tick_label_formatter)
            ax.tick_params(left='on', right='on', top='on', bottom='on',\
                labelleft='off', labelright='off', labeltop='off',\
                labelbottom='off', direction='inout')
            if (row == column):
                ax.tick_params(left='off', top='off', right='off')
            elif (row == (column + 1)):
                ax.tick_params(left='off')
                ax.tick_params(axis='y', direction='in')
                ax.tick_params(left='on')
            if (row + 1) == num_samples:
                ax.set_xlabel(column_label, size=fontsize, rotation=15)
                ax.tick_params(labelbottom='on')
            if column == 0:
                if row == 0:
                    ax.tick_params(labelleft='off')
                else:
                    ax.set_ylabel(row_label, size=fontsize, rotation=60,\
                        labelpad=30)
                    ax.tick_params(labelleft='on')
    fig.subplots_adjust(wspace=0, hspace=0)
    if show:
        pl.show()
    else:
        return fig

