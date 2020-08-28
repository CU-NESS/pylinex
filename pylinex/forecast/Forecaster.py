"""
File: pylinex/forecaster/Forecaster.py
Author: Keith Tauscher
Date: 23 Jan 2018

Description: File containing a class which performs forecasts of
             reconstructions of quantities with the classes defined in the
             pylinex.fitter module (i.e. the Extractor class). It takes as
             inputs the noise level of the data, training sets for each
             component of the data, the numbers of terms to allow for each
             component of the data, the Quantity objects which should be
             tracked over all performed fits, and the Quantity object which
             should be minimized to determine the number of terms to use for
             each component of the data. As outputs, it can produce histograms
             of statistics, sample reconstructions of signals, and more.
"""
import os, h5py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl
from distpy import ChiSquaredDistribution
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value
from ..expander import NullExpander
from ..quantity import CompiledQuantity, FunctionQuantity
from ..fitter import Extractor
from ..hdf5 import ExtractionPlotter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class Forecaster(object):
    """
    Class which performs forecasts of reconstructions of quantities with the
    classes defined in the pylinex.fitter module (i.e. the Extractor class). It
    takes as inputs the noise level of the data, training sets for each
    component of the data, the numbers of terms to allow for each component of
    the data, the Quantity objects which should be tracked over all performed
    fits, and the Quantity object which should be minimized to determine the
    number of terms to use for each component of the data. As outputs, it can
    produce histograms of statistics, sample reconstructions of signals, and
    more.
    """
    def __init__(self, file_name, num_curves_to_create=None, error=None,\
        names=None, training_sets=None, input_curve_sets=None,\
        dimensions=None, compiled_quantity=CompiledQuantity('empty'),\
        quantity_to_minimize='bias_score', expanders=None,\
        num_curves_to_score=None, use_priors_in_fit=False,\
        prior_covariance_expansion_factor=1., prior_covariance_diagonal=False,\
        seed=None, target_subbasis_name=None, verbose=True):
        """
        Initializes a Forecaster.
        
        file_name: the string file location indicating where to place the hdf5
                   file with the results of this Forecaster
        num_curves_to_create: the number of data curves to simulate in the
                              forecast. This can only be None if file already
                              exists at file_name!
        error: the level of Gaussian noise in the data. This should only be
               None if file already exists at file_name!
        names: a list of strings containing the names of the different
               components of the data in a list of strings. Can only be None if
               file already exists at file_name!
        training_sets: list of 2D numpy.ndarrays of shape (ntrain, nchannel)
                       containing the sets of curves used to define the SVD
                       basis vectors. This can only be None if file already
                       exists at file_name!
        input_curve_sets: list of 2D numpy.ndarrays of shape
                          (ncurves, nchannel) containing the sets of curves
                          used to define the generated data curves. if None,
                          training_sets are used for this purpose
        dimensions: the dimensions of the grid in which to search for the
                    chosen solution. Should be a list of dictionaries of arrays
                    where each element of the list is a dimension and the
                    arrays in each dictionary must be of equal length. The
                    combined keys of the dictionaries should be equal to names.
                    Can only be None if file already exists at file_name!
        compiled_quantity: CompiledQuantity object containing the Quantity
                           objects which should be tracked over all data curves
                           and all SVD truncations
        quantity_to_minimize: string name of quantity (in compiled_quantity)
                              which should be minimized to determine which SVD
                              truncation to use
        expanders: list of Expander objects describing the expansion matrix of
                   each component of the data
        num_curves_to_score: if 0, bias_score is not run and not added to the
                                   given CompiledQuantity
                             if positive int, gives number of training set
                                              curves which are scored using the
                                              bias_score function
                             if None, all curves are scored using the
                                      bias_score function (this may be slow)
        use_priors_in_fit: boolean determining whether priors derived from the
                           training sets should be used in the fits
        prior_covariance_expansion_factor: factor by which prior covariance
                                           matrices should be expanded (if they
                                           are used), default 1
        prior_covariance_diagonal: boolean determining whether off-diagonal
                                   elements of the prior covariance are used or
                                   not, default False (meaning they are used).
                                   Setting this to true will weaken priors but
                                   should enhance numerical stability
        seed: seed for random number generator. if None, this exact calculation
              may be difficult (not impossible) to reproduce
        target_subbasis_name: the name of the subbasis for which statistics
                              should be kept (the signal bias statistic of
                              Tauscher et al. 2018).
        verbose: boolean determining the level of output printed by this object
        """
        self.file_name = file_name
        if os.path.exists(self.file_name):
            if verbose:
                print(("{!s} already exists. Will perform analysis using " +\
                    "existing file. If you wish to rerun, move or delete " +\
                    "existing file.").format(self.file_name))
        else:
            if type(input_curve_sets) is type(None):
                curve_sets = training_sets
            else:
                curve_sets = []
                for (index, input_curve_set) in enumerate(input_curve_sets):
                    if type(input_curve_set) is type(None):
                        curve_sets.append(training_sets[index])
                    else:
                        curve_sets.append(input_curve_set)
            (data, curve_set_indices) = self.make_data_curves(\
                num_curves_to_create, curve_sets, expanders, error, seed=seed)
            if type(target_subbasis_name) is not type(None):
                iname = names.index(target_subbasis_name)
                input_targets = curve_sets[iname][curve_set_indices[iname]]
                compiled_quantity = compiled_quantity +\
                    FunctionQuantity('subbasis_bias_statistic',\
                    name=target_subbasis_name, true_curve=input_targets)
            extractor = Extractor(data, error, names, training_sets,\
                dimensions, compiled_quantity=compiled_quantity,\
                quantity_to_minimize=quantity_to_minimize,\
                expanders=expanders, num_curves_to_score=num_curves_to_score,\
                use_priors_in_fit=use_priors_in_fit,\
                prior_covariance_expansion_factor=\
                prior_covariance_expansion_factor,\
                prior_covariance_diagonal=prior_covariance_diagonal,\
                verbose=verbose)
            hdf5_file = h5py.File(self.file_name, 'w')
            if type(target_subbasis_name) is not type(None):
                hdf5_file.attrs['target'] = target_subbasis_name
            group = hdf5_file.create_group('input_curves')
            for (name, curve_set, indices) in\
                zip(names, curve_sets, curve_set_indices):
                create_hdf5_dataset(group, name, data=curve_set[indices,:])
            extractor.fill_hdf5_group(hdf5_file, save_all_fitters=True,\
                save_training_sets=False, save_channel_estimates=False)
            hdf5_file.close()
    
    @property
    def target(self):
        """
        Property storing the name of the component of the data being focused on
        """
        if not hasattr(self, '_target'):
            try:
                self._target = self.plotter.file.attrs['target']
            except:
                try:
                    compiled_quantity = self.plotter.compiled_quantity
                    bias_statistic_quantity =\
                        compiled_quantity['subbasis_bias_statistic']
                    self._target =\
                        bias_statistic_quantity.function_kwargs['name']
                except:
                    raise ValueError("Couldn't ascertain target for some " +\
                        "reason.")
        return self._target
    
    @property
    def input_curves(self):
        """
        Property storing the curves used as input to test ability of inference.
        """
        if not hasattr(self, '_input_curves'):
            self._input_curves = {}
            group = self.plotter.file['input_curves']
            for name in group:
                self._input_curves[name] = get_hdf5_value(group[name])
        return self._input_curves
    
    def make_data_curves(self, ncurves, curve_sets, expanders, error,\
        seed=None):
        """
        Makes ncurves data curves to fit (with all components of the data and
        noise) for with this Forecaster by drawing random curves to combine
        from the given training sets, expanding them with the given Expander
        objects and adding noise.
        
        ncurves: the number of data curves to generate
        curve_sets: the sets of curves from which to draw random samples of
                    each component of the data
        expanders: the Expander objects used to expand the curves from each set
                   into the same space. can be None if all curve sets live in
                   the same space
        error: the magnitude of the Gaussian noise to add to the data
        seed: seed for the random number generator to allow for repeatability
        
        returns: (data_curves, curve_set_indices) where data_curves is a 2D
                 numpy.ndarray of shape (ncurves, nchannels) and
                 curve_set_indices is a list of 1D arrays each of length
                 ncurves corresponding to the indices in the given curve_sets
        """
        nsets = len(curve_sets)
        if type(expanders) is type(None):
            expanders = [NullExpander()] * nsets
        if type(seed) is not type(None):
            np.random.seed(seed)
        curve_set_indices = []
        for curve_set in curve_sets:
            curve_set_indices.append(\
                np.random.randint(0, len(curve_set), ncurves))
        data_curves =\
            np.random.normal(0, 1, (ncurves, len(error))) * error[np.newaxis,:]
        for iset in range(nsets):
            expander = expanders[iset]
            curve_set = curve_sets[iset]
            indices = curve_set_indices[iset]
            data_curves = data_curves + expander(curve_set[indices,:])
        return (data_curves, curve_set_indices)
    
    @property
    def file_name(self):
        """
        Property storing the name of the file where the results of this
        Forecaster will be saved.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the file name where the results of this Forecaster will be
        saved.
        
        value: must be a valid file location where an hdf5 file can be placed
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was not a string.")
    
    @property
    def plotter(self):
        """
        ExtractionPlotter object which can be used to access all of the results
        of the extractions performed as part of this forecaster.
        """
        if not hasattr(self, '_plotter'):
            self._plotter = ExtractionPlotter(self.file_name)
        return self._plotter
    
    def plot_subbasis_fits(self, indices, width, height, nsigma=[1, 2],\
        x_values=None, subtract_truth=False, scale_factor=1, verbose=True,\
        xlabel=None, ylabel=None, fontsize=24, figsize=(20,20), show=False):
        """
        Plots individual fits of the target subbasis on a grid.
        
        indices: the indices of the input curves to show fits for
        width: the number of plots horizontally across the figure
        height: the number of plots vertically across the figure
        nsigma: the numbers of sigma to plots in the figure
        x_values: the x values associated with the fit targets
        subtract_truth: if True, the true input targets are subtracted from the
                                 fits to show residuals
        scale_factor: number to multiply signals by (usually used for
                      conversion between units used in internals and units used
                      for fits)
        verbose: if True, more info is printed to the console
        xlabel: string label for x axis
        ylabel: string label for y axis
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        """
        fig = pl.figure(figsize=figsize)
        for (iindex, index) in enumerate(indices):
            ax = fig.add_subplot(width, height, iindex + 1)
            self.plotter.plot_subbasis_fit(icurve=index, nsigma=nsigma,\
                name=self.target, title='', xlabel=xlabel, ylabel=ylabel,\
                subtract_truth=subtract_truth, plot_truth=True,\
                verbose=verbose, ax=ax, true_curves=\
                {self.target: self.input_curves[self.target][index]},\
                scale_factor=scale_factor, channels=x_values, show=False,\
                fontsize=fontsize)
        if show:
            pl.show()
        else:
            return fig
    
    def plot_subbasis_bias_statistic_histogram(self, take_sqrt=True,\
        bins=1000, quantity_to_minimize=None, label=None, ax=None, color=None,\
        plot_reference_curve=True, fontsize=24, figsize=(12,9), show=False,\
        **kwargs):
        """
        Plots the subbasis bias statistic histogram for the target subbasis
        name given when this Forecaster was written.
        
        take_sqrt: if True, takes square root of subbasis bias statistic (as is
                            done in Tauscher et al. 2018)
        bins: bins argument to pass on to matplotlib.axes.Axes.hist function,
              default 1000
        ax: matplotlib.axes.Axes instance on which to plot this histogram
            if None, new figure is created
        color: color(s) of histogram(s)
        quantity_to_minimize: name of quantity to minimize to choose SVD
                              truncations. if None (default), the
                              quantity_to_minimize given when this forecaster
                              was written
        plot_reference_curve: if True, the chi(1) cdf is plotted as a dashed
                                       black line on the plot.
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        """
        self.close()
        if type(ax) is type(None):
            fig = pl.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        def plot_single_histogram(qtm, this_color=None, bins=None, label=None):
            statistics = self.plotter.statistics_by_minimized_quantity(\
                minimized_quantity=qtm,\
                grid_quantities=self.plotter.compiled_quantity.quantities)
            statistics = statistics['subbasis_bias_statistic']
            if take_sqrt:
                statistics = np.sqrt(statistics)
            (numbers, bins, patches) = ax.hist(statistics, bins=bins,\
                histtype='step', cumulative=True, density=True,\
                color=this_color, label=label, **kwargs)
            return bins
        if type(quantity_to_minimize) is type(None):
            bins = plot_single_histogram(\
                self.plotter.quantity_to_minimize, this_color=color,\
                bins=bins, label=label)
        elif isinstance(quantity_to_minimize, basestring):
            bins = plot_single_histogram(quantity_to_minimize,\
                this_color=color, bins=bins, label=label)
        elif type(quantity_to_minimize) in sequence_types:
            final_bins = []
            for (quantity, this_color) in zip(quantity_to_minimize, color):
                final_bins.append(plot_single_histogram(quantity,\
                    this_color=this_color, bins=bins,\
                    label='{!s}'.format(quantity)))
            bins = np.sort(np.concatenate(final_bins))
        if plot_reference_curve:
            x_values = np.concatenate([[0],\
                np.linspace(bins[0], bins[-1], 10 * len(bins))])
            if take_sqrt:
                y_values = stats.chi.cdf(x_values, 1)
            else:
                y_values = stats.chi2.cdf(x_values, 1)
            ax.plot(x_values, y_values, color='k', linestyle='--',\
                label='$\chi(1)$')
        ax.set_xlim((bins[0], bins[-1]))
        ax.set_xlabel('# of $\sigma$', size=fontsize)
        ax.set_ylabel('Confidence level [%]', size=fontsize)
        ax.set_yticks(np.linspace(0, 1, 6), minor=False)
        ax.set_yticklabels(['{:d}'.format(20 * integer)\
            for integer in range(6)], minor=False)
        ax.set_yticks(np.linspace(0, 1, 21), minor=True)
        ax.set_yticklabels([], minor=True)
        ax.set_title('Signal bias statistic histogram', size=fontsize)
        ax.tick_params(length=7.5, width=2.5, labelsize=fontsize,\
            which='major')
        ax.tick_params(length=4.5, width=1.5, labelsize=fontsize,\
            which='minor')
        ax.legend(loc='lower right', fontsize=fontsize)
        if show:
            pl.show()
        else:
            return ax
    
    def close(self):
        """
        Closes the hdf5 file which stores the data relevant to this Forecaster.
        """
        self.plotter.close()
        del self._plotter
    
    def __enter__(self):
        """
        Enters into a with statement using this object as a context manager.
        
        returns: this object
        """
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        """
        Exits a with statement where this Forecaster was used as a context
        manager.
        
        exception_type: type of Exception raised
        exception_value: Exception raised
        traceback: the traceback corresponding to the raised Exception
        """
        self.close()

