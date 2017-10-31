"""
File: pylinex/hdf5/ExtractionPlotter.py
Author: Keith Tauscher
Date: 20 Sep 2017

Description: File containing class which, given the string name of an hdf5
             file, reads in results from an Extractor object defined in the
             pylinex.fitter module.
"""
import os
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm
from ..util import get_hdf5_value
from ..basis import load_basis_sum_from_hdf5_group
from ..fitter import Extractor
from ..expander import ExpanderSet, load_expander_set_from_hdf5_group
try:
    import h5py
except:
    have_h5py = False
else:
    have_h5py = True
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ExtractionPlotter(object):
    """
    Class which interfaces with a saved Extractor object to plot (or,
    sometimes) results such as quantity grids.
    """
    def __init__(self, file_name):
        """
        Initializes this plotter with the given hdf5 file.
        
        file_name: string name of extant Extractor-saved hdf5 file
        """
        self.file_name = file_name
    
    @property
    def file_name(self):
        """
        Property storing the string file name of the Extractor-saved hdf5 file
        with which this object interfaces.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name

    @file_name.setter
    def file_name(self, value):
        """
        Setter for the file_name.
        
        file_name: string name of extant Extractor-saved hdf5 file
        """
        if isinstance(value, basestring):
            if os.path.exists(value):
                self._file_name = value
            else:
                raise ValueError("No file located at {!s}".format(value))
        else:
            raise TypeError("file_name was set to a non-string.")
    
    @property
    def file(self):
        """
        Property storing the h5py-opened hdf5 file storing the desired
        extraction results.
        """
        if not hasattr(self, '_file'):
            self._file = h5py.File(self.file_name, 'r')
        return self._file
    
    @property
    def dimensions(self):
        """
        Property storing the dimension list which contains dictionaries with
        numbers of terms to use.
        """
        if not hasattr(self, '_dimensions'):
            group = self.file['meta_fitter/dimensions']
            idim = 0
            self._dimensions = []
            while 'dimension_{}'.format(idim) in group:
                subgroup = group['dimension_{}'.format(idim)]
                this_dimension = {}
                for key in subgroup:
                    this_dimension[key] = subgroup[key]
                self._dimensions.append(this_dimension)
                idim += 1
        return self._dimensions
    
    @property
    def dimension_lengths(self):
        """
        Property storing a list containing the number of elements in each
        dimension.
        """
        if not hasattr(self, '_dimension_lengths'):
            self._dimension_lengths = []
            for (idimension, dimension) in enumerate(self.dimensions):
                lengths = []
                for key in dimension:
                    lengths.append(len(dimension[key]))
                lengths = list(set(lengths))
                if len(lengths) == 1:
                    self._dimension_lengths.append(lengths[0])
                else:
                    raise ValueError(("dimension #{} has values with " +\
                        "different lengths.").format(idimension))
        return self._dimension_lengths
    
    def get_grid(self, name):
        """
        Gets the grid associated with the quantity of the given name.
        
        name: name of quantity which desired grid concerns
        """
        return get_hdf5_value(self.file['meta_fitter/grids/{!s}'.format(name)])
    
    def __getitem__(self, key):
        """
        Allows for square-bracket indexing. This returns the same thing as
        self.get_grid(key)
        
        key: name of quantity which desired grid concerns
        """
        return self.get_grid(key)
    
    def ticks_from_dimension_number(self, idimension):
        """
        Finds the ticks on a given dimension's axis.
        
        idimension: dimension of given axis
        
        returns: list of string tick labels
        """
        dimension = self.dimensions[0]
        mode_numbers = [dimension[key] for key in dimension]
        if all([np.allclose(mode_numbers[i], mode_numbers[0])\
            for i in range(len(mode_numbers))]):
            ticks = mode_numbers[0]
        else:
            ticks = np.arange(self.dimension_lengths[0])
        return ['{:d}'.format(tick) for tick in ticks]
    
    @property
    def xticks(self):
        """
        Property storing the ticks put on the x-axis when grids are plotted.
        """
        if not hasattr(self, '_xticks'):
            self._xticks = self.ticks_from_dimension_number(0)
        return self._xticks
    
    @property
    def yticks(self):
        """
        Property storing the ticks put on the y-axis when grids are plotted.
        """
        if not hasattr(self, '_yticks'):
            self._yticks = self.ticks_from_dimension_number(1)
        return self._yticks
    
    def label_from_dimension_number(self, idimension):
        """
        Function which gets grid dimension label from dimension index.
        
        idimension: integer satisfying 0<=idimension<len(self.dimensions)
        
        returns: string label to put on axis of specified dimension
        """
        dimension = self.dimensions[idimension]
        dimension_keys = [key for key in dimension.keys()]
        return '{!s} terms'.format('/'.join(dimension_keys))
    
    @property
    def xlabel(self):
        """
        When grids are plotted, this property stores the string label to put on
        the x-axis. It is taken from the names of the basis sets included in
        dimension #0.
        """
        if not hasattr(self, '_xlabel'):
            self._xlabel = self.label_from_dimension_number(0)
        return self._xlabel
    
    @property
    def ylabel(self):
        """
        When grids are plotted, this property stores the string label to put on
        the y-axis. It is taken from the names of the basis sets included in
        dimension #1.
        """
        if not hasattr(self, '_ylabel'):
            self._ylabel = self.label_from_dimension_number(1)
        return self._ylabel
    
    @property
    def channel_mean(self):
        """
        Property storing the channel mean of the fit to the data.
        """
        if not hasattr(self, '_channel_mean'):
            if self.multiple_data_curves:
                self._channel_mean = []
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/optimal_fitters/' +\
                        'data_curve_{}').format(icurve)]
                    if 'channel_mean' in fitter_group['posterior']:
                        this_mean_dataset_name = ('meta_fitter/' +\
                            'optimal_fitters/data_curve_{}/posterior/' +\
                            'channel_mean').format(icurve)
                        this_mean = get_hdf5_value(\
                            fitter_group['posterior/channel_mean'])[icurve]
                    else:
                        basis_sum = load_basis_sum_from_hdf5_group(\
                            fitter_group['basis_sum'])
                        parameter_mean = get_hdf5_value(\
                            fitter_group['posterior/parameter_mean'])
                        this_mean =\
                            np.dot(parameter_mean, basis_sum.basis)[icurve]
                    self._channel_mean.append(this_mean)
                self._channel_mean = np.array(self._channel_mean)
            elif 'channel_mean' in\
                self.file['meta_fitter/optimal_fitter/posterior']:
                self._channel_mean = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/channel_mean'])
            else:
                basis_sum = load_basis_sum_from_hdf5_group(\
                    self.file['meta_fitter/optimal_fitter/basis_sum'])
                parameter_mean = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/parameter_mean'])
                self._channel_mean = np.dot(parameter_mean, basis_sum.basis)
        return self._channel_mean
    
    @property
    def channel_means(self):
        """
        Property storing a dictionary of channel means indexed by name of data
        component under concern.
        """
        if not hasattr(self, '_channel_means'):
            if self.multiple_data_curves:
                self._channel_means = {name: [] for name in self.names}
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/' +\
                        'optimal_fitters/data_curve_{}').format(icurve)]
                    basis_sum = load_basis_sum_from_hdf5_group(\
                        fitter_group['basis_sum'])
                    for name in self.names:
                        if 'channel_mean' in\
                            fitter_group['posterior/{!s}'.format(name)]:
                            this_mean = get_hdf5_value( fitter_group[\
                                'posterior/{!s}/channel_mean'.format(name)])[\
                                icurve]
                        else:
                            parameter_mean = get_hdf5_value(fitter_group[\
                                'posterior/{!s}/parameter_mean'.format(name)])
                            this_mean = np.dot(parameter_mean[icurve],\
                                basis_sum[name].basis)
                        self._channel_means[name].append(this_mean)
                for name in self.names:
                    self._channel_means[name] =\
                        np.array(self._channel_means[name])
            else:
                self._channel_means = {}
                fitter_group = self.file['meta_fitter/optimal_fitter']
                basis_sum = load_basis_sum_from_hdf5_group(\
                    fitter_group['basis_sum'])
                for name in self.names:
                    if 'channel_mean' in\
                        fitter_group['posterior/{!s}'.format(name)]:
                        self._channel_means[name] = get_hdf5_value(\
                            fitter_group['posterior/{!s}/channel_mean'.format(\
                            name)])
                    else:
                        parameter_mean = get_hdf5_value(fitter_group[\
                            'posterior/{!s}/parameter_mean'.format(name)])
                        self._channel_means[name] =\
                            np.dot(parameter_mean, basis_sum.basis)
        return self._channel_means
    
    @property
    def channel_error(self):
        """
        Property storing the channel error of the full data fit.
        """
        if not hasattr(self, '_channel_error'):
            if self.multiple_data_curves:
                self._channel_error = []
                for icurve in range(self.num_data_curves):
                    error_name = ('meta_fitter/optimal_fitters/' +\
                        'data_curve_{}/posterior/channel_error').format(icurve)
                    self._channel_error.append(\
                        get_hdf5_value(self.file[error_name]))
                self._channel_error = np.array(self._channel_error)
            else:
                error_name =\
                    'meta_fitter/optimal_fitter/posterior/channel_error'
                self._channel_error = get_hdf5_value(self.file[error_name])
        return self._channel_error
    
    @property
    def channel_errors(self):
        """
        Property storing the channel errors of the fits to each component of
        the data.
        """
        if not hasattr(self, '_channel_errors'):
            if self.multiple_data_curves:
                self._channel_errors = {name: [] for name in self.names}
                for icurve in range(self.num_data_curves):
                    for name in self.names:
                        error_name = ('meta_fitter/optimal_fitters/' +\
                            'data_curve_{0}/posterior/{1!s}/' +\
                            'channel_error').format(icurve, name)
                        this_error = get_hdf5_value(self.file[error_name])
                        self._channel_errors[name].append(this_error)
                for name in self.names:
                    self._channel_errors[name] =\
                        np.array(self._channel_errors[name])
            else:
                self._channel_errors = {}
                for name in self.names:
                    error_name = ('meta_fitter/optimal_fitter/posterior/' +\
                        '{!s}/channel_error').format(name)
                    self._channel_errors[name] =\
                        get_hdf5_value(self.file[error_name])
        return self._channel_errors
    
    @property
    def channel_RMS(self):
        """
        Property storing the RMS of the full data fit.
        """
        if not hasattr(self, '_channel_RMS'):
            self._channel_RMS =\
                np.sqrt(np.mean(np.power(self.channel_error, 2), axis=-1))
        return self._channel_RMS
    
    @property
    def channel_RMSs(self):
        """
        Property storing the RMS of the fits to each component of the data in
        the fit.
        """
        if not hasattr(self, '_channel_RMSs'):
            self._channel_RMSs = {}
            for name in self.names:
                self._channel_RMSs[name] = np.sqrt(np.mean(np.power(\
                    self.channel_errors[name], 2), axis=-1))
        return self._channel_RMSs
    
    def get_true_curve(self, name, true_curves, icurve=None):
        """
        Infers a true curve from the given true curves and the data associated
        with this object.
        
        name: the name of the component of the data under consideration here
        true_curves: dictionary of "true" curves in the data. If the desired
                     curve is in this dictionary, it is returned. Otherwise,
                     this function attempts to infer the true value of a curve
                     from those given in this dictionary. If the amount of
                     curves in this dictionary is insufficient, a ValueError
                     will be raised
        icurve: if more than one data curve is in this ExtractorPlotter, the
                icurve argument describes which curve is under consideration.
                If only one data curve is contained herein, this argument can
                safely be ignored
        
        returns: 1D numpy.ndarray containing "true" value of desired curve
                 (with noise added if name not in true_curves)
        """
        if name is None:
            if self.multiple_data_curves:
                return self.data[icurve]
            else:
                return self.data
        elif name in true_curves:
            return true_curves[name]
        else:
            if self.multiple_data_curves:
                expander_set =\
                    self.expander_set.reset_data(self.data[icurve])
            else:
                expander_set = self.expander_set
            for true_curve_name in true_curves:
                expander_set = expander_set.marginalize(true_curve_name,\
                    true_curves[true_curve_name])
            try:
                (separated_curves, residual) = expander_set.separate()
            except RuntimeError:
                raise ValueError("Not enough true curves were given in " +\
                    "order to infer the desired one.")
            else:
                return separated_curves[name]

    def plot_subbasis_fit(self, icurve=0, quantity_to_minimize=None, nsigma=1,\
        name=None, true_curves={}, title='Subbasis fit', subtract_truth=False,\
        plot_truth=False, yscale='linear', error_to_plot='posterior',\
        verbose=True, ax=None, scale_factor=1, show=False):
        """
        Plots a subbasis fit from this Extractor.
        
        icurve: argument describing the index of the data curve under
                consideration. If only one data curve is stored in this
                ExtractionPlotter, this can be safely ignored
        nsigma: number of sigma levels of to which to plot error
        name: name of the subbasis of the data under consideration
        true_curves: dictionary whose keys are names of data components and
                     whose values are the "true" curves representing those
                     components
        title: title of the created plot
        subtract_truth: if True, the curve to plot has "true" curve subtracted
                                 from it
        plot_truth: if True and subtract_truth is False, plot "true" curve in
                                                         addition to estimate
        yscale: 'linear', 'log', or 'symlog'
        error_to_plot: if 'posterior', error plotted comes from posterior
                                       distribution (i.e. it is the "real"
                                       estimate)
                       if 'likelihood', 
        verbose: if True, useful reminders to the user are printed
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        """
        if ax is None:
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if subtract_truth or plot_truth:
            true_curve = self.get_true_curve(name, true_curves, icurve=icurve)
        error_to_plot = error_to_plot.lower() # allow capital letters
        if error_to_plot == 'likelihood':
            if verbose:
                print("Plotting likelihood error (i.e. noise level) " +\
                    "instead of posterior error at user's request. If this " +\
                    "isn't what you want, change the error_to_plot " +\
                    "argument of the plot_subbasis_fit* function you called.")
        elif error_to_plot != 'posterior':
            raise ValueError("error_to_plot wasn't recognized. It should " +\
                "be either 'posterior' or 'likelihood'.")
        fitter_group_name = 'meta_fitter/'
        if quantity_to_minimize is not None:
            fitter_group_name =\
                fitter_group_name + '{!s}_'.format(quantity_to_minimize)
        fitter_group_name = fitter_group_name + 'optimal_fitter'
        if self.multiple_data_curves:
            fitter_group_name =\
                fitter_group_name + 's/data_curve_{}'.format(icurve)
        basis_sum_group_name = '{!s}/basis_sum'.format(fitter_group_name)
        basis_sum =\
            load_basis_sum_from_hdf5_group(self.file[basis_sum_group_name])
        posterior_group_name = '{!s}/posterior'.format(fitter_group_name)
        if name is None:
            basis = basis_sum.basis
        else:
            basis = basis_sum[name].basis
            posterior_group_name = ('{0!s}/{1!s}').format(\
                posterior_group_name, name)
        if error_to_plot == 'likelihood':
            expander = self.expanders[self.names.index(name)]
            channel_error = expander.contract_error(self.error)
        else:
            channel_error = get_hdf5_value(self.file[('{!s}/' +\
                'channel_error').format(posterior_group_name)])
        parameter_mean = get_hdf5_value(self.file[('{!s}/' +\
            'parameter_mean').format(posterior_group_name)])
        if self.multiple_data_curves:
            parameter_mean = parameter_mean[icurve]
        channel_mean = np.dot(parameter_mean, basis)
        channel_mean = scale_factor * channel_mean
        channel_error = scale_factor * channel_error
        if plot_truth or subtract_truth:
            true_curve = scale_factor * true_curve
        channels = np.arange(len(channel_error))
        if subtract_truth:
            mean_to_plot = channel_mean - true_curve
        else:
            mean_to_plot = channel_mean
        ax.plot(channels, mean_to_plot, color='r', linewidth=1)
        if subtract_truth and (error_to_plot == 'likelihood'):
            ax.fill_between(channels, -nsigma * channel_error,\
                nsigma * channel_error, color='r', alpha=0.3)
        else:
            ax.fill_between(channels, mean_to_plot - (nsigma * channel_error),\
                mean_to_plot + (nsigma * channel_error), color='r', alpha=0.3)
        ax.plot(channels, np.zeros_like(channels), color='k', linewidth=1)
        if subtract_truth:
            ax.plot(channels, np.zeros_like(channels), color='k',\
                linewidth=1)
        elif plot_truth:
            ax.plot(channels, true_curve, color='k', linewidth=1)
        ax.set_yscale(yscale)
        ax.set_title(title, size='xx-large')
        if show:
            pl.show()
        
    
    def plot_subbasis_fit_grid(self, icurve=0, nsigma=1, name=None,\
        true_curves={}, title='Subbasis fit grid', subtract_truth=False,\
        plot_truth=False, low_indices=(0,0), high_indices=(-1,-1),\
        yscale='linear', show=False):
        """
        Plots a grid of subbasis fits.
        
        nsigma: number of sigmas determining the width of the band
        name: string name of desired data component
        true_curve: the true data to either plot or subtract
        title: string title of plot
        subtract_truth: if True, true_curve is subtracted (if name is None, no
                                 true_curve needs to be given, as the true
                                 curve in that case is the data itself)
        yscale: 'linear', 'log', or 'symlog'
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        """
        if subtract_truth or plot_truth:
            true_curve = self.get_true_curve(name, true_curves, icurve=icurve)
        low_indices = (low_indices[0] % self.dimension_lengths[0],\
            low_indices[1] % self.dimension_lengths[1])
        high_indices = (high_indices[0] % self.dimension_lengths[0],\
            high_indices[1] % self.dimension_lengths[1])
        fig = pl.figure()
        nrows = high_indices[1] - low_indices[1] + 1
        ncols = high_indices[0] - low_indices[0] + 1
        xtick_locations = [((i + 0.5) / ncols) for i in range(ncols)]
        ytick_locations = [((i + 0.5) / nrows) for i in range(nrows)][-1::-1]
        pl.xticks(xtick_locations,\
            self.xticks[low_indices[0]:high_indices[0]+1])
        pl.yticks(ytick_locations,\
            self.yticks[low_indices[1]:high_indices[1]+1])
        pl.tick_params(labelsize='xx-large', pad=40)
        pl.xlabel(self.xlabel, size='xx-large')
        pl.ylabel(self.ylabel, size='xx-large')
        pl.title(title, size='xx-large')
        naxes = nrows * ncols
        for iplot in range(naxes):
            irow = (iplot // ncols) + low_indices[1]
            icol = (iplot % ncols) + low_indices[0]
            ax = fig.add_subplot(nrows, ncols, iplot + 1)
            group_name =\
                'meta_fitter/fitters/{}_{}/posterior'.format(icol, irow)
            if name is not None:
                group_name = group_name + '/{!s}'.format(name)
            channel_mean = get_hdf5_value(\
                self.file['{!s}/channel_mean'.format(group_name)])
            if self.multiple_data_curves:
                channel_mean = channel_mean[icurve]
            channel_error = get_hdf5_value(\
                self.file['{!s}/channel_error'.format(group_name)])
            if subtract_truth:
                mean_to_plot = channel_mean - true_curve
            else:
                mean_to_plot = channel_mean
            channels = np.arange(mean_to_plot.shape[-1])
            ax.plot(channels, mean_to_plot, color='r', linewidth=2)
            ax.fill_between(channels, mean_to_plot - (nsigma * channel_error),\
                mean_to_plot + (nsigma * channel_error), color='r', alpha=0.3)
            ax.plot(channels, np.zeros_like(channels), color='k', linewidth=1)
            if subtract_truth:
                ax.plot(channels, np.zeros_like(channels), color='k',\
                    linewidth=2)
            elif plot_truth:
                ax.plot(channels, true_curve, color='k', linewidth=2)
            ax.set_yscale(yscale)
            #ax.set_xticks([])
            #ax.set_xticklabels([])
            #ax.set_yticks([])
            #ax.set_yticklabels([])
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        if show:
            pl.show()
    
    def plot_grid(self, name, icurve=None, log_from_min=False, show=False,\
        **kwargs):
        """
        Plots the saved grid of the given quantity.
        
        name: name of the quantity whose grid should be plotted
        log_from_min: boolean determining whether the log from the minimum grid
                      value should be shown
        show: boolean determining whether matplotlib.pyplot.show() is called
              before this function returns
        **kwargs: extra keyword arguments to pass to matplotlib.pyplot.imshow()
        """
        grid = self[name]
        if self.multiple_data_curves:
            if icurve is None:
                raise ValueError("Since the hdf5 file given to this " +\
                    "plotter contains many data curves, a value of icurve " +\
                    "must be provided.")
            else:
                grid = grid[...,icurve]
        if grid.ndim == 1:
            grid = grid[:,np.newaxis]
            oneD = True
        elif grid.ndim == 2:
            oneD = False
        else:
            raise ValueError("grid must be 1D or 2D to be plotted.")
        if log_from_min:
            grid_min = np.min(grid)
            pl.imshow((grid - grid_min).T, norm=LogNorm(), **kwargs)
        else:
            pl.imshow(grid.T, **kwargs)
        pl.colorbar()
        pl.xlim((-0.5, self.dimension_lengths[0] - 0.5))
        pl.xticks(np.arange(self.dimension_lengths[0]), self.xticks)
        pl.xlabel(self.xlabel)
        if not oneD:
            pl.ylim((-0.5, self.dimension_lengths[1] - 0.5))
            pl.yticks(np.arange(self.dimension_lengths[1]), self.yticks)
            pl.ylabel(self.ylabel)
        if log_from_min:
            pl.title('{0!s} grid (log from min={1:.3g})'.format(name,\
                grid_min))
        else:
            pl.title('{!s} grid'.format(name))
        if show:
            pl.show()
    
    @property
    def statistics(self):
        """
        Property storing a dictionary of statistics from the "optimal" fit.
        """
        if not hasattr(self, '_statistics'):
            self._statistics = self.statistics_by_minimized_quantity()
        return self._statistics
    
    def minimum_indices(self, quantity_name=None, icurve=None):
        if quantity_name is None:
            quantity_name = self.quantity_to_minimize
        grid = self.get_grid(quantity_name)
        if icurve is not None:
            grid = grid[...,icurve]
        return np.unravel_index(np.argmin(grid), grid.shape)
    
    def statistics_by_minimized_quantity(self, minimized_quantity=None,\
        grid_quantities=[]):
        """
        """
        if not hasattr(self, '_statistics_by_minimized_quantity'):
            self._statistics_by_minimized_quantity = {}
        if minimized_quantity not in self._statistics_by_minimized_quantity:
            no_saved_statistics_error = ValueError("No saved statistics " +\
                "for given minimized_quantity to minimize.")
            if self.multiple_data_curves:
                new_statistics = {}
                group_name = 'meta_fitter'
                if minimized_quantity is None:
                    group_name = '{!s}/optimal_fitters/data_curve_'.format(\
                        group_name)
                else:
                    group_name = ('{0!s}/{1!s}_optimal_fitters/' +\
                        'data_curve_').format(group_name, minimized_quantity)
                for icurve in range(self.num_data_curves):
                    try:
                        group =\
                            self.file['{0!s}{1}'.format(group_name, icurve)]
                    except KeyError:
                        raise no_saved_statistics_error
                    else:
                        for key in group.attrs:
                            this_attribute = group.attrs[key]
                            if isinstance(this_attribute, np.ndarray):
                                this_attribute = this_attribute[icurve]
                            if key in new_statistics:
                                new_statistics[key].append(this_attribute)
                            else:
                                new_statistics[key] = [this_attribute]
                    for quantity_name in grid_quantities:
                        if quantity_name not in group.attrs:
                            minimum_indices = self.minimum_indices(\
                                minimized_quantity, icurve)
                            this_attribute = get_hdf5_value(\
                                self.file['meta_fitter/grids/{!s}'.format(\
                                quantity_name)])
                            this_attribute =\
                                this_attribute[minimum_indices][icurve]
                            if quantity_name in new_statistics:
                                new_statistics[quantity_name].append(\
                                    this_attribute)
                            else:
                                new_statistics[quantity_name] =\
                                    [this_attribute]
                for key in new_statistics:
                    new_statistics[key] = np.array(new_statistics[key])
                self._statistics_by_minimized_quantity[minimized_quantity] =\
                    new_statistics
            else:
                 group_name = 'meta_fitter'
                 if minimized_quantity is None:
                     group_name = '{!s}/optimal_fitter'.format(group_name)
                 else:
                     group_name = '{0!s}/{1!s}_optimal_fitter'.format(\
                         group_name, minimized_quantity)
                 try:
                     group = self.file[group_name]
                 except KeyError:
                     raise no_saved_statistics_error
                 else:
                     new_statistics =\
                         {key: group.attrs[key] for key in group.attrs}
                 for quantity_name in grid_quantities:
                     if quantity_name not in new_statistics:
                         indices = self.minimum_indices(minimized_quantity)
                         new_statistics[quantity_name] = get_hdf5_value(\
                             self.file['meta_fitter/grids/{!s}'.format(\
                             quantity_name)])[indices]
                 self._statistics_by_minimized_quantity[\
                     minimized_quantity] = new_statistics
        return self._statistics_by_minimized_quantity[minimized_quantity]
    
    @property
    def multiple_data_curves(self):
        """
        Property storing whether the Extractor which made the file at the heart
        of this ExtractionPlotter contained multiple data curves or just one.
        """
        if not hasattr(self, '_multiple_data_curves'):
            self._multiple_data_curves =\
                ('meta_fitter/optimal_fitters' in self.file)
        return self._multiple_data_curves
    
    @property
    def num_data_curves(self):
        """
        """
        if not hasattr(self, '_num_data_curves'):
            if self.multiple_data_curves:
                self._num_data_curves = 0
                while 'meta_fitter/optimal_fitters/data_curve_{}'.format(\
                    self._num_data_curves) in self.file:
                    self._num_data_curves += 1
            else:
                self._num_data_curves = 1
        return self._num_data_curves
    
    @property
    def data(self):
        """
        Property storing the data of the Extractor which saved trhe hdf5 file
        at the heart of this ExtractionPlotter.
        """
        if not hasattr(self, '_data'):
            self._data = get_hdf5_value(self.file['data'])
        return self._data
    
    @property
    def error(self):
        """
        Property storing the error associated with the data of the Extractor
        which saved trhe hdf5 file at the heart of this ExtractionPlotter.
        """
        if not hasattr(self, '_error'):
            self._error = get_hdf5_value(self.file['error'])
        return self._error
    
    @property
    def quantity_to_minimize(self):
        """
        Property storing string name of Quantity which was minimized to define
        optimal fitter.
        """
        if not hasattr(self, '_quantity_to_minimize'):
            self._quantity_to_minimize =\
                self.file['meta_fitter'].attrs['quantity_to_minimize']
        return self._quantity_to_minimize
    
    @property
    def names(self):
        """
        Property storing the list of names of different basis types in the
        data.
        """
        if not hasattr(self, '_names'):
            self._names = [key for key in self.file['names'].attrs]
        return self._names
    
    @property
    def expanders(self):
        """
        Property storing the list of Expander objects associated with the
        different components of the data (in the same order as names).
        """
        if not hasattr(self, '_expanders'):
            self._expanders = [self.expander_set[name] for name in self.names]
        return self._expanders
    
    @property
    def expander_set(self):
        """
        Property yielding an ExpanderSet object organizing the expanders here
        so that "true" curves for (e.g.) systematics can be found by using the
        data as well as a "true" curve for the signal.
        """
        if not hasattr(self, '_expander_set'):
            self._expander_set =\
                load_expander_set_from_hdf5_group(self.file['expanders'])
        return self._expander_set
    
    @property
    def training_sets(self):
        """
        Property storing the list of training sets associated with the
        different components of the data (in same order as names and
        expanders).
        """
        if not hasattr(self, '_training_sets'):
            if 'training_sets' in self.file:
                self._training_sets = []
                for name in self.names:
                    self._training_sets.append(get_hdf5_value(\
                        self.file['training_sets/{!s}'.format(name)]))
            else:
                raise ValueError("training_sets cannot be retrieved from " +\
                    "hdf5 file because save_training_sets flag was set to " +\
                    "False in original Extractor.")
        return self._training_sets
    
    @property
    def compiled_quantity(self):
        """
        Property storing the CompiledQuantity object used at each grid of the
        Extractor which made the file at the heart of this ExtractionPlotter.
        """
        if not hasattr(self, '_compiled_quantity'):
            group = self.file['meta_fitter/compiled_quantity']
            self._compiled_quantity =\
                load_quantity_from_hdf5_group(group)
        return self._compiled_quantity
    
    @property
    def extractor(self):
        """
        Property storing a (newly created) copy of the extractor being saved.
        This copy will have its save_* properties set to False as it shouldn't
        be saved (it already is saved!).
        """
        if not hasattr(self, '_extractor'):
            self._extractor = Extractor(self.data, self.error, self.names,\
                self.training_sets, self.dimensions,\
                compiled_quantity=self.compiled_quantity,\
                quantity_to_minimize=self.quantity_to_minimize,\
                expanders=self.expanders, save_training_sets=False,\
                save_all_fitters=False, num_curves_to_score=0, verbose=True)
        return self._extractor
    
    @property
    def meta_fitter(self):
        """
        Property storing the MetaFitter object at the heart of the (newly
        created) Extractor object recreated through the hdf5 file it saved.
        """
        if not hasattr(self, '_meta_fitter'):
            self._meta_fitter = self.extractor.meta_fitter
        return self._meta_fitter
    
    @property
    def fitter(self):
        """
        Property storing the "optimal" fitter recreated from the (newly
        created) Extractor object at the heart of this object.
        """
        if not hasattr(self, '_fitter'):
            self._fitter = self.extractor.fitter
        return self._fitter
    
    def close(self):
        """
        Closes the hdf5 file containing extraction information with which this
        plotter is an interface.
        """
        self.file.close()
        del self._file
    

