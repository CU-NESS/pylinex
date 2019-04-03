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
from distpy import GaussianDistribution, ChiSquaredDistribution
from ..util import get_hdf5_value, sequence_types
from ..quantity import load_quantity_from_hdf5_group
from ..expander import ExpanderSet
from ..basis import BasisSum
from ..fitter import Extractor
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
    def training_set_ranks(self):
        """
        Property storing a dictionary of the effective ranks of the training
        sets (indexed by name) used by the Extractor which generated the hdf5
        file at the heart of this object.
        """
        if not hasattr(self, '_training_set_ranks'):
            self._training_set_ranks =\
                {name: self.file['ranks'].attrs[name] for name in self.names}
        return self._training_set_ranks
    
    @property
    def training_set_rank_indices(self):
        """
        Property storing a dictionary with tuples containing the dimension and
        rank index as values indexed by the associated subbasis name.
        """
        if not hasattr(self, '_training_set_rank_indices'):
            self._training_set_rank_indices = {}
            for name in self.names:
                rank = self.training_set_ranks[name]
                for (idimension, dimension) in enumerate(self.dimensions):
                    if name in dimension:
                        if rank in dimension[name]:
                            rank_index =\
                                np.where(dimension[name] == rank)[0][0]
                        else:
                            print(("rank of {0!s} ({1:d}) not in its grid " +\
                                "dimension. Are you sure you're using " +\
                                "enough terms?").format(name, rank))
                            rank_index = None
                        self._training_set_rank_indices[name] =\
                            (idimension, rank_index)
                        break
                    else:
                        pass
        return self._training_set_rank_indices
    
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
                    this_dimension[key] = subgroup[key][()]
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
              if None, the quantity_to_minimize is used
        
        returns: the numpy.ndarray grid associated with the given name
        """
        if type(name) is type(None):
            return get_hdf5_value(self.file[\
                'meta_fitter/grids/{!s}'.format(self.quantity_to_minimize)])
        else:
            return get_hdf5_value(\
                self.file['meta_fitter/grids/{!s}'.format(name)])
    
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
        dimension = self.dimensions[idimension]
        mode_numbers = [dimension[key] for key in dimension]
        if all([np.allclose(mode_numbers[i], mode_numbers[0])\
            for i in range(len(mode_numbers))]):
            ticks = mode_numbers[0]
        else:
            ticks = np.arange(self.dimension_lengths[idimension])
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
    def basis_sum(self):
        """
        Property storing the basis_sum which is used to fit the single data
        curve at the heart of this object. If there is more than one data
        curve, this property will throw an error. In that case, you should use
        the get_basis_sum function instead.
        """
        if not hasattr(self, '_basis_sum'):
            if self.multiple_data_curves:
                raise AttributeError("The basis_sum property does not make " +\
                    "sense when there are multiple data curves. Use the " +\
                    "get_basis_sum function of the ExtractionPlotter class " +\
                    "instead.")
            else:
                self._basis_sum = self.get_basis_sum()
        return self._basis_sum
    
    def get_basis_sum(self, icurve=None):
        """
        Gets the basis_sum deteremined optimal for the given data curve.
        
        icurve: index of data curve for which to find the basis_sum (only
                necessary if multiple data curves are given)
        
        returns: basis_sum object determined optimal to fit the given data
        """
        group_name = 'meta_fitter/optimal_fitter'
        if self.multiple_data_curves:
            if type(icurve) is type(None):
                raise ValueError("Since multiple data curves are included " +\
                    "in this Extractor, an index of the data curve for " +\
                    "which the basis_sum is desired is required to be " +\
                    "passed to get_basis_sum.")
            else:
                group_name = '{0!s}s/data_curve_{1}'.format(group_name, icurve)
        group_name = '{!s}/basis_sum'.format(group_name)
        return BasisSum.load_from_hdf5_group(self.file[group_name])
    
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
                        basis_sum = BasisSum.load_from_hdf5_group(\
                            fitter_group['basis_sum'])
                        parameter_mean = get_hdf5_value(\
                            fitter_group['posterior/parameter_mean'])[icurve]
                        this_mean =\
                            np.dot(parameter_mean, basis_sum.basis)
                    self._channel_mean.append(this_mean)
                self._channel_mean = np.array(self._channel_mean)
            elif 'channel_mean' in\
                self.file['meta_fitter/optimal_fitter/posterior']:
                self._channel_mean = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/channel_mean'])
            else:
                basis_sum = BasisSum.load_from_hdf5_group(\
                    self.file['meta_fitter/optimal_fitter/basis_sum'])
                parameter_mean = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/parameter_mean'])
                self._channel_mean = np.dot(parameter_mean, basis_sum.basis)
        return self._channel_mean
    
    @property
    def parameter_mean(self):
        """
        Property storing the parameter mean of the fit to the data.
        """
        if not hasattr(self, '_parameter_mean'):
            if self.multiple_data_curves:
                self._parameter_mean = []
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/optimal_fitters/' +\
                        'data_curve_{}').format(icurve)]
                    parameter_mean = get_hdf5_value(\
                        fitter_group['posterior/parameter_mean'])
                    self._parameter_mean.append(parameter_mean)
                self._parameter_mean = np.array(self._parameter_mean)
            else:
                self._parameter_mean = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/parameter_mean'])
        return self._parameter_mean
    
    @property
    def parameter_covariance(self):
        """
        Property storing the parameter covariance of the fit to the data.
        """
        if not hasattr(self, '_parameter_covariance'):
            if self.multiple_data_curves:
                self._parameter_covariance = []
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/optimal_fitters/' +\
                        'data_curve_{}').format(icurve)]
                    parameter_covariance = get_hdf5_value(\
                        fitter_group['posterior/parameter_covariance'])
                    self._parameter_covariance.append(parameter_covariance)
                self._parameter_covariance =\
                    np.array(self._parameter_covariance)
            else:
                self._parameter_covariance = get_hdf5_value(self.file[\
                    'meta_fitter/optimal_fitter/posterior/' +\
                    'parameter_covariance'])
        return self._parameter_covariance
    
    @property
    def parameter_distribution(self):
        """
        Property storing a GaussianDistribution object containing the
        parameter_mean property as its mean and parameter_covariance property
        as its covariance.
        """
        if not hasattr(self, '_parameter_distribution'):
            self._parameter_distribution = GaussianDistribution(\
                self.parameter_mean, self.parameter_covariance)
        return self._parameter_distribution
    
    def get_fit(self, grid_indices):
        """
        Gets the fit associated with the given grid indices.
        
        grid_indices: list of indices into the grid for which to return the fit
                      (indices may be negative)
        
        returns: (basis_sum, parameter_mean, parameter_covariance)
        """
        grid_indices = [index % length\
            for (index, length) in zip(grid_indices, self.dimension_lengths)]
        grid_string =\
            '_'.join(['{}'.format(grid_index) for grid_index in grid_indices])
        fitter_group = self.file['meta_fitter/fitters/{}'.format(grid_string)]
        basis_sum = BasisSum.load_from_hdf5_group(fitter_group['basis_sum'])
        parameter_mean =\
            get_hdf5_value(fitter_group['posterior/parameter_mean'])
        parameter_covariance =\
            get_hdf5_value(fitter_group['posterior/parameter_covariance'])
        return (basis_sum, parameter_mean, parameter_covariance)
    
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
                    basis_sum = BasisSum.load_from_hdf5_group(\
                        fitter_group['basis_sum'])
                    for name in self.names:
                        if 'channel_mean' in\
                            fitter_group['posterior/{!s}'.format(name)]:
                            this_mean = get_hdf5_value(fitter_group[\
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
                basis_sum = BasisSum.load_from_hdf5_group(\
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
                            np.dot(parameter_mean, basis_sum[name].basis)
        return self._channel_means
    
    @property
    def parameter_means(self):
        """
        Property storing a dictionary of parameter means indexed by name of
        data component under concern.
        """
        if not hasattr(self, '_parameter_means'):
            if self.multiple_data_curves:
                self._parameter_means = {name: [] for name in self.names}
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/' +\
                        'optimal_fitters/data_curve_{}').format(icurve)]
                    for name in self.names:
                        this_mean = get_hdf5_value(fitter_group[\
                            'posterior/{!s}/parameter_mean'.format(name)])
                        self._parameter_means[name].append(this_mean)
                for name in self.names:
                    self._parameter_means[name] =\
                        np.array(self._parameter_means[name])
            else:
                self._parameter_means = {}
                fitter_group = self.file['meta_fitter/optimal_fitter']
                for name in self.names:
                    self._parameter_means[name] = get_hdf5_value(fitter_group[\
                        'posterior/{!s}/parameter_mean'.format(name)])
        return self._parameter_means
    
    @property
    def parameter_covariances(self):
        """
        Property storing a dictionary of parameter covariance indexed by name
        of data component under concern.
        """
        if not hasattr(self, '_parameter_covariances'):
            if self.multiple_data_curves:
                self._parameter_covariances = {name: [] for name in self.names}
                for icurve in range(self.num_data_curves):
                    fitter_group = self.file[('meta_fitter/' +\
                        'optimal_fitters/data_curve_{}').format(icurve)]
                    for name in self.names:
                        this_covariance = get_hdf5_value(fitter_group[\
                            'posterior/{!s}/parameter_covariance'.format(\
                            name)])
                        self._parameter_covariances[name].append(\
                            this_covariance)
                for name in self.names:
                    self._parameter_covariances[name] =\
                        np.array(self._parameter_covariances[name])
            else:
                self._parameter_covariances = {}
                fitter_group = self.file['meta_fitter/optimal_fitter']
                for name in self.names:
                    self._parameter_covariances[name] = get_hdf5_value(\
                        fitter_group[('posterior/{!s}/' +\
                        'parameter_covariance').format(name)])
        return self._parameter_covariances
    
    def get_parameter_indices(self, names):
        """
        Gets the indices of parameters describing the given subbases.
        
        names: list of names of subbases of the optimal BasisSum
        
        returns: 1D numpy.ndarray of indices of parameters
        """
        num_basis_vectors = self.basis_sum.num_basis_vectors
        if isinstance(names, basestring):
            names = [names]
        slices = [self.basis_sum.slices_by_name[name] for name in names]
        parameter_indices =\
            [range(*slc.indices(num_basis_vectors)) for slc in slices]
        return np.concatenate(parameter_indices)
    
    def marginalized_parameter_mean(self, names):
        """
        Marginalizes the parameter mean so as to only include the given names
        (in the given order).
        
        names: sequence of subbasis names to include in the output
        
        returns: 1D numpy.ndarray of parameter means whose length is determined
                 by the number of parameters which describe the given names
        """
        parameter_indices = self.get_parameter_indices(names)
        return self.parameter_mean[...,parameter_indices]
    
    def marginalized_parameter_covariance(self, names):
        """
        Marginalizes the parameter covariance so as to only include the given
        names (in the given order).
        
        names: sequence of subbasis names to include in the output
        
        returns: square 2D numpy.ndarray whose shape is determined the number
                 of parameters which describe the given names
        """
        parameter_indices = self.get_parameter_indices(names)
        return self.parameter_covariance[...,parameter_indices,:]\
            [...,:,parameter_indices]
    
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
        if type(name) is type(None):
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
    
    def plot_residual(self, icurve=0, title=None, xlabel=None, ylabel=None,\
        ax=None, scale_factor=1, channels=None, fontsize=24, label=None,\
        show=False, **plot_kwargs):
        """
        Plots the residual of the full model fit to the data (with the chosen
        minimized quantity).
        
        icurve: Index of residual to plot. Only necessary/used when multiple
                data curves are present
        title: title to put on plot
        xlabel: label to put on x-axis
        ylabel: label to put on y-axis
        ax: Axes instance on which to make this plot
        scale_factor: factor by which to multiply residual before plotting
        channels: x-values to use for the plot. If None is given, then
                  np.arange(num_channels) is used as the x-values
        fontsize: size of fonts for tick labels, axis labels, title, and legend
        label: if None, no legend label is applied. Otherwise label is taken to
               be a string which can be formatted as string.format(rms=rms)
               where rms is the rms residual multiplied by the scale factor
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        plot_kwargs: extra keyword arguments to matplotlib.pyplot.plot
        
        returns: None if show is True, Axes instance containing plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        residual = self.data - self.channel_mean
        if self.multiple_data_curves:
            if type(icurve) in int_types:
                residual = residual[icurve]
            else:
                raise TypeError("Since there are multiple data curves " +\
                    "stored in this ExtractionPlotter, icurve must be set " +\
                    "to an integer.")
        if type(channels) is type(None):
            channels = np.arange(len(residual))
        if type(label) is not type(None):
            rms = np.sqrt(np.mean(np.power(residual, 2))) * scale_factor
            label = label.format(rms=rms)
        ax.plot(channels, residual * scale_factor, label=label, **plot_kwargs)
        if type(xlabel) is not type(None):
            ax.set_xlabel(xlabel, size=fontsize)
        if type(ylabel) is not type(None):
            ax.set_ylabel(ylabel, size=fontsize)
        if type(title) is not type(None):
            ax.set_title(title, size=fontsize)
        if type(label) is not type(None):
            ax.legend(fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(width=1.5, length=4.5, which='minor')
        if show:
            pl.show()
        else:
            return ax

    def plot_subbasis_fit(self, icurve=0, quantity_to_minimize=None, nsigma=1,\
        name=None, true_curves={}, title=None, xlabel=None, ylabel=None,\
        subtract_truth=False, plot_truth=False, yscale='linear',\
        error_to_plot='posterior', plot_zero=False, verbose=True, ax=None,\
        scale_factor=1, channels=None, fontsize=24, color='r', figsize=(12,9),\
        show=False):
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
        xlabel: label of x axis
        ylabel: label of y axis
        subtract_truth: if True, the curve to plot has "true" curve subtracted
                                 from it
        plot_truth: if True and subtract_truth is False, plot "true" curve in
                                                         addition to estimate
        yscale: 'linear', 'log', or 'symlog'
        error_to_plot: if 'posterior', error plotted comes from posterior
                                       distribution (i.e. it is the "real"
                                       estimate)
                       if 'likelihood', 
        plot_zero: if True (default: False), zero line is plotted
        verbose: if True, useful reminders to the user are printed
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, Axes instance containing plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure(figsize=figsize)
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
        if type(quantity_to_minimize) is not type(None):
            fitter_group_name =\
                fitter_group_name + '{!s}_'.format(quantity_to_minimize)
        fitter_group_name = fitter_group_name + 'optimal_fitter'
        if self.multiple_data_curves:
            fitter_group_name =\
                fitter_group_name + 's/data_curve_{}'.format(icurve)
        basis_sum_group_name = '{!s}/basis_sum'.format(fitter_group_name)
        basis_sum =\
            BasisSum.load_from_hdf5_group(self.file[basis_sum_group_name])
        posterior_group_name = '{!s}/posterior'.format(fitter_group_name)
        if type(name) is type(None):
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
        if type(channels) is type(None):
            channels = np.arange(len(channel_error))
        if subtract_truth:
            mean_to_plot = channel_mean - true_curve
        else:
            mean_to_plot = channel_mean
        ax.plot(channels, mean_to_plot, color=color, linewidth=1)
        if type(nsigma) not in [list, tuple]:
            nsigma = [nsigma, None]
        if subtract_truth and (error_to_plot == 'likelihood'):
            ax.fill_between(channels, -nsigma[0] * channel_error,\
                nsigma[0] * channel_error, color=color, alpha=0.5)
            if type(nsigma[1]) is not type(None):
                ax.fill_between(channels, -nsigma[1] * channel_error,\
                    nsigma[1] * channel_error, color=color, alpha=0.3)
        else:
            ax.fill_between(channels,\
                mean_to_plot - (nsigma[0] * channel_error),\
                mean_to_plot + (nsigma[0] * channel_error), color=color,\
                alpha=0.5)
            if type(nsigma[1]) is not type(None):
                ax.fill_between(channels,\
                    mean_to_plot - (nsigma[1] * channel_error),\
                    mean_to_plot + (nsigma[1] * channel_error), color=color,\
                    alpha=0.3)
        if subtract_truth or plot_zero:
            ax.plot(channels, np.zeros_like(channels), color='k', linewidth=1)
        if plot_truth and (not subtract_truth):
            ax.plot(channels, true_curve, color='k', linewidth=1)
        ax.set_yscale(yscale)
        if type(title) is not type(None):
            ax.set_title(title, size=fontsize)
        if type(xlabel) is not type(None):
            ax.set_xlabel(xlabel, size=fontsize)
        if type(ylabel) is not type(None):
            ax.set_ylabel(ylabel, size=fontsize)
        ax.set_xlim((channels[0], channels[-1]))
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5)
        if show:
            pl.show()
        else:
            return ax
    
    def plot_subbasis_fit_grid(self, icurve=0, nsigma=1, name=None,\
        true_curves={}, title='Subbasis fit grid', subtract_truth=False,\
        plot_truth=False, low_indices=(0,0), high_indices=(-1,-1),\
        yscale='linear', color='r', show=False):
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
            if type(name) is not type(None):
                group_name = group_name + '/{!s}'.format(name)
            if type(name) is type(None):
                channel_mean = self.channel_mean
            else:
                channel_mean = self.channel_means[name]
            if self.multiple_data_curves:
                channel_mean = channel_mean[icurve]
            channel_error = get_hdf5_value(\
                self.file['{!s}/channel_error'.format(group_name)])
            if subtract_truth:
                mean_to_plot = channel_mean - true_curve
            else:
                mean_to_plot = channel_mean
            channels = np.arange(mean_to_plot.shape[-1])
            ax.plot(channels, mean_to_plot, color=color, linewidth=2)
            ax.fill_between(channels, mean_to_plot - (nsigma * channel_error),\
                mean_to_plot + (nsigma * channel_error), color=color,\
                alpha=0.3)
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
            if type(icurve) is type(None):
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
        xlim = (-0.5, self.dimension_lengths[0] - 0.5)
        pl.xlim(xlim)
        pl.xticks(np.arange(self.dimension_lengths[0]), self.xticks)
        pl.xlabel(self.xlabel)
        if not oneD:
            ylim = (-0.5, self.dimension_lengths[1] - 0.5)
            pl.ylim(ylim)
            pl.yticks(np.arange(self.dimension_lengths[1]), self.yticks)
            pl.ylabel(self.ylabel)
            try:
                rank_indices = []
                for dimension in self.dimensions:
                    for basis_name in dimension:
                        rank_indices.append(\
                            self.training_set_rank_indices[basis_name][1])
                        break
                (xrank_index, yrank_index) = rank_indices
                if type(xrank_index) is not type(None):
                    pl.plot([xrank_index - 0.5] * 2, ylim, color='r',\
                        linestyle='--')
                if type(yrank_index) is not type(None):
                    pl.plot(xlim, [yrank_index - 0.5] * 2, color='r',\
                        linestyle='--')
            except:
                pass # this says "don't worry about it if you can't plot ranks"
        if log_from_min:
            pl.title('{0!s} grid (log from min={1:.3g})'.format(name,\
                grid_min))
        else:
            pl.title('{!s} grid'.format(name))
        if show:
            pl.show()
    
    def plot_parameter_number_grid(self, quantity_to_minimize=None,
        cmap='binary', vmin=None, vmax=None, rank=True, ax=None,\
        show=False):
        """
        Plots a histogram of the number of parameters used in each dimension.
        
        quantity_to_minimize: if None, the default quantity_to_minimize is used
                              otherwise, this should be the string name of a
                                         quantity whose minimization can be
                                         used to decide on a basis truncation
        cmap: the name of the colormap to use for the histogram
        vmin, vmax: range of data to be plotted by hist2d
        rank: boolean which determines whether rank should be plotted
        ax: matplotlib.Axes object on which to plot the histogram. if None, a
            new figure is created. 
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        minimization_grid = self[quantity_to_minimize]
        if not self.multiple_data_curves:
            minimization_grid = minimization_grid[...,np.newaxis]
        grid_shape = minimization_grid.shape[:-1]
        for name in self.dimensions[0]:
            left = self.dimensions[0][name][0]
            right = self.dimensions[0][name][-1]
            if rank:
                try:
                    xrank = self.training_set_ranks[name]
                except:
                    pass # don't worry if ranks are unavailable
                break
        for name in self.dimensions[1]:
            bottom = self.dimensions[1][name][0]
            top = self.dimensions[1][name][-1]
            try:
                yrank = self.training_set_ranks[name]
            except:
                pass # don't worry if ranks are unavailable
            break
        (xs, ys) = (np.arange(left, right + 1), np.arange(bottom, top + 1))
        xbins = np.arange(left, right + 2) - 0.5
        xlim = (xbins[0], xbins[-1])
        ybins = np.arange(bottom, top + 2) - 0.5
        ylim = (ybins[0], ybins[-1])
        minimization_grid =\
            np.resize(minimization_grid, (-1, self.num_data_curves))
        flattened_argmins = np.argmin(minimization_grid, axis=0)
        (x_indices, y_indices) =\
            np.unravel_index(flattened_argmins, grid_shape)
        (x_values, y_values) = (xs[x_indices], ys[y_indices])
        pl.hist2d(x_values, y_values, cmap=cmap, bins=(xbins, ybins),\
                  vmin=vmin, vmax=vmax)
        try:
            pl.plot([xrank + 0.5] * 2, ylim, color='r', linestyle='--')
            pl.plot(xlim, [yrank + 0.5] * 2, color='r', linestyle='--')
        except:
            pass # don't worry if ranks are unavailable
        pl.xlim(xlim)
        pl.ylim(ylim)
        pl.colorbar()
        pl.xlabel(self.xlabel)
        pl.ylabel(self.ylabel)
        pl.xticks(np.arange(left, right + 1), self.xticks)
        pl.yticks(np.arange(bottom, top + 1), self.yticks)
        if show:
            pl.show()
    
    def plot_histogram(self, quantity_name, quantity_to_minimize=None,\
        color=None, ax=None, bins=None, cumulative=False, normed=False,\
        label=None, restricted=False, show=False):
        """
        Plots a histogram (or multiple if many quantities to minimize are
        given) of the given quantity. NOTE: matplotlib's histogram functions
        may yield strange and subtly wrong results when using cumulative=True
        AND normed=True!
        
        quantity_name: the name of the quantity for which to plot a histogram
        quantity_to_minimize: the quantity which was calculated by this
                              Forecaster object which should be minimized to
                              choose the SVD truncation. If None, the
                              quantity_to_minimize passed to the initializer of
                              the Extractor saved at self.file_name
                              is minimized. Can also be a list of quantities if
                              multiple histograms should be plotted.
        ax: matplotlib.Axes object on which to plot the histogram. if None, a
            new figure is created.
        bins: the bins to use for the Axes.hist function
        cumulative: True for cdf, False for pdf
        normed: boolean determining whether to norm the histogram
        label: the label to add associated with the histogram. If
               quantity_to_minimize is a sequence, label goes unused.
        restricted: if True, grid is restricted at ranks when minimizing
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: bins, array or list of arrays
        """
        if not self.multiple_data_curves:
            raise NotImplementedError("Only one data curve is included in " +\
                "this ExtractionPlotter, so a histogram doesn't make much " +\
                "sense.")
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        if type(quantity_to_minimize) in sequence_types:
            bins_to_return = []
            for (iquantity, quantity) in enumerate(quantity_to_minimize):
                if type(color) is type(None):
                    this_color = None
                elif type(color) in sequence_types:
                    this_color = color[iquantity]
                else:
                    raise TypeError("color argument not understood. If " +\
                        "multiple quantities to minimize are given, it " +\
                        "should be either None or a sequence of strings of " +\
                        "the same length as the number of quantities to " +\
                        "minimize.")
                bins_to_return.append(self.plot_histogram(quantity_name,\
                    quantity, ax=ax, normed=normed, cumulative=cumulative,\
                    bins=bins, color=this_color, label=quantity,\
                    restricted=restricted, show=False))
                ax.legend()
            return bins_to_return
        else:
            if restricted:
                to_hist = self.restricted_statistics(quantity_name,\
                    quantity_to_minimize=quantity_to_minimize)
            else:
                statistics = self.statistics_by_minimized_quantity(\
                    minimized_quantity=quantity_to_minimize,\
                    grid_quantities=self.compiled_quantity.quantities)
                if quantity_name in statistics:
                    to_hist = statistics[quantity_name]
                else:
                    raise ValueError(("{0!s} was not a statistic that was " +\
                        "stored in the file. The available statistics are " +\
                        "{1!s}.").format(quantity_name,\
                        [key for key in statistics.keys()]))
            (nums, bins, patches) = ax.hist(to_hist, bins=bins,\
                histtype='step', cumulative=cumulative, normed=normed,\
                color=color, label=label)
            return bins
    
    def plot_normalized_deviance_histograms(self, quantity_to_minimize=None,\
        bins=None, fontsize=24, show=False):
        """
        Plots normalized deviance histograms for restricted and unrestricted
        grids on the same axes.
        
        quantity_to_minimize: string name of quantity to minimize when
                              computing deviances
        bins: integer number of bins to use for each histogram
        fontsize: size of labels and title
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        """
        fig = pl.figure()
        ax = fig.add_subplot(111)
        self.plot_normalized_deviance_histogram(\
            quantity_to_minimize=quantity_to_minimize, color='b', ax=ax,\
            bins=bins, fontsize=fontsize, restricted=False, show=False,\
            label='unrestricted')
        self.plot_normalized_deviance_histogram(\
            quantity_to_minimize=quantity_to_minimize, color='r', ax=ax,\
            bins=bins, fontsize=fontsize, restricted=True, show=show,\
            label='rank-restricted')
    
    def plot_normalized_deviance_histogram(self, quantity_to_minimize=None,\
        color=None, ax=None, bins=None, fontsize=24, restricted=False,\
        label=None, show=False):
        """
        Plots a histogram of the normalized deviance and the expected reduced
        chi squared distributions.
        
        quantity_to_minimize: if None, only the statistics for the default
                                       quantity_to_minimize are shown
                              if a string, only the statistics with the given
                                           quantity_to_minimize are shown
                              if a sequence of string(s), describes the
                                                          quantities whose
                                                          deviance histogram
                                                          should be plotted
        color: the color (if quantity_to_minimize is None or a string) or
               sequence of colors to use (if quantity_to_minimize is a
               sequence)
        ax: either None or a matplotlib.axes.Axes object
        bins: bins argument to pass on to matplotlib.axes.Axes.hist function
        fontsize: size of font for title and labels, default 24
        restricted: if True, grid for minimization is truncated at ranks
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        
        returns: None if show is True, Axes instance containing plot otherwise
        """
        if type(ax) is type(None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        bins = self.plot_histogram('normalized_bias_statistic',\
            quantity_to_minimize=quantity_to_minimize, color=color, ax=ax,\
            bins=bins, cumulative=False, normed=True, restricted=restricted,\
            label=label, show=False)
        if type(quantity_to_minimize) not in sequence_types:
            quantity_to_minimize = [quantity_to_minimize]
            bins = [bins]
            if type(color) not in sequence_types:
                color = [color]
        for (index, minimized_quantity) in enumerate(quantity_to_minimize):
            this_color = color[index]
            these_bins = bins[index]
            num_bins = len(these_bins)
            if restricted:
                degrees_of_freedom =\
                    self.restricted_statistics('degrees_of_freedom',\
                    quantity_to_minimize=minimized_quantity)
            else:
                degrees_of_freedom =\
                    self.statistics_by_minimized_quantity(\
                    minimized_quantity)['degrees_of_freedom']
            median_degrees_of_freedom = int(np.median(degrees_of_freedom))
            degrees_of_freedom = np.unique(degrees_of_freedom)
            median_distribution =\
                ChiSquaredDistribution(median_degrees_of_freedom, reduced=True)
            distributions = [ChiSquaredDistribution(dof, reduced=True)\
                for dof in degrees_of_freedom]
            x_values =\
                np.linspace(these_bins[0], these_bins[-1], 10 * num_bins)
            y_values = np.array([np.exp(distribution.log_value(x_values))\
                for distribution in distributions])
            ax.plot(x_values, np.exp(median_distribution.log_value(x_values)),\
                color=this_color)
            ax.fill_between(x_values, np.min(y_values, axis=0),\
                np.max(y_values, axis=0), color=this_color)
        title = 'Normalized deviance histogram'
        if restricted:
            title = title + ' (rank-restricted)'
        ax.set_title(title, size=fontsize)
        ax.set_xlabel('Deviance, $D$', size=fontsize)
        ax.set_ylabel('PDF', size=fontsize)
        ax.tick_params(width=2.5, length=7.5, labelsize=fontsize)
        if show:
            pl.legend()
            pl.show()
        else:
            return ax
    
    @property
    def statistics(self):
        """
        Property storing a dictionary of statistics from the "optimal" fit.
        """
        if not hasattr(self, '_statistics'):
            self._statistics = self.statistics_by_minimized_quantity()
        return self._statistics
    
    def minimum_indices(self, quantity_name=None, icurve=None):
        """
        Finds the grid indices which minimize the given quantity for the given
        input data curve.
        
        quantity_name: name of the quantity to minimize
        icurve: index of the data curve for which to minimize the given
                quantity (if only one data curve is in the data, then icurve is
                not necessary)
        
        returns: tuple of indices describing the grid pixel which minimizes the
                 quantity for the given input data curve
        """
        if type(quantity_name) is type(None):
            quantity_name = self.quantity_to_minimize
        grid = self.get_grid(quantity_name)
        if type(icurve) is not type(None):
            grid = grid[...,icurve]
        return np.unravel_index(np.argmin(grid), grid.shape)
    
    def statistics_by_minimized_quantity(self, minimized_quantity=None,\
        grid_quantities=[]):
        """
        Finds the statistics (i.e. attributes of Fitter h5py groups) associated
        with the minimization of the given quantity.
        
        minimized_quantity: the quantity to minimize when finding the stats
        grid_quantities: quantities in the CompiledQuantity of the Extractor
                         which created the hdf5 file at the heart of this
                         object which are desired to be in the returned
                         dictionary
        
        returns: dictionary of statistics
        """
        if not hasattr(self, '_statistics_by_minimized_quantity'):
            self._statistics_by_minimized_quantity = {}
        if minimized_quantity not in self._statistics_by_minimized_quantity:
            no_saved_statistics_error = ValueError("No saved statistics " +\
                "for given minimized_quantity to minimize.")
            if self.multiple_data_curves:
                new_statistics = {}
                group_name = 'meta_fitter'
                if type(minimized_quantity) is type(None):
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
                    for grid_quantity in grid_quantities:
                        quantity_name = grid_quantity.name
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
                 if type(minimized_quantity) is type(None):
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
    
    def restricted_statistics(self, statistic, quantity_to_minimize=None):
        """
        Finds grid-restricted statistics.
        
        statistic: string name of the statistic to find
                   (e.g. 'normalized_bias_statistic')
        quantity_to_minimize: if None, self.quantity_to_minimize is minimized
        
        returns: 1D array of the given statistics
        """
        if type(quantity_to_minimize) is type(None):
            quantity_to_minimize = self.quantity_to_minimize
        grid = get_hdf5_value(self.file['meta_fitter/grids/{!s}'.format(\
            quantity_to_minimize)])
        restriction_slice = []
        for dimension in self.dimensions:
            for name in dimension:
                rank_index = self.training_set_rank_indices[name][1]
                if type(rank_index) is type(None):
                    restriction_slice.append(slice(None))
                else:
                    restriction_slice.append(slice(rank_index + 1))
                break
        if self.multiple_data_curves:
            restriction_slice.append(slice(None))
        else:
            restriction_slice.append(None)
        restriction_slice = tuple(restriction_slice)
        restricted_grid = grid[restriction_slice]
        grid_shape = restricted_grid.shape[:-1]
        num_curves = restricted_grid.shape[-1]
        flattened_grid = np.reshape(restricted_grid, (-1, num_curves))
        argmins = np.argmin(flattened_grid, axis=0)
        unraveled_indices = np.unravel_index(argmins, grid_shape)
        statistics = []
        for icurve in range(num_curves):
            grid_string = ('{}' + ('_{}' * (len(grid_shape) - 1)))
            grid_string =\
                grid_string.format(*[ui[icurve] for ui in unraveled_indices])
            fitter_group =\
                self.file['meta_fitter/fitters/{!s}'.format(grid_string)]
            statistic_values = fitter_group.attrs[statistic]
            if self.multiple_data_curves and\
                isinstance(statistic_values, np.ndarray):
                statistics.append(fitter_group.attrs[statistic][icurve])
            else:
                statistics.append(fitter_group.attrs[statistic])
        return np.array(statistics)
    
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
        Property storing the number of data curves in the hdf5 file at the
        heart of this object.
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
    def num_channels(self):
        """
        Property storing the integer number of data channels being fit in the
        Extractor which made the file at the heart of this ExtractionPlotter
        object.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.data.shape[-1]
        return self._num_channels
    
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
                ExpanderSet.load_from_hdf5_group(self.file['expanders'])
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
    
    def __enter__(self):
        """
        Enters into a with-statement with this plotter as the context manager.
        
        returns: self
        """
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        """
        Exits a with-statement which used this plotter as the context manager.
        Closes the file associated with this plotter.
        
        exception_type: type of exception thrown in with-statement
        exception_value: the exception thrown in with-statement
        traceback: a traceback of the exception thrown in with-statement
        """
        self.close()

