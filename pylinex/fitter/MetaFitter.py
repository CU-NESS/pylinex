"""
File: pylinex/fitter/MetaFitter.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class which employs many Fitter objects to form
             grids of fit statistics with which to perform parameter number
             optimization.
"""
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import LogNorm, SymLogNorm
from distpy import GaussianDistribution 
from ..util import VariableGrid, int_types, sequence_types
from ..quantity import QuantityFinder
from .Fitter import Fitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class MetaFitter(Fitter, VariableGrid, QuantityFinder):
    """
    Class which performs fits using the BasisSet it is given as well as subsets
    of the BasisSet given. By doing so for grids of different subsets, it
    chooses the optimal number of parameters.
    """
    def __init__(self, basis_set, data, error, compiled_quantity, *dimensions,\
        **priors):
        """
        Initializes a new MetaFitter object using the given inputs.
        
        basis_set: a BasisSet object (or a Basis object, which is converted
                   internally to a BasisSet of one Basis with the name 'Sole')
        data: 1D vector of same length as vectors in basis_set
        error: 1D vector of same length as vectors in basis_set containing only
               positive numbers
        compiled_quantity: CompiledQuantity object representing quantities to
                           retrieve
        *dimensions: list of lists of dictionaries indicating slices to take
                     for each subbasis.
        **priors: keyword arguments where the keys are exactly the names of the
                  basis sets with '_prior' appended to them
        """
        Fitter.__init__(self, basis_set, data, error, **priors)
        self.dimensions = dimensions
        self.compiled_quantity = compiled_quantity

    @property
    def grids(self):
        """
        Property storing the grids calculated by the full grid calculations. It
        is a list of numpy.ndarray objects.
        """
        if not hasattr(self, '_grids'):
            self._grids = [np.zeros(self.shape + self.data.shape[:-1])\
                for index in range(self.num_quantities)]
            for indices in np.ndindex(*self.shape):
                fitter = self.fitter_from_indices(indices)
                quantity_values = self.compiled_quantity(fitter)
                for (iquantity, quantity) in enumerate(quantity_values):
                    self._grids[iquantity][indices] = quantity
        return self._grids
    
    def fitter_from_subsets(self, **subsets):
        """
        Finds the Fitter object associated with the given subbasis
        subsets.
        
        subsets: dict where the keys are basis names and the values are index
                 slices corresponding to the subsets to take
        
        returns: Fitter corresponding to the given subbasis subsets
        """
        sub_basis_set = self.basis_set.basis_subsets(**subsets)
        sub_prior_sets = self.prior_subsets(**subsets)
        return Fitter(sub_basis_set, self.data, self.error, **sub_prior_sets)
    
    def fitter_from_indices(self, indices):
        """
        Finds the Fitter object corresponding to the given tuple of indices.
        
        indices: tuple of ints with length given by the number of dimensions
        
        returns: Fitter corresponding to the given grid position
        """
        return self.fitter_from_subsets(**self.point_from_indices(indices))
    
    def prior_subsets(self, **subsets):
        """
        Applies given subbasis subsets to the priors for each subbasis.
        
        subsets: dict where the keys are basis names and the values are index
                 slices corresponding to the subsets to take
        
        returns: dict of priors with subsets taken from each subbasis
        """
        result = {}
        if self.has_priors:
            for name in self.basis_set.names:
                key = name + '_prior'
                old_prior = self.priors[key]
                if name in subsets:
                    mi = subsets[name]
                    new_mean = old_prior.mean.A[0,:mi]
                    new_covariance = old_prior.covariance.A[:mi][:,:mi]
                    result[key] =\
                        GaussianDistribution(new_mean, new_covariance)
                else:
                    result[key] = old_prior
        return result
    
    def __getitem__(self, index):
        """
        Gets the grid associated with the given index.
        
        index: if int, it is taken to be the internal index of the quantity to
                       retrieve
               if str, it is taken to be the name of the quantity to retrieve
                       (in this case,
                       self.compiled_quantity.can_index_by_string must be True)
        
        returns: numpy.ndarray of shape given by shape property containing
                 values of the given quantity
        """
        if type(index) in int_types:
            return self.grids[index]
        elif isinstance(index, basestring):
            return self.grids[self.compiled_quantity.index_dict[index]]
        else:
            raise AttributeError("index of MetaFitter must be an index " +\
                                 "or a string. If it is a string, the " +\
                                 "CompiledQuantity at the center of the " +\
                                 "MetaFitter must have can_index_by_string " +\
                                 "be True.")
    
    def minimize_quantity(self, index=0, which_data=None, verbose=True):
        """
        Minimizes the quantity associated with the given index and returns the
        Fitter from that given set of subbasis subsets.
        
        index: if int, it is taken to be the internal index of the quantity to
                       retrieve
               if str, it is taken to be the name of the quantity to retrieve
                       (in this case,
                       self.compiled_quantity.can_index_by_string must be True)
        which_data: if None, 
        verbose: if True, prints the name of the quantity being minimized
        
        returns: Fitter corresponding to set of subbasis subsets which
                 minimizes the Quantity under concern
        """
        grid_slice = ((slice(None),) * self.ndim)
        if self.data.ndim == 2:
            if which_data is None:
                raise ValueError("which_data must be given if data is not 1D.")
            else:
                grid_slice = grid_slice + (which_data,)
        grid = self[index][grid_slice]
        quantity = self.compiled_quantity[index]
        if verbose:
            print("Minimizing {!s} over grid.".format(quantity.name))
        indices_of_minimum = np.unravel_index(np.argmin(grid), grid.shape)
        return self.fitter_from_indices(indices_of_minimum)
    
    def save_all_fitters_to_hdf5_group(self, group):
        """
        Saves all fitters to an hdf5 group. This should be used cautiously, as
        it would take an unreasonably long time for large grids.
        
        group: hdf5 file group to fill with Fitter information
        """
        for indices in np.ndindex(*self.shape):
            format_string = (('{}_' * (len(self.shape) - 1)) + '{}')
            subgroup = group.create_group(format_string.format(*indices))
            self.fitter_from_indices(indices).fill_hdf5_group(subgroup)
    
    def plot_grid(self, grid, title='', fig=None, ax=None, xticks=None,\
        yticks=None, xlabel='', ylabel='', show=False, norm=None, **kwargs):
        """
        Plots the given grid using matplotlib.pyplot.imshow().
        
        grid: the 2D grid to plot. Its shape should be equal to the shape
              property of thie MetaFitter
        title: string title of plot
        fig: the matplotlib.figure objects on which to plot the grid
        ax: the matplotlib.axes objects on which to plot the grid
        xticks: the string labels to put at each of the x-values of the grid
        yticks: the string labels to put at each of the y-values of the grid
        xlabel: string label to place on the x-axis of the grid
        ylabel: string label to place on the y-axis of the grid
        show: if True, matplotlib.pyplot.show() is called before this function
                       returns
        norm: if None, linear color scale is used
              if 'log', log scale is used with no value adjustment
              if 'log_from_min', log scale is used to show the distance of each
                                 point above the minimum of the grid
              if 'log_from_max', log scale is used to show the distance of each
                                 point below the maximum of the grid
        kwargs: extra keyword arguments to pass on to matplotlib.pyplot.imshow
        """
        if self.shape != grid.shape:
            raise ValueError("Only grids of the same shape as this " +\
                             "MetaFitter can be plotted.")
        if len(self.shape) != 2:
            raise ValueError("Only 2D grids can be plotted.")
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        def_kwargs = {'interpolation': 'none'}
        def_kwargs.update(kwargs)
        if norm in ['log', 'log_from_min', 'log_from_max']:
            if norm == 'log_from_min':
                grid = grid - np.min(grid)
            elif norm == 'log_from_max':
                grid = np.max(grid) - grid
            norm = LogNorm()
        pl.imshow(grid.T, norm=norm, **def_kwargs)
        if xticks is not None:
            pl.xticks(np.arange(self.shape[0]), xticks)
        if yticks is not None:
            pl.yticks(np.arange(self.shape[1]), yticks)
        pl.colorbar()
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        pl.title(title)
        if show:
            pl.show()
        return grid
    
    def plot_quantity_grid(self, index, **kwargs):
        """
        Plots the grid which was calculated for the indicated Quantity.
        
        index: if int, it is taken to be the internal index of the quantity to
                       retrieve
               if str, it is taken to be the name of the quantity to retrieve
                       (in this case,
                       self.compiled_quantity.can_index_by_string must be True)
        kwargs: keyword arguments to pass on to the plot_grid function of this
                class
        """
        if 'title' not in kwargs:
            kwargs['title'] = self.compiled_quantity[index].name
        return self.plot_grid(self[index], **kwargs)
        
    def plot_subbasis_fit_grid(self, name=None, true_curve=None,\
        title='Subbasis fit grid', xticks=None, yticks=None, xlabel='',\
        ylabel='', subtract_truth=False, show=False):
        """
        Plots many subbasis fits at the same time on a single grid.
        
        name: string identifying subbasis under concern
        true_curve: the "truth" (i.e. input) to plot along with the fit
        title: string title of grid of plots
        xticks: the string labels to put at each of the x-values of the grid
        yticks: the string labels to put at each of the y-values of the grid
        xlabel: string label to place on the x-axis of the grid
        ylabel: string label to place on the y-axis of the grid
        subtract_truth: if True, true_curve is subtracted in plot so that the
                                 plot shows residuals (true_curve must be given
                                 if subtract_truth is True)
        show: if True, matplotlib.pyplot.show() is called before this function
        """
        fig = pl.figure()
        pl.title(title)
        nrows = self.shape[1]
        ncols = self.shape[0]
        xtick_locations = [((i + 0.5) / ncols) for i in range(ncols)]
        ytick_locations = [((i + 0.5) / nrows) for i in range(nrows)][-1::-1]
        if xticks is None:
            xtick_strings = ([''] * ncols)
        else:
            xtick_strings = xticks
        if yticks is None:
            ytick_strings = ([''] * nrows)
        else:
            ytick_strings = yticks
        pl.xticks(xtick_locations, xtick_strings)
        pl.yticks(ytick_locations, ytick_strings)
        pl.xlabel(xlabel)
        pl.ylabel(ylabel)
        naxes = nrows * ncols
        for iplot in range(naxes):
            irow = iplot // ncols
            icol = iplot % ncols
            ax = fig.add_subplot(nrows, ncols, iplot + 1)
            fitter = self.fitter_from_indices((icol, irow))
            fitter.plot_subbasis_fit(name=name, true_curve=true_curve,\
                fig=fig, ax=ax, title='', xlabel='', ylabel='',\
                subtract_truth=subtract_truth, show=False)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
        fig.subplots_adjust(hspace=0, wspace=0)
        if show:
            pl.show()

