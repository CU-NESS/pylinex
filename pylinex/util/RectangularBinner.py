"""
File: pylinex/util/RectangularBinner.py
Author: Keith Tauscher
Date: 20 Sep 2018

Description: File containing a class and function which bins using a
             rectangular window function defined by bin edges.
"""
from __future__ import division
import numpy as np
from distpy import Savable, Loadable, sequence_types

class RectangularBinner(Savable, Loadable):
    """
    Class which bins using a rectangular window function defined by bin edges.
    """
    def __init__(self, unbinned_x_values, bin_edges):
        """
        Initializes a RectangularBinner with unbinned x values and edges of
        bins with which to define new points.
        
        unbinned_x_values: x_values associated with points to bin
        bin_edges: edges of bins with which to create new x values
        """
        self.unbinned_x_values = unbinned_x_values
        self.bin_edges = bin_edges
        self.digitize()
    
    def digitize(self):
        """
        Digitizes the bins so that binning can proceed quickly once data values
        are given.
        """
        bin_indices = np.digitize(self.unbinned_x_values, self.bin_edges)
        (bins_to_keep, unique_indices, unique_counts) =\
            np.unique(bin_indices, return_index=True, return_counts=True)
        if bins_to_keep[0] == 0:
            bins_to_keep = bins_to_keep[1:]
            unique_indices = unique_indices[1:]
            unique_counts = unique_counts[1:]
        if bins_to_keep[-1] == len(self.bin_edges):
            bins_to_keep = bins_to_keep[:-1]
            unique_indices = unique_indices[:-1]
            unique_counts = unique_counts[:-1]
        self._bins_to_keep = bins_to_keep - 1
        self._unique_indices = unique_indices
        self._unique_counts = unique_counts
    
    @property
    def unique_indices(self):
        """
        Property storing indices in the space of unbinned_x_values which share
        bins.
        """
        if not hasattr(self, '_unique_indices'):
            raise AttributeError("unique_indices was referenced before " +\
                "bins were digitized.")
        return self._unique_indices
    
    @property
    def unique_counts(self):
        """
        """
        if not hasattr(self, '_unique_counts'):
            raise AttributeError("unique_counts was referenced before bins " +\
                "were digitized.")
        return self._unique_counts
    
    @property
    def unbinned_x_values(self):
        """
        Property storing the x values of the unbinned points.
        """
        if not hasattr(self, '_unbinned_x_values'):
            raise AttributeError("unbinned_x_values was referenced before " +\
                "it was set.")
        return self._unbinned_x_values
    
    @unbinned_x_values.setter
    def unbinned_x_values(self, value):
        """
        Setter for the independent variable of the underlying model.
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._unbinned_x_values = value
            else:
                raise ValueError("unbinned_x_values was set to something " +\
                    "other than a 1D numpy.ndarray.")
        else:
            raise TypeError("unbinned_x_values was set to something other " +\
                "than a 1D numpy.ndarray.")
    
    @property
    def bin_edges(self):
        """
        Property storing the bin edges.
        """
        if not hasattr(self, '_bin_edges'):
            raise AttributeError("bin_edges was referenced before it was set.")
        return self._bin_edges
    
    @bin_edges.setter
    def bin_edges(self, value):
        """
        Setter for the bin edges.
        
        value: 1D array of bin edges (must be of length num_channels+1)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._bin_edges = value
            else:
                raise ValueError("bin_edges was set to something other " +\
                    "than a 1D numpy.ndarray.")
        else:
            raise TypeError("bin_edges was set to something other than a " +\
                "1D numpy.ndarray.")
    
    @property
    def nbins(self):
        """
        Property storing the number of bins implied by the bin edges.
        """
        if not hasattr(self, '_nbins'):
            self._nbins = len(self.bin_edges) - 1
        return self._nbins
    
    @property
    def bins_to_keep(self):
        """
        Property storing the indices of bin with nonzero weight in order.
        """
        if not hasattr(self, '_bins_to_keep'):
            raise AttributeError("bins_to_keep referenced before it was set.")
        return self._bins_to_keep
    
    @property
    def nbins_to_keep(self):
        """
        Property storing the number of bins in the results of this binner.
        """
        if not hasattr(self, '_nbins_to_keep'):
            self._nbins_to_keep = len(self.bins_to_keep)
        return self._nbins_to_keep
    
    @property
    def binned_x_values(self):
        """
        Property storing the binned x values.
        """
        if not hasattr(self, '_binned_x_values'):
            self._binned_x_values = (self.bin_edges[self.bins_to_keep] +\
                self.bin_edges[self.bins_to_keep+1]) / 2
        return self._binned_x_values
    
    @property
    def x_samples(self):
        """
        Property storing the x values in each bin. It is a list of arrays.
        """
        if not hasattr(self, '_x_samples'):
            self._x_samples = [np.array([])] * self.nbins
            for (final_bin_index, original_bin_index) in\
                enumerate(self.bins_to_keep):
                index = self.unique_indices[final_bin_index]
                count = self.unique_counts[final_bin_index]
                self._x_samples[original_bin_index] =\
                    self.unbinned_x_values[index:index+count]
        return self._x_samples
    
    def bin(self, old_y_values, weights=None, return_weights=False):
        """
        Bins the given data.
        
        old_y_values: data to bin
        weights: weights to associate with each unbinned y value (should be of
                 same shape as old_y_values)
        return_weights: if True, weights are returned alongside binned data
        
        returns: if return_weights is False, new_y_values
                 if return_weights is True, (new_y_values, new_weights)
        """
        shape = old_y_values.shape[:-1] + (len(self.bins_to_keep),)
        new_y_values = np.zeros(shape)
        if weights is None:
            weights = np.ones_like(old_y_values)
        if return_weights:
            new_weights = np.zeros(shape)
        for final_bin_index in range(len(self.bins_to_keep)):
            unique_index = self.unique_indices[final_bin_index]
            unique_count = self.unique_counts[final_bin_index]
            where = slice(unique_index, unique_index + unique_count)
            old_y_slice = old_y_values[...,where]
            weight_slice = weights[...,where]
            new_weight = np.sum(weight_slice, axis=-1)
            new_y_values[...,final_bin_index] =\
                np.sum(old_y_slice * weight_slice, axis=-1) / new_weight
            if return_weights:
                new_weights[...,final_bin_index] = new_weight
        if return_weights:
            return (new_y_values, new_weights)
        else:
            return new_y_values
    
    def bin_error(self, old_error, weights=None, return_weights=False):
        """
        Bins the given error vector(s).
        
        old_error: error to bin containing positive numbers with the last axis
                   being the binning axis.
        weights: weights to associate with each unbinned y value (should be of
                 same shape as old_y_values)
        return_weights: if True, weights are returned alongside binned data
        
        returns: if return_weights is False, new_y_values
                 if return_weights is True, (new_y_values, new_weights)
        """
        shape = old_error.shape[:-1] + (self.nbins_to_keep,)
        new_error = np.zeros(shape)
        if weights is None:
            weights = np.ones_like(old_error)
        if return_weights:
            new_weights = np.zeros(shape)
        for final_bin_index in range(self.nbins_to_keep):
            unique_index = self.unique_indices[final_bin_index]
            unique_count = self.unique_counts[final_bin_index]
            where = slice(unique_index, unique_index + unique_count)
            old_error_slice = old_error[...,where]
            weight_slice = weights[...,where]
            new_weight = np.sum(weight_slice, axis=-1)
            new_error[...,original_bin_index] = np.sqrt(np.sum(np.power(\
                weight_slice * old_error_slice, 2), axis=-1)) / new_weight
            if return_weights:
                new_weights[...,original_bin_index] = new_weight
        if return_weights:
            return (new_error, new_weights)
        else:
            return new_error
    
    def __call__(self, old_y_values, weights=None, return_weights=False):
        """
        Bins the given data.
        
        old_y_values: data to bin
        weights: weights to associate with each unbinned y value (should be of
                 same shape as old_y_values)
        return_weights: if True, weights are returned alongside binned data
        
        returns: if return_weights is False, new_y_values
                 if return_weights is True, (new_y_values, new_weights)
        """
        return self.bin(old_y_values, weights=weights,\
            return_weights=return_weights)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this RectangularBinner.
        """
        group.create_dataset('unbinned_x_values', data=self.unbinned_x_values)
        group.create_dataset('bin_edges', data=self.bin_edges)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a RectangularBinner from the given hdf5 file group.
        
        group: group where RectangularBinner was once saved
        
        returns: RectangularBinner whose info was saved in the given group
        """
        unbinned_x_values = group['unbinned_x_values'].value
        bin_edges = group['bin_edges'].value
        return RectangularBinner(unbinned_x_values, bin_edges)
    
    def __eq__(self, other):
        """
        Checks if other is the same binner as this.
        
        other: object to check for equality
        
        returns: True only if other is a RectangularBinner with the same
                 unbinned x values and bin edges. False otherwise
        """
        if not isinstance(other, RectangularBinner):
            return False
        if (self.unbinned_x_values.shape != other.unbinned_x_values.shape) or\
            np.any(self.unbinned_x_values != other.unbinned_x_values):
            return False
        if (self.bin_edges.shape != other.bin_edges.shape) or\
            np.any(self.bin_edges != other.bin_edges):
            return False
        return True
    
    def __ne__(self, other):
        """
        Checks if other is not the same binner as this.
        
        other: object to check for inequality
        
        returns: False only if other is a RectangularBinner with the same
                 unbinned x values and bin edges. True otherwise
        """
        return (not self.__eq__(other))

def rect_bin(bin_edges, unbinned_x_values, old_y_values, weights=None,\
    return_weights=False):
    """
    Bins on final axis.
    
    bin_edges: 1D numpy.ndarray of length Nafter+1 containing bin edges
    unbinned_x_values: 1D numpy.ndarray of length Nbefore containing x values
                       of points
    old_y_values: ND numpy.ndarray whose final axis length is Nbefore
    return_weights: if True, extra array of new weights is returned
                    (Default False)
    
    returns: (binned_x_values, new_y_values[, new_weights])
             binned_x_values: 1D numpy.ndarray of length Nafter
             new_y_values: ND numpy.ndarray of the same shape as old_y_values
                           with Nafter instead of Nbefore as the final axis
                           length
             [new_weights]: new weights to associate with the binned data
                            points, given by the sum of the weights that went
                            into each bin (returned only if return_weights is
                            True)
    """
    binner = RectangularBinner(unbinned_x_values, bin_edges)
    bin_results =\
        binner(old_y_values, weights=weights, return_weights=return_weights)
    if return_weights:
        return (binner.binned_x_values,) + bin_results
    else:
        return (binner.binned_x_values, binner.new_y_values)

