"""
Module containing base class containing common properties for any class of
linear fitter, such as a basis, priors, and data and error vectors.

**File**: $PYLINEX/pylinex/fitter/BaseFitter.py  
**Author**: Keith Tauscher  
**Date**: 25 May 2021
"""
import numpy as np
import scipy.linalg as scila
from distpy import GaussianDistribution, SparseSquareBlockDiagonalMatrix
from ..util import sequence_types, create_hdf5_dataset, get_hdf5_value
from ..basis import Basis, BasisSum

class BaseFitter(object):
    """
    Base class containing common properties for any class of linear fitter,
    such as a basis, priors, and data and error vectors.
    """
    @property
    def basis_sum(self):
        """
        The `pylinex.basis.BasisSum.BasisSum` object containing the basis
        vectors used in the fit, which represent the columns of matrix,
        \\(\\boldsymbol{G}\\).
        """
        if not hasattr(self, '_basis_sum'):
            raise AttributeError("basis_sum was referenced before it was " +\
                                 "set. This shouldn't happen. Something is " +\
                                 "wrong.")
        return self._basis_sum
    
    @basis_sum.setter
    def basis_sum(self, value):
        """
        Setter for `BaseFitter.basis_sum`
        
        Parameters
        ----------
        value : `pylinex.basis.BasisSum.BasisSum` or\
        `pylinex.basis.Basis.Basis`
            the basis used to model the data, represented in equations by
            \\(\\boldsymbol{G}\\) alongside the translation component
            \\(\\boldsymbol{\\mu}\\). Two types of inputs are accepted:
            
            - If `basis_sum` is a `pylinex.basis.BasisSum.BasisSum`, then it is
            assumed to have constituent bases for each modeled component
            alongside `pylinex.expander.Expander.Expander` objects determining
            how those components enter into the data
            - If `basis_sum` is a `pylinex.basis.Basis.Basis`, then it is
            assumed that this single basis represents the only component that
            needs to be modeled. The
            `BaseFitter.basis_sum` property will be set to a
            `pylinex.basis.BasisSum.BasisSum` object with this
            `pylinex.basis.Basis.Basis` as its only component, labeled with the
            string name `"sole"`
        """
        if isinstance(value, BasisSum):
            self._basis_sum = value
        elif isinstance(value, Basis):
            self._basis_sum = BasisSum('sole', value)
        else:
            raise TypeError("basis_sum was neither a BasisSum or a " +\
                            "different Basis object.")
    
    @property
    def num_channels(self):
        """
        The integer number of data channels in this fit. This should be equal
        to the number of rows in \\(\\boldsymbol{y}\\), \\(\\boldsymbol{G}\\),
        and \\(\\boldsymbol{C}\\).
        """
        return self.basis_sum.num_larger_channel_set_indices
    
    @property
    def sizes(self):
        """
        A dictionary with basis names as keys and the number of basis vectors
        in that basis as values.
        """
        if not hasattr(self, '_sizes'):
            self._sizes = self.basis_sum.sizes
        return self._sizes
    
    @property
    def names(self):
        """
        A list of the names of the component bases of `BaseFitter.basis_sum`.
        """
        if not hasattr(self, '_names'):
            self._names = self.basis_sum.names
        return self._names
    
    @property
    def data(self):
        """
        The data vector(s), \\(\\boldsymbol{y}\\), to fit.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data wasn't set before it was " +\
                                 "referenced. Something is wrong. This " +\
                                 "shouldn't happen.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for `BaseFitter.data`
        
        Parameters
        ----------
        value : numpy.ndarray
            the data to fit, represented in equations by \\(\\boldsymbol{y}\\)
            - if `data` is 1D, then its length should be the same as the
            (expanded) vectors in `basis_sum`, i.e. the number of rows of
            \\(\\boldsymbol{G}\\), `nchannels`
            - if `data` is 2D, then it should have shape `(ncurves, nchannels)`
            and it will be interpreted as a list of data vectors to fit
            independently
        """
        value = np.array(value)
        if value.ndim in [1, 2]:
            if value.shape[-1] == self.num_channels:
                self._data = value
            else:
                raise ValueError("data curve(s) did not have the same " +\
                                 "length as the basis functions.")
        else:
            raise ValueError("data was neither 1- or 2-dimensional.")
    
    @property
    def multiple_data_curves(self):
        """
        A boolean describing whether this `BaseFitter` contains multiple data
        curves or not.
        """
        if not hasattr(self, '_multiple_data_curves'):
            self._multiple_data_curves = (self.data.ndim == 2)
        return self._multiple_data_curves
    
    @property
    def error(self):
        """
        The 1D error vector with which to weight the least square fit (if
        `BaseFitter.non_diagonal_noise_covariance` is False) or a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
        (if `BaseFitter.non_diagonal_noise_covariance` if True).
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error wasn't set before it was " +\
                "referenced. Something is wrong. This shouldn't happen.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for `BaseFitter.error`.
        
        Parameters
        ----------
        error : numpy.ndarray or\
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            the noise level of the data that determines the covariance matrix,
            represented in equations by \\(\\boldsymbol{C}\\):
            
            - if `error` is a 1D `numpy.ndarray`, it should have the same
            length as the (expanded) vectors in `basis_sum`, i.e. the number of
            rows of \\(\\boldsymbol{G}\\), `nchannels` and should only contain
            positive numbers. In this case, \\(\\boldsymbol{C}\\) is a diagonal
            matrix whose elements are the squares of the values in `error`
            - if `error` is a
            `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`,
            then it is assumed to represent a block diagonal
            \\(\\boldsymbol{C}\\) directly
        """
        if type(value) is type(None):
            self._error = np.ones(self.num_channels)
            self._non_diagonal_noise_covariance = False
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.num_channels,):
                self._error = value
            else:
                raise ValueError("error didn't have the same length as the " +\
                                 "basis functions.")
            self._non_diagonal_noise_covariance = False
        elif isinstance(value, SparseSquareBlockDiagonalMatrix):
            if value.dimension == self.num_channels:
                self._error = value
            else:
                raise ValueError("error was set to a " +\
                    "SparseSquareBlockDiagonalMatrix with the wrong " +\
                    "dimension.")
            self._non_diagonal_noise_covariance = True
            self._inverse_square_root_noise_covariance =\
                value.inverse_square_root()
        else:
            raise TypeError("error was neither None, a sequence, nor a " +\
                "SparseSquareBlockDiagonalMatrix.")
    
    @property
    def non_diagonal_noise_covariance(self):
        """
        Boolean describing whether noise level is stored as a
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
        (True) or not (False)
        """
        if not hasattr(self, '_non_diagonal_noise_covariance'):
            self._non_diagonal_noise_covariance =\
                isinstance(self.error, SparseSquareBlockDiagonalMatrix)
        return self._non_diagonal_noise_covariance
    
    
    def weight(self, array, axis):
        """
        Weights the given array by the inverse square root of the noise
        covariance matrix.
        
        Parameters
        ----------
        array : numpy.ndarray
            the array to weight, can be any number of dimensions as long as the
            specified one has length given by `BaseFitter.num_channels`
        axis : int
            index of the axis corresponding to the parameters
        
        Returns
        -------
        weighted : numpy.ndarray
            array of same shape as array corresponding to
            \\(\\boldsymbol{C}^{-1/2}\\boldsymbol{A}\\), where
            \\(\\boldsymbol{A}\\) is array shaped so that the matrix
            multiplication makes sense.
        """
        axis = axis % array.ndim
        if self.non_diagonal_noise_covariance:
            if array.ndim == 1:
                return self._inverse_square_root_noise_covariance.__matmul__(\
                    array)
            elif array.ndim == 2:
                if axis == 0:
                    return\
                        self._inverse_square_root_noise_covariance.__matmul__(\
                        array.T).T
                else:
                    return\
                        self._inverse_square_root_noise_covariance.__matmul__(\
                        array)
            else:
                before_shape = array.shape[:axis]
                after_shape = array.shape[(axis+1):]
                if axis == (array.ndim - 1):
                    weighted_array = array
                else:
                    weighted_array = np.rollaxis(array, axis, start=array.ndim)
                weighted_array =\
                    self._inverse_square_root_noise_covariance.__matmul__(\
                    weighted_array)
                if axis != (array.ndim - 1):
                    weighted_array =\
                        np.rollaxis(weighted_array, array.ndim - 1, start=axis)
                return weighted_array
        else:
            before_dims = axis
            after_dims = array.ndim - axis - 1
            error_slice = ((np.newaxis,) * before_dims) + (slice(None),) +\
                ((np.newaxis,) * after_dims)
            return array / self.error[error_slice]
    
    @property
    def translated_data(self):
        """
        The data with the translation vector subtracted, i.e. the part of the
        data that will be fit with a linear model,
        \\(\\boldsymbol{y}-\\boldsymbol{\\mu}\\).
        """
        if not hasattr(self, '_translated_data'):
            self._translated_data = self.data - self.basis_sum.translation
        return self._translated_data
    
    @property
    def weighted_data(self):
        """
        The data vector weighted by the noise covariance. This is represented
        mathematically by \\(\\boldsymbol{C}^{-1/2}\\boldsymbol{y}\\) and is
        the quantity that actually goes into the calculation of the mean and
        covariance of the posterior parameter distribution.
        """
        if not hasattr(self, '_weighted_data'):
            self._weighted_data = self.weight(self.data, -1)
        return self._weighted_data
    
    @property
    def weighted_translated_data(self):
        """
        The translated data vector weighted by the noise covariance. This is
        represented mathematically by
        \\(\\boldsymbol{C}^{-1/2}(\\boldsymbol{y}-\\boldsymbol{\\mu})\\) and is
        the quantity that actually goes into the calculation of the mean and
        covariance of the posterior parameter distribution. It has the same
        shape as `BaseFitter.data`.
        """
        if not hasattr(self, '_weighted_translated_data'):
            self._weighted_translated_data =\
                self.weight(self.translated_data, -1)
        return self._weighted_translated_data
    
    @property
    def weighted_basis(self):
        """
        The basis functions of the basis set weighted down by the error. This
        is represented mathematically by
        \\(\\boldsymbol{C}^{-1/2}\\boldsymbol{G}\\). It has the same shape as
        the `basis` property of `BaseFitter.basis_sum`.
        """
        if not hasattr(self, '_weighted_basis'):
            self._weighted_basis = self.weight(self.basis_sum.basis, -1)
        return self._weighted_basis
    
    @property
    def basis_overlap_matrix(self):
        """
        The matrix of overlaps between the weighted basis vectors. It is
        represented mathematically as
        \\(\\boldsymbol{G}^T\\boldsymbol{C}^{-1}\\boldsymbol{G}\\) and is
        stored as a \\(n\\times n\\) `numpy.ndarray`, where \\(n\\) is the
        total number of basis vectors, i.e. the `num_basis_vectors` property of
        `BaseFitter.basis_sum`.
        """
        if not hasattr(self, '_basis_overlap_matrix'):
            self._basis_overlap_matrix =\
                np.dot(self.weighted_basis, self.weighted_basis.T)
        return self._basis_overlap_matrix
    
    @property
    def has_priors(self):
        """
        Boolean describing whether or not any priors will be used in the fit.
        """
        if not hasattr(self, '_has_priors'):
            self._has_priors = bool(self.priors)
        return self._has_priors
    
    @property
    def priors(self):
        """
        The `priors` dictionary provided to this `BaseFitter` at initialization
        (by the subclass). It should be a dictionary with keys of the form
        `(name+'_prior')` and values which are
       `distpy.distribution.GaussianDistribution.GaussianDistribution` objects.
        """
        if not hasattr(self, '_priors'):
            self._priors = {}
        return self._priors
    
    @priors.setter
    def priors(self, value):
        """
        Sets the `BaseFitter.priors`.
        
        Parameters
        ----------
        value : dict
            keyword arguments where the keys are exactly the names of the
            `basis_sum` with `'_prior'` appended to them and the values are
            `distpy.distribution.GaussianDistribution.GaussianDistribution`
            objects. Priors are optional and can be included or excluded for
            any given component. If `basis_sum` was given as a
            `pylinex.basis.Basis.Basis`, then `priors` should either be empty
            or a dictionary of the form
            `{'sole_prior': gaussian_distribution}`. The means and inverse
            covariances of all priors are combined into a full parameter prior
            mean and full parameter prior inverse covariance, represented in
            equations by \\(\\boldsymbol{\\nu}\\) and
            \\(\\boldsymbol{\\Lambda}^{-1}\\), respectively. Having no prior is
            equivalent to having an infinitely wide prior, i.e. a prior with an
            inverse covariance matrix of \\(\\boldsymbol{0}\\)
        """
        self._priors = value
        self._has_all_priors = False
        if self.has_priors:
            self._has_all_priors = True
            self._prior_mean = []
            self._prior_covariance = []
            self._prior_inverse_covariance = []
            for name in self.names:
                key = '{!s}_prior'.format(name)
                if key in self._priors:
                    self._prior_mean.append(\
                        self._priors[key].internal_mean.A[0])
                    self._prior_covariance.append(\
                        self._priors[key].covariance.A)
                    self._prior_inverse_covariance.append(\
                        self._priors[key].inverse_covariance.A)
                else:
                    nparams = self.basis_sum[name].num_basis_vectors
                    self._prior_mean.append(np.zeros(nparams))
                    self._prior_covariance.append(np.zeros((nparams, nparams)))
                    self._prior_inverse_covariance.append(\
                        np.zeros((nparams, nparams)))
                    self._has_all_priors = False
            self._prior_mean = np.concatenate(self._prior_mean)
            self._prior_covariance = scila.block_diag(*self._prior_covariance)
            self._prior_inverse_covariance =\
                scila.block_diag(*self._prior_inverse_covariance)
    
    @property
    def has_all_priors(self):
        """
        Boolean describing whether all basis sets have priors or not.
        """
        if not hasattr(self, '_has_all_priors'):
            raise AttributeError("has_all_priors was referenced before it " +\
                "was set. It can't be referenced until the priors dict " +\
                "exists.")
        return self._has_all_priors
    
    @property
    def prior_mean(self):
        """
        The mean parameter vector, \\(\\boldsymbol{\\nu}\\), of the prior
        distribution. It is a 1D `numpy.ndarray` with an element for each basis
        vector.
        """
        if not hasattr(self, '_prior_mean'):
            raise AttributeError("prior_mean was referenced before it was " +\
                "set. Something is wrong. This shouldn't happen.")
        return self._prior_mean
    
    @property
    def prior_channel_mean(self):
        """
        The prior mean in the space of the data. This is represented
        mathematically as \\(\\boldsymbol{G}\\boldsymbol{\\nu} +\
        \\boldsymbol{\\mu}\\) and is stored
        in a 1D `numpy.ndarray` of length `BaseFitter.num_channels`.
        """
        if not hasattr(self, '_prior_channel_mean'):
            self._prior_channel_mean = self.basis_sum.translation +\
                np.dot(self.prior_mean, self.basis_sum.basis)
        return self._prior_channel_mean
    
    @property
    def weighted_prior_channel_mean(self):
        """
        The error-weighted channel mean. It is represented mathematically as
        \\(\\boldsymbol{C}^{-1/2}(\\boldsymbol{G}\\boldsymbol{\\nu}+\
        \\boldsymbol{\\mu})\\) and is stored in a 1D `numpy.ndarray` of length
        `BaseFitter.num_channels`.
        """
        if not hasattr(self, '_weighted_prior_channel_mean'):
            self._weighted_prior_channel_mean =\
                self.weight(self.prior_channel_mean, -1)
        return self._weighted_prior_channel_mean
    
    @property
    def weighted_shifted_data(self):
        """
        An error-weighted version of the data vector, shifted by the basis
        translation and the prior channel mean. This is represented
        mathematically as \\(\\boldsymbol{C}^{-1/2}(\\boldsymbol{y}-\
        \\boldsymbol{G}\\boldsymbol{\\nu}-\\boldsymbol{\\mu})\\) and is stored
        in a `numpy.ndarray` with the same shape as `BaseFitter.data`.
        """
        if not hasattr(self, '_weighted_shifted_data'):
            if self.has_priors:
                if self.multiple_data_curves:
                    self._weighted_shifted_data = self.weighted_data -\
                        self.weighted_prior_channel_mean[np.newaxis,:]
                else:
                    self._weighted_shifted_data =\
                        self.weighted_data - self.weighted_prior_channel_mean
            else:
                self._weighted_shifted_data = self.weighted_translated_data
        return self._weighted_shifted_data
    
    @property
    def prior_inverse_covariance(self):
        """
        The inverse covariance matrix of the prior parameter distribution. It
        is represented mathematically as \\(\\boldsymbol{\\Lambda}^{-1}\\) and
        is stored as a square 2D `numpy.ndarray` with a dimension given by the
        total number of basis vectors.
        """
        if not hasattr(self, '_prior_inverse_covariance'):
            raise AttributeError("prior_inverse_covariance was referenced " +\
                "before it was set. Something is wrong. This shouldn't " +\
                "happen.")
        return self._prior_inverse_covariance
    
    @property
    def prior_inverse_covariance_times_mean(self):
        """
        The vector result of the matrix multiplication of the prior inverse
        covariance matrix and the prior mean vector. This is represented
        mathematically as \\(\\boldsymbol{\\Lambda}^{-1}\\boldsymbol{\\nu}\\)
        and is stored in a 1D `numpy.ndarray` whose length is the number of
        basis vectors.
        """
        if not hasattr(self, '_prior_inverse_covariance_times_mean'):
            self._prior_inverse_covariance_times_mean =\
                np.dot(self.prior_inverse_covariance, self.prior_mean)
        return self._prior_inverse_covariance_times_mean
    
    def save_data(self, root_group, data_link=None):
        """
        Saves `BaseFitter.data` in the given hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group this `BaseFitter` is being saved in
        data_link : None or `HDF5Link` or h5py.Group sequence
            link to existing data dataset, if it exists (see
            `distpy.util.h5py_extensions.create_hdf5_dataset` docs for info
            about accepted formats)
        """
        create_hdf5_dataset(root_group, 'data', data=self.data, link=data_link)
    
    @staticmethod
    def load_data(root_group):
        """
        Loads the data from an hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group in which this `BaseFitter` was saved
        
        Returns
        -------
        data : numpy.ndarray
            the data vector(s) loaded
        """
        return get_hdf5_value(root_group['data'])
    
    def save_error(self, root_group, error_link=None):
        """
        Saves `BaseFitter.error` in the given hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group this `BaseFitter` is being saved in
        error_link : None or `HDF5Link` or h5py.Group or sequence
            link to existing error dataset, if it exists (see
            `distpy.util.h5py_extensions.create_hdf5_dataset` docs for info
            about accepted formats) or the group at which the error is already
            stored if `BaseFitter.non_diagonal_noise_covariance` is True.
        """
        if self.non_diagonal_noise_covariance:
            if type(error_link) is type(None):
                self.error.fill_hdf5_group(root_group.create_group('error'))
            else:
                root_group['error'] = error_link
        else:
            create_hdf5_dataset(root_group, 'error', data=self.error,\
                link=error_link)
    
    @staticmethod
    def load_error(root_group):
        """
        Loads the error from an hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group in which this `BaseFitter` was saved
        
        Returns
        -------
        error : numpy.ndarray or\
        `distpy.util.SparseSquareBlockDiagonalMatrix.SparseSquareBlockDiagonalMatrix`
            the error vector loaded
        """
        try:
            return get_hdf5_value(root_group['error'])
        except AttributeError:
            return SparseSquareBlockDiagonalMatrix.load_from_hdf5_group(\
                root_group['error'])
    
    def save_basis_sum(self, root_group, basis_links=None,\
        expander_links=None):
        """
        Saves `BaseFitter.basis_sum` using the given root hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group this `BaseFitter` is being saved in
        basis_links : sequence
            list of links to basis functions saved elsewhere (see
            `pylinex.basis.BasisSum.BasisSum.fill_hdf5_group` docs for info
            about accepted formats)
        expander_links : sequence
            list of links to existing saved Expander (see
            `pylinex.basis.BasisSum.BasisSum.fill_hdf5_group` docs for info
            about accepted formats)
        """
        self.basis_sum.fill_hdf5_group(root_group.create_group('basis_sum'),\
            basis_links=basis_links, expander_links=expander_links)
    
    @staticmethod
    def load_basis_sum(root_group):
        """
        Loads the `pylinex.basis.BasisSum.BasisSum` saved in the given hdf5
        file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group in which this `BaseFitter` was saved
        
        Returns
        -------
        basis_sum : `pylinex.basis.BasisSum.BasisSum`
            the basis loaded
        """
        return BasisSum.load_from_hdf5_group(root_group['basis_sum'])
    
    def save_priors(self, root_group, prior_mean_links=None,\
        prior_covariance_links=None):
        """
        Saves priors using the given root hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group in which this `BaseFitter` is being saved
        prior_mean_links : dict
            dictionary of links to existing saved prior means (see
            `distpy.util.h5py_extensions.create_hdf5_dataset` docs for info
            about accepted formats)
        prior_covariance_links : dict
            dictionary of links to existing saved prior covariances (see
            `distpy.util.h5py_extensions.create_hdf5_dataset` docs for info
            about accepted formats)
        """
        if self.has_priors:
            group = root_group.create_group('prior')
            if type(prior_mean_links) is type(None):
                prior_mean_links = {name: None for name in self.names}
            if type(prior_covariance_links) is type(None):
                prior_covariance_links = {name: None for name in self.names}
            for name in self.names:
                key = '{!s}_prior'.format(name)
                if key in self.priors:
                    subgroup = group.create_group(name)
                    self.priors[key].fill_hdf5_group(subgroup,\
                        mean_link=prior_mean_links[name],\
                        covariance_link=prior_covariance_links[name])
    
    @staticmethod
    def load_priors(root_group):
        """
        Loads priors dictionary saved in the given hdf5 file group.
        
        Parameters
        ----------
        root_group : h5py.Group
            the hdf5 file group in which this `BaseFitter` was saved
        
        Returns
        -------
        priors : dict
            the priors that can be supplied when creating a new `BaseFitter`
            (through subclasses)
        """
        priors = {}
        if 'prior' in root_group:
            group = root_group['prior']
            for name in group:
                key = '{!s}_prior'.format(name)
                subgroup = group[name]
                distribution =\
                    GaussianDistribution.load_from_hdf5_group(subgroup)
                priors[key] = distribution
        return priors

