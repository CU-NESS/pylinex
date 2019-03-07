"""
File: pylinex/basis/Basis.py
Author: Keith Tauscher
Update date: 10 Apr 2018

Description: File containing a class which represents a single set of basis
             vectors. It allows for memory-efficient expansion of the data
             (e.g. repetition, modulation, FFT, etc.) through the use of the
             pylinex.expander module. It also has convenience methods which
             compute projection matrices, gram matrices, and values at specific
             points in parameter space. Basis can also be concatenated through
             the use of the '+' operator and a subbasis can be taken using
             square bracket notation.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from ..util import Savable, Loadable, get_hdf5_value, create_hdf5_dataset
from ..expander import Expander, NullExpander, load_expander_from_hdf5_group

class Basis(Savable, Loadable):
    """
    Class surrounding a set of basis vectors and a way to expand those basis
    vectors into a larger data space. It contains methods which generate
    priors, concatenate bases (through the '+' operator), take subsets of the
    basis, and find the overlap of the basis with another.
    """
    def __init__(self, basis_vectors, expander=None):
        """
        Initializes this Basis with the given basis vectors and expander.
        
        basis_vectors: 2D numpy.ndarray of shape (k,N) where k is the number of
                       basis vectors and N is the number of data channels in
                       the smaller (i.e. unexpanded) channel set
        expander: if None, no expansion is performed and value of basis at each
                           parameter space point is of the same length as the
                           basis vectors themselves
                  otherwise, expander must be an instance of pylinex's
                             Expander class.
        """
        self.basis = basis_vectors
        self.expander = expander
    
    def dot(self, other, error=None):
        """
        Finds the degree of overlap between the two basis objects.
        
        other: another Basis object with the same number of channels (in the
               expanded space) as this one
        
        returns: a single number in [0,1) indicating the degree of overlap of
                 the two Basis objects. This number gives a measure of how
                 successful a simultaneous least square fit with the two bases
                 and the given data error would be.
        """
        if not isinstance(other, Basis):
            raise TypeError("Basis objects can only be multiplied by other " +\
                            "Basis objects.")
        if self.rank_deficient or other.rank_deficient:
            raise ValueError("The overlap between bases cannot be defined " +\
                "unless they are both full rank.")
        if self.num_larger_channel_set_indices !=\
            other.num_larger_channel_set_indices:
            raise ValueError("The number of channels in the expanded bases " +\
                             "of the two Basis objects were not equal.")
        if error is None:
            error = np.ones(self.num_larger_channel_set_indices)
        this_expanded_basis = self.expanded_basis / error[np.newaxis,:]
        other_expanded_basis = other.expanded_basis / error[np.newaxis,:]
        overlap_matrix = np.dot(this_expanded_basis, other_expanded_basis.T)
        this_normalization =\
            la.inv(np.dot(this_expanded_basis, this_expanded_basis.T))
        other_normalization =\
            la.inv(np.dot(other_expanded_basis, other_expanded_basis.T))
        to_trace = np.dot(overlap_matrix, other_normalization)
        to_trace = np.dot(this_normalization, to_trace)
        to_trace = np.dot(overlap_matrix.T, to_trace)
        return np.trace(to_trace)
    
    def __mul__(self, other):
        """
        Finds the degree of overlap using the dot() function between this Basis
        and other. When the '*' operator is used, the error in the data is
        assumed to be flat across the different data channels
        
        other: another Basis object with the same number of channels (in the
               expanded space) as this one
        
        returns: a single number in [0,1) indicating the degree of overlap of
                 the two Basis objects. This number gives a measure of how
                 successful a simultaneous least square fit with the two bases
                 and a flat error spectrum would be.
        """
        return self.dot(other)
    
    def combine(self, other):
        """
        Combines this basis with other.
        
        other: Basis object with the same exact expander and number of channels
               as this one
        
        returns: Basis object including the basis vectors from both of the
                 constituent objects
        """
        if isinstance(other, Basis):
            expanders_equal = (self.expander == other.expander)
            nscsi_equal = (self.num_smaller_channel_set_indices ==\
                other.num_smaller_channel_set_indices)
            if expanders_equal and nscsi_equal:
                new_basis = np.concatenate([self.basis, other.basis], axis=0)
                return Basis(new_basis, expander=self.expander)
            else:
                raise NotImplementedError("Two basis objects cannot be " +\
                                          "combined when their expanders " +\
                                          "and basis vector lengths are " +\
                                          "not both identical.")
        else:
            raise TypeError("Cannot add Basis to object which is not " +\
                            "another Basis.")
    
    def __add__(self, other):
        """
        Allows for the combination of two Basis objects using the '+' operator.
        
        other: another Basis object with the same exact expander and number of
               channels in the smaller (i.e. unexpanded) space
        
        returns: Basis object including the basis vectors from both of the
                 constituent objects
        """
        return self.combine(other)
    
    @property
    def basis(self):
        """
        Property storing the actual basis vectors. It is a 2D numpy.ndarray of
        shape (k,N) where k is the number of basis vectors and N is the number
        of data channels.
        """
        if not hasattr(self, '_basis'):
            raise AttributeError("basis was referenced before it was set.")
        return self._basis
    
    @basis.setter
    def basis(self, value):
        """
        Allows user to supply basis vectors as a 2D numpy.ndarray.
        
        value: a 2D numpy.ndarray of shape (k,N), where k<=N for
               non-rank-deficient bases (although k>N is allowed)
        """
        value = np.array(value)
        if value.ndim == 2:
            self._basis = value
        else:
            raise ValueError("basis given to Basis class was not 2D. It " +\
                             "should be of shape (Nbasis, Nchannel).")
    
    @property
    def rank_deficient(self):
        """
        Property storing a boolean describing whether or not this Basis is rank
        deficient (true if and only if there are more basis vectors than data
        channels). Note that, technically speaking, a basis may be rank
        deficient even if it contains fewer vectors than data channels if the
        basis vectors are degenerate. That degeneracy is not handled here.
        """
        if not hasattr(self, '_rank_deficient'):
            self._rank_deficient = (self.basis.shape[0] > self.basis.shape[1])
        return self._rank_deficient
    
    @property
    def num_basis_vectors(self):
        """
        Property storing the number of basis vectors stored in this object.
        """
        if not hasattr(self, '_num_basis_vectors'):
            self._num_basis_vectors = self.basis.shape[0]
        return self._num_basis_vectors
    
    @property
    def expander(self):
        """
        Property storing the expander of this basis. It is an Expander object
        from the extactpy.expander submodule which expands the data from the
        space of the basis vectors to the space of the data which they are
        meant to fit.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        Allows user to set Expander object used by this Basis.
        
        value: if None, no expansion is performed
               otherwise, value must be an Expander object which expands the
                          data from the space of the basis vectors to the space
                          of the data which they are meant to fit.
        """
        if value is None:
            self._expander = NullExpander()
        elif isinstance(value, Expander):
            self._expander = value
        else:
            raise TypeError("expander property must be set to either None " +\
                            "or an Expander object.")
    
    @property
    def expanded_basis(self):
        """
        Property storing the basis vectors after they have been expanded by the
        expander.
        """
        if not hasattr(self, '_expanded_basis'):
            self._expanded_basis = np.array([self.expander(self.basis[index])\
                for index in range(len(self))])
        return self._expanded_basis
    
    def __len__(self):
        """
        Alias for the num_basis_vectors property. The implementation of this
        function allows the built-in function len to be called on Basis
        objects.
        
        returns: number of vectors in this basis
        """
        return self.num_basis_vectors
    
    def generate_gaussian_prior(self, curves, error=None,\
        covariance_expansion_factor=1., diagonal=False):
        """
        Generates a Gaussian prior using this basis and the given curves.
        
        curves: the curves to fit to create the sample with which to create the
                Gaussian prior
        error: error to use in the least square fit to the given curves. If
               error is None, a raw least square fit is done (default: None)
        covariance_expansion_factor: the factor by which the covariance matrix
                                     should be multiplied (default: 1)
        diagonal: if True, the off-diagonal components of the covariance matrix
                  are ignored (default: False)
        """
        if self.rank_deficient:
            raise ValueError("Gaussian priors cannot be generated by rank " +\
                "deficient basis objects (those with more basis vectors " +\
                "than data channels.")
        if error is None:
            weighted_basis = self.basis
            weighted_curves = curves
        else:
            weighted_basis = self.basis / error[np.newaxis,:]
            weighted_curves = curves / error[np.newaxis,:]
        max_likelihood_parameters = np.dot(np.dot(\
            la.inv(np.dot(weighted_basis, weighted_basis.T)),\
            weighted_basis), weighted_curves.T).T
        mean = np.mean(max_likelihood_parameters, axis=0)
        max_likelihood_parameters =\
            max_likelihood_parameters - mean[np.newaxis,:]
        covariance =\
            np.dot(max_likelihood_parameters.T, max_likelihood_parameters) /\
            (len(curves) - 1)
        cov_to_return = covariance * covariance_expansion_factor
        if diagonal:
            cov_to_return = np.diag(np.diag(cov_to_return))
        self._gaussian_prior = GaussianDistribution(mean, cov_to_return)
    
    @property
    def gaussian_prior(self):
        """
        Property storing the GaussianDistribution of this Basis if it has been
        generated using the Basis.generate_gaussian_prior() method. If not, an
        AttributeError is raised.
        """
        if not hasattr(self, '_gaussian_prior'):
            raise AttributeError("gaussian_prior was referenced before it " +\
                                 "was generated. Call " +\
                                 "basis.generate_gaussian_prior(curves) " +\
                                 "with some training set curves to " +\
                                 "generate one.")
        return self._gaussian_prior
    
    @property
    def num_smaller_channel_set_indices(self):
        """
        Property storing the number of channels in the basis vectors.
        """
        if not hasattr(self, '_num_smaller_channel_set_indices'):
            self._num_smaller_channel_set_indices = self.basis.shape[1]
        return self._num_smaller_channel_set_indices
    
    @property
    def num_larger_channel_set_indices(self):
        """
        Property storing the number of channels in the expanded basis vectors.
        """
        if not hasattr(self, '_num_larger_channel_set_indices'):
            test_vector = np.zeros(self.num_smaller_channel_set_indices)
            self._num_larger_channel_set_indices =\
                len(self.expander(test_vector))
        return self._num_larger_channel_set_indices
    
    @property
    def euclidean_gram_matrix(self):
        """
        Property storing the standard Gram matrix of this set of basis vectors
        (i.e. the one with no weighting of the data). The property takes the
        form of a symmetric 2D numpy.ndarray with dimension length
        num_basis_vectors.
        """
        if not hasattr(self, '_euclidean_gram_matrix'):
            self._euclidean_gram_matrix = np.dot(self.basis, self.basis.T)
        return self._euclidean_gram_matrix
    
    def euclidean_projection_matrix(self):
        """
        Property storing matrix which projects any given vector into the space
        defined by this basis (using the Euclidean data-weighting).
        """
        if not hasattr(self, '_euclidean_projection_matrix'):
            if self.rank_deficient:
                raise ValueError("The projection matrix cannot be defined " +\
                    "for rank deficient Basis objects (those with more " +\
                    "basis vectors than data channels).")
            inverse_gram_matrix = la.inv(self.euclidean_gram_matrix)
            self._euclidean_projection_matrix =\
                np.dot(np.dot(self.basis.T, inverse_gram_matrix), self.basis)
        return self._euclidean_projection_matrix
    
    def gram_matrix(self, error=None):
        """
        Finds the Gram matrix when the given error is used to weight the basis
        vectors.
        
        error: the 1D array full of errors with which to weight the basis
               vectors. error can also be None (its default value), in which
               case the euclidean gram matrix is returned.
        
        returns: symmetric 2D numpy.ndarray of shape (num_basis_vectors,)*2
                 containing the weighted dot products of the vectors.
        """
        if error is None:
            return self.euclidean_gram_matrix
        elif len(error) == self.num_larger_channel_set_indices:
            weighted_expanded_basis = self.expanded_basis / error[np.newaxis,:]
            return np.dot(weighted_expanded_basis, weighted_expanded_basis.T)
        elif len(error) == self.num_smaller_channel_set_indices:
            weighted_basis = self.basis / error[np.newaxis,:]
            return np.dot(weighted_basis, weighted_basis.T)
        else:
            raise ValueError("error with which to compute Gram matrix was " +\
                             "neither of the expected lengths.")
    
    def projection_matrix(self, error=None):
        """
        Finds the matrix which projects any given vector into the space of this
        set of basis vectors (when the dot product is defined through the given
        error).
        
        error: the 1D array full of errors with which to weight the basis
               vectors. error can also be None (its default value), in which
               case the euclidean gram matrix is returned.
        
        returns: square 2D numpy.ndarray of dimension length same as error
        """
        if self.rank_deficient:
            raise ValueError("The projection matrix cannot be defined " +\
                    "for rank deficient Basis objects (those with more " +\
                    "basis vectors than data channels).")
        if error is None:
            return self.euclidean_projection_matrix
        elif len(error) == self.num_larger_channel_set_indices:
            inverse_gram_matrix = la.inv(self.gram_matrix(error=error))
            double_weighted_basis =\
                self.expanded_basis / (error[np.newaxis,:] ** 2)
            return np.dot(np.dot(self.expanded_basis.T, inverse_gram_matrix),\
                double_weighted_basis)
        elif len(error) == self.num_smaller_channel_set_indices:
            inverse_gram_matrix = la.inv(self.gram_matrix(error=error))
            double_weighted_basis = self.basis / (error[np.newaxis,:] ** 2)
            return np.dot(np.dot(self.basis.T, inverse_gram_matrix),\
                double_weighted_basis)
        else:
            raise ValueError("error with which to compute projection " +\
                             "matrix was neither of the expected lengths.")
    
    def get_value(self, parameter_values, expanded=False):
        """
        Gets the value of this basis given a specific set of coefficients.
        
        parameter values: 1 or 2 dimensional numpy.ndarray of shape (...,k)
                          where k is the number of basis vectors in this object
                          containing the coefficients by which each basis
                          vector should be multiplied. 
        expanded: Boolean determining whether the basis vectors alone should be
                  used (False) or the expanded basis vectors should be used
                  (True). Default: True
        
        returns: 1D numpy.ndarray of length of the (expanded if expanded True)
                 basis vectors
        """
        if expanded:
            return np.dot(parameter_values, self.expanded_basis)
        else:
            return np.dot(parameter_values, self.basis)
    
    def __call__(self, parameter_values, expanded=False):
        """
        Gets the value of this basis given a specific set of coefficients.
        
        parameter values: 1D numpy.ndarray of length k where k is the number of
                          basis vectors in this object containing the
                          coefficients by which each basis vector should be
                          multiplied.
        expanded: Boolean determining whether the basis vectors alone should be
                  used (False) or the expanded basis vectors should be used
                  (True). Default: True
        
        returns: 1D numpy.ndarray of length of the (expanded if expanded True)
                 basis vectors
        """
        return Basis.get_value(self, parameter_values, expanded=expanded)
    
    def subbasis(self, basis_indices):
        """
        basis_indices: indices of the basis vectors to include in the subbasis
        
        returns: a Basis object with only the given basis vectors
        """
        return Basis(self.basis[basis_indices], expander=self.expander)
    
    def __getitem__(self, basis_indices_to_keep):
        """
        Allows for square bracket indexing to create a subbasis.
        
        basis_indices_to_keep: the indices of the basis vectors to include in
                               the subbasis
        
        returns: a Basis object with only the given basis vectors
        """
        return Basis.subbasis(self, basis_indices_to_keep)
    
    def fill_hdf5_group(self, group, basis_link=None, expander_link=None):
        """
        Fills the given hdf5 file group with data about this basis.
        Particularly, this function records the basis array and the expander.
        
        group: hdf5 file group to fill       
        """
        group.attrs['class'] = 'Basis'
        create_hdf5_dataset(group, 'basis', data=self.basis, link=basis_link)
        try:
            create_hdf5_dataset(group, 'expander', link=expander_link)
        except ValueError:
            self.expander.fill_hdf5_group(group.create_group('expander'))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Basis object from the given hdf5 group.
        
        group: hdf5 file group from which to load a Basis object
        
        returns: a Basis which was stored in the given hdf5 file group
        """
        basis = get_hdf5_value(group['basis'])
        expander = load_expander_from_hdf5_group(group['expander'])
        return Basis(basis, expander=expander)
    
    def plot(self, basis_indices=slice(None), x_values=None, title='Basis',\
        xlabel=None, ylabel=None, fontsize=20, fig=None, ax=None,\
        figsize=(12, 9), show=True, **kwargs):
        """
        Plots the basis vectors stored in this Basis object.
        
        basis_indices: the indices of the basis vectors to include in the plot.
                       Default: slice(None) (all basis vectors are included)
        title: title of the plot. Default: 'Basis'
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to pass to matplotlib.pyplot.plot()
        """
        if x_values is None:
            x_values = np.arange(self.num_smaller_channel_set_indices)
        if (fig is None) and (ax is None):
            fig = pl.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)
        ax.plot(x_values, self.basis[basis_indices].T, **kwargs)
        ax.plot(x_values, np.zeros_like(x_values), linewidth=1, color='k')
        ax.set_xlim((x_values[0], x_values[-1]))
        if title is not None:
            ax.set_title(title, size=fontsize)
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return ax
    
    def copy(self):
        """
        Creates and returns a deep copy of this basis (except it is guaranteed
        to have Basis class).
        
        returns: a Basis object with the same (but copied to a different RAM
                 location) basis array and expander as this one. It will have
                 the base Basis class, however, not the same class as self.
        """
        return Basis(self.basis.copy(), self.expander.copy())
    
    def __eq__(self, other):
        """
        Checks if other represents the same basis as this one. However, this
        method ignores class differences.
        
        other: object with which to check for equality
        
        returns: True if other has the same basis vectors and expander.
                 False otherwise
        """
        if isinstance(other, Basis):
            basis_vectors_equal =\
                np.allclose(self.basis, other.basis, rtol=1e-6, atol=0)
            expanders_equal = (self.expander == other.expander)
            return (basis_vectors_equal and expanders_equal)
        else:
            return False
    
    def __ne__(self, other):
        """
        Checks whether other is a functionally different Basis than this one.
        This function enforces (self != other) == (not (self == other)).
        
        other: object with which to check for inequality
        
        returns: the opposite of __eq__ called with same arguments
        """
        return (not self.__eq__(other))

