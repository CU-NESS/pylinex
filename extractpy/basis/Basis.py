"""
File: extractpy/basis/Basis.py
Author: Keith Tauscher
Date: 26 Aug 2017

Description: File containing a class which represents a single set of basis
             vectors.
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl
from distpy import GaussianDistribution
from ..util import Savable
from ..expander import Expander, NullExpander, load_expander_from_hdf5_group

class Basis(Savable):
    """
    Class surrounding a set of basis vectors and a way to expand those basis
    vectors into a larger data space. It contains methods which generate
    priors, concatenate bases (through the '+' operator), take subsets of the
    basis, and find the overlap of the basis with another.
    """
    def __init__(self, basis_vectors, expander=None, normed_importances=None):
        """
        Initializes this Basis with the given basis vectors and expander.
        
        basis_vectors: 2D numpy.ndarray of shape (k,N) where k is the number of
                       basis vectors and N is the number of data channels in
                       the smaller (i.e. unexpanded) channel set
        expander: if None, no expansion is performed
                  otherwise, expander must be Expander object
        normed_importances: (optional) the importances of the constituent modes
                            normed so that they add to 1. Default: None.
        """
        self.basis = basis_vectors
        self.expander = expander
        self.normed_importances = normed_importances
    
    @property
    def normed_importances(self):
        """
        Property storing the importances of each mode. This is a 1D array of
        nonnegative numbers. If none is provided at initialization, then all of
        the importances are set to 0.
        """
        if not hasattr(self, '_normed_importances'):
            self._normed_importances = np.zeros(self.num_basis_vectors)
        return self._normed_importances
    
    @normed_importances.setter
    def normed_importances(self, value):
        """
        Allows the user to supply the importances of the basis vectors.
        
        value: must be a 1D numpy.ndarray of non-negative real numbers. The
               array should have an element for each basis vector.
        """
        if value is not None:
            value = np.array(value)
            if value.shape == (self.num_basis_vectors,):
                self._normed_importances = value
            else:
                raise ValueError("normed_importances was not a " +\
                                 "numpy.ndarray with the same length as " +\
                                 "the number of basis vectors.")
    
    @property
    def total_normed_importance(self):
        """
        Property storing the sum of the importances of all modes.
        """
        if not hasattr(self, '_total_normed_importance'):
            self._total_normed_importance = np.sum(self.normed_importances)
        return self._total_normed_importance
    
    def dot(self, other, error=None):
        """
        Finds the degree of overlap between the two basis objects.
        
        other: another Basis object 
        
        returns: a number in [0,1) indicating the degree of overlap of the two
                 Basis objects.
        """
        if not isinstance(other, Basis):
            raise TypeError("Basis objects can only be multiplied by other " +\
                            "Basis objects.")
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
    
    def __add__(self, other):
        """
        Allows for the addition of two Basis objects.
        
        other: another Basis object
        
        returns: Basis object with the basis vectors from both of the
                 constituent objects
        """
        if isinstance(other, Basis):
            if self.expander == other.expander:
                new_basis = np.concatenate([self.basis, other.basis], axis=0)
                new_importances = np.concatenate(\
                    [self.normed_importances, other.normed_importances])
                return Basis(new_basis, expander=self.expander,\
                    normed_importances=new_importances)
            else:
                raise NotImplementedError("Two basis objects cannot be " +\
                                          "added together when their " +\
                                          "expanders are not identical.")
        else:
            raise TypeError("Cannot add Basis to object which is not " +\
                            "another Basis.")
    
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
        
        value: a 2D numpy.ndarray of shape (k,N) where k<N
        """
        value = np.array(value)
        if value.ndim == 2:
            if value.shape[0] <= value.shape[1]:
                self._basis = value
            else:
                raise ValueError("The number of basis functions (1st " +\
                                 "dimension of given basis) can not be " +\
                                 "greater than the dimension of the space " +\
                                 "in which the basis vectors lie (2nd " +\
                                 "dimension of given basis).")
        else:
            raise ValueError("basis given to Basis class was not 2D. It " +\
                             "should be of shape (Nbasis, Nchannel).")
    
    @property
    def num_basis_vectors(self):
        """
        Property storing the number of basis vectors stored in this object.
        """
        return self.basis.shape[0]
    
    @property
    def expander(self):
        """
        Property storing the expander of this basis. It is an Expander object
        which expands the data from the space of the basis vectors to the space
        of the data which they are meant to fit.
        """
        if not hasattr(self, '_expander'):
            raise AttributeError("expander was referenced before it was set.")
        return self._expander
    
    @expander.setter
    def expander(self, value):
        """
        The function called when the expander is set.
        
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
        Property storing the basis after it is expanded by the expander.
        """
        if not hasattr(self, '_expanded_basis'):
            self._expanded_basis = np.array([self.expander(self.basis[i])\
                for i in xrange(self.num_basis_vectors)])
        return self._expanded_basis
    
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
        generated using the Basis.generate_gaussian_prior() method.
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
    
    def get_value(self, parameter_values, expanded=True):
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
        if expanded:
            return np.dot(parameter_values.T, self.expanded_basis)
        else:
            return np.dot(parameter_values.T, self.basis)
    
    def __call__(self, parameter_values, expanded=True):
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
        new_basis = self.basis[basis_indices]
        new_normed_importances = self.normed_importances[basis_indices]
        return Basis(new_basis, expander=self.expander,\
            normed_importances=new_normed_importances)
    
    def __getitem__(self, basis_indices_to_keep):
        """
        Allows for square bracket indexing to create a subbasis.
        
        basis_indices_to_keep: the indices of the basis vectors to include in
                               the subbasis
        
        returns: a Basis object with only the given basis vectors
        """
        return Basis.subbasis(self, basis_indices_to_keep)
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this basis.
        Particularly, this function records the basis array and the expander.
        
        group: hdf5 file group to fill       
        """
        group.attrs['class'] = 'Basis'
        group.create_dataset('basis', data=self.basis)
        self.expander.fill_hdf5_group(group.create_group('expander'))
    
    def plot(self, basis_indices=slice(None), title='Basis', x_values=None,\
        show=True, fig=None, ax=None, **kwargs):
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
        if (fig is None) or (ax is None):
            fig = pl.figure()
            ax = fig.add_subplot(111)
        ax.plot(x_values, self.basis[basis_indices].T, **kwargs)
        ax.plot(x_values, np.zeros_like(x_values), linewidth=1, color='k')
        ax.set_title(title)
        if show:
            pl.show()

def load_basis_from_hdf5_group(group):
    """
    Allows for Basis objects to be read from hdf5 file groups.
    
    group: hdf5 file group from which to load the basis
    
    returns: Basis object stored in the hdf5 group
    """
    basis = group['basis'].value
    expander = load_expander_from_hdf5_group(group['expander'])
    return Basis(basis, expander=expander)

