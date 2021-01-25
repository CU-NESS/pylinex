"""
File: pylinex/basis/Basis.py
Author: Keith Tauscher
Update date: 12 Dec 2020

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
from ..util import Savable, Loadable, get_hdf5_value, create_hdf5_dataset,\
    numerical_types, sequence_types, real_numerical_types
from ..expander import Expander, NullExpander, load_expander_from_hdf5_group

class Basis(Savable, Loadable):
    """
    Class surrounding a set of basis vectors and a way to expand those basis
    vectors into a larger data space. It contains methods which generate
    priors, concatenate bases (through the '+' operator), take subsets of the
    basis, and find the overlap of the basis with another.
    """
    def __init__(self, basis_vectors, expander=None, translation=None):
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
        translation: if None, no constant additive is included in the results
                              of calling this Basis.
                     otherwise: should be a 1D numpy.ndarray of length N (see
                                above definition of shape of basis_vectors). It
                                is the (unexpanded) result of calling the basis
                                on a zero-vector of parameters
        """
        self.basis = basis_vectors
        self.expander = expander
        self.translation = translation
    
    def RMS_of_training_set_fits(self, training_set, error=None):
        """
        Finds the RMS vs number of terms when using this basis to fit the
        curves in the given training set.
        
        training_set: training_set to fit, 2D array of shape
                      (num_curves, num_channels)
        error: error vector used to define dot product (usually a 1D array)
               if error is None, function uses error=np.ones(num_channels)
               if error is a single number, function uses that noise level at
                                            every channel
        
        returns: single number representing the RMS residual when fitting the
                 given training set with this basis. If error is None, then
                 this number is given in the same units and scale as the
                 training set itself. If error is a single number noise level
                 or a 1D array of noise levels, then the error is used to
                 normalize residuals before RMS'ing
        """
        if type(training_set) in sequence_types:
            training_set = np.array(training_set)
        else:
            raise TypeError("training_set was not array-like.")
        if type(error) in sequence_types:
            error = np.array(error)
            if error.shape[-1] != self.basis.shape[1]:
                raise ValueError("error was not of length num_channels.")
            if (error.ndim == 2) and (len(error) != len(training_set)):
                raise ValueError("error was two-dimensional but .")
            if error.ndim > 2:
                raise ValueError("error had more than two dimensions.")
        elif type(error) is type(None):
            error = np.ones(self.basis.shape[1])
        elif type(error) in real_numerical_types:
            error = error * np.ones(self.basis.shape[1])
        else:
            raise TypeError("error was not None, a single number, or a 1D " +\
                "array.")
        if training_set.ndim != 2:
            raise ValueError("training_set was not a 2D array.")
        if training_set.shape[1] != self.basis.shape[1]:
            raise ValueError("training_set curves do not have same length " +\
                "as (unexpanded) basis vectors.")
        if error.ndim == 1:
            weighted_basis = self.basis / error[np.newaxis,:]
            weighted_centered_training_set =\
                (training_set - self.translation[np.newaxis,:]) /\
                error[np.newaxis,:]
            inverse_self_overlap =\
                la.inv(np.dot(weighted_basis, weighted_basis.T))
            training_set_overlap =\
                np.dot(weighted_basis, weighted_centered_training_set.T)
            fit_parameters = np.dot(inverse_self_overlap, training_set_overlap).T
            fit_residuals = weighted_centered_training_set -\
                np.dot(fit_parameters, weighted_basis)
        else:
            fit_residuals = []
            for (curve, err) in zip(training_set, error):
                weighted_basis = self.basis / err[np.newaxis,:]
                weighted_centered_curve = (curve - self.translation) / err
                inverse_self_overlap =\
                    la.inv(np.dot(weighted_basis, weighted_basis.T))
                training_set_overlap =\
                    np.dot(weighted_basis, weighted_centered_curve)
                fit_parameters =\
                    np.dot(inverse_self_overlap, training_set_overlap)
                fit_residual = weighted_centered_curve -\
                    np.dot(weighted_basis.T, fit_parameters)
                fit_residuals.append(fit_residual)
            fit_residuals = np.array(fit_residuals)
        return np.sqrt(np.mean(np.power(fit_residuals, 2)))
    
    def RMS_spectrum_of_training_set_fits(self, training_set, error=None):
        """
        Finds the RMS spectrum of this training set when fit with this basis,
        i.e. the RMS of all residuals in the training set after fits with this
        basis.
        
        training_set: training_set to fit, 2D array of shape
                      (num_curves, num_channels)
        error: error vector used to define dot product (usually a 1D array)
               if error is None, function uses error=np.ones(num_channels)
               if error is a single number, function uses that noise level at
                                            every channel
        
        returns: 1D array of RMS's when using np.arange(1 + num_basis_vectors)
                 terms. If error is None, the RMS values are the same units and
                 scale as the training_set. If error is a single number or 1D
                 array, the residuals are normalized by this noise level before
                 being RMS'd
        """
        if type(error) is type(None):
            temporary_error = 1
        elif type(error) in real_numerical_types:
            temporary_error = error
        elif type(error) in sequence_types:
            temporary_error = np.array(error)
            if np.array(error).ndim == 1:
                temporary_error = temporary_error[np.newaxis,:]
            else:
                temporary_error = temporary_error
        else:
            raise ValueError("error was not None, a single number, or a 1D " +\
                "or 2Darray.")
        spectrum = [np.sqrt(np.mean(np.power((training_set -\
            self.translation[np.newaxis,:]) / temporary_error, 2)))]
        for num_terms in range(1, 1 + self.num_basis_vectors):
            spectrum.append(self[:num_terms].RMS_of_training_set_fits(\
                training_set, error=error))
        return np.array(spectrum)
    
    def dot(self, other, error=None):
        """
        Finds the degree of overlap between the two basis objects. The returned
        value can be thought of as the fraction of this basis that can be fit
        by the other basis. Due to its construction,
        basis1.dot(basis2)*basis1.num_basis_vectors is the same as
        basis2.dot(basis1)*basis2.num_basis_vectors. Therefore, this operation
        is not commutative. Note that this operation does not account for the
        translations associated with the two bases.
        
        other: another Basis object with the same number of channels (in the
               expanded space) as this one
        error: error with which to define inner product of basis vectors
        
        returns: a single number in [0,1] indicating the degree of overlap of
                 the two Basis objects. Lower numbers indicate less similarity.
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
        if type(error) is type(None):
            error = np.ones(self.num_larger_channel_set_indices)
        this_expanded_basis = self.expanded_basis / error[np.newaxis,:]
        other_expanded_basis = other.expanded_basis / error[np.newaxis,:]
        overlap_matrix = np.dot(this_expanded_basis, other_expanded_basis.T)
        this_normalization =\
            la.inv(np.dot(this_expanded_basis, this_expanded_basis.T))
        other_normalization =\
            la.inv(np.dot(other_expanded_basis, other_expanded_basis.T))
        to_trace = np.dot(this_normalization, overlap_matrix)
        to_trace = np.dot(to_trace, other_normalization)
        to_trace = np.dot(to_trace, overlap_matrix.T)
        return np.trace(to_trace) / self.num_basis_vectors
    
    def __mul__(self, other):
        """
        Finds the degree of overlap between the two basis objects. The returned
        value can be thought of as the fraction of this basis that can be fit
        by the other basis. Due to its construction,
        (basis1*basis2)*basis1.num_basis_vectors is the same as
        (basis2*basis1)*basis2.num_basis_vectors. Therefore, this operation
        is not commutative. When using the * operator, the inner product
        between basis vectors assume a weighting matrix equal to the identity
        matrix. To use a non-identity weighting matrix, use the Basis.dot
        function instead of the * operator and give an error keyword argument.
        Note that this operation does not account for the translations
        associated with the two bases.
        
        other: another Basis object with the same number of channels (in the
               expanded space) as this one
        
        returns: a single number in [0,1] indicating the degree of overlap of
                 the two Basis objects. Lower numbers indicate less similarity.
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
                new_translation = self.translation + other.translation
                return Basis(new_basis, expander=self.expander,\
                    translation=new_translation)
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
        if type(value) is type(None):
            self._expander = NullExpander()
        elif isinstance(value, Expander):
            self._expander = value
        else:
            raise TypeError("expander property must be set to either None " +\
                            "or an Expander object.")
    
    @property
    def translation(self):
        """
        Property storing the vector that is returned when this basis is called
        with the zero vector as parameters, given in the space of the
        unexpanded channels.
        """
        if not hasattr(self, '_translation'):
            raise AttributeError("translation was referenced before it was " +\
                "set.")
        return self._translation
    
    @translation.setter
    def translation(self, value):
        """
        Setter for the translation vector that is returned when this basis is
        called on a zero-vector of parameters.
        
        value: if None, no constant additive is included in the results of
                        calling this Basis.
               otherwise: should be a 1D numpy.ndarray of length N given by the
                          length of the axis with index 1 of the basis vectors
                          at the heart of this Basis object (i.e. the size of
                          the unexpanded channel space) It is the (unexpanded)
                          result of calling the basis on a zero-vector of
                          parameters
        """
        if type(value) is type(None):
            self._translation = np.zeros(self.basis.shape[1])
        elif type(value) in sequence_types:
            value = np.array(value)
            if value.shape == (self.basis.shape[1],):
                self._translation = value
            else:
                raise ValueError(("translation was set to an array of the " +\
                    "wrong shape. It should be a 1D array of length {0:d}, " +\
                    "but instead it had shape {1}.").format(\
                    self.basis.shape[1], value.shape))
        else:
            raise TypeError("translation was set to neither None nor an " +\
                "array.")
    
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
    
    @property
    def expanded_translation(self):
        """
        Property storing an expanded version of the translation vector. This is
        the vector returned if the Basis is called (with expanded=True) with
        all zero parameters.
        """
        if not hasattr(self, '_expanded_translation'):
            self._expanded_translation = self.expander(self.translation)
        return self._expanded_translation
    
    def __len__(self):
        """
        Alias for the num_basis_vectors property. The implementation of this
        function allows the built-in function len to be called on Basis
        objects.
        
        returns: number of vectors in this basis
        """
        return self.num_basis_vectors
    
    def generate_gaussian_prior(self, curves, error=None,\
        covariance_expansion_factor=1., diagonal=False, epsilon=0):
        """
        Generates a Gaussian prior using this basis and the given curves.
        
        curves: the curves to fit to create the sample with which to create the
                Gaussian prior (in the unexpanded space)
        error: error to use in the least square fit to the given curves. If
               error is None, a raw least square fit is done (default: None)
        covariance_expansion_factor: if a positive number, the factor by which
                                                           the covariance
                                                           matrix should be
                                                           multiplied
                                                           (default: 1)
                                     if None, then covariance is expanded so
                                              that all curves lie within 1
                                              sigma
        diagonal: if True, the off-diagonal components of the covariance matrix
                  are ignored (default: False)
        epsilon: amount to add to diagonal (default, 0)
        """
        if self.rank_deficient:
            raise ValueError("Gaussian priors cannot be generated by rank " +\
                "deficient basis objects (those with more basis vectors " +\
                "than data channels.")
        if type(error) is type(None):
            weighted_basis = self.basis
            weighted_centered_curves = curves - self.translation[np.newaxis,:]
        else:
            weighted_basis = self.basis / error[np.newaxis,:]
            weighted_centered_curves =\
                (curves - self.translation[np.newaxis,:]) / error[np.newaxis,:]
        max_likelihood_parameters = np.dot(np.dot(\
            la.inv(np.dot(weighted_basis, weighted_basis.T)),\
            weighted_basis), weighted_centered_curves.T).T
        mean = np.mean(max_likelihood_parameters, axis=0)
        max_likelihood_parameters =\
            max_likelihood_parameters - mean[np.newaxis,:]
        covariance =\
            np.dot(max_likelihood_parameters.T, max_likelihood_parameters) /\
            (len(curves) - 1)
        if diagonal:
            covariance = np.diag(np.diag(covariance))
        if type(covariance_expansion_factor) is type(None):
            sigma_squareds = np.sum(max_likelihood_parameters *\
                np.dot(max_likelihood_parameters, la.inv(covariance)), axis=1)
            covariance_expansion_factor = np.max(sigma_squareds)
        cov_to_return = covariance * covariance_expansion_factor
        if epsilon != 0:
            if type(epsilon) in numerical_types:
                cov_to_return = cov_to_return +\
                    np.diag(np.ones(self.num_basis_vectors) * epsilon)
            elif type(epsilon) in sequence_types:
                if len(epsilon) == self.num_basis_vectors:
                    cov_to_return = cov_to_return + np.diag(epsilon)
                else:
                    raise ValueError("epsilon was a sequence of a length " +\
                        "different than num_basis_vectors.")
            else:
                raise TypeError("epsilon was neither a number or a " +\
                    "sequence of numbers of length num_basis_vectors.")
        self._gaussian_prior = GaussianDistribution(mean, cov_to_return)
        return self._gaussian_prior
    
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
    def num_channels(self):
        """
        Alias for the num_larger_channel_set_indices property. It is assumed if
        a user is merely asking for a number of channels, they are referring to
        the number of channels after expansion.
        """
        return self.num_larger_channel_set_indices
    
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
        if type(error) is type(None):
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
        if type(error) is type(None):
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
            return np.dot(parameter_values, self.expanded_basis) +\
                self.expanded_translation
        else:
            return np.dot(parameter_values, self.basis) + self.translation
    
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
        return Basis(self.basis[basis_indices], expander=self.expander,\
            translation=self.translation)
    
    def __getitem__(self, basis_indices_to_keep):
        """
        Allows for square bracket indexing to create a subbasis.
        
        basis_indices_to_keep: the indices of the basis vectors to include in
                               the subbasis
        
        returns: a Basis object with only the given basis vectors
        """
        return Basis.subbasis(self, basis_indices_to_keep)
    
    def fill_hdf5_group(self, group, basis_link=None, expander_link=None,\
        translation_link=None):
        """
        Fills the given hdf5 file group with data about this basis.
        Particularly, this function records the basis array and the expander.
        
        group: hdf5 file group to fill
        basis_link: link to basis already in file, if it exists
        expander_link: link to expander already in file, if it exists
        translation_link: link to translation already in file, if it exists
        """
        group.attrs['class'] = 'Basis'
        create_hdf5_dataset(group, 'basis', data=self.basis, link=basis_link)
        try:
            create_hdf5_dataset(group, 'expander', link=expander_link)
        except ValueError:
            self.expander.fill_hdf5_group(group.create_group('expander'))
        create_hdf5_dataset(group, 'translation', data=self.translation,\
            link=translation_link)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a Basis object from the given hdf5 group.
        
        group: hdf5 file group from which to load a Basis object
        
        returns: a Basis which was stored in the given hdf5 file group
        """
        basis = get_hdf5_value(group['basis'])
        expander = load_expander_from_hdf5_group(group['expander'])
        if 'translation' in group:
            translation = get_hdf5_value(group['translation'])
        else:
            translation = None
        return Basis(basis, expander=expander, translation=translation)
    
    def plot(self, basis_indices=slice(None), x_values=None,\
        correct_sign=False, include_translation=False,\
        matplotlib_function='plot', title='Basis', xlabel=None, ylabel=None,\
        fontsize=20, fig=None, ax=None, figsize=(12, 9), show=False, **kwargs):
        """
        Plots the basis vectors stored in this Basis object.
        
        basis_indices: the indices of the basis vectors to include in the plot.
                       Default: slice(None) (all basis vectors are included)
        x_values: the x values to use in plotting the basis, defaults to
                  np.arange(num_channels) if x_values is None
        correct_sign: if True, the first channel of each plotted basis vector
                      is forced to be non-negative by multiplying by a negative
                      sign if necessary
        include_translation: if True, the translation vector is plotted before
                                      the basis vectors (default False)
        matplotlib_function: type of plot to make, either 'plot' or 'scatter'
        title: title of the plot. Default: 'Basis'
        xlabel, ylabel: labels for the x and y axes
        fontsize: size of fonts for title and labels
        fig: existing figure on which to put plot, if it exists (default None)
        ax: existing axes on which to put plot, if they exist (default None)
        figsize: size of figure to create if no such figure exists already
        show: if True, matplotlib.pyplot.show() is called before this function
              returns
        **kwargs: keyword arguments to pass to matplotlib_function
        """
        supported_matlpotlib_functions = ['plot', 'scatter']
        if type(x_values) is type(None):
            x_values = np.arange(self.num_smaller_channel_set_indices)
        if (type(fig) is type(None)) and (type(ax) is type(None)):
            fig = pl.figure(figsize=figsize)
        if type(ax) is type(None):
            ax = fig.add_subplot(111)
        if matplotlib_function == 'plot':
            sign_correction =\
                (np.where(self.basis[:,0] >= 0, 1, -1)[:,np.newaxis]\
                if correct_sign else 1)
            if include_translation:
                ax.plot(x_values, self.translation, **kwargs)
            ax.plot(x_values,\
                (self.basis * sign_correction)[basis_indices].T, **kwargs)
        elif matplotlib_function == 'scatter':
            (minimum, maximum) = (np.inf, -np.inf)
            if include_translation:
                ax.scatter(x_values, self.translation, **kwargs)
            for vector in self.basis[basis_indices]:
                sign_correction =\
                    ((1 if (vector[0] >= 0) else (-1)) if correct_sign else 1)
                ax.scatter(x_values, vector * sign_correction, **kwargs)
                minimum = min(minimum, np.min(vector))
                maximum = max(maximum, np.max(vector))
            ax.set_ylim((minimum, maximum))
        else:
            raise ValueError("matplotlib_function, which was set to " +\
                "'{0!s}', was not one of the supported types, {1}.".format(\
                matplotlib_function, supported_matplotlib_functions))
        ax.plot(x_values, np.zeros_like(x_values), linewidth=1, color='k')
        ax.set_xlim((x_values[0], x_values[-1]))
        if type(title) is not type(None):
            ax.set_title(title, size=fontsize)
        if type(xlabel) is not type(None):
            ax.set_xlabel(xlabel, size=fontsize)
        if type(ylabel) is not type(None):
            ax.set_ylabel(ylabel, size=fontsize)
        ax.tick_params(labelsize=fontsize, width=2.5, length=7.5,\
            which='major')
        ax.tick_params(labelsize=fontsize, width=1.5, length=4.5,\
            which='minor')
        if show:
            pl.show()
        else:
            return ax
    
    def change_expander(self, new_expander):
        """
        Returns a Basis object with the same basis vectors and translation but
        with a different expander.
        """
        return Basis(self.basis, expander=new_expander,\
            translation=self.translation)
    
    def expanderless(self):
        """
        Returns a Basis object with the same basis vectors and translation but
        with no expander.
        """
        return self.change_expander(None)
    
    def change_translation(self, new_translation):
        """
        Returns a Basis object with the same basis vectors and expander but
        with a different translation vector.
        """
        return Basis(self.basis, expander=self.expander,\
            translation=new_translation)
    
    def scale(self, scale_factor):
        """
        Scales the basis vectors and translation of this basis, so that the
        same parameters lead to vectors that are scaled by the given scale
        factor.
        
        scale_factor: factor by which basis and translation vectors are
                      multiplied
        
        returns: a new Basis object that is scaled by the given factor
        """
        return Basis(self.basis * scale_factor, expander=self.expander,\
            translation=self.translation * scale_factor)
    
    def copy(self):
        """
        Creates and returns a deep copy of this basis (except it is guaranteed
        to have Basis class).
        
        returns: a Basis object with the same (but copied to a different RAM
                 location) basis array and expander as this one. It will have
                 the base Basis class, however, not the same class as self.
        """
        return Basis(self.basis.copy(), expander=self.expander.copy(),\
            translation=self.translation.copy())
    
    def __eq__(self, other):
        """
        Checks if other represents the same basis as this one. However, this
        method ignores class differences.
        
        other: object with which to check for equality
        
        returns: True if other has the same basis vectors and expander.
                 False otherwise
        """
        if isinstance(other, Basis):
            if self.expander == other.expander:
                if self.basis.shape == other.basis.shape:
                    basis_close =\
                        np.allclose(self.basis, other.basis, rtol=1e-6, atol=0)
                    translation_close = np.allclose(self.translation,\
                        other.translation, rtol=1e-6, atol=0)
                    return (basis_close and translation_close)
                else:
                    return False
            else:
                return False
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

