"""
File: plyinex/model/BasisFitModel.py
Author: Keith Tauscher
Date: 4 Jun 2019

Description: File containing class representing a model based on a basis or
             bases that could be truncated. The only parameter(s) of the model
             is (are) the number of terms to include in each basis.
"""
import numpy as np
from ..util import create_hdf5_dataset, get_hdf5_value, numerical_types,\
    sequence_types
from ..basis import Basis, BasisSum
from ..fitter import Fitter
from .LoadableModel import LoadableModel

class BasisFitModel(LoadableModel):
    """
    A class representing a model based on a basis or bases that could be
    truncated. The only parameter(s) of the model is (are) the number of terms
    to include in each basis.
    """
    def __init__(self, basis_sum, data, error=None):
        """
        Creates a new BasisFitModel with the given basis sum, data, and error.
        
        basis_sum: a BasisSum object containing vectors with which to fit data
        data: data vector to use when conditionalizing. Ideally, all outputs of
              this model should be similar to this data.
        error: noise level of the data,
               if None, the noise level is 1 everywhere
        """
        self.basis_sum = basis_sum
        self.data = data
        self.error = error
    
    @property
    def basis_sum(self):
        """
        Property storing the basis sum containing the vectors that are used to
        fit data.
        """
        if not hasattr(self, '_basis_sum'):
            raise AttributeError("basis_sum was referenced before it was set.")
        return self._basis_sum
    
    @basis_sum.setter
    def basis_sum(self, value):
        """
        Setter for the basis sum containing vectors with which to fit data.
        
        value: either a BasisSum object, or a single basis object
        """
        if isinstance(value, BasisSum):
            self._basis_sum = value
        elif isinstance(value, Basis):
            self._basis_sum = BasisSum(['sole'], [value])
        else:
            raise TypeError("basis_sum was neither a Basis or a BasisSum " +\
                "object.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.basis_sum.num_channels
        return self._num_channels
    
    @property
    def data(self):
        """
        Property storing the data vector to use when conditionalizing. Ideally,
        all results from this model are similar to this data vector.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data was referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data vector to use when conditionalizing.
        
        value: 1D numpy.ndarray object
        """
        if type(value) in sequence_types:
            if len(value) == self.num_channels:
                if all([(type(element) in numerical_types)\
                    for element in value]):
                    self._data = np.array(value)
                else:
                    raise TypeError("data was set to a sequence whose " +\
                        "elements are not numbers.")
            else:
                raise ValueError("data does not have same length as basis " +\
                    "vectors.")
        else:
            raise TypeError("data was set to a non-sequence.")
    
    @property
    def error(self):
        """
        Property storing the noise level on the data vector.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the noise level on the data vector.
        
        value: 1D numpy.ndarray object of same length as data vector
        """
        if type(value) is type(None):
            self._error = np.ones_like(self.data)
        elif type(value) in sequence_types:
            if len(value) == self.num_channels:
                if all([(type(element) in numerical_types)\
                    for element in value]):
                    self._error = np.array(value)
                else:
                    raise TypeError("error was set to a sequence whose " +\
                        "elements are not numbers.")
            else:
                raise ValueError("error does not have same length as data.")
        else:
            raise TypeError("error was set to a non-sequence.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters =\
                ['{!s}_nterms'.format(name) for name in self.basis_sum.names]
        return self._parameters
    
    @property
    def bounds(self):
        """
        Property storing the bounds of the parameters, taken from the bounds of
        the submodels.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {'{!s}_nterms'.format(name):\
                (1, self.basis_sum[name].num_basis_vectors)\
                for name in self.basis_sum.names}
        return self._bounds
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values, which should be
                    integers
        
        returns: array of size (num_channels,)
        """
        int_parameters = parameters.astype(int)
        if parameters != int_parameters:
            raise ValueError("Some of the parameters given to a " +\
                "BasisFitModel were not integers (i.e. after casting to " +\
                "integers, the parameters were not equal to their uncasted " +\
                "selves).")
        basis_sum_to_use =\
            self.basis_sum[dict(zip(self.basis_sum.names, parameters))]
        return\
            Fitter(basis_sum_to_use, self.data, error=self.error).channel_mean
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return False
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable.
        """
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'BasisFitModel'
        self.basis_sum.fill_hdf5_group(group.create_group('basis_sum'))
        create_hdf5_dataset(group, 'data', data=self.data)
        create_hdf5_dataset(group, 'error', data=self.error)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a BasisFitModel from the given hdf5 group.
        
        group: h5py.Group object from which to load a BasisFitModel
        
        returns: a BasisFitModel that was previously saved in the given group
        """
        try:
            assert('BasisFitModel' == group.attrs['class'])
        except:
            raise ValueError("The given group does not seem to contain a " +\
                "BasisFitModel.")
        basis_sum = BasisSum.load_from_hdf5_group(group)
        data = get_hdf5_value(group['data'])
        error = get_hdf5_value(group['error'])
        return BasisFitModel(basis_sum, data, error=error)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this BasisFitModel.
        
        other: object to check for equality
        
        returns: False unless other is a BasisFitModel with the same
                 basis sum, data, and error
        """
        if isinstance(other, BasisFitModel):
            if self.basis_sum == other.basis_sum:
                return (np.all(self.data == other.data) and\
                    np.all(self.error == other.error))
            else:
                return False
        else:
            return False

