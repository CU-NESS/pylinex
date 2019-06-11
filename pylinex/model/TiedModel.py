"""
File: pylinex/model/TiedModel.py
Author: Keith Tauscher
Date: 6 Sep 2018

Description: File containing a class representing a model which ties together
             multiple instances of one other model class by setting some of
             their parameters equal.
"""
import numpy as np
from ..util import sequence_types, create_hdf5_dataset
from .Model import Model
from .SumModel import SumModel

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TiedModel(Model):
    """
    Class representing a model which ties together multiple instances of one
    other model class by setting some of their parameters equal.
    """
    def __init__(self, names, models, shared_name, *tied_parameters):
        """
        Initializes a new TiedModel based on the given models and the names of
        the parameters which should be tied together.
        
        names: sequence of string names to associate with each model
        models: sequence of models
        shared_name: the name to place before the parameters which are tied
                     together, if any. Default: None
        tied_parameters: string names of parameters to keep fixed
        """
        self.models = models
        self.names = names
        self.shared_name = shared_name
        self.tied_parameters = tied_parameters
    
    @property
    def sum_model(self):
        """
        Property storing the SumModel which will be used internally to
        implement the __call__, gradient, and hessian functions.
        """
        if not hasattr(self, '_sum_model'):
            self._sum_model = SumModel(self.names, self.models)
        return self._sum_model
    
    @property
    def models(self):
        """
        Property storing the models which are tied together by this TiedModel.
        """
        if not hasattr(self, '_models'):
            raise AttributeError("models referenced before it was set.")
        return self._models
    
    @models.setter
    def models(self, value):
        """
        Setter for the models which are tied together by this TiedModel.
        
        value: sequence of Model objects to tie together
        """
        if type(value) in sequence_types:
            if all([isinstance(element, Model) for element in value]):
                self._models = [element for element in value]
            else:
                raise ValueError("Not all elements of models were Model " +\
                    "objects.")
        else:
            raise TypeError("models was set to a non-sequence.")
    
    @property
    def names(self):
        """
        Property storing the names associated with each submodel.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("names referenced before it was set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Setter for the names of the models given to this TiedModel.
        
        value: sequence of strings with same length as models property
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                if len(value) == len(self.models):
                    self._names = [element for element in value]
                else:
                    raise ValueError("The number of names given was not " +\
                        "the same as the number of submodels.")
            else:
                raise TypeError("Not all elements of given names are strings.")
        else:
            raise TypeError("names was set to a non-sequence.")
    
    @property
    def shared_name(self):
        """
        Property storing the name to place before the parameters of this model
        which are tied together, if any.
        """
        if not hasattr(self, '_shared_name'):
            raise AttributeError("shared_name was referenced before it was " +\
                "set.")
        return self._shared_name
    
    @shared_name.setter
    def shared_name(self, value):
        """
        Setter for the name to place before the parameters of this model which
        are tied together.
        
        value: None or a string
        """
        if (type(value) is type(None)) or isinstance(value, basestring):
            self._shared_name = value
        else:
            raise TypeError("shared_name was set to neither None nor a " +\
                "string.")
    
    @property
    def shared_prefix(self):
        """
        Property storing the string placed before parameters which are tied
        together (including underscore if shared_name is not None).
        """
        if not hasattr(self, '_shared_prefix'):
            if type(self.shared_name) is type(None):
                self._shared_prefix = ''
            else:
                self._shared_prefix = '{!s}_'.format(self.shared_name)
        return self._shared_prefix
    
    @property
    def tied_parameters(self):
        """
        Property storing a list of names of parameters which are tied together.
        """
        if not hasattr(self, '_tied_parameters'):
            raise AttributeError("tied_parameters was referenced before it " +\
                "was set.")
        return self._tied_parameters
    
    @tied_parameters.setter
    def tied_parameters(self, value):
        """
        Setter for the list of parameters which are tied together.
        
        value: list of parameter names shared by all submodels which should be
               forced to be identical
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                if all([all([(element in model.parameters)\
                    for model in self.models]) for element in value]):
                    self._tied_parameters = [element for element in value]
                else:
                    raise ValueError("Not all elements of tied_parameters " +\
                        "given were names of parameters of all submodels.")
            else:
                raise TypeError("Not all elements of tied_parameters given " +\
                    "were strings.")
        else:
            raise TypeError("tied_parameters was set to a non-sequence.")
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            shared_parameters = ['{0!s}{1!s}'.format(self.shared_prefix,\
                tied_parameter) for tied_parameter in self.tied_parameters]
            split_parameters = sum([['{0!s}_{1!s}'.format(name, parameter)\
                for parameter in model.parameters\
                if parameter not in self.tied_parameters]\
                for (name, model) in zip(self.names, self.models)], [])
            self._parameters = shared_parameters + split_parameters
        return self._parameters
    
    @property
    def first_indices_of_models(self):
        """
        Property storing the sequence of indices which mark the first parameter
        of each model after the parameters have been reprocessed.
        """
        if not hasattr(self, '_first_indices_of_models'):
            self._first_indices_of_models = []
            current = 0
            for model in self.models:
                self._first_indices_of_models.append(current)
                current += model.num_parameters
        return self._first_indices_of_models
    
    @property
    def parameter_indices(self):
        """
        Sequence of numpy.ndarrays of differing sizes. Each array contains the
        index/indices which the given input parameter will be mapped to before
        calling the models.
        """
        if not hasattr(self, '_parameter_indices'):
            self._parameter_indices = []
            for parameter in self.tied_parameters:
                these_indices = [(index + model.parameters.index(parameter))\
                    for (index, model) in\
                    zip(self.first_indices_of_models, self.models)]
                self._parameter_indices.append(np.array(these_indices))
            for (index, model) in\
                zip(self.first_indices_of_models, self.models):
                for parameter in model.parameters:
                    if parameter not in self.tied_parameters:
                        self._parameter_indices.append(np.array(\
                            [index + model.parameters.index(parameter)]))
        return self._parameter_indices
    
    def form_parameters(self, parameters):
        """
        Forms the parameter array to pass to the underlying model from the one
        to passed to this model.
        
        parameters: 1D array of length self.num_parameters
        
        returns: 1D array of length
                 sum([model.num_parameters for model in self.models])
        """
        if len(parameters) != self.num_parameters:
            raise ValueError("parameters given to TiedModel did not have a " +\
                "length given by the number of this model's parameters.")
        processed_parameters = np.ndarray((\
            sum([model.num_parameters for model in self.models]),))
        for (parameter, indices) in zip(parameters, self.parameter_indices):
            processed_parameters[indices] = parameter
        return processed_parameters
    
    def __call__(self, parameters):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of size (num_channels,)
        """
        return self.sum_model(self.form_parameters(parameters))
    
    @property
    def gradient_computable(self):
        """
        Property storing a boolean describing whether the gradient of this
        model is computable.
        """
        return self.sum_model.gradient_computable
    
    def gradient(self, parameters):
        """
        Evaluates the gradient of the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters)
        """
        unsummed_gradient =\
            self.sum_model.gradient(self.form_parameters(parameters))
        return np.stack([np.sum(unsummed_gradient[:,indices], axis=1)\
            for indices in self.parameter_indices], axis=1)
    
    @property
    def hessian_computable(self):
        """
        Property storing a boolean describing whether the hessian of this model
        is computable. Currently, this is not yet implemented for any
        underlying models.
        """
        return False
    
    def hessian(self, parameters):
        """
        Evaluates the hessian of this model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        
        returns: array of shape (num_channels, num_parameters, num_parameters)
        """
        raise NotImplementedError("hessian is not yet implemented for the " +\
            "TiedModel class.")
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'TiedModel'
        group.attrs['shared_name'] = self.shared_name
        create_hdf5_dataset(group, 'tied_parameters',\
            data=self.tied_parameters)
        subgroup = group.create_group('models')
        for (iname, name) in enumerate(self.names):
            subsubgroup = subgroup.create_group('{}'.format(iname))
            subsubgroup.attrs['name'] = name
            self.models[iname].fill_hdf5_group(subsubgroup)
    
    def __eq__(self, other):
        """
        Checks for equality with other.
        
        other: object to check for equality
        
        returns: True if other is equal to this mode, False otherwise
        """
        if not isinstance(other, TiedModel):
            return False
        if self.sum_model != other.sum_model:
            return False
        if self.shared_name != other.shared_name:
            return False
        if self.tied_parameters != other.tied_parameters:
            return False
        return True
    
    def quick_fit(self, data, error):
        """
        Fits this SlicedModel by marginalizing out the constant parameters of
        the posterior parameter distribution of the unsliced model.
        
        data: 1D numpy.ndarray of data points to fit with this model
        error: 1D array of same length as data containing errors on the data
               if None, all channels are treated equally
        
        returns: (parameter_mean, parameter_covariance) where both are
                 marginalized over the constant parameters, not conditionalized
        """
        raise NotImplementedError("quick_fit is not implemented for the " +\
            "TiedModel class.")
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for (index, indices) in enumerate(self.parameter_indices):
                parameter = self.parameters[index]
                sum_model_parameter = self.sum_model.parameters[indices[0]]
                self._bounds[parameter] =\
                    self.sum_model.bounds[sum_model_parameter]
        return self._bounds

