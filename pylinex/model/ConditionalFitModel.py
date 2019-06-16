"""
File: plyinex/model/ConditionalFitModel.py
Author: Keith Tauscher
Date: 31 May 2019

Description: File containing class representing a model where part(s) of the
             model is conditionalized and the parameters of the other part(s)
             of the model are automatically evaluated at parameters that
             maximize a likelihood.
"""
import numpy as np
from distpy import GaussianDistribution
from ..util import create_hdf5_dataset, numerical_types, sequence_types
from .Model import Model
from .SumModel import SumModel
from .DirectSumModel import DirectSumModel
from .ProductModel import ProductModel
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class ConditionalFitModel(Model):
    """
    An abstract class representing a model where part(s) of the model is
    conditionalized and the parameters of the other part(s) of the model are
    automatically evaluated at parameters that maximize a likelihood.
    """
    def __init__(self, model, data, error, unknown_name_chain, prior=None):
        """
        Creates a new ConditionalFitModel with the given full model, data, and
        error, assuming the specified parameters are unknown.
        
        model: a SumModel, DirectSumModel, or ProductModel object
        data: data vector to use when conditionalizing. Ideally, all outputs of
              this model should be similar to this data.
        error: noise level of the data,
               if None, the noise level is 1 everywhere
        unknown_name_chain: name (or chain of names) of the single submodel
                            which has a quick_fit function which will be solved
                            for
        prior: either None or a GaussianDistribution object for all of the
               parameters of the unknown submodel
        """
        self.model = model
        self.data = data
        self.error = error
        self.unknown_name_chain = unknown_name_chain
        self.prior = prior
    
    @property
    def prior(self):
        """
        Property storing the prior to use when calling the 
        """
        if not hasattr(self, '_prior'):
            raise AttributeError("prior was referenced before it was set.")
        return self._prior
    
    @prior.setter
    def prior(self, value):
        """
        Setter for the prior distribution to use, if applicable.
        
        value: either None or a GaussianDistribution object
        """
        if type(value) is type(None):
            self._prior = None
        elif isinstance(value, GaussianDistribution):
            if value.numparams == self.unknown_submodel.num_parameters:
                self._prior = value
            else:
                raise ValueError("The prior given did not have the same " +\
                    "number of parameters as the unknown submodel.")
        else:
            raise TypeError("prior was neither None nor a " +\
                "GaussianDistribution object.")
    
    @property
    def model(self):
        """
        Property storing the full model that is used to fit data.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the full model.
        
        value: either a SumModel, a DirectSumModel, or a ProductModel
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was not a Model object.")
    
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
            if all([(type(element) in numerical_types) for element in value]):
                self._data = np.array(value)
            else:
                raise TypeError("data was set to a sequence whose elements " +\
                    "are not numbers.")
        else:
            raise TypeError("data was set to a non-sequence.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output which is the same
        as the number of x_values given.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = len(self.data)
        return self._num_channels
    
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
    def unknown_name_chain(self):
        """
        Property storing the chain of name (or chain of names) of the single
        submodel which has a quick_fit function which will be solved for.
        """
        if not hasattr(self, '_unknown_name_chain'):
            raise AttributeError("unknown_name_chain was referenced before " +\
                "it was set.")
        return self._unknown_name_chain
    
    @unknown_name_chain.setter
    def unknown_name_chain(self, value):
        """
        Setter for the unknown chain of names.
        
        value: name (or chain of names) of the single submodel which has a
               quick_fit function which will be solved for
        """
        if type(value) is type(None):
            self._unknown_name_chain = []
        elif isinstance(value, basestring):
            self._unknown_name_chain = [value]
        elif type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._unknown_name_chain = [element for element in value]
            else:
                raise TypeError("At least one element of the given " +\
                    "unknown_name_chain was not a string.")
        else:
            raise TypeError("unknown_name_chain was set to neither a " +\
                "string or a sequence of strings.")
    
    def _load_parameters_and_models(self):
        """
        Goes through the unknown model chain, loading the unknown submodel, the
        known models, the names of the known parameters, and whether each chain
        link is a sum or product.
        """
        parameters = []
        known_model_chain = []
        is_sum_chain = []
        unknown_submodel = self.model
        bounds = {}
        parameter_prefix = None
        for unknown_name in self.unknown_name_chain:
            known_names = [name for name in unknown_submodel.names\
                if (name != unknown_name)]
            known_models = [unknown_submodel[name] for name in known_names]
            if isinstance(unknown_submodel, DirectSumModel):
                known_model = DirectSumModel(known_names, known_models)
                is_sum = True
            elif isinstance(unknown_submodel, SumModel):
                known_model = SumModel(known_names, known_models)
                is_sum = True
            elif isinstance(unknown_submodel, ProductModel):
                known_model = ProductModel(known_names, known_models)
                is_sum = False
            else:
                raise ValueError("The unknown_name_chain given to this " +\
                    "LikelihoodDistributionHarmonizer doesn't seem to " +\
                    "match up with the structure of the model in the " +\
                    "given Loglikelihood.")
            these_parameters = known_model.parameters
            if parameter_prefix is None:
                parameter_prefix = unknown_name
            else:
                these_parameters = ['{0!s}_{1!s}'.format(\
                    parameter_prefix, parameter)\
                    for parameter in these_parameters]
                parameter_prefix =\
                    '{0!s}_{1!s}'.format(parameter_prefix, unknown_name)
            parameters.extend(these_parameters)
            known_model_chain.append(known_model)
            is_sum_chain.append(is_sum)
            unknown_submodel = unknown_submodel[unknown_name]
        quick_fit_parameter_prefix = '_'.join(self.unknown_name_chain)
        parameters.extend(['{0!s}_{1!s}'.format(quick_fit_parameter_prefix,\
            parameter) for parameter in unknown_submodel.quick_fit_parameters])
        self._known_model_chain = known_model_chain
        self._is_sum_chain = is_sum_chain
        self._parameters = parameters
        self._unknown_submodel = unknown_submodel
    
    @property
    def known_model_chain(self):
        """
        Property storing the chain of known models. The chain is the same
        length as unknown_name_chain sequence and its elements are the known
        parts of the model at each part of the chain.
        """
        if not hasattr(self, '_known_model_chain'):
            self._load_parameters_and_models()
        return self._known_model_chain
    
    @property
    def is_sum_chain(self):
        """
        Property string a chain of booleans describing whether the link at the
        given chain position is a sum link (True) or a product link (False).
        """
        if not hasattr(self, '_is_sum_chain'):
            self._load_parameters_and_models()
        return self._is_sum_chain
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._load_parameters_and_models()
        return self._parameters
    
    @property
    def indices_of_parameters(self):
        """
        Property storing the indices of this model's parameters in the
        underlying model's parameters.
        """
        if not hasattr(self, '_indices_of_parameters'):
            self._indices_of_parameters =\
                np.array([self.model.parameters.index(parameter)\
                for parameter in self.parameters])
        return self._indices_of_parameters
    
    @property
    def unknown_submodel(self):
        """
        Property storing the unknown model (the one whose quick_fit function is
        called to best fit the given data).
        """
        if not hasattr(self, '_unknown_submodel'):
            self._load_parameters_and_models()
        return self._unknown_submodel
    
    @property
    def bounds(self):
        """
        Property storing the bounds of the parameters, taken from the bounds of
        the submodels.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {parameter: self.model.bounds[parameter]\
                for parameter in self.parameters}
        return self._bounds
    
    @property
    def num_links(self):
        """
        Property storing the number of links in the unknown_name_chain.
        """
        if not hasattr(self, '_num_links'):
            self._num_links = len(self.unknown_name_chain)
        return self._num_links
    
    def __call__(self, parameters, return_conditional_mean=False,\
        return_conditional_covariance=False):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        return_conditional_covariance: if True (return False), then conditional
                                       parameter covariance matrix is returned
                                       alongside the data recreation
        
        returns: data_recreation, an array of size (num_channels,).
                 if return_conditional_covariance is True, the conditional
                 parameter covariance matrix is also returned
        """
        data_to_fit = self.data
        error_to_fit = self.error
        pars_used = 0
        known_model_values = []
        for index in range(self.num_links):
            known_model = self.known_model_chain[index]
            parameter_array =\
                parameters[pars_used:pars_used+known_model.num_parameters]
            known_model_value = known_model(parameter_array)
            if self.is_sum_chain[index]:
                data_to_fit = data_to_fit - known_model_value
            else:
                data_to_fit = data_to_fit / known_model_value
                error_to_fit = error_to_fit / np.abs(known_model_value)
            known_model_values.append(known_model_value)
            pars_used += known_model.num_parameters
        try:
            (conditional_mean, conditional_covariance) =\
                self.unknown_submodel.quick_fit(data_to_fit, error_to_fit,\
                quick_fit_parameters=parameters[pars_used:], prior=self.prior)
        except NotImplementedError:
            raise NotImplementedError(("The submodel (class: {!s}) " +\
                "concerning the parameters whose distribution is not known " +\
                "does not have a quick_fit function implemented, so the " +\
                "ConditionalFitModel class cannot be used.").format(\
                type(self.unknown_submodel)))
        recreation = self.unknown_submodel(conditional_mean)
        for index in range(self.num_links - 1, -1, -1):
            if self.is_sum_chain[index]:
                recreation = recreation + known_model_values[index]
            else:
                recreation = recreation * known_model_values[index]
        return_value = (recreation,)
        if return_conditional_mean:
            return_value = return_value + (conditional_mean,)
        if return_conditional_covariance:
            return_value = return_value + (conditional_covariance,)
        if len(return_value) == 1:
            return_value = return_value[0]
        return return_value
    
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
        group.attrs['class'] = 'ConditionalFitModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        create_hdf5_dataset(group, 'data', data=self.data)
        create_hdf5_dataset(group, 'error', data=self.error)
        create_hdf5_dataset(group, 'unknown_name_chain',\
            data=self.unknown_name_chain)
        if type(self.prior) is not type(None):
            self.prior.fill_hdf5_group(group.create_group('prior'))
    
    def change_data(self, new_data):
        """
        Creates a new ConditionalFitModel which has everything kept constant
        except the given new data vector is used.
        
        new_data: 1D numpy.ndarray data vector of the same length as the data
                  vector of this ConditionalFitModel
        
        returns: a new ConditionalFitModel object
        """
        return ConditionalFitModel(self.model, new_data, self.error,\
            self.unknown_name_chain)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this ConditionalFitModel.
        
        other: object to check for equality
        
        returns: False unless other is a ConditionalFitModel with the same
                 model, data, error, and unknown_name_chain
        """
        if isinstance(other, ConditionalFitModel):
            if self.model == other.model:
                if self.unknown_name_chain == other.unknown_name_chain:
                    if self.data.shape == other.data.shape:
                        return (np.all(self.data == other.data) and\
                            np.all(self.error == other.error))
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

