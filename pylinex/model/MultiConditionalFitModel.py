"""
File: plyinex/model/MultiConditionalFitModel.py
Author: Keith Tauscher
Date: 31 May 2019

Description: File containing class representing a model where part(s) of the
             model is conditionalized and the parameters of the other part(s)
             of the model are automatically evaluated at parameters that
             maximize a likelihood.
"""
import numpy as np
from distpy import GaussianDistribution
from ..util import create_hdf5_dataset, sequence_types
from .SumModel import SumModel
from .DirectSumModel import DirectSumModel
from .ProductModel import ProductModel
from .BasisModel import BasisModel
from .ConditionalFitModel import ConditionalFitModel
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class MultiConditionalFitModel(ConditionalFitModel):
    """
    An class representing a model where part(s) of the model is
    conditionalized and the parameters of the other part(s) of the model are
    automatically evaluated at parameters that maximize a likelihood.
    """
    def __init__(self, model, data, error, unknown_name_chains, priors=None):
        """
        Creates a new MultiConditionalFitModel with the given full model, data,
        and error, assuming the specified parameters are unknown.
        
        model: a SumModel, DirectSumModel, or ProductModel object
        data: data vector to use when conditionalizing. Ideally, all outputs of
              this model should be similar to this data.
        error: noise level of the data,
               if None, the noise level is 1 everywhere
        unknown_name_chains: names (or chains of names) of the submodels which
                             has a quick_fit function which will be solved for
        priors: tuple of length nunknown whose values are either None or
               GaussianDistribution objects for all of the parameters of each
               of the unknown submodels
        """
        self.model = model
        self.data = data
        self.error = error
        self.unknown_name_chains = unknown_name_chains
        self.priors = priors
        self.check_if_valid()
    
    def change_priors(self, new_priors):
        """
        Creates a copy of this MultiConditionalFitModel with the given prior
        replacing the current priors.
        
        new_prior: tuple of length nunknown containing either None or a
                   GaussianDistribution object
        
        returns: new MultiConditionalFitModel with the given priors replacing
                 the current priors
        """
        return MultiConditionalFitModel(self.model, self.data, self.error,\
            self.unknown_name_chains, new_priors)
    
    def priorless(self):
        """
        Creates a new MultiConditionalFitModel with no prior but has everything
        else the same.
        
        returns: a new MultiConditionalFitModel copied from this one without
                 any priors
        """
        return self.change_prior((None,) * len(self.unknown_name_chains))
    
    @property
    def priors(self):
        """
        Property storing the priors to use when fitting the unknown models.
        """
        if not hasattr(self, '_priors'):
            raise AttributeError("priors was referenced before it was set.")
        return self._priors
    
    @priors.setter
    def priors(self, value):
        """
        Setter for the prior distributions to use, if applicable.
        
        value: tuple of length nunknown of either None or a
               GaussianDistribution object
        """
        if type(value) in sequence_types:
            if len(value) == len(self.unknown_name_chains):
                if all([((type(element) is type(None)) or\
                    isinstance(element, GaussianDistribution))\
                    for element in value]):
                    self._priors = [element for element in value]
                else:
                    raise TypeError("At least one prior was neither None " +\
                        "nor a GaussianDistribution object.")
            else:
                raise ValueError("Length of priors was not the same as the " +\
                    "length of the unknown_name_chains.")
        else:
            raise TypeError("priors was set to a non-sequence")
    
    @property
    def unknown_name_chains(self):
        """
        Property storing the chains of name (or chains of names) of the
        submodels which will be solved for.
        """
        if not hasattr(self, '_unknown_name_chains'):
            raise AttributeError("unknown_name_chains was referenced " +\
                "before it was set.")
        return self._unknown_name_chains
    
    @unknown_name_chains.setter
    def unknown_name_chains(self, value):
        """
        Setter for the unknown chains of names.
        
        value: names (or chains of names) of the submodel which will be solved
               for
        """
        if type(value) in sequence_types:
            chains = []
            for element in value:
                if isinstance(element, basestring):
                    chains.append([element])
                elif type(element) in sequence_types:
                    if all([isinstance(subelement, basestring)\
                        for subelement in element]):
                        chains.append([subelement for subelement in element])
                    else:
                        raise TypeError("At least one unknown_name_chain " +\
                            "contains a non-string.")
                else:
                    raise TypeError("At least one unknown_name_chain was " +\
                        "neither a string or a sequence of strings.")
        else:
            raise TypeError("unknown_name_chain was set to a non-sequence.")
    
    @property
    def num_unknown_models(self):
        """
        Property storing the number of unknown models.
        """
        if not hasattr(self, '_num_unknown_models'):
            self._num_unknown_models = len(self.unknown_name_chains)
        return self._num_unknown_models
    
    @property
    def model_tree(self):
        """
        Property storing the ModelTree class associated with the full model of
        this class.
        """
        if not hasattr(self, '_model_tree'):
            self._model_tree = ModelTree(self.model)
        return self._model_tree
    
    @property
    def parameters_by_known_leaf(self):
        """
        Property storing the parameters associated with each known leaf.
        """
        if not hasattr(self, '_parameters_by_known_leaf'):
            self._parameters_by_known_leaf =\
                [leaf.parameters for (leaf, name_chain) in\
                zip(self.model_tree.leaves, self.model_tree.name_chains)\
                if name_chain not in self.unknown_name_chains]
        return self.parameters_by_known_leaf
    
    @property
    def num_parameters_by_known_leaf(self):
        """
        Property storing the number of parameters for each known leaf.
        """
        if not hasattr(self, '_num_parameters_by_known_leaf'):
            self._num_parameters_by_known_leaf = [len(parameters)\
                for parameters in self.parameters_by_known_leaf]
        return self._num_parameters_by_known_leaf
    
    @property
    def parameters(self):
        """
        Property storing a list of strings associated with the parameters
        necessitated by this model.
        """
        if not hasattr(self, '_parameters'):
            self._parameters = sum(self.parameters_by_known_leaf, [])
        return self._parameters
    
    @property
    def unknown_leaf_indices(self):
        """
        Property storing a list of the leaf indices of the unknown models.
        """
        if not hasattr(self, '_unknown_leaf_indices'):
            self._unknown_leaf_indices =\
                [self.model_tree.name_chains.index(name_chain)\
                for name_chain in self.unknown_name_chains]
        return self._unknown_leaf_indices
    
    @property
    def unknown_submodels(self):
        """
        Property storing the list of unknown submodels.
        """
        if not hasattr(self, '_unknown_submodels'):
            self._unknown_submodels = [self.model_tree.leaves[leaf_index]\
                for leaf_index in self.unknown_leaf_indices]
        return self._unknown_submodels
    
    def check_if_valid(self):
        """
        Checks if this MultiConditionalFitModel is valid by ensuring that all
        unknown submodels are basis models. If not, an error is thrown.
        """
        if any([(not isinstance(model, BasisModel))\
            for model in self.unknown_submodels]):
            raise ValueError("This MultiConditionalFitModel object is " +\
                "ill-formed because all unknown submodels must be " +\
                "BasisModels so that they can be combined into a model " +\
                "that still has a quick_fit function.")
    
    @property
    def num_unknown_parameters_per_model(self):
        """
        Property storing the number of parameters in each unknown model.
        """
        if not hasattr(self, '_num_unknown_parameters_per_model'):
            self._num_unknown_parameters_per_model =\
                [model.num_parameters for model in self.unknown_submodels]
        return self._num_unknown_parameters_per_model
    
    @property
    def unknown_leaf_modulators(self):
        """
        Property storing the modulating models of the unknown leaves.
        """
        if not hasattr(self, '_unknown_leaf_modulators'):
            self._unknown_leaf_modulators =\
                [self.model_tree.modulators[leaf_index]\
                for leaf_index in self.unknown_leaf_indices]
        return self._unknown_leaf_modulators
    
    @property
    def unknown_leaf_modulator_leaf_lists(self):
        """
        Property storing the indices of the leaves that compose the modulating
        models of the unknown leaves.
        """
        if not hasattr(self, '_unknown_leaf_modulator_leaf_lists'):
            self._unknown_leaf_modulator_leaf_lists =\
                [self.model_tree.modulator_leaf_lists[leaf_index]\
                for leaf_index in self.unknown_leaf_indices]
        return self._unknown_leaf_modulator_leaf_lists
    
    def split_parameter_vector(self, vector):
        """
        Splits the given parameter vectors into sets to evaluate each known
        leaf.
        
        vector: 1D numpy array of length self.num_parameters
        
        returns: list of vectors with which to evaluate known leaves
        """
        (vectors, accounted_for) = ([], 0)
        for num_parameters in self.num_parameters_by_known_leaf:
            vectors.append(vector[accounted_for:accounted_for+num_parameters])
            accounted_for = accounted_for + num_parameters
        return vectors
    
    def evaluate_leaves(self, parameters):
        """
        Evaluates the leaves given the parameters of this ConditionalFitModel.
        Leaves corresponding to unknown_name_chains are replaced by all zeros
        so that if these leaf evaluations are given to the full model tree of
        this class, it will produce a version without the unknown leaves.
        
        vector: 1D numpy array of length self.num_parameters
        
        returns: list of evaluated leaf models
        """
        (leaf_evaluations, known_leaves_accounted_for) = ([], 0)
        parameter_vectors = self.split_parameter_vector(parameters)
        for (leaf, name_chain) in\
            zip(self.model_tree.leaves, self.model_tree.name_chains):
            if name_chain in self.unknown_name_chains:
                leaf_evaluations.append(np.zeros(leaf.num_channels))
            else:
                leaf_evaluations.append(\
                    parameter_vectors[known_leaves_accounted_for])
                known_leaves_accounted_for += 1
        return leaf_evaluations
    
    def __call__(self, parameters, return_conditional_mean=False,\
        return_conditional_covariance=False,\
        return_log_prior_at_conditional_mean=False):
        """
        Evaluates the model at the given parameters.
        
        parameters: 1D numpy.ndarray of parameter values
        return_conditional_mean: if True (default False), then conditional
                                 parameter mean is returned alongside the data
                                 recreation
        return_conditional_covariance: if True (default False), then
                                       conditional parameter covariance matrix
                                       is returned alongside the data
                                       recreation
        return_log_prior_at_conditional_mean: if True (default False), then the
                                              log value of the prior evaluated
                                              at the conditional mean is
                                              returned
        
        returns: data_recreation, an array of size (num_channels,).
                 if return_conditional_mean and/or
                 return_conditional_covariance and/or
                 return_log_prior_at_conditional_mean is True, the conditional
                 parameter mean vector and/or covariance matrix and/or the log
                 value of the prior is also returned
        """
        leaf_evaluations = self.evaluate_leaves(parameters)
        known_leaf_contribution =\
            self.model_tree.evaluate_from_leaves(leaf_evaluations)
        data_less_known_leaves = self.data - known_leaf_contribution
        modulating_factors = []
        for (modulator, modulator_leaf_list) in\
            zip(self.unknown_leaf_modulators,\
            self.unknown_leaf_modulator_leaf_lists):
            if modulator_leaf_list:
                modulating_factors.append(\
                    ModelTree.evaluate_model_from_leaves(modulator,\
                    [leaf_evaluations[leaf_index]\
                    for leaf_index in self.unknown_leaf_indices]))
            else:
                modulating_factors.append(modulator([]))
        temporary_combined_basis =\
            np.concatenate([(basis_model.basis.expanded_basis *\
            modulating_factor[np.newaxis,:]) for (basis_model, factor) in\
            zip(self.unknown_submodels, modulating_factors)], axis=0)
        temporary_basis_model = BasisModel(Basis(temporary_combined_basis))
        (conditional_mean, conditional_covariance) =\
            temporary_basis_model.quick_fit(data_less_known_leaves, self.error)
        recreation =\
            known_leaf_contribution + temporary_basis_model(conditional_mean)
        return_value = (recreation,)
        if return_conditional_mean:
            return_value = return_value + (conditional_mean,)
        if return_conditional_covariance:
            return_value = return_value + (conditional_covariance,)
        if return_log_prior_at_conditional_mean:
            return_value =\
                return_value + (self.log_prior_value(conditional_mean),)
        if len(return_value) == 1:
            return_value = return_value[0]
        return return_value
    
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
    
    def log_prior_value(self, unknown_parameters):
        """
        Computes the value of the log prior of the unknown model parameters at
        a specific point in parameter space.
        
        unknown_parameters: parameters at which to evaluate log prior of
                            unknown parameters
        
        returns: a single float describing the prior value
        """
        accounted_for = 0
        log_value = 0.
        for (prior, num_parameters) in\
            zip(self.priors, self.num_unknown_parameters_per_model):
            these_parameters =\
                unknown_parameters[accounted_for:accounted_for+num_parameters]
            if type(prior) is not type(None):
                log_value += prior.log_value(these_parameters)
            accounted_for += num_parameters
        return log_value
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with information about this model.
        
        group: the hdf5 file group to fill with information about this model
        """
        group.attrs['class'] = 'MultiConditionalFitModel'
        self.model.fill_hdf5_group(group.create_group('model'))
        create_hdf5_dataset(group, 'data', data=self.data)
        create_hdf5_dataset(group, 'error', data=self.error)
        group.attrs['num_unknown_models'] = self.num_unknown_models
        subgroup = group.create_group('unknown_name_chains')
        for (iunknown, unknown_name_chain) in\
            enumerate(self.unknown_name_chains):
            create_hdf5_dataset(subgroup, '{:d}'.format(iunknown),\
                data=unknown_name_chain)
        subgroup = group.create_group('priors')
        for (iunknown, prior) in enumerate(self.priors):
            if type(prior) is not type(None):
                self.prior.fill_hdf5_group(\
                    subgroup.create_group('{:d}'.format(unknown)))
    
    def change_data(self, new_data):
        """
        Creates a new MultiConditionalFitModel which has everything kept
        constant except the given new data vector is used.
        
        new_data: 1D numpy.ndarray data vector of the same length as the data
                  vector of this MultiConditionalFitModel
        
        returns: a new MultiConditionalFitModel object
        """
        return MultiConditionalFitModel(self.model, new_data, self.error,\
            self.unknown_name_chains, priors=self.priors)
    
    def __eq__(self, other):
        """
        Checks if other is an equivalent to this MultiConditionalFitModel.
        
        other: object to check for equality
        
        returns: False unless other is a MultiConditionalFitModel with the same
                 model, data, error, and unknown_name_chains
        """
        if isinstance(other, MultiConditionalFitModel):
            if self.model == other.model:
                if self.unknown_name_chains == other.unknown_name_chains:
                    if self.data.shape == other.data.shape:
                        if (np.all(self.data == other.data) and\
                            np.all(self.error == other.error)):
                            return self.priors == other.priors
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False

