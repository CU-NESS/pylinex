"""
File: pylinex/model/ModelTree.py
Author: Keith Tauscher
Date: 1 Oct 2020

Description: File containing a class representing a tree of models connected
             through sums and products. The sums and products are represented
             by branch nodes while the submodels are represented by leaf nodes.
"""
import numpy as np
from .Model import Model
from .SumModel import SumModel
from .DirectSumModel import DirectSumModel
from .ProductModel import ProductModel
from .FixedModel import FixedModel
from .RenamedModel import RenamedModel

allowed_compound_model_classes = [SumModel, ProductModel]

class ModelTree(object):
    """
    Class representing a tree of models connected through sums and products.
    The sums and products are represented by branch nodes while the submodels
    are represented by leaf nodes.
    """
    def __init__(self, model):
        """
        Initializes a new ModelTree with the given root.
        
        model: the root of the model tree
        """
        ModelTree.check_if_well_formed(model)
        self.model = model
    
    @property
    def model(self):
        """
        Property storing the model that is the root of the tree.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the model that is the root of the tree.
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @staticmethod
    def check_if_well_formed(model):
        """
        Throws an error if any two successive internal nodes in the tree
        specified by the given model are the same type. If a SumModel
        (ProductModel) has a SumModel (ProductModel) as one of its submodels,
        the same model can and should be represented with only one SumModel
        (ProductModel).
        
        model: the root of the tree to examine
        """
        model_type = type(model)
        if model_type in allowed_compound_model_classes:
            for submodel in model.models:
                if model_type == type(submodel):
                    raise ValueError(("The model tree is not well formed. " +\
                        "A {0!s} contains another {0!s} as a submodel. " +\
                        "These can be combined into a single {0!s}.").format(\
                        model_type))
                ModelTree.check_if_well_formed(submodel)
    
    @staticmethod
    def find_leaves(model):
        """
        Static method that finds the submodels of a given model.
        
        model: root of the tree that one wishes to find the leaves for
        
        returns: list of submodels
        """
        if type(model) in allowed_compound_model_classes:
            return sum([ModelTree.find_leaves(submodel)\
                for submodel in model.models], [])
        else:
            return [model]
    
    @staticmethod
    def find_num_leaves(model):
        """
        Finds the number of leaves that compose the given model.
        
        model: root of the tree that one wishes to find the number of leaves of
        
        returns: integer number of leaf models
        """
        return len(ModelTree.find_leaves(model))
    
    @staticmethod
    def find_name_chains(model, prefix=''):
        """
        Finds the chains of names that connect the given model to its leaves.
        
        model: root of the tree that one wishes to find name chains for
        prefix: the prefix to the current root
        
        returns: list of strings connected by '_' indicating the path to its
                 leaves
        """
        if type(model) in allowed_compound_model_classes:
            return sum([ModelTree.find_name_chains(submodel,\
                prefix='{0!s}___{1!s}'.format(prefix, name))\
                for (submodel, name) in zip(model.models, model.names)], [])
        else:
            return [prefix]
    
    @staticmethod
    def evaluate_model_from_leaves(model, leaf_evaluations):
        """
        Evaluates the given model from the evaluations of its leaves.
        Essentially, this evaluates the model except the leaf models have
        already been evaluated.
        
        model: the Model to evaluate
        leaf_evaluations: list of leaf evaluations with which to evaluate the
                          model (i.e. the leaves have already been evaluated)
        
        returns: 1D numpy array of length model.num_channels
        """
        if ModelTree.find_num_leaves(model) != len(leaf_evaluations):
            raise ValueError("Length of the leaf_evaluations list was not " +\
                "equal to number of leaves of the model.")
        if type(model) in allowed_compound_model_classes:
            (submodel_evaluations, accounted_for) = ([], 0)
            for submodel in model.models:
                this_num_leaves = ModelTree.find_num_leaves(submodel)
                this_evaluation_slice =\
                    slice(accounted_for, accounted_for + this_num_leaves)
                these_evaluations = leaf_evaluations[this_evaluation_slice]
                this_evaluation = ModelTree.evaluate_model_from_leaves(\
                    submodel, these_evaluations)
                submodel_evaluations.append(this_evaluation)
                accounted_for = accounted_for + this_num_leaves
            if isinstance(model, SumModel):
                return np.sum(submodel_evaluations, axis=0)
            else:
                return np.prod(submodel_evaluations, axis=0)
        else:
            return leaf_evaluations[0]
    
    def evaluate_from_leaves(self, leaf_evaluations):
        """
        Evaluates the model of this ModelTree from the evaluation of its
        leaves. Essentially, this evaluates the model except the leaf models
        have already been evaluated.
        
        leaf_evaluations: list of evaluated leaves with which to evaluate the
                          model (i.e. the leaves have already been evaluated)
        
        returns: 1D numpy array of length self.model.num_channels
        """
        return\
            ModelTree.evaluate_model_from_leaves(self.model, leaf_evaluations)
    
    @property
    def leaves(self):
        """
        Property storing the leaves of the tree. These are the nodes that are
        not SumModel or ProductModel objects.
        """
        if not hasattr(self, '_leaves'):
            self._leaves = ModelTree.find_leaves(self.model)
        return self._leaves
    
    @property
    def parameters_per_leaf(self):
        """
        Property storing the number of parameters associated with each leaf
        model.
        """
        if not hasattr(self, '_parameters_per_leaf'):
            self._parameters_per_leaf =\
                [model.num_parameters for model in self.leaves]
        return self._parameters_per_leaf
    
    @property
    def name_chains(self):
        """
        Property storing the chains of names that lead to each leaf of the
        tree. It is a list of the same length as leaves where each element is a
        list of strings specifying how to get to the associated leaf.
        """
        if not hasattr(self, '_name_chains'):
            chains = ModelTree.find_name_chains(self.model)
            self._name_chains = [chain.split('___')[1:] for chain in chains]
        return self._name_chains
    
    def load_leaf_list_by_name_chain(self, name_chain=[]):
        """
        Stores (in the leaf_list_by_string_name_chain property) and returns the
        list of leaves associated with the given node (that can be either a
        leaf or a branch node).
        
        name_chain: string of names that specifies path to given node (empty
                    list represents root node)
        
        returns: list of leaf indices composing the model given by the node
                 specified by name_chain
        """
        model = self.model
        for name in name_chain:
            model = model.models[model.names.index(name)]
        if type(model) in allowed_compound_model_classes:
            to_return = sum([self.load_leaf_list_by_name_chain(\
                name_chain=name_chain+[new_name])\
                for new_name in model.names], [])
        else:
            to_return = [self.name_chains.index(name_chain)]
        self.leaf_list_by_string_name_chain['_'.join(name_chain)] = to_return
        return to_return
    
    @property
    def leaf_list_by_string_name_chain(self):
        """
        Property storing a dictionary are the name chains leading to all nodes,
        including both leaf nodes and branch nodes, and whose values are lists
        of leaf indices composing the specified node.
        """
        if not hasattr(self, '_leaf_list_by_string_name_chain'):
            self._leaf_list_by_string_name_chain = {}
            self.load_leaf_list_by_name_chain()
        return self._leaf_list_by_string_name_chain
    
    def form_modulator(self, name_chain):
        """
        Forms the model that modulates the leaf associated with the given
        name chain.
        
        name_chain: list of strings specifying how to get to a specific leaf
        
        returns: Model object that modulates the given leaf
        """
        current_model = self.model
        (factor_models, factor_names, factor_leaf_list, prefix) =\
            ([], [], [], '')
        for name in name_chain:
            if isinstance(current_model, ProductModel):
                for (new_name, new_model) in\
                    zip(current_model.names, current_model.models):
                    if new_name != name:
                        if prefix == '':
                            factor_names.append(new_name)
                            new_leaf_list =\
                                self.leaf_list_by_string_name_chain[new_name]
                        else:
                            factor_names.append('_'.join([prefix, new_name]))
                            new_leaf_list =\
                                self.leaf_list_by_string_name_chain[\
                                '_'.join([prefix, new_name])]
                        factor_models.append(new_model)
                        factor_leaf_list.extend(new_leaf_list)
            current_model =\
                current_model.models[current_model.names.index(name)]
            if prefix == '':
                prefix = name
            else:
                prefix = '_'.join([prefix, name])
        if len(factor_models) == 0:
            return (FixedModel(np.ones(self.model.num_channels)),\
                factor_leaf_list)
        elif len(factor_models) == 1:
            return (factor_models[0], factor_leaf_list)
        else:
            return (ProductModel(factor_names, factor_models),\
                factor_leaf_list)
    
    def load_modulators(self):
        """
        Loads the modulators and modulator_leaf_lists into this object's
        properties.
        """
        (modulators, leaf_lists) = ([], [])
        for name_chain in self.name_chains:
            (modulator, leaf_list) = self.form_modulator(name_chain)
            modulators.append(modulator)
            leaf_lists.append(leaf_list)
        self._modulators = modulators
        self._modulator_leaf_lists = leaf_lists
    
    @property
    def modulators(self):
        """
        Property storing the models that modulate each leaf node.
        """
        if not hasattr(self, '_modulators'):
            self.load_modulators()
        return self._modulators
    
    @property
    def modulator_leaf_lists(self):
        """
        Property storing the indices of leaves that compose each modulator in
        the order they appear in that modulator.
        """
        if not hasattr(self, '_modulator_leaf_lists'):
            self.load_modulators()
        return self._modulator_leaf_lists
    

