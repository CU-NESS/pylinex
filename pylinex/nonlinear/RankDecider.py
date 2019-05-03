"""
File: pylinex/nonlinear/RankDecider.py
Author: Keith Tauscher
Date: 20 Apr 2018

Description: File containing a class which represents an IC-minimizer over a
             discrete grid defined by a set of basis vector groups.
"""
import numpy as np
from distpy import Expression, KroneckerDeltaDistribution, DistributionSet
from ..util import int_types, real_numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from ..basis import Basis, BasisSet
from ..model import Model, BasisModel, CompositeModel,\
    load_model_from_hdf5_group
from ..loglikelihood import GaussianLoglikelihood
from .LeastSquareFitter import LeastSquareFitter
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class RankDecider(object):
    """
    Class which chooses a rank for multiple data components using DIC-like
    loglikelihoods which varies the number of coefficients to use in each of a
    number of bases as the parameters of the likelihood.
    """
    def __init__(self, names, basis_set, data, error, expression,\
        parameter_penalty=1, **non_basis_models):
        """
        Initializes a new TruncationLoglikelihood with the given basis_sum,
        data, and error.
        
        names: list of string names of data components
        basis_set: BasisSet objects containing basis with the largest number
                   of basis vectors allowed for each component. All of
                   basis_set.names must be in names, but some of names may not
                   be in basis_set.names; those that are not in basis_set.names
                   should have non_basis_models given as kwargs
        data: 1D data vector to fit
        error: 1D vector of noise level estimates for data
        expression: Expression object which forms full model from submodels.
                    The ith submodel (with i starting at 0), corresponding to
                    the ith name in the names list, should be represented by
                    {i} in the expression string
        parameter_penalty: the logL parameter penalty for adding a parameter in
                           any given model. Should be a non-negative constant.
                           It defaults to 1, which is the penalty used for the
                           Deviance Information Criterion (DIC)
        **non_basis_models: extra keyword arguments whose keys are elements of
                            names which are not in basis_set.names, if any
                            exist, and whose values are model objects
                            corresponding to those data components
        """
        self.names = names
        self.basis_set = basis_set
        self.data = data
        self.error = error
        self.expression = expression
        self.parameter_penalty = parameter_penalty
        self.non_basis_models = non_basis_models
    
    @property
    def names(self):
        """
        Property storing the names of the data components.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("names was referenced before it was set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Setter for the names of data components.
        
        value: sequence of string names
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._names = [element for element in value]
            else:
                raise TypeError("Not all elements of names were strings.")
        else:
            raise TypeError("names was set to a non-sequence.")
    
    @property
    def basis_set(self):
        """
        Property storing the BasisSet object 
        """
        if not hasattr(self, '_basis_set'):
            raise AttributeError("basis_set was referenced before it was set.")
        return self._basis_set
    
    @basis_set.setter
    def basis_set(self, value):
        """
        Setter for the basis_set object.
        
        value: a BasisSet object
        """
        if isinstance(value, BasisSet):
            if all([(name in self.names) for name in value.names]):
                self._basis_set = value
            else:
                raise ValueError("basis_set had at least one key that was " +\
                    "not in names.")
        else:
            raise TypeError("basis_set was set to a non-BasisSet object.")
    
    @property
    def data(self):
        """
        Property storing the data to fit.
        """
        if not hasattr(self, '_data'):
            raise AttributeError("data referenced before it was set.")
        return self._data
    
    @data.setter
    def data(self, value):
        """
        Setter for the data to fit.
        
        value: must be a 1-dimensional numpy.ndarray
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                self._data = value
            else:
                raise ValueError("Data for RankDecider must be 1D.")
        else:
            raise TypeError("data was not given as a sequence.")
    
    @property
    def error(self):
        """
        Property storing the error on the data given.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error used to define the likelihood.
        
        value: must be a numpy.ndarray of the same shape as the data property
        """
        value = np.array(value)
        if value.shape == self.data.shape:
            self._error = value
        elif value.shape == (self.data.shape * 2):
            self._error = value
        else:
            raise ValueError("error given was not the same shape as the data.")
    
    @property
    def expression(self):
        """
        Property storing the Expression object which allows for the combination
        of all of the sets of basis vectors.
        """
        if not hasattr(self, '_expression'):
            raise AttributeError("expression was referenced before it was " +\
                "set.")
        return self._expression
    
    @expression.setter
    def expression(self, value):
        """
        Setter for the Expression object which allows for the combination of
        all of the sets of basis vectors.
        
        value: an Expression object which has as many arguments as the
               basis_set has names.
        """
        if isinstance(value, Expression):
            if value.num_arguments == len(self.names):
                self._expression = value
            else:
                raise ValueError("expression had a different number of " +\
                    "arguments than the RankDecider had submodels.")
        else:
            raise TypeError("expression was set to a non-Expression object.")
    
    @property
    def parameter_penalty(self):
        """
        Property storing the penalty imposed on the log-likelihood when an
        extra parameter is included in any given model.
        """
        if not hasattr(self, '_parameter_penalty'):
            raise AttributeError("parameter_penalty was referenced before " +\
                "it was set.")
        return self._parameter_penalty
    
    @parameter_penalty.setter
    def parameter_penalty(self, value):
        """
        Setter for the penalty assessed when an extra parameter is included in
        any given model.
        
        value: a non-negative number
        """
        if type(value) in real_numerical_types:
            if value >= 0:
                self._parameter_penalty = value
            else:
                raise ValueError("parameter_penalty was set to a negative " +\
                    "number.")
        else:
            raise TypeError("parameter_penalty was set to a non-number.")
    
    @property
    def non_basis_models(self):
        """
        Property storing the non-basis models, whose numbers of terms don't
        vary.
        """
        if not hasattr(self, '_non_basis_models'):
            raise AttributeError("non_basis_models was referenced before " +\
                "it was set.")
        return self._non_basis_models
    
    @non_basis_models.setter
    def non_basis_models(self, value):
        """
        Setter for the non-basis models.
        
        value: dictionary whose keys are elements of names which are not in
               basis_set.names, if any exist, and whose values are model
               objects corresponding to those data components
        """
        if isinstance(value, dict):
            keys = [key for key in value]
            if all([isinstance(key, basestring) for key in keys]):
                if (set(keys) & set(self.basis_set.names)):
                    raise ValueError("A key of non_basis_models also has a " +\
                        "basis in basis_set.")
                else:
                    all_names_set = (set(keys) | set(self.basis_set.names))
                    if (all_names_set == set(self.names)):
                        if all([isinstance(value[key], Model)\
                            for key in keys]):
                            self._non_basis_models = value
                        else:
                            raise TypeError("Not all values of " +\
                                "non_basis_models were Model objects.")
                    else:
                        raise ValueError(("The following names were " +\
                            "neither keys of non_basis_models or names in " +\
                            "the basis_set: {0}. The following names were " +\
                            "keys of non_basis_models, but were not in the " +\
                            "names given at initialization: {1}.").format(\
                            set(self.names) - all_names_set,\
                            set(keys) - set(self.names)))
            else:
                raise TypeError("Not all keys of non_basis_models were " +\
                    "strings.")
        else:
            raise TypeError("non_basis_models was set to a non-dict.")
    
    def fill_hdf5_group(self, group, data_link=None, error_link=None):
        """
        Fills the given hdf5 group with information about this RankDecider.
        
        group: the group to fill with information about this RankDecider
        data_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        error_link: link like that returned by pylinex.h5py_extensions.HDF5Link
        """
        group.attrs['class'] = 'RankDecider'
        create_hdf5_dataset(group, 'names', data=self.names)
        create_hdf5_dataset(group, 'data', data=self.data, link=data_link)
        create_hdf5_dataset(group, 'error', data=self.error, link=error_link)
        self.basis_set.fill_hdf5_group(group.create_group('basis_set'))
        self.expression.fill_hdf5_group(group.create_group('expression'))
        group.attrs['parameter_penalty'] = self.parameter_penalty
        subgroup = group.create_group('non_basis_models')
        for name in self.non_basis_models:
            self.non_basis_models[name].fill_hdf5_group(\
                subgroup.create_group(name))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads a RankDecider object from an hdf5 file group in which it was
        previously saved.
        
        group: the hdf5 file group from which to load a RankDecider object
        
        returns: the RankDecider object loaded from the given hdf5 file group
        """
        try:
            assert group.attrs['class'] == 'RankDecider'
        except:
            raise ValueError("group doesn't appear to point to a " +\
                "RankDecider object.")
        names = get_hdf5_value(group['names'])
        data = get_hdf5_value(group['data'])
        error = get_hdf5_value(group['error'])
        basis_set = BasisSet.load_from_hdf5_group(group['basis_set'])
        expression = Expression.load_from_hdf5_group(group['expression'])
        parameter_penalty = group.attrs['parameter_penalty']
        (subgroup, non_basis_models) = (group['non_basis_models'], {})
        for name in subgroup:
            non_basis_models[name] = load_model_from_hdf5_group(subgroup[name])
        return RankDecider(names, basis_set, data, error, expression,\
            parameter_penalty=parameter_penalty, **non_basis_models)
    
    def __eq__(self, other):
        """
        Checks if self is equal to other.
        
        other: an object to check for equality
        
        returns: True if other and self have the same properties
        """
        if not isinstance(other, RankDecider):
            return False
        if self.names != other.names:
            return False
        if self.basis_set != other.basis_set:
            return False
        if not np.allclose(self.data, other.data):
            return False
        if not np.allclose(self.error, other.error):
            return False
        if self.expression != other.expression:
            return False
        if self.parameter_penalty != other.parameter_penalty:
            return False
        if set(self.non_basis_models.keys()) ==\
            set(other.non_basis_models.keys()):
            for key in self.non_basis_models:
                if self.non_basis_models[key] != other.non_basis_models[key]:
                    return False
            return True
        else:
            return False
    
    def model_from_nterms(self, **nterms):
        """
        Creates a model from the given number of terms for each basis.
        
        nterms: kwargs with names in self.basis_set.names as keys and integer
                numbers of terms as values
        
        returns: a CompositeModel object that includes all submodels
        """
        if set(nterms.keys()) == set(self.basis_set.names):
            models = []
            for name in self.names:
                if name in nterms:
                    model = BasisModel(self.basis_set[name][:nterms[name]])
                else:
                    model = self.non_basis_models[name]
                models.append(model)
            return CompositeModel(self.expression, self.names, models)
        else:
            raise ValueError("The keys of nterms were not identical to the " +\
                "names of basis_set.")
    
    def loglikelihood_from_nterms(self, **nterms):
        """
        Creates a loglikelihood with a model created from the given number of
        terms for each basis.
        
        nterms: kwargs with names in self.basis_set.names as keys and integer
                numbers of terms as values
        
        returns: a GaussianLoglikelihood object with a CompositeModel
        """
        return GaussianLoglikelihood(self.data, self.error,\
            self.model_from_nterms(**nterms))
    
    def starting_point_from_nterms(self, true_parameters, true_curves, nterms):
        """
        
        
        true_parameters: dictionary containing true parameter vectors indexed
                         by name
        true_curves: dictionary of the form {(true_curve[name], suberror[name])
                     for name in true_curve_names}
        nterms: dictionary with names in self.basis_set.names as keys and
                integer numbers of terms as values
        
        returns: (loglikelihood, starting_parameters)
        """
        loglikelihood = self.loglikelihood_from_nterms(**nterms)
        starting_point = []
        for (iname, name) in enumerate(self.names):
            if name in true_parameters:
                starting_point.append(true_parameters[name])
            else:
                submodel = loglikelihood.model.models[iname]
                if name in true_curves:
                    starting_point.append(\
                        submodel.quick_fit(*true_curves[name])[0])
                elif submodel.num_parameters != 0:
                    raise ValueError("A submodel has parameters but was " +\
                        "given neither of the true_parameters or " +\
                        "true_curves dictionaries.")
        return (loglikelihood, np.concatenate(starting_point))
    
    def best_parameters_from_nterms(self, true_parameters, true_curves,\
        nterms, **bounds):
        """
        Finds the best parameters (and the loglikelihood to which they apply)
        for given nterms.
        
        true_parameters: dictionary containing true parameter vectors indexed
                         by name
        true_curves: dictionary of the form {(true_curve[name], suberror[name])
                     for name in true_curve_names}
        nterms: dictionary with names in self.basis_set.names as keys and
                integer numbers of terms as values
        bounds: tuples of form (minimum, maximum) where either may be None for
                each parameter for which bounds should be obeyed
        
        returns: (loglikelihood, max_likelihood_parameters)
        """
        (loglikelihood, starting_point) = self.starting_point_from_nterms(\
            true_parameters, true_curves, nterms)
        guess_distribution = KroneckerDeltaDistribution(starting_point)
        guess_distribution_set = DistributionSet([(guess_distribution,\
            loglikelihood.parameters, None)])
        least_square_fitter = LeastSquareFitter(\
            loglikelihood=loglikelihood, prior_set=guess_distribution_set,\
            **bounds)
        least_square_fitter.run()
        return (loglikelihood, least_square_fitter.argmin)
    
    def information_criterion_from_nterms(self, true_parameters, true_curves,\
        nterms, **bounds):
        """
        Finds the best parameters (and the loglikelihood to which they apply)
        for given nterms.
        
        true_parameters: dictionary containing true parameter vectors indexed
                         by name
        true_curves: dictionary of the form {(true_curve[name], suberror[name])
                     for name in true_curve_names}
        nterms: dictionary with names in self.basis_set.names as keys and
                integer numbers of terms as values
        bounds: tuples of form (minimum, maximum) where either may be None for
                each parameter for which bounds should be obeyed
        
        returns: (loglikelihood, max_likelihood_parameters)
        """
        (loglikelihood, max_likelihood_parameters) =\
            self.best_parameters_from_nterms(true_parameters, true_curves,\
            nterms, **bounds)
        loglikelihood_value = loglikelihood(max_likelihood_parameters)
        varying_num_parameters = sum([nterms[key] for key in nterms])
        penalty = (varying_num_parameters * self.parameter_penalty)
        return ((-2.) * (loglikelihood_value - penalty))
    
    def minimize_information_criterion(self, starting_nterms, true_parameters,\
        true_curves, return_trail=False, can_backtrack=False,\
        verbose=True, **bounds):
        """
        Minimizes the information criterion over the grid of possible nterms
        through finite difference descent.
        
        true_parameters: dictionary containing true parameter vectors indexed
                         by name
        true_curves: dictionary of the form {(true_curve[name], suberror[name])
                     for name in true_curve_names}
        return_trail: if True, return value contains not only the final nterms
                      dictionary, but also the trail used to get there from the
                      start
        bounds: tuples of form (minimum, maximum) where either may be None for
                each parameter for which bounds should be obeyed
        
        returns: (last_nterms, nterms_trail) if return_trail else last_nterms
        """
        nterms = {name: starting_nterms[name] for name in self.basis_set.names}
        information_criterion = self.information_criterion_from_nterms(\
            true_parameters, true_curves, nterms, **bounds)
        previous_nterms = []
        done = False
        iteration_number = 0
        while not done:
            iteration_number += 1
            if verbose:
                print("Iteration #{:d} starting nterms: {}".format(\
                    iteration_number, nterms))
            possible_next_nterms =\
                [{} for index in range(2 * len(self.basis_set.names))]
            for (iname, name) in enumerate(self.basis_set.names):
                for index in range(2 * len(self.basis_set.names)):
                    difference = (((2 * (index % 2)) - 1) *\
                        (1 if (iname == (index / 2)) else 0))
                    possible_next_nterms[index][name] =\
                        nterms[name] + difference
            indices_to_delete = []
            for (nti, nt) in enumerate(possible_next_nterms):
                deleted = False
                if can_backtrack and (nt in previous_nterms):
                    indices_to_delete.append(nti)
                elif (not can_backtrack) and (len(previous_nterms) != 0) and\
                    (nt == previous_nterms[-1]):
                    indices_to_delete.append(nti)
                else:
                    for name in self.basis_set.names:
                        if deleted:
                            continue
                        if (nt[name] < 1) or (nt[name] >\
                            self.basis_set[name].num_basis_vectors):
                            indices_to_delete.append(nti)
                            deleted = True
            pnt = []
            for index in range(len(possible_next_nterms)):
                if index not in indices_to_delete:
                    pnt.append(possible_next_nterms[index])
            possible_next_nterms = pnt
            if len(possible_next_nterms) == 0:
                done = True
                continue
            information_criteria = [self.information_criterion_from_nterms(\
                true_parameters, true_curves, nt, **bounds)\
                for nt in possible_next_nterms]
            information_criteria_argmin = np.argmin(information_criteria)
            information_criteria_min =\
                information_criteria[information_criteria_argmin]
            if information_criterion < information_criteria_min:
                done = True
                continue
            information_criterion = information_criteria_min
            previous_nterms.append(nterms)
            nterms = possible_next_nterms[information_criteria_argmin]
        if verbose:
            print("Final nterms: {}".format(nterms))
        if return_trail:
            return (nterms, previous_nterms)
        else:
            return nterms

