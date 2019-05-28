"""
File: pylinex/loglikelihood/LikelihoodDistributionHarmonizer.py
Author: Keith Tauscher
Date: 30 Jun 2018

Description: File containing class which implements distpy's
             DistributionHarmonizer in the context of a likelihood function
             whose model is a sum of other models.
"""
import numpy as np
from distpy import GaussianDistribution, DistributionSet,\
    DistributionHarmonizer
from ..model import SumModel, DirectSumModel, ProductModel
from .GaussianLoglikelihood import GaussianLoglikelihood

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class LikelihoodDistributionHarmonizer(DistributionHarmonizer):
    """
    Class which implements distpy's DistributionHarmonizer in the context of a
    likelihood.
    """
    def __init__(self, incomplete_guess_distribution_set,\
        gaussian_loglikelihood, unknown_name_chain, marginal_draws,\
        conditional_draws=None, **transforms):
        """
        Creates a new LikelihoodDistributionHarmonizer from the loglikelihood.
        
        incomplete_guess_distribution_set: DistributionSet(s) describing
                                           parameters not to be solved for
        gaussian_loglikelihood: GaussianLoglikelihood object whose model is a
                                SumModel or ProductModel
        unknown_name_chain: name (or chain of names) of the single submodel
                            which has a quick_fit function which will be solved
                            for to generate samples from this
                            LikelihoodDistributionHarmonizer
        marginal_draws: positive integer number of samples to draw from the
                        marginal distribution (given by the
                        incomplete_guess_distribution_set)
        conditional_draws: if None (default), then the conditional distribution
                                              is effectively degenerate, with
                                              only the maximum likelihood value
                                              being drawn
                           otherwise, then this should be a positive integer
                                      determining the number of times the
                                      conditional distribution should be drawn
                                      from for each of the marginal_draws
                                      number of draws from the marginal
                                      distribution (given by the
                                      incomplete_guess_distribution_set).
        transforms: transforms indexed by parameter. parameters not given are
                    assumed to remain untransformed throughout
        """
        if not isinstance(gaussian_loglikelihood, GaussianLoglikelihood):
            raise TypeError("gaussian_loglikelihood was not a " +\
                "GaussianLoglikelihood object.")
        model = gaussian_loglikelihood.model
        if isinstance(unknown_name_chain, basestring):
            unknown_name_chain = [unknown_name_chain]
        unknown_submodel = model
        known_model_chain = []
        is_sum_chain = []
        for unknown_name in unknown_name_chain:
            known_names = [name\
                for name in unknown_submodel.names if (name != unknown_name)]
            known_models = [unknown_submodel[name] for name in known_names]
            if isinstance(unknown_submodel, DirectSumModel):
                known_model_chain.append(\
                    DirectSumModel(known_names, known_models))
                is_sum_chain.append(True)
            elif isinstance(unknown_submodel, SumModel):
                known_model_chain.append(SumModel(known_names, known_models))
                is_sum_chain.append(True)
            elif isinstance(unknown_submodel, ProductModel):
                known_model_chain.append(\
                    ProductModel(known_names, known_models))
                is_sum_chain.append(False)
            else:
                raise ValueError("The unknown_name_chain given to this " +\
                    "LikelihoodDistributionHarmonizer doesn't seem to " +\
                    "match up with the structure of the model in the given " +\
                    "Loglikelihood.")
            unknown_submodel = unknown_submodel[unknown_name]
        if unknown_name_chain:
            unknown_parameter_names =\
                ['{0!s}_{1!s}'.format('_'.join(unknown_name_chain), parameter)\
                for parameter in unknown_submodel.parameters]
        else:
            unknown_parameter_names = unknown_submodel.parameters
        def remaining_parameter_solver(incomplete_parameters):
            #
            # Solves for the unknown_parameters by using the drawn values of
            # the parameters whose distribution is known (or assumed).
            # 
            # incomplete_parameters: dictionary with names of parameters whose
            #                        distributions are known as keys and their
            #                        drawn floats as values
            # 
            # returns: dictionary of same format as incomplete_parameters,
            #          except the keys and values are associated with the
            #          parameters whose distribution is not known
            #
            data_to_fit = gaussian_loglikelihood.data
            error_to_fit = gaussian_loglikelihood.error
            for index in range(len(unknown_name_chain)):
                if index == 0:
                    parameter_prefix = ''
                else:
                    parameter_prefix =\
                        '{!s}_'.format('_'.join(unknown_name_chain[:index]))
                known_model = known_model_chain[index]
                parameter_array = np.array([incomplete_parameters[\
                    '{0!s}{1!s}'.format(parameter_prefix, parameter)]\
                    for parameter in known_model.parameters])
                known_model_value = known_model(parameter_array)
                if is_sum_chain[index]:
                    data_to_fit = data_to_fit - known_model_value
                else:
                    data_to_fit = data_to_fit / known_model_value
                    error_to_fit = error_to_fit / np.abs(known_model_value)
            try:
                quick_fit =\
                    unknown_submodel.quick_fit(data_to_fit, error_to_fit)
            except NotImplementedError:
                raise NotImplementedError(("The submodel (class: {!s}) " +\
                    "concerning the parameters whose distribution is not " +\
                    "known does not have a quick_fit function implemented, " +\
                    "so the LikelihoodDistributionHarmonizer class cannot " +\
                    "be used.").format(type(unknown_submodel)))
            if type(conditional_draws) is type(None):
                return {parameter: value for (parameter, value) in\
                    zip(unknown_parameter_names, quick_fit[0])}
            else:
                return DistributionSet([(GaussianDistribution(*quick_fit),\
                    unknown_parameter_names, None)])
        DistributionHarmonizer.__init__(self,\
            incomplete_guess_distribution_set, remaining_parameter_solver,\
            marginal_draws, conditional_draws=conditional_draws, **transforms)

