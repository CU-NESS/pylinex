"""
File: pylinex/loglikelihood/LikelihoodDistributionHarmonizer.py
Author: Keith Tauscher
Date: 30 Jun 2018

Description: File containing class which implements distpy's
             DistributionHarmonizer in the context of a likelihood function
             whose model is a sum of other models.
"""
import numpy as np
from distpy import DistributionHarmonizer
from ..model import SumModel
from .GaussianLoglikelihood import GaussianLoglikelihood

class LikelihoodDistributionHarmonizer(DistributionHarmonizer):
    """
    Class which implements distpy's DistributionHarmonizer in the context of a
    likelihood.
    """
    def __init__(self, incomplete_guess_distribution_set,\
        gaussian_loglikelihood, unknown_name, ndraw):
        """
        Creates a new LikelihoodDistributionHarmonizer from the loglikelihood.
        
        incomplete_guess_distribution_set: DistributionSet(s) describing
                                           parameters not to be solved for
        gaussian_loglikelihood: GaussianLoglikelihood object whose model is a
                                SumModel
        unknown_name: name of the single submodel which has a quick_fit
                      function which will be solved for to generate samples
                      from this LikelihoodDistributionHarmonizer
        ndraw: positive integer number of desired samples
        """
        if isinstance(gaussian_loglikelihood, GaussianLoglikelihood):
            sum_model = gaussian_loglikelihood.model
            if not isinstance(sum_model, SumModel):
                raise TypeError("gaussian_loglikelihood's model was not a " +\
                    "SumModel, so solving for things will have to be more " +\
                    "customized. You should probably implement a distpy " +\
                    "DistributionHarmonizer manually.")
        else:
            raise TypeError("gaussian_loglikelihood was not a " +\
                "GaussianLoglikelihood object.")
        known_names =\
            [name for name in sum_model.names if (name != unknown_name)]
        known_submodels = [sum_model[name] for name in known_names]
        unknown_submodel = sum_model[unknown_name]
        unknown_parameter_names =\
            ['{0!s}_{1!s}'.format(unknown_name, parameter)\
            for parameter in unknown_submodel.parameters]
        known_sum_model = SumModel(known_names, known_submodels)
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
            parameter_array = np.array([incomplete_parameters[parameter]\
                for parameter in known_sum_model.parameters])
            data_to_fit =\
                gaussian_loglikelihood.data - known_sum_model(parameter_array)
            error_to_fit = gaussian_loglikelihood.error
            try:
                solved_for_parameters =\
                    unknown_submodel.quick_fit(data_to_fit, error_to_fit)[0]
            except NotImplementedError:
                raise NotImplementedError(("The submodel (class: {!s}) " +\
                    "concerning the parameters whose distribution is not " +\
                    "known does not have a quick_fit function implemented, " +\
                    "so the LikelihoodDistributionHarmonizer class cannot " +\
                    "be used.").format(type(unknown_submodel)))
            return {parameter: value for (parameter, value) in\
                zip(unknown_parameter_names, solved_for_parameters)}
        DistributionHarmonizer.__init__(self,\
            incomplete_guess_distribution_set, remaining_parameter_solver,\
            ndraw)

