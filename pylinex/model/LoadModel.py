"""
File: pylinex/model/LoadModel.py
Author: Keith Tauscher
Update date: 9 Apr 2018

Description: File containing a function which can load a Model object from an
             hdf5 file group.
"""
import importlib
import numpy as np
from distpy import load_transform_from_hdf5_group
from ..util import get_hdf5_value, Expression
from ..expander import load_expander_from_hdf5_group
from ..basis import Basis
from .FixedModel import FixedModel
from .ConstantModel import ConstantModel
from .BasisModel import BasisModel
from .TruncatedBasisHyperModel import TruncatedBasisHyperModel
from .GaussianModel import GaussianModel
from .LorentzianModel import LorentzianModel
from .SinusoidalModel import SinusoidalModel
from .TanhModel import TanhModel
from .TransformedModel import TransformedModel
from .DistortedModel import DistortedModel
from .ProjectedModel import ProjectedModel
from .SumModel import SumModel
from .TiedModel import TiedModel
from .DirectSumModel import DirectSumModel
from .ProductModel import ProductModel
from .CompositeModel import CompositeModel
from .ExpressionModel import ExpressionModel
from .RenamedModel import RenamedModel
from .RestrictedModel import RestrictedModel
from .ExpandedModel import ExpandedModel
from .ScaledModel import ScaledModel
from .SlicedModel import SlicedModel

# These are the model classes where it's valid to load the model using
# XXXXX.load_from_hdf5_group(group) or XXXXX.load(hdf5_file_name). Other
# classes are only loadable from the function in this file
self_loadable_model_classes =\
[\
    'BasisModel', 'ConstantModel', 'ExpressionModel', 'GaussianModel',\
    'SinusoidalModel', 'TanhModel', 'TruncatedBasisHyperModel',\
    'LorentzianModel', 'FixedModel'\
]

# Model classes which are simple wrappers around exactly one other Model class
meta_model_classes =\
[\
    'ExpandedModel', 'RenamedModel', 'RestrictedModel', 'TransformedModel',\
    'DistortedModel', 'ProjectedModel', 'SlicedModel', 'ScaledModel'\
]

# Model classes which are wrappers around an arbitrary number of Model classes
compound_model_classes =\
[\
    'CompositeModel', 'SumModel', 'DirectSumModel', 'ProductModel',\
    'TiedModel'\
]

def load_model_from_hdf5_group(group):
    """
    Loads a Model object from the given hdf5 group.
    
    group: hdf5 file group from which to load a Model object
    
    returns: a Model object of type specified by the given group's 'class'
             attribute
    """
    class_name = group.attrs['class']
    try:
        context = {}
        if 'import_string' in group.attrs:
            import_string = group.attrs['import_string']
            words = import_string.split(' ')
            num_words = len(words)
            is_simple_import = (num_words == 2) and (words[0] == 'import')
            is_as_import = (num_words == 4) and\
                ((words[0], words[2]) == ('import', 'as'))
            is_from_import = (num_words == 4) and\
                ((words[0], words[2]) == ('from', 'import'))
            if is_simple_import:
                context[words[1]] = importlib.import_module(words[1])
            elif is_as_import:
                context[words[3]] = importlib.import_module(words[1])
            elif is_from_import:
                module = importlib.import_module(words[1])
                context[words[3]] = getattr(module, words[3])
            else:
                raise ValueError("Form of import string not recognized.")
        context.update(globals())
        context.update(locals())
        return eval(class_name, context).load_from_hdf5_group(group)
    except:
        if class_name in meta_model_classes:
            model = load_model_from_hdf5_group(group['model'])
            if class_name == 'TransformedModel':
                transform = load_transform_from_hdf5_group(group['transform'])
                return TransformedModel(model, transform)
            elif class_name == 'DistortedModel':
                transform_list =\
                    TransformList.load_from_hdf5_group(group['transform_list'])
                return DistortedModel(model, transform_list)
            elif class_name == 'ProjectedModel':
                basis = Basis.load_from_hdf5_group(group['basis'])
                error = group['error'].value
                return ProjectedModel(model, basis, error=error)
            elif class_name == 'ScaledModel':
                scale_factor = group.attrs['scale_factor']
                return ScaledModel(model, scale_factor)
            elif class_name == 'ExpandedModel':
                expander = load_expander_from_hdf5_group(group['expander'])
                return ExpandedModel(model, expander)
            elif class_name == 'RenamedModel':
                subgroup = group['parameters']
                iparameter = 0
                parameters = []
                while '{:d}'.format(iparameter) in subgroup.attrs:
                    parameters.append(subgroup.attrs['{:d}'.format(iparameter)])
                    iparameter += 1
                return RenamedModel(model, parameters)
            elif class_name == 'RestrictedModel':
                minima = get_hdf5_value(group['minima'])
                maxima = get_hdf5_value(group['maxima'])
                bounds = [bound_tuple for bound_tuple in zip(minima, maxima)]
                return RestrictedModel(model, bounds)
            elif class_name == 'SlicedModel':
                subgroup = group['constant_parameters']
                constant_parameters =\
                    {key: subgroup.attrs[key] for key in subgroup.attrs}
                return SlicedModel(model, **constant_parameters)
            else:
                raise RuntimeError("This should never happen. Is there a " +\
                    "model loading function missing from LoadModel.py?")
        elif class_name in compound_model_classes:
            subgroup = group['models']
            imodel = 0
            names = []
            models = []
            while '{}'.format(imodel) in subgroup:
                subsubgroup = subgroup['{}'.format(imodel)]
                names.append(subsubgroup.attrs['name'])
                models.append(load_model_from_hdf5_group(subsubgroup))
                imodel += 1
            if class_name == 'SumModel':
                return SumModel(names, models)
            elif class_name == 'DirectSumModel':
                return DirectSumModel(names, models)
            elif class_name == 'TiedModel':
                shared_name = group.attrs['shared_name']
                tied_parameters =\
                    [element for element in group['tied_parameters'].value]
                return TiedModel(names, models, shared_name, *tied_parameters)
            elif class_name == 'ProductModel':
                return ProductModel(names, models)
            elif class_name == 'CompositeModel':
                expression =\
                    Expression.load_from_hdf5_group(group['expression'])
                if 'gradient_expressions' in group:
                    subgroup = group['gradient_expressions']
                    gradient_expressions = np.ndarray((len(names),), dtype=object)
                    for iname in range(len(names)):
                        subsubgroup = subgroup['{}'.format(iname)]
                        gradient_expressions[iname] =\
                            Expression.load_from_hdf5_group(subsubgroup)
                else:
                    gradient_expressions = None
                if 'hessian_expressions' in group:
                    subgroup = group['hessian_expressions']
                    hessian_expressions =\
                        np.ndarray((len(names),) * 2, dtype=object)
                    for iname1 in range(len(names)):
                        for iname2 in range(iname1 + 1):
                            this_block = Expression.load_from_hdf5_group(\
                                subgroup['{}_{}'.format(iname1, iname2)])
                            hessian_expressions[iname1,iname2] = this_block
                            hessian_expressions[iname2,iname1] = this_block
                else:
                    hessian_expressions = None
                return CompositeModel(expression, names, models,\
                    gradient_expressions, hessian_expressions)
            else:
                raise RuntimeError("This should never happen. Is there a " +\
                    "model loading function missing from LoadModel.py?")
        else:
            print("Got the following error with a {!s}:".format(class_name))
            raise

