"""
File: pylinex/model/CompoundModel.py
Author: Keith Tauscher
Date: 15 Jan 2018

Description: File containing an abstract class representing a Model composed of
             many submodels.
"""
from .Model import Model

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class CompoundModel(Model):
    """
    Abstract class representing a Model composed of many submodels.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializer always throws error because this is an abstract class.
        """
        raise NotImplementedError("CompoundModel cannot be initialized " +\
            "directly. It must be subclassed and the initializer of the " +\
            "subclass must be defined.")
    
    @property
    def names(self):
        """
        Property storing the list of names of submodels.
        """
        if not hasattr(self, '_names'):
            raise AttributeError("names referenced before it was set.")
        return self._names
    
    @names.setter
    def names(self, value):
        """
        Setter for the list of names of submodels.
        
        value: must be a list of strings
        """
        if all([isinstance(element, basestring) for element in value]):
            self._names = [element for element in value]
        else:
            raise TypeError("At least one of the given names was not a " +\
                "string.")
    
    @property
    def models(self):
        """
        Property storing the submodels of this CompoundModel.
        """
        if not hasattr(self, '_models'):
            raise AttributeError("models was referenced before it was set.")
        return self._models
   
    @models.setter
    def models(self, value):
        """
        Setter for the submodels of this CompoundModel. This must be called
        after the setter for the names of the submodels.
        
        value: sequence of models of same length as names
        """
        if len(value) == len(self.names):
            if all([isinstance(element, Model) for element in value]):
                self._models = [element for element in value]
            else:
                raise TypeError("At least one of the given models was not " +\
                    "a Model object.")
        else:
            raise ValueError("Length of models sequence was not the same " +\
                "as the number of names given.")
    
    def __getitem__(self, key):
        """
        Gets the model associated with the given name.
        
        key: string describing desired submodel; either name string of the
             submodel or (only if this CompoundModel object is composed of
             other CompoundModel objects) sequence of nested name strings
             joined by '/'
        
        returns: Model object representing desired (possibly nested) submodel
        """
        return self.submodel(key)
    
    def submodel(self, key):
        """
        Finds the submodel associated with the given key.
        
        key: string describing desired submodel; either name string of the
             submodel or (only if this CompoundModel object is composed of
             other CompoundModel objects) sequence of nested name strings
             joined by '/'
        
        returns: Model object representing desired (possibly nested) submodel
        """
        split_by_slash = key.split('/')
        more_expected = (len(split_by_slash) > 1)
        name = split_by_slash[0]
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError("The given name of a submodel was not found.")
        else:
            model = self.models[index]
        more_had = isinstance(model, CompoundModel)
        if more_expected:
            if more_had:
                new_key = '/'.join(split_by_slash[1:])
                return model.submodel(new_key)
            else:
                raise KeyError("The key given to the submodel function " +\
                    "implied more nesting of CompoundModel's than actually " +\
                    "exists.")
        else:
            return model
    
    @property
    def parameters(self):
        """
        Property storing the list of string parameters describing this model.
        It is formed by prepending each parameter of a submodel with the name
        of the submodel and combining the submodels' lists of parameters in the
        order they are given in the models/names property.
        """
        if not hasattr(self, '_parameters'):
             self._parameters = []
             for (iname, name) in enumerate(self.names):
                 self._parameters = self._parameters +\
                     ['{0!s}_{1!s}'.format(name, parameter)\
                     for parameter in self.models[iname].parameters]
        return self._parameters
    
    @property
    def partition_slices(self):
        """
        Property storing the slices which can be used to extract the parameters
        necessary for a given submodel from the parameters given to the full
        model.
        """
        if not hasattr(self, '_partition_slices'):
            self._partition_slices = []
            current_index = 0
            for model in self.models:
                next_index = current_index + model.num_parameters
                self._partition_slices.append(slice(current_index, next_index))
                current_index = next_index
        return self._partition_slices
    
    def submodel_partition_slice(self, key):
        """
        Finds the slice of parameters associated with the given (possibly
        nested) model.
        
        key: string describing desired submodel; either name string of the
             submodel or (only if this CompoundModel object is composed of
             other CompoundModel objects) sequence of nested name strings
             joined by '/'
        
        returns: slice describing how to extract parameters for the submodel
                 described by the given key
        """
        split_by_slash = key.split('/')
        name = split_by_slash[0]
        more_expected = (len(split_by_slash) > 1)
        try:
            index = self.names.index(name)
        except ValueError:
            raise KeyError("The given name of a submodel was not found.")
        else:
            model = self.models[index]
            partition = self.partition_slices[index]
        more_had = isinstance(model, CompoundModel)
        if more_expected:
            if more_had:
                new_key = '/'.join(split_by_slash[1:])
                subpartition = model.submodel_partition_slice(new_key)
                return slice(partition.start + subpartition.start,\
                    partition.start + subpartition.stop)
            else:
                raise KeyError("The key given to the submodel function " +\
                    "implied more nesting of CompoundModel's than actually " +\
                    "exists.")
        else:
            return partition
    
    def partition_parameters(self, parameters):
        """
        Partitions the given array of parameters into a list of arrays of
        parameters to pass on to the submodels.
        
        parameters: 1D numpy.ndarray with length given by sum of all submodel
                    num_parameters values
        
        returns: list of same length as models containing arrays of parameters
                 to pass on to each model
        """
        return [parameters[partition] for partition in self.partition_slices]
    
    @property
    def bounds(self):
        """
        Property storing natural bounds for this Model. They are taken from the
        submodels.
        """
        if not hasattr(self, '_bounds'):
            self._bounds = {}
            for (model_name, model) in zip(self.names, self.models):
                for parameter_name in model.parameters:
                    full_name =\
                        '{0!s}_{1!s}'.format(model_name, parameter_name)
                    self._bounds[full_name] = model.bounds[parameter_name]
        return self._bounds

