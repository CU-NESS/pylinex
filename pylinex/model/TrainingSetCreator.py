"""
File: pylinex/model/TrainingSetCreator.py
Author: Keith Tauscher
Date: 20 Jun 2018

Description: File containing class which creates sets of curves given models
             and prior distributions.
"""
from __future__ import division
import os, time, h5py
import numpy as np
from ..util import bool_types, int_types
from distpy import DistributionSet
from .Model import Model
from .LoadModel import load_model_from_hdf5_group
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TrainingSetCreator(object):
    """
    Class which creates sets of curves given models and prior distributions.
    """
    def __init__(self, model, prior_set, num_curves, file_name, seed=None,\
        allow_errors=True, verbose=True):
        """
        Creates a training set of curves from the given models and the given
        prior region.
        
        model: Model object which will be applied to sample parameters
        prior_set: DistributionSet object from which parameters will be sampled
        num_curves: integer number of curves this object will generate
        file_name: either a file location where an hdf5 file can be saved or a
                   location where it has already begun being saved
        seed: either None if no seed or a 32-bit unsigned integer
        verbose: boolean determining if message is printed after each
                 convolution (i.e. pair of beam+maps)
        """
        self.verbose = verbose
        self.allow_errors = allow_errors
        self.seed = seed
        self.file_name = file_name
        self.num_curves = num_curves
        self.model = model
        self.prior_set = prior_set
    
    @property
    def allow_errors(self):
        """
        Property storing a boolean which determines whether errors are allowed
        (True) or disallowed (False). If they are disallowed, they are raised
        directly.
        """
        if not hasattr(self, '_allow_errors'):
            raise AttributeError("allow_errors was referenced before it " +\
                "was set.")
        return self._allow_errors
    
    @allow_errors.setter
    def allow_errors(self, value):
        """
        Setter for the allow_errors switch.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._allow_errors = value
        else:
            raise TypeError("allow_errors was set to a non-bool.")
    
    @property
    def seed(self):
        """
        Property storing the seed for the random number generator (for
        reproducibility).
        """
        if not hasattr(self, '_seed'):
            raise AttributeError("seed was referenced before it was set.")
        return self._seed
    
    @seed.setter
    def seed(self, value):
        """
        Setter for the seed for the random number generator (for
        reproducibility).
        
        value: either None for no seed or a 32-bit unsigned integer
        """
        if (value is None) or (type(value) in int_types):
            self._seed = value
        else:
            raise TypeError("seed was set to neither None nor an integer.")
    
    @property
    def model(self):
        """
        Property storing the Model object which will be used to create the
        training set curves.
        """
        if not hasattr(self, '_model'):
            raise AttributeError("model was referenced before it was set.")
        return self._model
    
    @model.setter
    def model(self, value):
        """
        Setter for the model which will be used to create the training set
        curves.
        
        value: a Model object
        """
        if isinstance(value, Model):
            self._model = value
        else:
            raise TypeError("model was set to a non-Model object.")
    
    @property
    def prior_set(self):
        """
        Property storing the DistributionSet which is used to draw model
        parameter vector realizations.
        """
        if not hasattr(self, '_prior_set'):
            raise AttributeError("prior_set was referenced before it was set.")
        return self._prior_set
    
    @prior_set.setter
    def prior_set(self, value):
        """
        Setter for the DistributionSet which is used to draw model parameter
        vector realizations.
        
        value: DistributionSet object describing the same set of parameters as
               the model which will be used to generate training set curves
        """
        if isinstance(value, DistributionSet):
            if set(value.params) == set(self.model.parameters):
                self._prior_set = value
            else:
                not_needed_parameters =\
                    set(value.params) - set(self.model.parameters)
                missing_parameters =\
                    set(self.model.parameters) - set(value.params)
                if not_needed_parameters and missing_parameters:
                    raise ValueError(("There were some parameters ({0}) " +\
                        "which were described in the prior_set but were " +\
                        "not parameters of the model and there were some " +\
                        "parameters ({1}) of the model which were not " +\
                        "described in the prior_set.").format(\
                        not_needed_parameters, missing_parameters))
                elif not_needed_parameters:
                    raise ValueError(("Some parameters ({}) which were " +\
                        "described in the prior_set were not parameters of " +\
                        "the model.").format(not_needed_parameters))
                else:
                    raise ValueError(("Some parameters ({}) which were " +\
                        "included in the model were not described in the " +\
                        "prior_set.").format(missing_parameters))
        else:
            raise TypeError("prior_set was set to a non-DistributionSet " +\
                "object.")
    
    @property
    def verbose(self):
        """
        Property storing a boolean which determines whether or not a message is
        printed after each curve is created.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the boolean which determines whether or not a message is
        printed after each curve is created.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise TypeError("verbose was set to a non-bool.")
    
    @property
    def file_name(self):
        """
        Property storing the name of the hdf5 file in which to save the data
        generated by this object.
        """
        if not hasattr(self, '_file_name'):
            raise AttributeError("file_name referenced before it was set.")
        return self._file_name
    
    @file_name.setter
    def file_name(self, value):
        """
        Setter for the name of the hdf5 file in which to save the data
        generated by this object.
        
        value: string name of hdf5 file, which may or may not exist
        """
        if isinstance(value, basestring):
            self._file_name = value
        else:
            raise TypeError("file_name was set to a non-string.")
    
    @property
    def num_curves(self):
        """
        Property storing the integer number of curves this TrainingSetCreator
        will create.
        """
        if not hasattr(self, '_num_curves'):
            raise AttributeError("num_curves was referenced before it was " +\
                "set.")
        return self._num_curves
    
    @num_curves.setter
    def num_curves(self, value):
        """
        Setter for the integer number of curves this TrainingSetCreator will
        create.
        
        value: positive integer number of curves to generate
        """
        if type(value) in int_types:
            if value > 0:
                self._num_curves = value
            else:
                raise ValueError("num_curves was set to a non-positive " +\
                    "integer.")
        else:
            raise TypeError("num_curves was set to a non-integer.")
    
    @property
    def file(self):
        """
        Property storing the h5py File object in which the data generated by
        this object will be saved.
        """
        if not hasattr(self, '_file'):
            if os.path.exists(self.file_name):
                self._file = h5py.File(self.file_name, 'r+')
            else:
                self._file = h5py.File(self.file_name, 'w')
                self.model.fill_hdf5_group(self._file.create_group('model'))
                self.prior_set.fill_hdf5_group(\
                    self._file.create_group('prior_set'))
                self._file.create_group('parameters')
                self._file.create_group('curves')
                self._file.attrs['next_index'] = 0
                if self.seed is not None:
                    self._file.attrs['seed'] = self.seed
        return self._file
    
    def generate(self):
        """
        Generates (or continues generating) curves for a set of training
        curves. These curves are saved in the hdf5 file at file_name as they
        are generated.
        """
        completed = self.file.attrs['next_index']
        try:
            np.random.seed(self.seed)
            for icurve in range(self.num_curves):
                curve_string = 'curve_{:d}'.format(icurve)
                parameter_draw = self.prior_set.draw()
                if icurve < completed:
                    continue
                parameter_draw = np.array([parameter_draw[parameter]\
                    for parameter in self.model.parameters])
                try:
                    curve = self.model(parameter_draw)
                except KeyboardInterrupt:
                    raise
                except:
                    if self.allow_errors:
                        curve = np.array([np.nan])
                    else:
                        raise
                self.file['parameters'].create_dataset(curve_string,\
                    data=parameter_draw)
                self.file['curves'].create_dataset(curve_string, data=curve)
                completed += 1
                self.file.attrs['next_index'] = completed
                self.close()
                if self.verbose:
                    print("Finished curve #{0:d}/{1:d} at {2!s}.".format(\
                        completed, self.num_curves, time.ctime()))
                low_floor =\
                    int(np.floor(((100 * completed) - 50) / self.num_curves))
                high_floor =\
                    int(np.floor(((100 * completed) + 50) / self.num_curves))
                if (low_floor != high_floor) and\
                    (completed != self.num_curves):
                    print("Done with {0:d}% of {1:d} curves at {2!s}.".format(\
                        high_floor, self.num_curves, time.ctime()))
        except KeyboardInterrupt:
            completed_according_to_file = self.file.attrs['next_index']
            curve_string_to_delete_if_present =\
                'curve_{:d}'.format(completed_according_to_file)
            parameters_string =\
                'parameters/{!s}'.format(curve_string_to_delete_if_present)
            curves_string =\
                'curves/{!s}'.format(curve_string_to_delete_if_present)
            if parameters_string in self.file:
                del self.file[parameters_string]
            if curves_string in self.file:
                del self.file[curves_string]
            self.file.close()
            print(("Stopping curve generation due to KeyboardInterrupt at " +\
                "{!s}.").format(time.ctime()))
    
    def get_training_set(self, return_parameters=False, return_model=False,\
        return_prior_set=False):
        """
        Gets the (assumed already generated) training set in the file of this
        TrainingSetCreator.
        
        return_model: if True, model is also returned
        return_prior_set: if True, prior_set is also returned
        
        returns: numpy.ndarray whose shape is (num_curves, num_channels),
                 followed by parameters, model and/or prior_set if applicable
        """
        group = self.file['curves']
        de_facto_num_curves = 0
        while 'curve_{:d}'.format(de_facto_num_curves) in group:
            de_facto_num_curves += 1
        if de_facto_num_curves == 0:
            raise RuntimeError("No curves have been generated yet.")
        else:
            num_channels = np.max([group['curve_{:d}'.format(icurve)].size\
                for icurve in range(de_facto_num_curves)])
        training_set = np.ndarray((de_facto_num_curves, num_channels))
        to_keep = []
        for icurve in range(de_facto_num_curves):
            curve = group['curve_{:d}'.format(icurve)][()]
            if np.any(np.isnan(curve)):
                training_set[icurve,:] = np.nan
            else:
                training_set[icurve,:] = curve
                to_keep.append(icurve)
        to_keep = np.array(to_keep)
        self.close()
        return_value = [training_set[to_keep,:]]
        if return_parameters:
            parameters =\
                np.ndarray((de_facto_num_curves, self.model.num_parameters))
            group = self.file['parameters']
            for icurve in range(de_facto_num_curves):
                parameters[icurve,:] = group['curve_{:d}'.format(icurve)][()]
            parameters = parameters[to_keep,:]
            parameters = {param: parameters[:,iparam]\
                for (iparam, param) in enumerate(self.model.parameters)}
            return_value = return_value + [parameters]
            self.close()
        if return_model:
            return_value = return_value + [self.model]
        if return_prior_set:
            return_value = return_value + [self.prior_set]
        if len(return_value) == 1:
            return return_value[0]
        else:
            return tuple(return_value)
    
    def get_bad_parameters(self):
        """
        Gets the bad parameters which were encountered by this
        TrainingSetCreator (i.e. the parameter values which produce curves that
        throw errors or return nan's)
        
        returns: (model, parameters) where model is a Model object and
                 parameters is a dictionary of arrays indexed by parameter name
        """
        group = self.file['curves']
        de_facto_num_curves = 0
        while 'curve_{:d}'.format(de_facto_num_curves) in group:
            de_facto_num_curves += 1
        to_keep = []
        for icurve in range(de_facto_num_curves):
            if np.any(np.isnan(group['curve_{:d}'.format(icurve)][()])):
                to_keep.append(icurve)
        self.close()
        parameters = np.ndarray((len(to_keep), self.model.num_parameters))
        group = self.file['parameters']
        for (iicurve, icurve) in enumerate(to_keep):
            parameters[iicurve,:] = group['curve_{:d}'.format(icurve)][()]
        parameters = {param: parameters[:,iparam]\
            for (iparam, param) in enumerate(self.model.parameters)}
        return (self.model, parameters)
    
    @staticmethod
    def load_training_set(file_name, return_parameters=False,\
        return_model=False, return_prior_set=False):
        """
        Loads a training set from the TrainingSetCreator which was saved to
        the given file name.
        
        file_name: the file in which a TrainingSetCreator was saved
        return_parameters: if True, dictionary of parameter values is also
                           returned
        return_model: if True, model is also returned
        return_prior_set: if True, prior_set is also returned
        
        returns: numpy.ndarray whose shape is (num_curves, num_channels),
                 followed by model and/or prior_set if applicable
        """
        hdf5_file = h5py.File(file_name, 'r')
        group = hdf5_file['curves']
        num_curves = 0
        while 'curve_{:d}'.format(num_curves) in group:
            num_curves += 1
        if num_curves == 0:
            raise RuntimeError("No curves have been stored in the given file.")
        else:
            num_channels = np.max([group['curve_{:d}'.format(icurve)].size\
                for icurve in range(de_facto_num_curves)])
        training_set = np.ndarray((num_curves, num_channels))
        to_keep = []
        for icurve in range(num_curves):
            curve = group['curve_{:d}'.format(icurve)][()]
            if np.any(np.isnan(curve)):
                training_set[icurve,:] = np.nan
            else:
                training_set[icurve,:] = curve
                to_keep.append(icurve)
        to_keep = np.array(to_keep)
        return_value = [training_set[to_keep,:]]
        model = load_model_from_hdf5_group(hdf5_file['model'])
        if return_parameters:
            parameters = np.ndarray((num_curves, model.num_parameters))
            group = hdf5_file['parameters']
            for icurve in range(num_curves):
                parameters[icurve,:] = group['curve_{:d}'.format(icurve)][()]
            parameters = parameters[to_keep,:]
            parameters = {parameter: parameters[:,iparameter]\
                for (iparameter, parameter) in enumerate(model.parameters)}
            return_value = return_value + [parameters]
        if return_model:
            return_value = return_value + [model]
        if return_prior_set:
            prior_set =\
                DistributionSet.load_from_hdf5_group(hdf5_file['prior_set'])
            return_value = return_value + [prior_set]
        hdf5_file.close()
        if len(return_value) == 1:
            return return_value[0]
        else:
            return return_value
    
    @staticmethod
    def load_bad_parameters(file_name):
        """
        Gets the bad parameters which were encountered by this
        TrainingSetCreator (i.e. the parameter values which produce curves that
        throw errors or return nan's)
        
        returns: (model, parameters) where model is a Model object and
                 parameters is a dictionary of arrays indexed by parameter name
        """
        hdf5_file = h5py.File(file_name, 'r')
        group = hdf5_file['curves']
        num_curves = 0
        while 'curve_{:d}'.format(num_curves) in group:
            num_curves += 1
        to_keep = []
        for icurve in range(num_curves):
            if np.any(np.isnan(group['curve_{:d}'.format(icurve)][()])):
                to_keep.append(icurve)
        model = load_model_from_hdf5_group(hdf5_file['model'])
        parameters = np.ndarray((len(to_keep), model.num_parameters))
        group = hdf5_file['parameters']
        for (iicurve, icurve) in enumerate(to_keep):
            parameters[iicurve,:] = group['curve_{:d}'.format(icurve)][()]
        parameters = {parameter: parameters[:,iparameter]\
            for (iparameter, parameter) in enumerate(model.parameters)}
        hdf5_file.close()
        return (model, parameters)
    
    def close(self):
        """
        Closes the file containing the curves made by this object.
        """
        if hasattr(self, '_file'):
            self.file.close()
            del self._file

