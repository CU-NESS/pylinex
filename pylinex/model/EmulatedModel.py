"""
File: pylinex/model/EmulatedModel.py
Author: Keith Tauscher
Date: 8 Jul 2019

Description: File containing a class that uses the emupy package's emulator to
             quickly map inputs to outputs, best used when the true mapping is
             very slow and a large training set has been built up.
"""
import numpy as np
from ..util import bool_types, int_types, numerical_types, sequence_types,\
    create_hdf5_dataset, get_hdf5_value
from .LoadableModel import LoadableModel
try:
    from sklearn.gaussian_process.kernels import Kernel, ConstantKernel,\
        DotProduct, ExpSineSquared, Matern, RBF, RationalQuadratic,\
        WhiteKernel, Exponentiation, Product, Sum, PairwiseKernel
    from emupy import Emu as Emulator
except:
    have_emupy_and_sklearn = False
else:
    have_emupy_and_sklearn = True
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

def save_kernel_to_hdf5_group(group, kernel):
    """
    Saves the given Kernel into the given hdf5 file group.
    
    group: hdf5 file group in which to save the Kernel.
    kernel: a Kernel object from sklearn.gaussian_process.Kernels
    """
    if isinstance(kernel, ConstantKernel):
        group.attrs['class'] = 'ConstantKernel'
        group.attrs['constant_value'] = kernel.constant_value
    elif isinstance(kernel, DotProduct):
        group.attrs['class'] = 'DotProduct'
        group.attrs['sigma_0'] = kernel.sigma_0
    elif isinstance(kernel, ExpSineSquared):
        group.attrs['class'] = 'ExpSineSquared'
        group.attrs['length_scale'] = kernel.length_scale
        group.attrs['periodicity'] = kernel.periodicity
    elif isinstance(kernel, Matern):
        group.attrs['class'] = 'Matern'
        group.attrs['length_scale'] = kernel.length_scale
        group.attrs['nu'] = kernel.nu
    elif isinstance(kernel, RBF):
        group.attrs['class'] = 'RBF'
        group.attrs['length_scale'] = kernel.length_scale
    elif isinstance(kernel, RationalQuadratic):
        group.attrs['class'] = 'RationalQuadratic'
        group.attrs['length_scale'] = kernel.length_scale
        group.attrs['alpha'] = kernel.alpha
    elif isinstance(kernel, WhiteKernel):
        group.attrs['class'] = 'WhiteKernel'
        group.attrs['noise_level'] = kernel.noise_level
    elif isinstance(kernel, Exponentiation):
        group.attrs['class'] = 'Exponentiation'
        group.attrs['exponent'] = kernel.exponent
        save_kernel_to_hdf5_group(group.create_group('kernel'), kernel.kernel)
    elif isinstance(kernel, Product):
        group.attrs['class'] = 'Product'
        save_kernel_to_hdf5_group(group.create_group('k1'), kernel.k1)
        save_kernel_to_hdf5_group(group.create_group('k2'), kernel.k2)
    elif isinstance(kernel, Sum):
        group.attrs['class'] = 'Sum'
        save_kernel_to_hdf5_group(group.create_group('k1'), kernel.k1)
        save_kernel_to_hdf5_group(group.create_group('k2'), kernel.k2)
    elif isinstance(kernel, PairwiseKernel):
        raise NotImplementedError("The PairwiseKernel is not savable or " +\
            "loadable with the functions in EmulatedModel.py.")
    else:
        raise TypeError("The object to be saved does not seem to be a Kernel.")

def load_kernel_from_hdf5_group(group):
    """
    Loads a Kernel from the given hdf5 file group.
    
    group: hdf5 file group from which to load the Kernel.
    
    returns: a Kernel object previously saved into group
    """
    try:
        class_name = group.attrs['class']
    except:
        raise ValueError("The given group does not contain a 'class' " +\
            "attribute, so a Kernel cannot be loaded from it.")
    if class_name == 'ConstantKernel':
        return ConstantKernel(constant_value=group.attrs['constant_value'])
    elif class_name == 'DotProduct':
        return DotProduct(sigma_0=group.attrs['sigma_0'])
    elif class_name == 'ExpSineSquared':
        return ExpSineSquared(length_scale=group.attrs['length_scale'],\
            periodicity=group.attrs['periodicity'])
    elif class_name == 'Matern':
        return Matern(length_scale=group.attrs['length_scale'],\
            nu=group.attrs['nu'])
    elif class_name == 'RBF':
        return RBF(length_scale=group.attrs['length_scale'])
    elif class_name == 'RationalQuadratic':
        return RationalQuadratic(length_scale=group.attrs['length_scale'],\
            alpha=group.attrs['alpha'])
    elif class_name == 'WhiteKernel':
        return WhiteKernel(noise_level=group.attrs['noise_level'])
    elif class_name == 'Exponentiation':
        return Exponentiation(load_kernel_from_hdf5_group(group['kernel']),\
            group.attrs['exponent'])
    elif class_name == 'Product':
        return Product(load_kernel_from_hdf5_group(group['k1']),\
            load_kernel_from_hdf5_group(group['k2']))
    elif class_name == 'Sum':
        return Sum(load_kernel_from_hdf5_group(group['k1']),\
            load_kernel_from_hdf5_group(group['k2']))
    else:
        raise ValueError("The class saved in this group does not seem to " +\
            "be a Kernel object.")

class EmulatedModel(LoadableModel):
    """
    Class that uses the emupy package's emulator to quickly map inputs to
    outputs, best used when the true mapping is very slow and a large training
    set has been built up.
    """
    def __init__(self, parameters, inputs, outputs, error=None,\
        num_modes=None, kernel=None, verbose=True):
        """
        Initializes a new EmulatedModel object with the given training set.
        
        parameters: sequence of strings containing parameter names
        inputs: 2D numpy array of shape (num_samples, num_parameters)
                containing input parameter vectors
        outputs: 2D numpy array of shape (num_samples, num_channels) containing
                 output data vectors
        error: if None, error is 1 at all channels
               if a number, error is that number at all channels
               otherwise, should be a 1D array of length num_channels
        num_modes: integer number of modes, None (default) means all modes used
        kernel: kernel to pass to Gaussian process regressor. A sequence of
                num_modes kernels can also be passed here, in which case it is
                assumed that the Emulator is already trained
        verbose: boolean determining if messages should be printed during train
                 (default: True). Not used if kernel given is a sequence
        """
        if not have_emupy_and_sklearn:
            raise ImportError("The EmulatedModel class can only be used if " +\
                "emupy is installed. See https://github.com/nkern/emupy " +\
                "for download and/or details.")
        self.verbose = verbose
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs
        self.error = error
        self.num_modes = num_modes
        self.kernel = kernel
        self.emulator
    
    @property
    def verbose(self):
        """
        Property storing whether messages are printed during train.
        """
        if not hasattr(self, '_verbose'):
            raise AttributeError("verbose was referenced before it was set.")
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        """
        Setter for the verbose property.
        
        value: True or False
        """
        if type(value) in bool_types:
            self._verbose = value
        else:
            raise TypeError("verbose was set to neither True nor False.")
    
    @property
    def parameters(self):
        """
        Property storing the names of the parameters of this model.
        """
        if not hasattr(self, '_parameters'):
            raise AttributeError("parameters was referenced before it was " +\
                "set.")
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        """
        Setter for the names of the parameters of this model.
        
        value: sequence of string objects
        """
        if type(value) in sequence_types:
            if all([isinstance(element, basestring) for element in value]):
                self._parameters = [element for element in value]
            else:
                raise TypeError("At least one element of parameters was " +\
                    "not a string.")
        else:
            raise TypeError("parameters was set to a non-sequence.")
    
    @property
    def kernel(self):
        """
        Property storing the kernel to use in Gaussian process regression. This
        should come from sklearn.gaussian_process.kernels
        """
        if not hasattr(self, '_kernel'):
            raise AttributeError("kernel was referenced before it was set.")
        return self._kernel
    
    @kernel.setter
    def kernel(self, value):
        """
        Setter for the kernel to use in Gaussian process regression.
        
        value: either None for a default kernel (not recommended) or a kernel
               from sklearn.gaussian_process.kernels
        """
        if type(value) is type(None):
            print("Warning: No kernel given. Using a default one. This " +\
                "could be bad.")
            self._kernel = None
        elif isinstance(value, Kernel):
            self._kernel = value
        elif type(value) in sequence_types:
            if len(value) == self.num_modes:
                if all([isinstance(element, Kernel) for element in value]):
                    self._kernel = [element for element in value]
                else:
                    raise TypeError("At least one of the elements of the " +\
                        "kernel given to the EmulatedModel was not a " +\
                        "Kernel object.")
            else:
                raise ValueError("The number of Kernel objects given was not ")
        else:
            raise TypeError("kernel was set to neither a Kernel object nor " +\
                "a sequence of Kernel objects. See " +\
                "sklearn.gaussian_process.kernels for details on available " +\
                "kernels.")
    
    @property
    def trained_already(self):
        """
        Property storing whether or not this Emulator has already been trained.
        """
        if not hasattr(self, '_trained_already'):
            self._trained_already = (type(self.kernel) in sequence_types)
        return self._trained_already
    
    @property
    def inputs(self):
        """
        Property storing the inputs used for training.
        """
        if not hasattr(self, '_inputs'):
            raise AttributeError("inputs was referenced before it was set.")
        return self._inputs
    
    @inputs.setter
    def inputs(self, value):
        """
        Setter for inputs used for training the emulator at the heart of this
        model.
        
        value: 2D numpy array of shape (num_samples, num_parameters)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                value = value[:,np.newaxis]
            if value.ndim == 2:
                if value.shape[-1] == self.num_parameters:
                    self._inputs = value
                else:
                    raise ValueError(("Last axis of inputs ({0:d}) was not " +\
                        "the same as the number of parameters " +\
                        "({1:d}).").format(value.shape[-1],\
                        self.num_parameters))
            else:
                raise ValueError("inputs was neither 1- or 2-dimensional.")
        else:
            raise TypeError("inputs was set to a non-sequence.")
    
    @property
    def num_samples(self):
        """
        Property storing the number of training samples used in this emulator.
        """
        if not hasattr(self, '_num_samples'):
            self._num_samples = self.inputs.shape[0]
        return self._num_samples
    
    @property
    def outputs(self):
        """
        Property storing the outputs used for training this emulator.
        """
        if not hasattr(self, '_outputs'):
            raise AttributeError("outputs was referenced before it was set.")
        return self._outputs
    
    @outputs.setter
    def outputs(self, value):
        """
        Setter for the outputs used to train this emulator.
        
        value: 2D numpy array of shape (num_samples, num_channels)
        """
        if type(value) in sequence_types:
            value = np.array(value)
            if value.ndim == 1:
                value = value[:,np.newaxis]
            if value.ndim == 2:
                if value.shape[0] == self.num_samples:
                    self._outputs = value
                else:
                    raise ValueError(("First axis of outputs ({0:d}) was " +\
                        "not the same as the number of samples " +\
                        "({1:d}).").format(value.shape[0], self.num_samples))
            else:
                raise ValueError("outputs was neither 1- or 2-dimensional.")
        else:
            raise TypeError("outputs was set to a non-sequence.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in outputs of this model.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.outputs.shape[-1]
        return self._num_channels
    
    @property
    def num_modes(self):
        """
        Property storing the number of modes to use in PCA.
        None means all modes are used.
        """
        if not hasattr(self, '_num_modes'):
            raise AttributeError("num_modes was referenced before it was set.")
        return self._num_modes
    
    @num_modes.setter
    def num_modes(self, value):
        """
        Setter for the number of modes to use in PCA.
        
        value: integer number of modes (0 means don't do PCA) or None (which
               also means don't do PCA).
        """
        if type(value) is type(None):
            value = self.num_channels
        if type(value) in int_types:
            if value > 0:
                self._num_modes = value
            else:
                raise ValueError("num_modes was set to a negative integer.")
        else:
            raise TypeError("num_modes was set to a non-integer.")
    
    def _fiducial(self):
        """
        Computes fiducial inputs and outputs to give to emulator.
        """
        mean_input = np.mean(self.inputs, axis=0)
        input_ranges =\
            np.max(self.inputs, axis=0) - np.min(self.inputs, axis=0)
        squared_scaled_input_distances =\
            np.sum(np.power((self.inputs - mean_input[np.newaxis,:]) /\
            input_ranges[np.newaxis,:], 2), axis=-1)
        index_closest_to_mean = np.argmin(squared_scaled_input_distances)
        self._fiducial_input = self.inputs[index_closest_to_mean]
        self._fiducial_output = self.outputs[index_closest_to_mean]
    
    @property
    def fiducial_input(self):
        """
        Property storing a fiducial input vector to give to the emulator.
        """
        if not hasattr(self, '_fiducial_input'):
            self._fiducial()
        return self._fiducial_input
    
    @property
    def fiducial_output(self):
        """
        Property storing the output associated with the fiducial input vector
        to give to the emulator.
        """
        if not hasattr(self, '_fiducial_output'):
            self._fiducial()
        return self._fiducial_output
    
    @property
    def error(self):
        """
        Property storing the error which is used to define the inner product in
        output space.
        """
        if not hasattr(self, '_error'):
            raise AttributeError("error was referenced before it was set.")
        return self._error
    
    @error.setter
    def error(self, value):
        """
        Setter for the error used to define the inner product in output space.
        
        value: if None, error is 1 at all channels
               if a number, error is that number at all channels
               otherwise, should be a 1D array of length num_channels
        """
        if type(value) is type(None):
            self._error = np.ones(self.num_channels)
        elif type(value) in numerical_types:
            self.error = value * np.ones(self.num_channels)
        elif type(value) in sequence_types:
            value = np.array(value)
            expected_shape = (self.num_channels,)
            if value.shape == expected_shape:
                if np.all(value > 0):
                    self._error = value
                else:
                    raise ValueError("At least one element of error was " +\
                        "non-positive.")
            else:
                raise ValueError(("The shape of the given error, {0}, was " +\
                    "not the expected shape, {1}.").format(value.shape,\
                    expected_shape))
        else:
            raise TypeError("error was neither None, a number, or a 1D array.")
    
    @property
    def emulator(self):
        """
        Property storing the emulator from emupy at the heart of this model.
        """
        if not hasattr(self, '_emulator'):
            emulator = Emulator()
            emulator.reg_meth =  'gaussian'
            emulator.N_modes = self.num_modes
            emulator.N_data = self.num_channels
            emulator.N_samples = self.num_samples
            emulator.cov_whiten = False
            emulator.cov_rescale = False
            emulator.lognorm = False
            emulator.use_pca = True
            emulator.fid_grid = self.fiducial_input
            emulator.sphere(self.inputs, fid_grid=emulator.fid_grid,\
                save_chol=True, norotate=True)
            emulator.fid_data = self.fiducial_output / self.error
            weighted_outputs = self.outputs / self.error[np.newaxis,:]
            emulator.klt(weighted_outputs, normalize=True)
            if self.trained_already:
                gp_kwargs_arr = [{'kernel': kernel, 'optimizer': None}\
                    for kernel in self.kernel]
                verbose = False
            else:
                gp_kwargs_arr = None
                emulator.gp_kwargs =\
                    {'kernel': self.kernel, 'n_restarts_optimizer': 5}
                verbose = self.verbose
            emulator.train(weighted_outputs, self.inputs, verbose=verbose,\
                gp_kwargs_arr=gp_kwargs_arr)
            self._emulator = emulator
            self._kernel = [self._emulator.GP[index].kernel_\
                for index in range(len(self._emulator.GP))]
        return self._emulator
    
    def __call__(self, parameters):
        """
        Evaluates the emulator at the given parameters.
        
        parameters: 1D array of length num_parameters
        """
        return self.error * self.emulator.predict(parameters,\
            fast=True, sphere=True, output=True)[0][0]
    
    def fill_hdf5_group(self, group):
        """
        Saves this model to group in a way such that it can be loaded later.
        
        group: hdf5 file group in which to save this model
        """
        group.attrs['class'] = 'EmulatedModel'
        group.attrs['num_modes'] = self.num_modes
        create_hdf5_dataset(group, 'parameters', data=self.parameters)
        create_hdf5_dataset(group, 'inputs', data=self.inputs)
        create_hdf5_dataset(group, 'outputs', data=self.outputs)
        create_hdf5_dataset(group, 'error', data=self.error)
        subgroup = group.create_group('kernel')
        if hasattr(self, '_emulator'):
            for (ikernel, kernel) in enumerate(self.kernel):
                subsubgroup =\
                    subgroup.create_group('kernel{:d}'.format(ikernel))
                save_kernel_to_hdf5_group(subsubgroup, kernel)
        else:
            save_kernel_to_hdf5_group(subgroup.create_group('sole'),\
                self.kernel)
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an EmulatedModel from the given hdf5 file group.
        
        group: hdf5 file group in which an EmulatedModel object was once saved
        
        returns: an EmulatedModel object loaded from the given group
        """
        if ('class' in group.attrs) and\
            (group.attrs['class'] == 'EmulatedModel'):
            num_modes = group.attrs['num_modes']
            parameters = get_hdf5_value(group['parameters'])
            inputs = get_hdf5_value(group['inputs'])
            outputs = get_hdf5_value(group['outputs'])
            error = get_hdf5_value(group['error'])
            subgroup = group['kernel']
            if 'sole' in subgroup:
                kernel = load_kernel_from_hdf5_group(subgroup['sole'])
            else:
                (ikernel, kernel) = (0, [])
                while 'kernel{:d}'.format(ikernel) in subgroup:
                    kernel.append(load_kernel_from_hdf5_group(\
                        subgroup['kernel{:d}'.format(ikernel)]))
                    ikernel += 1
            return EmulatedModel(parameters, inputs, outputs, error=error,\
                num_modes=num_modes, kernel=kernel)
        else:
            raise RuntimeError("The given group does not seem to contain " +\
                "an EmulatedModel object.")

