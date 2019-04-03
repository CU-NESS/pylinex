"""
File: pylinex/util/TrainingSetIterator.py
Author: Keith Tauscher
Date: 3 Sep 2017

Description: File containing class which combines multiple training sets into
             one by including every single combination of curves from each
             training set. It also contains functions which aid in the
             splitting up of the output into blocks so as to not overtax
             memory.
"""
from __future__ import division
import numpy as np
from ..util import int_types, sequence_types
from ..expander import Expander, NullExpander
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class TrainingSetIterator(object):
    """
    Class which combines multiple training sets into one by including every
    single combination of curves from each training set. It also contains
    functions which aid in the splitting up of the output into blocks so as to
    not overtax memory.
    """
    def __init__(self, training_sets, expanders=None,\
        max_block_size=2**20, mode='add', curves_to_return=None,\
        return_constituents=False):
        """
        Initializes this TrainingSetIterator with the given training sets.
        
        training_sets: sequence of training sets for many distinct bases
        max_block_size: the maximum number of numbers in a large array
                        (relative to the total memory on the system).
        mode: if 'add', training sets are combined through addition
              if 'multiply', training sets are combined through multiplication
              else, mode must be an expression which is calculable using numpy
                    whose pieces are of the form {###} where ### is the index
                    (starting at 0) of the specific training set
                    (e.g. '{0}*np.power({1},{2})')
        curves_to_return: the number of curves that the user wishes for this
                          iterator to return. The actual number of returned
                          curves is usually within a block or so of this number
        return_constituents: if True, return constituent training sets used to
                                      each output combined training set
                             if False, only combined training set curves are
                                       returned
        """
        self.max_block_size = max_block_size
        self.training_sets = training_sets
        self.expanders = expanders
        if self.num_channels > self.max_block_size:
            raise ValueError("max_block_size must be greater than " +\
                             "num_channels.")
        self.mode = mode
        self.return_constituents = return_constituents
        self.curves_to_return = curves_to_return
        self.iblock = 0
    
    @property
    def expanders(self):
        """
        Property storing the Expander objects (see pylinex.expander module)
        which expand the individual training set curves of each type.
        """
        if not hasattr(self, '_expanders'):
            raise AttributeError("expanders referenced before it was set.")
        return self._expanders
    
    @expanders.setter
    def expanders(self, value):
        """
        Setter for the expanders to use to expand each given training set.
        
        value: if None, expanders is set to a list of NullExpander objects
               otherwise, value should be a sequence of the same length as
                          self.training_sets of objects which are either None
                          or Expander objects
        """
        if type(value) is type(None):
            self._expanders =\
                [NullExpander() for i in range(self.num_training_sets)]
        elif type(value) in sequence_types:
            if len(value) == self.num_training_sets:
                self._expanders = []
                for element in value:
                    if type(element) is type(None):
                        self._expanders.append(NullExpander())
                    elif isinstance(element, Expander):
                        self._expanders.append(element)
                    else:
                        raise TypeError("At least one element in the " +\
                            "expanders sequence was neither None nor an " +\
                            "Expander object.")
            else:
                raise ValueError("expanders was set to a sequence of the " +\
                    "wrong length.")
        else:
            raise TypeError("expanders was set to a non-sequence.")
        
    
    @property
    def return_constituents(self):
        """
        Property storing whether constituent training set curves are returned
        along combined training set curves.
        """
        if not hasattr(self, '_return_constituents'):
            raise AttributeError("return_constituents was referenced " +\
                                 "before it was set.")
        return self._return_constituents
    
    @return_constituents.setter
    def return_constituents(self, value):
        """
        Allows user to choose whether constituent training sets are returned in
        blocks.
        
        value: if True, return constituent training sets used to each output
                        combined training set
               if False, only combined training set curves are returned
        """
        if isinstance(value, bool):
            self._return_constituents = value
        else:
            raise TypeError("return_constituents must be a bool.")
    
    @property
    def mode(self):
        """
        Property storing the string mode determining how training sets are
        combined. If mode=='add', training sets are combined through addition.
        If mode=='multiply', training sets are combined through multiplication.
        Otherwise, it is an expression which is calculable using numpy whose
        pieces are of the form {###} where ### is the index (starting at 0) of
        the specific training set (e.g. '{0}*np.power({1}, {2})')
        """
        if not hasattr(self, '_mode'):
            raise AttributeError("mode was referenced before it was set.")
        return self._mode
    
    @mode.setter
    def mode(self, value):
        """
        Sets the mode with the given string.
        
        value: if 'add', training sets are combined through addition
               if 'multiply', training sets are combined through multiplication
               else, mode must be an expression which is calculable using numpy
                     whose pieces are of the form {###} where ### is the index
                     (starting at 0) of the specific training set
                     (e.g. '{0}*np.power({1}, {2})')
        """
        if value in ['add', 'multiply']:
            self._mode = value
        elif isinstance(value, basestring):
            num_args = 0
            while ('{{{}}}'.format(num_args)) in value:
                num_args += 1
            if num_args == self.num_training_sets:
                test_expression = value
                for iarg in range(num_args):
                    test_expression =\
                        test_expression.split('{{{}}}'.format(iarg))
                    test_expression = ('0').join(test_expression)
                try:
                    test_value = eval(test_expression)
                except SyntaxError:
                    raise SyntaxError("If mode is not in ['add', " +\
                                      "'multiply'], then it should be an " +\
                                      "expression with '{i}' as its " +\
                                      "arguments (where i is the argument " +\
                                      "number). The given expression led " +\
                                      "to a SyntaxError when run with the " +\
                                      "training sets.")
                except:
                    pass
                else:
                    self._mode = value
            else:
                raise ValueError("The number of arguments in the mode " +\
                                 "expression must be the same as the " +\
                                 "number of training sets.")
        else:
            raise ValueError("mode can only be set to add or multiply.")
    
    @property
    def max_block_size(self):
        """
        Property storing the maximum number of numbers in a block returned by
        this iterator.
        """
        if not hasattr(self, '_max_block_size'):
            raise AttributeError("max_block_size referenced before it was " +\
                                 "set.")
        return self._max_block_size
    
    @max_block_size.setter
    def max_block_size(self, value):
        """
        Allows user to set the maximum number of numbers in a block returned by
        this iterator.
        
        value: reasonable (i.e. less than about 5% of the number of bytes in
               memory) positive integer
        """
        if type(value) in int_types:
            if value > 0:
                self._max_block_size = value
            else:
                raise ValueError("max_block_size must be a positive integer.")
        else:
            raise TypeError("max_block_size must be an integer.")
    
    @property
    def training_sets(self):
        """
        Property storing a sequence of training sets which are 2D arrays which
        all have the same second dimension length.
        """
        if not hasattr(self, '_training_sets'):
            raise AttributeError("training_sets referenced before it was set.")
        return self._training_sets
    
    @training_sets.setter
    def training_sets(self, value):
        """
        Setter for the component training sets to iterate over.
        
        value: sequence of 2D arrays which all have the same second dimension
               length
        """
        if type(value) in sequence_types:
            if all([(training_set.ndim == 2) for training_set in value]):
                self._training_sets = value
            else:
                raise ValueError("Each element of training_sets must be 2D.")
        else:
            raise TypeError("training_sets must be a sequence.")
    
    @property
    def training_set_lengths(self):
        """
        Property storing the number of channels in each kind of training set.
        """
        if not hasattr(self, '_training_set_lengths'):
            self._training_set_lengths =\
                [training_set.shape[1] for training_set in self.training_sets]
        return self._training_set_lengths
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the output training set
        curves.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels =\
                len(self.expanders[0](self.training_sets[0][0]))
        return self._num_channels
    
    @property
    def channels_per_output_curve(self):
        """
        Property storing the number of channels per output curve. If
        constituents aren't returned, this is simply the number of channels in
        the output curve. Otherwise, this is the sum of the number of channels
        in the output curve and the numbers of channels in each training set.
        """
        if not hasattr(self, '_channels_per_output_curve'):
            if self.return_constituents:
                self._channels_per_output_curve =\
                    self.num_channels + sum(self.training_set_lengths)
            else:
                self._channels_per_output_curve = self.num_channels
        return self._channels_per_output_curve
    
    @property
    def num_curves_in_block(self):
        """
        Property storing the positive integer number of output curves in each
        block.
        """
        if not hasattr(self, '_num_curves_in_block'):
            self._num_curves_in_block =\
                self.max_block_size // self.channels_per_output_curve
        return self._num_curves_in_block
    
    @property
    def block_size(self):
        """
        Property storing the integer number of numbers in each returned block.
        """
        if not hasattr(self, '_block_size'):
            self._block_size =\
                self.channels_per_output_curve * self.num_curves_in_block
        return self._block_size

    @property
    def shape(self):
        """
        Property storing the number of curves in each training set.
        """
        if not hasattr(self, '_shape'):
            self._shape =\
                [training_set.shape[0] for training_set in self.training_sets]
            self._shape = tuple(self._shape)
        return self._shape
    
    @property
    def num_training_sets(self):
        """
        Property storing the number of different training sets to combine.
        """
        if not hasattr(self, '_num_training_sets'):
            self._num_training_sets = len(self.training_sets)
        return self._num_training_sets
    
    @property
    def num_training_set_curves(self):
        """
        Property storing the integer total number of combined training set
        curves.
        """
        if not hasattr(self, '_num_training_set_curves'):
            self._num_training_set_curves = np.prod(self.shape)
        return self._num_training_set_curves
    
    @property
    def total_size(self):
        """
        Property storing the total number of numbers to return over the course
        of this iterator's execution.
        """
        if not hasattr(self, '_total_size'):
            self._total_size =\
                self.channels_per_output_curve * self.num_training_set_curves
        return self._total_size
    
    @property
    def num_blocks(self):
        """
        Property storing the integer number of blocks in which this iterator
        will give its output.
        """
        if not hasattr(self, '_num_blocks'):
            self._num_blocks =\
                int(np.ceil(self.total_size / self.block_size))
        return self._num_blocks
    
    def __iter__(self):
        """
        The iterator associated with this object is itself since it has its own
        self.next() method.
        
        returns: self
        """
        return self
    
    def get_block(self, iblock):
        """
        Gets the block of training set curves of the given index (starting at
        0).
        
        iblock: integer satisfying 0<=iblock<self.num_blocks
        
        returns: a 2D numpy.ndarray where the first dimension tracks the
                 indices of the training set curves and the second dimension
                 tracks the data channels
        """
        start = (iblock * self.num_curves_in_block)
        end = (start + self.num_curves_in_block)
        end = min(end, self.num_training_set_curves)
        indices = np.unravel_index(np.arange(start, end), self.shape)
        if self.mode == 'add':
            block = np.zeros((end - start, self.num_channels))
            for itraining_set in range(self.num_training_sets):
                block = block + self.expanders[itraining_set](\
                    self.training_sets[itraining_set][indices[itraining_set]])
        elif self.mode == 'multiply':
            block = np.ones((end - start, self.num_channels))
            for itraining_set in range(self.num_training_sets):
                block = block * self.expanders[itraining_set](\
                    self.training_sets[itraining_set][indices[itraining_set]])
        else:
            expression = self.mode
            for iarg in range(self.num_training_sets):
                to_remove = ('{{{}}}'.format(iarg))
                to_add = ('self.expanders[{0}](self.training_sets[{1}]' +\
                    '[indices[{2}]])').format(iarg, iarg, iarg)
                expression = expression.split(to_remove)
                expression = to_add.join(expression)
            block = eval(expression)
        if self.return_constituents:
            constituents = [self.training_sets[itrain][indices[itrain]]\
                for itrain in range(self.num_training_sets)]
            return block, constituents
        else:
            return block
    
    @property
    def curves_to_return(self):
        """
        Property storing the number of curves that the user wishes for this
        iterator to return. The actual number of returned curves is usually
        within a block or so of this number.
        """
        if not hasattr(self, '_curves_to_return'):
            self._curves_to_return = self.num_training_set_curves
        return self._curves_to_return
    
    @curves_to_return.setter
    def curves_to_return(self, value):
        """
        Setter for the number of curves that the user wishes for this iterator
        to return. The actual number of returned curves is usually within a
        block or so of this number. If the user has no limit of curves they'd
        like, this property can be set to None.
        
        value: if None, all curves in combined training set are returned
               otherwise, value should be an int number of curves that the user
                          wishes for this iterator to return
        """
        if type(value) is type(None):
            pass
        elif type(value) in int_types:
            if value > 0:
                if value > self.num_training_set_curves:
                    pass
                else:
                    self._curves_to_return = value
            else:
                raise ValueError("curves_to_return was non-positive! That " +\
                    "leaves the job of this TrainingSetIterator unnecessary.")
        else:
            raise TypeError("curves_to_return was set to a non-integer.")
    
    @property
    def blocks_to_return(self):
        """
        Property storing the number of blocks to return in total.
        """
        if not hasattr(self, '_blocks_to_return'):
            self._blocks_to_return =\
                (self.curves_to_return // self.num_curves_in_block)
            if (self.curves_to_return % self.num_curves_in_block) != 0:
                self._blocks_to_return = self._blocks_to_return + 1
        return self._blocks_to_return
    
    @property
    def block_sequence(self):
        """
        Property storing the sequence of numbers of blocks to return.
        """
        if not hasattr(self, '_block_sequence'):
            block_stride = (self.num_blocks // self.blocks_to_return)
            self._block_sequence = np.arange(0, self.num_blocks, block_stride)
        return self._block_sequence
    
    def next(self):
        """
        Gets the next block of combined training set curves.
        
        returns: a 2D numpy.ndarray where the first dimension tracks the
                 indices of the training set curves and the second dimension
                 tracks the data channels
        """
        if self.iblock == self.blocks_to_return:
            raise StopIteration
        desired_block = self.get_block(self.block_sequence[self.iblock])
        self.iblock = self.iblock + 1
        return desired_block
    
    def __next__(self):
        """
        Alias for next included for Python 2/3 compatibility.
        """
        return self.next()

