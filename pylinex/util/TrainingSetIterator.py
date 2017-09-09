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
import numpy as np
from .TypeCategories import int_types, sequence_types

def ceil_div(a, b):
    """
    Finds the result of integer division of a/b when rounding towards +inf.
    """
    return -(-a // b)

class TrainingSetIterator(object):
    """
    Class which combines multiple training sets into one by including every
    single combination of curves from each training set. It also contains
    functions which aid in the splitting up of the output into blocks so as to
    not overtax memory.
    """
    def __init__(self, training_sets, max_block_size=536870912, mode='add',\
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
                    (e.g. '{0}*np.power({1}, {2})')
        return_constituents: if True, return constituent training sets used to
                                      each output combined training set
                             if False, only combined training set curves are
                                       returned
        """
        self.max_block_size = max_block_size
        self.training_sets = training_sets
        if self.num_channels > self.max_block_size:
            raise ValueError("max_block_size must be greater than " +\
                             "num_channels.")
        self.mode = mode
        self.return_constituents = return_constituents
        self.iblock = 0
    
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
        elif isinstance(value, str):
            num_args = 0
            while ('{%i}' % (num_args,)) in value:
                num_args += 1
            if num_args == self.num_training_sets:
                test_expression = value
                for iarg in xrange(num_args):
                    test_expression = test_expression.split('{%i}' % (iarg,))
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
                if len(set([tset.shape[1] for tset in value])) == 1:
                    self._training_sets = value
                else:
                    raise ValueError("All elements of training_sets must " +\
                                     "have the same second dimension length.")
            else:
                raise ValueError("Each element of training_sets must be 2D.")
        else:
            raise TypeError("training_sets must be a sequence.")
    
    @property
    def num_channels(self):
        """
        Property storing the number of channels in the training set curves.
        """
        if not hasattr(self, '_num_channels'):
            self._num_channels = self.training_sets[0].shape[1]
        return self._num_channels
    
    @property
    def num_curves_in_block(self):
        """
        Property storing the positive integer number of curves in each block.
        """
        if not hasattr(self, '_num_curves_in_block'):
            self._num_curves_in_block =\
                self.max_block_size // self.num_channels
        return self._num_curves_in_block
    
    @property
    def block_size(self):
        """
        Property storing the integer number of numbers in each returned block.
        """
        if not hasattr(self, '_block_size'):
            self._block_size = self.num_channels * self.num_curves_in_block
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
            self._num_training_sets = len(self.shape)
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
            self._total_size = self.num_channels * self.num_training_set_curves
        return self._total_size
    
    @property
    def num_blocks(self):
        """
        Property storing the integer number of blocks in which this iterator
        will give its output.
        """
        if not hasattr(self, '_num_blocks'):
            self._num_blocks =\
                np.ceil((1. * self.total_size) / self.block_size)
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
            for itraining_set in xrange(self.num_training_sets):
                block = block +\
                    self.training_sets[itraining_set][indices[itraining_set]]
        elif self.mode == 'multiply':
            block = np.ones((end - start, self.num_channels))
            for itraining_set in xrange(self.num_training_sets):
                block = block *\
                    self.training_sets[itraining_set][indices[itraining_set]]
        else:
            expression = self.mode
            for iarg in xrange(self.num_training_sets):
                to_remove = ('{%i}' % (iarg,))
                to_add = ('self.training_sets[%i][indices[%i]]' % (iarg, iarg))
                expression = expression.split(to_remove)
                expression = to_add.join(expression)
            block = eval(expression)
        if self.return_constituents:
            constituents = [self.training_sets[i][indices[i]]\
                                       for i in xrange(self.num_training_sets)]
            return block, constituents
        else:
            return block
    
    def next(self):
        """
        Gets the next block of combined training set curves.
        
        returns: a 2D numpy.ndarray where the first dimension tracks the
                 indices of the training set curves and the second dimension
                 tracks the data channels
        """
        if self.iblock == self.num_blocks:
            raise StopIteration
        self.iblock = self.iblock + 1
        return self.get_block(self.iblock - 1)

