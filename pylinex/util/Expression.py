"""
File: Expression.py
Author: Keith Tauscher
Date: 30 Oct 2017

Description: File containing class which allows for the automation of the
             evaluation of generic models.
"""
import importlib
import numpy as np
from distpy import int_types, bool_types, Savable, Loadable,\
    create_hdf5_dataset, get_hdf5_value

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class Expression(Savable, Loadable):
    """
    Class which allows for the automation of the evaluation of generic models.
    """
    def __init__(self, string, num_arguments=None, import_strings=[],\
        kwargs={}, should_throw_error=False):
        """
        Initializes the expression with the given form and requiring the given
        imports.
        
        string: str form of the expression to evaluate with {#} in place of
                arguments (e.g. '{0}+{1}')
        num_arguments: if given, assumed to be the number of arguments accepted
                                 by this Expression
                       otherwise, dynamically found from the numbers of the
                                  arguments in the expression string.
        import_strings: sequence of import strings of one of the following
                        forms: 'import XXXXX', 'import XXXXX as XXXXX',
                        'from XXXXX import XXXXX'
        kwargs: keyword arguments needed to to evaluate the expression. These
                are only calculated once as opposed to being evaluated every
                time the Expression is evaluated
        should_throw_error: boolean determining whether an error should be
                            thrown if this Expression can't be saved.
        """
        self.should_throw_error = should_throw_error
        self.string = string
        self.import_strings = import_strings
        self.imports # performs imports once import_strings are loaded in
        self.kwargs = kwargs
        self.num_arguments = num_arguments
    
    @property
    def should_throw_error(self):
        """
        Property storing the boolean switch determining if errors are thrown
        when this Expression is being saved.
        """
        if not hasattr(self, '_should_throw_error'):
            raise AttributeError("should_throw_error was referenced before " +\
                "it was set.")
        return self._should_throw_error
    
    @should_throw_error.setter
    def should_throw_error(self, value):
        """
        Setter for the switch determining if errors are thrown when this
        Expression is being saved.
        
        value: either True or False
        """
        if type(value) in bool_types:
            self._should_throw_error = value
        else:
            raise TypeError("should_throw_error was set to a non-bool.")
    
    @property
    def string(self):
        """
        Property storing the string form of this expression where strings of
        the form '{#}' are in place of the arguments.
        """
        if not hasattr(self, '_string'):
            raise AttributeError("string referenced before it was set.")
        return self._string
    
    @string.setter
    def string(self, value):
        """
        Setter for the string form of this expression.
        
        value: str where strings of the form '{#}' are in place of the
               arguments (e.g. '{0}+{1}')
        """
        if isinstance(value, basestring):
            self._string = value
        else:
            raise TypeError("string was not a str.")
    
    @property
    def num_arguments(self):
        """
        Property storing the number of arguments this expression requires.
        """
        if not hasattr(self, '_num_arguments'):
            self._num_arguments = 0
            while '{{{}}}'.format(self._num_arguments) in self.string:
                self._num_arguments += 1
        return self._num_arguments
    
    @num_arguments.setter
    def num_arguments(self, value):
        """
        Setter for the number of arguments to supply to this Expression.
        
        value: non-negative integer
        """
        if value is None:
            return
        elif (type(value) in int_types) and (value >= 0):
            self._num_arguments = value
        else:
            raise TypeError("num_arguments was set to something which was " +\
                "neither None nor a non-negative integer.")
    
    @property
    def import_strings(self):
        """
        Property storing the strings describing the imports necessary to
        perform before evaluation of this Expression.
        """
        if not hasattr(self, '_import_strings'):
            raise AttributeError("import_strings referenced before it was " +\
                "set.")
        return self._import_strings
    
    @import_strings.setter
    def import_strings(self, value):
        """
        Setter for the import_strings property.
        
        value: list of strings representing simple import statements
        """
        if all([isinstance(element, basestring) for element in value]):
            self._import_strings = [element for element in value]
        else:
            raise TypeError("At least one of the import strings was not " +\
                "recognized.")
        
    @property
    def imports(self):
        """
        Property storing the modules/objects imported by the import_strings.
        """
        if not hasattr(self, '_imports'):
            self._imports = {}
            for import_string in self.import_strings:
                words = import_string.split(' ')
                num_words = len(words)
                is_simple_import = (num_words == 2) and (words[0] == 'import')
                is_as_import = (num_words == 4) and\
                    ((words[0], words[2]) == ('import', 'as'))
                is_from_import = (num_words == 4) and\
                    ((words[0], words[2]) == ('from', 'import'))
                if is_simple_import:
                    self.imports[words[1]] = importlib.import_module(words[1])
                elif is_as_import:
                    self.imports[words[3]] = importlib.import_module(words[1])
                elif is_from_import:
                    module = importlib.import_module(words[1])
                    self.imports[words[3]] = getattr(module, words[3])
                else:
                    raise ValueError("Form of import string not recognized.")
        return self._imports
    
    @property
    def kwargs(self):
        """
        Keyword arguments to use in evaluating the expression.
        """
        if not hasattr(self, '_kwargs'):
            raise AttributeError("kwargs was referenced before it was set.")
        return self._kwargs
    
    @kwargs.setter
    def kwargs(self, value):
        """
        Setter for the kwargs used when evaluating this Expression
        
        value: dict with string keys of values to use when evaluating
               expression
        """
        if isinstance(value, dict):
            if all([isinstance(key, basestring) for key in value]):
                self._kwargs = value
            else:
                raise TypeError("Not all keys of the kwargs dict were " +\
                    "strings.")
        else:
            raise TypeError("kwargs was set to a non-dict.")
    
    def fill_hdf5_group(self, group, kwargs_links=None):
        """
        Fills the given hdf5 file group with information about this Expression.
        
        group: hdf5 file group to fill with information about this Expression
        kwargs_links: dictionary of links to aid in saving kwargs, if possible
                      and necessary
        """
        group.attrs['class'] = 'Expression'
        group.attrs['string'] = self.string
        group.attrs['num_arguments'] = self.num_arguments
        subgroup = group.create_group('import_strings')
        for (iimport_string, import_string) in enumerate(self.import_strings):
            subgroup.attrs['{}'.format(iimport_string)] = import_string
        subgroup = group.create_group('kwargs')
        if kwargs_links is None:
            kwargs_links = {}
        for key in self.kwargs:
            value = self.kwargs[key]
            if key in kwargs_links:
                create_hdf5_dataset(subgroup, key, link=kwargs_links[key])
            elif isinstance(value, np.ndarray):
                create_hdf5_dataset(subgroup, key, data=value)
            elif isinstance(value, Savable):
                subsubgroup = subgroup.create_group(key)
                value.fill_hdf5_group(subsubgroup)
            elif type(value) in bool_types:
                subgroup.attrs[key] = value
            elif self.should_throw_error:
                raise TypeError(("kwargs element with key {!s} was not " +\
                    "savable.").format(key))
    
    @staticmethod
    def load_from_hdf5_group(group):
        """
        Loads an Expression object from the given hdf5 group.
        
        group: hdf5 file group from which to load an Expression object
        """
        if ('class' in group.attrs) and (group.attrs['class'] == 'Expression'):
            string = group.attrs['string']
            num_arguments = group.attrs['num_arguments']
            subgroup = group['import_strings']
            import_strings = []
            iimport_string = 0
            while '{}'.format(iimport_string) in subgroup.attrs:
                import_strings.append(\
                    subgroup.attrs['{}'.format(iimport_string)])
                iimport_string += 1
            subgroup = group['kwargs']
            kwargs = {key: get_hdf5_value(subgroup[key]) for key in subgroup}
            kwargs.update({key: subgroup.attrs[key] for key in subgroup.attrs})
            return Expression(string, num_arguments=num_arguments,\
                import_strings=import_strings, kwargs=kwargs)
        else:
            raise ValueError("group doesn't appear to point to an " +\
                "Expression object.")
    
    def __call__(self, *args, **kwargs):
        """
        Evaluates this expression at the given arguments.
        
        *args: arguments at which to evaluate expression
        **kwargs: extra keyword arguments with which to define context of
                  expression string
        
        returns: object dependent on given expression string
        """
        num_arguments_given = len(args)
        if num_arguments_given == self.num_arguments:
            context = locals().copy()
            context.update(self.imports)
            context.update(self.kwargs)
            context.update(kwargs)
            arg_strings = ['args[{}]'.format(index)\
                for index in range(self.num_arguments)]
            return eval(self.string.format(*arg_strings), context)
        else:
            raise ValueError(("The number of arguments supplied ({0}) was " +\
                "not equal to the number of arguments expected " +\
                "({1}).").format(num_arguments_given, self.num_arguments))
    
    def __eq__(self, other):
        """
        Checks for equality between other and this Expression.
        
        other: object to check for equality
        
        returns: False unless other is equivalent to this Expression
        """
        if not isinstance(other, Expression):
            return False
        if self.string != other.string:
            return False
        if set(self.import_strings) != set(other.import_strings):
            return False
        if self.num_arguments != other.num_arguments:
            return False
        if set(self.kwargs.keys()) != set(other.kwargs.keys()):
            return False
        for key in self.kwargs:
            if np.any(self.kwargs[key] != other.kwargs[key]):
                return False
        return True
    
    def __ne__(self, other):
        """
        Checks for inequality between other and this Expression.
        
        other: object to check for inequality
        
        returns: True unless other is equivalent to this Expression
        """
        return (not self.__eq__(other))
    

