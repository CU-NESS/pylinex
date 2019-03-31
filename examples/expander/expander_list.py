"""
File: examples/expander/expander_list.py
Author: Keith Tauscher
Date: 29 Mar 2019

Description: 
"""
import os
import numpy as np
from pylinex import PadExpander, ExpanderList

hdf5_file_name = 'TESTINGEXPANDERLIST.hdf5'
nchannels = 5
expanders = [PadExpander('{:d}*'.format(ichannel),\
    '{:d}*'.format(nchannels - ichannel - 1)) for ichannel in range(nchannels)]
expander_list = ExpanderList(*expanders)

expander_list.save(hdf5_file_name)
try:
    assert(expander_list == ExpanderList.load(hdf5_file_name))
except:
    os.remove(hdf5_file_name)
    raise
else:
    os.remove(hdf5_file_name)

expander_list_other_initialization = ExpanderList()
for expander in expanders:
    expander_list_other_initialization += expander
assert(expander_list == expander_list_other_initialization)

inputs = [np.array([index]) for index in range(nchannels)]
expected_output_with_sum = np.arange(nchannels)
expected_output_no_sum = [np.where(expected_output_with_sum == ichannel,\
    expected_output_with_sum, 0) for ichannel in range(nchannels)]
output_no_sum = expander_list(inputs, should_sum=False)
output_with_sum = expander_list(inputs, should_sum=True)
assert(np.all(output_with_sum == expected_output_with_sum))
assert(all([np.all(actual == expected) for (actual, expected) in\
    zip(output_no_sum, expected_output_no_sum)]))

expected_inverse_on_output = (inputs, np.zeros((nchannels,)))
inverse_on_output = expander_list.invert(output_with_sum,\
    np.ones((nchannels,)), return_residual=True)

assert(all([np.all(actual == expected) for (actual, expected) in\
    zip(inverse_on_output[0], expected_inverse_on_output[0])]))
assert(np.all(expected_inverse_on_output[1] == inverse_on_output[1]))

