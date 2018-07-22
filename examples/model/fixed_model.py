import numpy as np
from pylinex import FixedModel

curve = np.arange(100)
model = FixedModel(curve)

file_name = 'TESTING_FIXEDMODEL_CLASS_DELETE_THIS_IF_YOU_SEE_IT.hdf5'
try:
    assert(model == FixedModel.load(file_name))
    assert(model == load_model_from_hdf5_file(file_name))
except:
    os.remove(file_name)
    raise
else:
    os.remove(file_name)

assert(np.all(model(None) == curve))
assert(np.all(model(1) == curve))
assert(np.all(model(np.array([1])) == curve))
assert(np.all(model(np.array([1, 2])) == curve))

