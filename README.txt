======
pylinex
======
pylinex is a Python package, compatible with both Python 2.7+ and Python 3.5+, for linear extraction of signals from data. It is flexible enough to perform most any linear fit to data but was created with the purpose of using Singular Value Decomposition (SVD) to create models of components of data with which to separate those components. This package was introduced along with a paper submitted to the Astrophysical Journal (***UPDATE WITH CITATION***).

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed.

Getting started
---------------------
To clone a copy and install: ::

    hg clone https://bitbucket.org/ktausch/distpy
    cd distpy
    python setup.py develop
    cd ..
    hg clone https://bitbucket.org/ktausch/pylinex
    cd perses
    python setup.py develop

Dependencies
--------------------
You will need:

- `numpy <http://www.numpy.org/>`_
- `scipy <http://www.scipy.org/>`_
- `matplotlib <http://matplotlib.org/>`_
- `h5py <http://www.h5py.org/>`_
- `distpy <https://bitbucket.org/ktausch/distpy>`_

Examples
--------------
To get started, take a look at the examples in the examples directory of this repository.

Contributors
------------

Author: Keith Tauscher
