======
pylinex
======
pylinex is a Python package, compatible with both Python 2.7+ and Python 3.5+, for linear extraction of signals from data. It is flexible enough to perform most any linear fit to data but was created with the purpose of using Singular Value Decomposition (SVD) to create models of components of data with which to separate those components. This package was introduced along with a paper published in the Astrophysical Journal (Tauscher, K., Rapetti, D., Burns, J. O., & Switzer, E. 2018 , ApJ 853, 187) and has been extended as described in a second paper submitted to the Astrophysical Journal (Rapetti, D., Tauscher, K., Mirocha, J., & Burns, J.O. 2019, arXiv e-prints, arXiv:1912.02205). Please cite the relevant paper(s) if you use this code in your publication.

There are two different ways to keep up to data with changes to pylinex: 1) watch this repository on bitbucket and 2) subscribe to the mailing list `here <https://docs.google.com/forms/d/1nQA1nPP-d3BHwzPQwAwLw8w8Ydx_EhNQWNkRcLW-PCA>`_. The mailing list will receive emails about major changes to the code.

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
    cd pylinex
    python setup.py develop

The first four lines above are necessary only if you do not already have distpy installed.

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
