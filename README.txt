======
pylinex
======
pylinex is a Python package, compatible with both Python 2.7+ and Python 3.5+, for linear extraction of signals from data. It is flexible enough to perform most any linear fit to data but was created with the purpose of using Singular Value Decomposition (SVD) to create models of components of data with which to separate those components. This package was introduced and described in a series of four papers published in the Astrophysical journal: Paper I (Tauscher et al. 2018, ApJ 853, 187), Paper II (Rapetti et al. 2020, ApJ 897, 174), Paper III (Taucher et al. 2020, ApJ 897, 175), Paper IV (Tauscher et al. 2021, ApJ 915, 66). Please cite the relevant paper(s) if you use this code in your publication.

There are two different ways to keep up to data with changes to pylinex: 1) watch this repository on Github and 2) subscribe to the mailing list here: https://docs.google.com/forms/d/1nQA1nPP-d3BHwzPQwAwLw8w8Ydx_EhNQWNkRcLW-PCA. The mailing list will receive emails about major changes to the code.

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed.

Getting started
---------------------
To clone a copy and install:

    git clone https://github.com/CU-NESS/distpy.git
    cd distpy
    python setup.py develop --user
    cd ..
    git clone https://github.com/CU-NESS/pylinex.git
    cd pylinex
    python setup.py develop --user

The first four lines above are necessary only if you do not already have distpy installed. The --user option can be omitted if you would like to install pylinex globally (this may require sudo privileges).

To generate HTML documentation pages for the code:

    cd docs
    ./make_docs.sh

Dependencies
--------------------
You will need:

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [h5py](http://www.h5py.org/)
- [distpy](https://github.com/CU-NESS/distpy)

Examples
--------------
To get started, take a look at the examples in the examples directory of this repository.

Contributors
------------

Author: Keith Tauscher
