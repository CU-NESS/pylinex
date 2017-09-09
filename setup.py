#!/usr/bin/env python
"""
File: setup.py
Author: Keith Tauscher
Date: 25 Aug 2017

Description: Installs pylinex.
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
        
setup(name='pylinex', version='0.1',\
    description='Linear signal extraction in Python',\
    packages=['pylinex.quantity', 'pylinex.expander', 'pylinex.basis',\
    'pylinex.fitter'])

    
    
    
