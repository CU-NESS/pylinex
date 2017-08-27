#!/usr/bin/env python
"""
File: setup.py
Author: Keith Tauscher
Date: 25 Aug 2017

Description: Installs extractpy.
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
        
setup(name='extractpy', version='0.1',\
    description='Signal extraction in Python', packages=['extractpy'])

    
    
    
