#!/usr/bin/env python

import sys
import numpy as np

from setuptools import setup
from Cython.Build import cythonize


# Determine shared library suffix
if sys.platform == 'darwin':
    suffix = 'dylib'
else:
    suffix = 'so'

# Get version information from __init__.py. This is ugly, but more reliable than
# using an import.
# with open('openmc/__init__.py', 'r') as f:
#     version = f.readlines()[-1].split()[-1].strip("'")

kwargs = {
    # 'version': version,
    'ext_modules': cythonize('openmc/data/*.pyx'),
    'include_dirs': [np.get_include()]
}

setup(**kwargs)
