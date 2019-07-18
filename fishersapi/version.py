from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "fishersapi: An API for applying a fast Fisher's Exact Test to variable pairs in pandas DataFrames"
# Long description will go up on the pypi page
long_description = """

fishersapi
========
fishersapi provides an interface for running a fast implementation
of Fisher's exact test for 2x2 tables on categorical data in 
a pandas.DataFrame. The results are tested against scipy.stats.fishers_exact
and fallback on scipy if the faster brentp/fishers_exact_test (~1000x faster)
is not installed.

License
=======
``fishersapi`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
"""

NAME = "fishersapi"
MAINTAINER = "Andrew Fiore-Gartland"
MAINTAINER_EMAIL = "agartlan@fredhutch.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/agartland/fishersapi"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Andrew Fiore-Gartland"
AUTHOR_EMAIL = "agartlan@fredhutch.org"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {}
REQUIRES = ['numpy', 'scipy', 'pandas']
