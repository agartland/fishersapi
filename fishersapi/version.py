from __future__ import absolute_import, division, print_function
from os import path

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

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

# read the contents of your README file into long_description
this_directory = path.abspath(path.dirname(__file__))
print(this_directory)

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()    

NAME = "fishersapi"
MAINTAINER = "Andrew Fiore-Gartland"
MAINTAINER_EMAIL = "agartlan@fredhutch.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown' 
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
REQUIRES = ['numpy', 'scipy', 'pandas', 'fisher', 'statsmodels']
