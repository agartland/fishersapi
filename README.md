## fishersapi
[![Build Status](https://travis-ci.org/uwescience/fishersapi.svg?branch=master)](https://travis-ci.org/uwescience/fishersapi)

An API for applying a fast Fisher's Exact Test to variable pairs in pandas DataFrames

This package provides an interface for running a fast implementation of Fisher's exact test for 2x2 tables on categorical data in a pandas.DataFrame. The results are tested against scipy.stats.fishers_exact and fall back on scipy if the faster brentp/fishers_exact_test (~1000x faster) is not installed.

The fast version of the test is run using the package `fisher` developed by Haibao Tang and Brent Pedersen, which uses cython.
 - https://pypi.python.org/pypi/fisher/
 - https://github.com/brentp/fishers_exact_test