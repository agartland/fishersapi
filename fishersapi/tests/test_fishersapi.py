from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import numpy.testing as npt
from scipy import stats
import time

import fishersapi
from fishersapi import _scipy_fishers_vec

def _gen_rand_abcd(n=1000):
    np.random.seed(110820)
    a = np.random.randint(0, 50, size=n)
    b = np.random.randint(0, 50, size=n)
    c = np.random.randint(0, 100, size=n)
    d = np.random.randint(0, 100, size=n)
    return a, b, c, d

def test_fishers_vec():
    """Testing the vectorized version against scipy on random data."""
    a, b, c, d = _gen_rand_abcd()
    n = len(a)
    
    for alt in ['two-sided', 'less', 'greater']:
        ORs, pvalues = fishersapi.fishers_vec(a, b, c, d, alternative=alt)
        
        scipy_pvalues, scipy_ORs = np.zeros(n), np.zeros(n)
        for i in range(n):
            scipy_ORs[i], scipy_pvalues[i] = stats.fisher_exact([[a[i], b[i]], [c[i], d[i]]], alternative=alt)

    npt.assert_allclose(ORs, scipy_ORs, rtol=1e-4)
    npt.assert_allclose(pvalues, scipy_pvalues, rtol=1e-4)

def test_benchmark():
    a, b, c, d = _gen_rand_abcd()
    
    n = 200
    res = np.zeros(n)
    for i in range(n):
        startT = time.time()
        ORs, pvalues = fishersapi.fishers_vec(a, b, c, d, alternative='two-sided')
        res[i] = n / (time.time() - startT)
    print('Imported test: %1.2f tests per second' % np.mean(res))

    n = 2
    res_scipy = np.zeros(n)
    for i in range(n):
        startT = time.time()
        ORs, pvalues = fishersapi._scipy_fishers_vec(a, b, c, d, alternative='two-sided')
        res_scipy[i] = n / (time.time() - startT)
    print('scipy test: %1.2f tests per second' % np.mean(res_scipy))


