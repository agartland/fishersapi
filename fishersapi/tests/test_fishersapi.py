from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import numpy.testing as npt
from scipy import stats

import fishersapi

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
    
    for alt in ['two-tailed', 'less', 'greater']:
        ORs, pvalues = fishersapi.fishers_vec(a, b, c, d, alternative=alt)
        
        scipy_pvalues, scipy_ORs = np.zeros(n), np.zeros(n)
        for i in range(n):
            scipy_ORs[i], scipy_pvalues[i] = stats.fisher_exact([[a[i], b[i]], [c[i], d[i]]])

    npt.assert_allclose(ORs, scipy_ORs, rtol=1e-4)
    npt.assert_allclose(pvalues, scipy_pvalues, rtol=1e-4)