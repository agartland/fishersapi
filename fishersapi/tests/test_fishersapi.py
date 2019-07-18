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
    
    n = 100
    res = np.zeros(n)
    for i in range(n):
        startT = time.time()
        ORs, pvalues = fishersapi.fishers_vec(a, b, c, d, alternative='two-sided')
        res[i] = n / (time.time() - startT)
    print('Imported test: %1.2f tests per second' % np.mean(res))

    n = 1
    res_scipy = np.zeros(n)
    for i in range(n):
        startT = time.time()
        ORs, pvalues = fishersapi._scipy_fishers_vec(a, b, c, d, alternative='two-sided')
        res_scipy[i] = n / (time.time() - startT)
    print('scipy test: %1.2f tests per second' % np.mean(res_scipy))

def test_fishers_frame():
    np.random.seed(110820)
    n = 50
    df = pd.DataFrame({'VA':np.random.choice(['TRAV14', 'TRAV12', 'TRAV3', 'TRAV23', 'TRAV11', 'TRAV6'], n),
                       'JA':np.random.choice(['TRAJ4', 'TRAJ2', 'TRAJ3','TRAJ5', 'TRAJ21', 'TRAJ13'], n),
                       'VB':np.random.choice(['TRBV14', 'TRBV12', 'TRBV3', 'TRBV23', 'TRBV11', 'TRBV6'], n),
                       'JB':np.random.choice(['TRBJ4', 'TRBJ2', 'TRBJ3','TRBJ5', 'TRBJ21', 'TRBJ13'], n)})
    df = df.assign(Count=1)
    df.loc[:10, 'Count'] = 15

    resDf = fishersapi.fishers_frame(df, ['VA', 'JA', 'VB', 'JB'], count_col=None, alternative='two-sided')
    npt.assert_allclose(resDf.OR.values[:3], [3.85714286, 2.05714286, 0.72916667])

    resDf = fishersapi.fishers_frame(df, ['VA', 'JA', 'VB', 'JB'], count_col='Count', alternative='two-sided')