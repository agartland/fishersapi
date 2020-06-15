"""python -m pytest fishersapi/tests"""
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

def test_fishers_vec_minn():
    """Testing the vectorized version against scipy on random data."""
    a, b, c, d = _gen_rand_abcd()
    n = len(a)
    
    counts = a + b + c + d
    gtmin = np.sum(counts >= np.median(counts))
    for alt in ['two-sided', 'less', 'greater']:
        ORs, pvalues = fishersapi.fishers_vec(a, b, c, d, alternative=alt, min_n=np.median(counts))
        npt.assert_equal(gtmin, (~np.isnan(pvalues)).sum())
        
def test_integers():
    OR, pvalue = fishersapi.fishers_vec(10, 2, 15, 3)
    assert np.isscalar(OR)
    assert np.isscalar(pvalue)

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

def test_options():
    np.random.seed(110820)
    n = 200
    df = pd.DataFrame({'VA':np.random.choice(['TRAV14', 'TRAV12', 'TRAV3', 'TRAV23', 'TRAV11', 'TRAV6'], n),
                       'JA':np.random.choice(['TRAJ4', 'TRAJ2', 'TRAJ3','TRAJ5', 'TRAJ21', 'TRAJ13'], n),
                       'VB':np.random.choice(['TRBV14', 'TRBV12', 'TRBV3', 'TRBV23', 'TRBV11', 'TRBV6'], n),
                       'JB':np.random.choice(['TRBJ4', 'TRBJ2', 'TRBJ3','TRBJ5', 'TRBJ21', 'TRBJ13'], n)})

    res = fishersapi.fishers_frame(df, col_pairs=[('VA', 'JA'), ('VB', 'JB')], adj_method='holm')

def test_fishers_frame():
    np.random.seed(110820)
    n = 50
    df = pd.DataFrame({'VA':np.random.choice(['TRAV14', 'TRAV12', 'TRAV3', 'TRAV23', 'TRAV11', 'TRAV6'], n),
                       'JA':np.random.choice(['TRAJ4', 'TRAJ2', 'TRAJ3','TRAJ5', 'TRAJ21', 'TRAJ13'], n),
                       'VB':np.random.choice(['TRBV14', 'TRBV12', 'TRBV3', 'TRBV23', 'TRBV11', 'TRBV6'], n),
                       'JB':np.random.choice(['TRBJ4', 'TRBJ2', 'TRBJ3','TRBJ5', 'TRBJ21', 'TRBJ13'], n)})
    df = df.assign(Count=1)
    df.loc[:10, 'Count'] = 15

    res = fishersapi.fishers_frame(df, ['VA', 'JA', 'VB', 'JB'], count_col=None, alternative='two-sided')
    npt.assert_allclose(res.OR.values[:3], [3.85714286, 2.05714286, 0.72916667])

    res = fishersapi.fishers_frame(df, ['VA', 'JA', 'VB', 'JB'], count_col='Count', alternative='two-sided')

def test_fishers_frame2():
    example_df = pd.DataFrame({'count': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                                         10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1},
                               'j_b_gene': {0: 'TRBJ1-2*01', 1: 'TRBJ1-2*01', 2: 'TRBJ1-2*01',
                                            3: 'TRBJ1-2*01', 4: 'TRBJ1-2*01', 5: 'TRBJ1-2*01',
                                            6: 'TRBJ1-2*01', 7: 'TRBJ1-5*01', 8: 'TRBJ1-2*01',
                                            9: 'TRBJ1-2*01', 10: 'TRBJ1-2*01', 11: 'TRBJ1-2*01',
                                            12: 'TRBJ1-2*01', 13: 'TRBJ1-2*01', 14: 'TRBJ2-3*01',
                                            15: 'TRBJ1-5*01', 16: 'TRBJ2-7*01', 17: 'TRBJ1-1*01',
                                            18: 'TRBJ2-7*01', 19: 'TRBJ2-7*01'},
                                'j_a_gene': {0: 'TRAJ42*01', 1: 'TRAJ42*01', 2: 'TRAJ42*01',
                                             3: 'TRAJ50*01', 4: 'TRAJ42*01', 5: 'TRAJ42*01',
                                             6: 'TRAJ42*01', 7: 'TRAJ20*01', 8: 'TRAJ42*01',
                                             9: 'TRAJ42*01', 10: 'TRAJ42*01', 11: 'TRAJ42*01',
                                             12: 'TRAJ42*01', 13: 'TRAJ42*01', 14: 'TRAJ49*01',
                                             15: 'TRAJ33*01', 16: 'TRAJ42*01', 17: 'TRAJ49*01',
                                             18: 'TRAJ31*01', 19: 'TRAJ37*02'}})
    example_df = example_df.assign(count=np.array([52., 24., 14., 21., 10., 50., 13., 50., 19., 47., 50., 13., 35.,
                                                   60., 34., 11., 40., 54., 42., 33.]).astype(int))

    res = fishersapi.fishers_frame(example_df, cols=['j_b_gene', 'j_a_gene'], count_col='count')
    # print(res.iloc[0])
    assert res.shape[0] == 35
    assert res.shape[1] == 16
    assert res.loc[0, 'X+Y+'] == 387
    assert res.loc[0, 'X+Y-'] == 21
    npt.assert_almost_equal(res.loc[0, 'OR'], 103.2, decimal=1)
