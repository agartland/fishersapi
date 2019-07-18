from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import warnings
from scipy import stats
import itertools

__all__ = ['fishers_vec',
           'fishers_frame']
"""TODO: Attempt to import brentp/fisher_exact_test and/or vectorized version of 
painyeph/FishersExactTest and/or falls back on scipy.stats.fishers?"""

def _add_docstring(doc):
    def dec(obj):
        obj.__doc__ = doc
        return obj
    return dec

def _scipy_fishers_vec(a, b, c, d, alternative='two-sided'):
    assert len(a) == len(b)
    assert len(a) == len(c)
    assert len(a) == len(d)
    n = len(a)
    scipy_pvalues, scipy_ORs = np.zeros(n), np.zeros(n)
    for i in range(n):
        scipy_ORs[i], scipy_pvalues[i] = stats.fisher_exact([[a[i], b[i]], [c[i], d[i]]], alternative=alternative)
    return scipy_ORs, scipy_pvalues

fishers_vec_doc = """Vectorized Fisher's exact test performs n tests
on 4 length n numpy vectors a, b, c, and d representing
the 4 elements of a 2x2 contigency table.

Wrapper around fisher.pvalue_npy found in:
Fast Fisher's Exact Test (Haibao Tang, Brent Pedersen)
https://pypi.python.org/pypi/fisher/
https://github.com/brentp/fishers_exact_test

Loop and test are performed in C (>1000x speed-up)

Parameters
----------
a, b, c, d : shape (n,) ndarrays
    Vector of counts (will be cast as uint8 for operation)
alternative : string
    Specfies the alternative hypothesis (similar to scipy.fisher_exact)
    Options: 'two-sided', 'less', 'greater' where less is "left-tailed"

Returns
-------
OR : shape (n,) ndarray
    Vector of odds-ratios associated with each 2 x 2 table
p : shape (n,) ndarray
    Vector of p-values asspciated with each test and the alternative hypothesis"""

try:
    """Attempt to use the fisher library (cython) if available (>1000x speedup)"""
    import fisher
    print("Using Cython-powered Fisher's exact test")

    @_add_docstring(fishers_vec_doc)
    def fishers_vec(a, b, c, d, alternative='two-sided'):
        assert len(a) == len(b)
        assert len(a) == len(c)
        assert len(a) == len(d)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            OR = (a*d) / (b*c)

        res = fisher.pvalue_npy(a.astype(np.uint), b.astype(np.uint), c.astype(np.uint), d.astype(np.uint))
        if alternative in ['two-sided', 'two-tailed']:
            return (OR, res[2])
        elif alternative in ['less', 'left-tailed']:
            return (OR, res[0])
        elif alternative in ['greater', 'right-tailed']:
            return (OR, res[1])
        else:
            print('Please specify an alternative: two-sided, less, or greater')
            return OR, np.nan * np.zeros((len(a), 1))
    
except ImportError:
    from scipy import stats
    print("Using scipy.stats Fisher's exact test (slow)")

    @_add_docstring(fishers_vec_doc)
    def fishers_vec(a, b, c, d, alternative='two-sided'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            OR = (a*d) / (b*c)

        p = np.asarray([stats.fisher_exact([[aa, bb], [cc, dd]], alternative=alternative)[1] for aa, bb, cc, dd in zip(a, b, c, d)])
        return OR, p
    
def fishers_frame(df, cols, count_col=None, alternative='two-sided'):
    """Use Fisher's Exact Test to scan for associations between values
    in each of the columns of the DataFrame. Tests all pairs of columns in cols
    and all pairs of unique values in the column pairs.

    Tests should be subject to multiple hypothesis testing adjustment
    to protect against false discovery.

    Parameters
    ----------
    df : pd.DataFrame
    cols : tuple or list of strings
        Column names to test in df
    count_col : str
        Optionally specify a column containg integer counts of observations.
        (default is None, equivalent to vector of ones)
    alternative : string
        Specfies the alternative hypothesis (similar to scipy.fisher_exact)
        Options: 'two-sided', 'less', 'greater' where less is "left-tailed"

    Returns
    -------
    resDf : pd.DataFrame [n tests x colA, colB, valA, valB, A0_B0, A0_B1, A1_B0, A1_B1, OR, pvalue]
        Odds-ratio, pvalue and contingency table for each of the tests performed."""

    res = []
    for col1, col2 in itertools.combinations(cols, 2):
        for val1, val2 in itertools.product(df[col1].dropna().unique(), df[col2].dropna().unique()):
            tab = _count_2_by_2(df, (col1, val1), (col2, val2), count_col=count_col)
            res.append({'ColA':col1,
                        'ColB':col2,
                        'ValA':val1,
                        'ValB':val2,
                        'A0_B0':tab[0, 0],
                        'A0_B1':tab[0, 1],
                        'A1_B0':tab[1, 0],
                        'A1_B1':tab[1, 1]})
    resDf = pd.DataFrame(res)
    OR, pvalue = fishers_vec(resDf['A0_B0'].values,
                             resDf['A0_B1'].values,
                             resDf['A1_B0'].values,
                             resDf['A1_B1'].values, alternative=alternative)
    resDf = resDf.assign(OR=OR, pvalue=pvalue,
                            A0_B0=resDf.A0_B0.astype(int),
                            A0_B1=resDf.A0_B1.astype(int),
                            A1_B0=resDf.A1_B0.astype(int),
                            A1_B1=resDf.A1_B1.astype(int))
    return resDf

def _count_2_by_2(df, node1, node2, count_col=None):
    """Test if the occurence of nodeA paired with nodeB is more/less frequent than expected.

    Parameters
    ----------
    nodeX : tuple (column, value)
        Specify the node by its column name and the value.

    Returns
    -------
    OR : float
        Odds-ratio associated with the 2x2 contingency table
    pvalue : float
        P-value associated with the Fisher's exact test that H0: OR = 1"""
    
    col1, val1 = node1
    col2, val2 = node2
    
    tab = np.zeros((2, 2))
    if count_col is None:
        tmp = df[[col1, col2]].dropna()
        tab[0, 0] = (((tmp[col1]!=val1) & (tmp[col2]!=val2))).sum()
        tab[0, 1] = (((tmp[col1]!=val1) & (tmp[col2]==val2))).sum()
        tab[1, 0] = (((tmp[col1]==val1) & (tmp[col2]!=val2))).sum()
        tab[1, 1] = (((tmp[col1]==val1) & (tmp[col2]==val2))).sum()
    else:
        tmp = df[[col1, col2, count_col]].dropna()
        tab[0, 0] = (((tmp[col1]!=val1) & (tmp[col2]!=val2)) * tmp[count_col]).sum()
        tab[0, 1] = (((tmp[col1]!=val1) & (tmp[col2]==val2)) * tmp[count_col]).sum()
        tab[1, 0] = (((tmp[col1]==val1) & (tmp[col2]!=val2)) * tmp[count_col]).sum()
        tab[1, 1] = (((tmp[col1]==val1) & (tmp[col2]==val2)) * tmp[count_col]).sum()
    return tab