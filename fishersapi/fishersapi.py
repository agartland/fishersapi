from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import warnings
from scipy import stats
import itertools

try:
    import statsmodels.api as sm
    SM = True
except ImportError:
    SM= False

__all__ = ['fishers_vec',
           'fishers_frame',
           'adjustnonnan']

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
    Also accepts scalars, in which case output will also be a scalar.
alternative : string
    Specfies the alternative hypothesis (similar to scipy.fisher_exact)
    Options: 'two-sided', 'less', 'greater' where less is "left-tailed"
min_n : int
    Minimum total number of counts to trigger testing a table.
    Allows skipping of tables that are unlikely to be significant.

Returns
-------
OR : shape (n,) ndarray
    Vector of odds-ratios associated with each 2 x 2 table
p : shape (n,) ndarray
    Vector of p-values asspciated with each test and the alternative hypothesis"""

try:
    """Attempt to use the fisher library (cython) if available (>1000x speedup)"""
    import fisher
    # print("Using Cython-powered Fisher's exact test")

    @_add_docstring(fishers_vec_doc)
    def fishers_vec(a, b, c, d, alternative='two-sided', min_n=0):
        scalar = np.isscalar(a) and np.isscalar(b) and np.isscalar(c) and np.isscalar(d)
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        c = np.asarray(c).ravel()
        d = np.asarray(d).ravel()

        assert len(a) == len(b)
        assert len(a) == len(c)
        assert len(a) == len(d)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            OR = (a*d) / (b*c)

        n = a + b + c + d
        ind = n >= min_n

        tmp_res = fisher.pvalue_npy(a[ind].astype(np.uint),
                                    b[ind].astype(np.uint),
                                    c[ind].astype(np.uint),
                                    d[ind].astype(np.uint))
        res = np.nan * np.ones(len(a))
        if alternative in ['two-sided', 'two-tailed']:
            res[ind] = tmp_res[2]
            out = (OR, res)
        elif alternative in ['less', 'left-tailed']:
            res[ind] = tmp_res[0]
            out = (OR, res)
        elif alternative in ['greater', 'right-tailed']:
            res[ind] = tmp_res[1]
            out = (OR, res)
        else:
            print('Please specify an alternative: two-sided, less, or greater')
            out = OR, np.nan * np.zeros((len(a), 1))
        if scalar:
            out = (out[0][0], out[1][0])
        return out
    
except (ImportError, ValueError):
    from scipy import stats
    print("Using scipy.stats Fisher's exact test (1000x slower)")

    @_add_docstring(fishers_vec_doc)
    def fishers_vec(a, b, c, d, alternative='two-sided', min_n=0):
        scalar = np.isscalar(a) and np.isscalar(b) and np.isscalar(c) and np.isscalar(d)

        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        c = np.asarray(c).ravel()
        d = np.asarray(d).ravel()

        assert len(a) == len(b)
        assert len(a) == len(c)
        assert len(a) == len(d)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            OR = (a*d) / (b*c)

        n = a + b + c + d
        ind = n >= min_n

        p_tmp = np.asarray([stats.fisher_exact([[aa, bb], [cc, dd]], alternative=alternative)[1] for aa, bb, cc, dd in zip(a[ind], b[ind], c[ind], d[ind])])

        p = np.nan * np.ones(len(a))
        p[ind] = p_tmp

        if scalar:
            out = (OR[0], p[0])
        else:
            out = (OR, p)
        return out
    
def fishers_frame(df, cols=None, col_pairs=None, count_col=None, alternative='two-sided', adj_method=None, min_n=0):
    """Prepare counts tables and perform rapid testing for associations
    between two or more categorical variables (columns) of a DataFrame
    (df). Tallies pairs of values within pairs of columns, which
    are then evaluated as 2 x 2 contingency tables using Fisher's exact
    test. While this is not the best way to model associations among
    multinomial distributed variables, it will work as an initial screen.

    Unique values of each column in cols will be tallied against
    unique values of all other columns in cols. Column pairs (col_pairs) can
    be directly specified as an alternative. If col_pairs is None then all
    pairs of values in cols will be tallied. If both are None then all pairs
    of allcolumns will be used.

    Each row is a 2x2 contingency table:
    
       Y+  Y-
    X+ 4   8
    X- 9   3

    where X+/- indicates the number of row counts with x_col == x_val.

    A count of X+Y+, X+Y-, X-Y+, X-Y-, frequencies and odds-ratio are provided
    for each combination (row) of the output.

    Tests should be subject to multiple hypothesis testing adjustment
    to protect against false discovery.

    Example
    -------

    res = fishersapi.fishers_frame(df,
                                   cols=['VA', 'JA', 'VB', 'JB'],
                                   count_col='Count',
                                   alternative='two-sided')
   
    Parameters
    ----------
    df : pd.DataFrame
    cols : tuple or list of strings
        Column names to test in df. Will test all pairs unless specified by col_pairs.
    col_pairs : list of tuples
        Pairs of columns names for testing. If specified, will replace cols.
    count_col : str
        Optionally specify a column containg integer counts of observations.
        (default is None, equivalent to vector of ones)
    alternative : string
        Specfies the alternative hypothesis (similar to scipy.fisher_exact)
        Options: 'two-sided', 'less', 'greater' where less is "left-tailed"
    adj_method : str
        Method for multiplicity adjustment using statsmodels.stats.multipletests, e.g.
        holm, bonferroni, fdr_bh
    min_n : int
        Minimum number of X+Y+ counts required for it to be included in output and testing.

    Returns
    -------
    res : pd.DataFrame [n tests x result columns, OR, pvalue]
        Odds-ratio, pvalue and contingency table for each of the tests performed."""
    if cols is None:
        cols = df.columns
    
    if col_pairs is None:
        col_pairs = [pair for pair in itertools.combinations(cols, 2)]

    res = []
    for col1, col2 in col_pairs:
        for val1, val2 in itertools.product(df[col1].dropna().unique(), df[col2].dropna().unique()):
            tmp = _count_2_by_2(df, (col1, val1), (col2, val2), count_col=count_col)
            res.append(tmp)
    res = pd.DataFrame(res)

    """Drop rows that have fewer than min_n X+Y+ counts for efficiency"""
    if min_n > 0:
        res = res.loc[res['X+Y+'] >= min_n]

    OR, pvalue = fishers_vec(res['X+Y+'].values,
                             res['X+Y-'].values,
                             res['X-Y+'].values,
                             res['X-Y-'].values, alternative=alternative)
    
    res = res.assign(OR=OR, pvalue=pvalue)

    if SM and not adj_method is None:
        if adj_method in ['bonferroni', 'holm']:
            k = 'FWER-pvalue'
        elif 'fdr' in adj_method:
            k = 'FDR-qvalue'
        res = res.assign(**{k:adjustnonnan(res['pvalue'], method=adj_method)})
    return res

def _count_2_by_2(df, node1, node2, count_col=None):
    """Tally instances of node1 and node2 where a node is
    the combination of a column and a value.

    E.g. hair color (col1) red (val1) and eye color (col2) blue (val2)

    Parameters
    ----------
    df : pd.DataFrame
        Contains columns in node1 and node2 and optionally a count column
    nodeX : tuple (column, value)
        Specify the node by its column name and the value.
    count-col : str
        Column in df containing integer counts of the instances of each row

    Returns
    -------
    out : dict
        Various labels, frequencies and counts associated with the
        contingency table created by node1 and node2"""
    
    col1, val1 = node1
    col2, val2 = node2
    
    if count_col is None:
        count_col = ''
        counts = np.ones(df.shape[0])
    else:
        counts = df[count_col].values

    aind = (df[col1] == val1) & (df[col2] == val2)
    bind = (df[col1] == val1) & (df[col2] != val2)
    cind = (df[col1] != val1) & (df[col2] == val2)
    dind = (df[col1] != val1) & (df[col2] != val2)
    
    n = counts.sum()
    w = np.sum(aind.astype(int).values * counts)

    tmp = {'xcol':col1,
           'xval':val1,
           'ycol':col2,
           'yval':val2,
           'X+Y+':w,
           'X+Y-':np.sum(bind.astype(int).values * counts),
           'X-Y+':np.sum(cind.astype(int).values * counts),
           'X-Y-':np.sum(dind.astype(int).values * counts)}
    tmp.update({'X_marg':(tmp['X+Y+'] + tmp['X+Y-']) / n,
                'Y_marg':(tmp['X+Y+'] + tmp['X-Y+']) / n,
                'X|Y+':tmp['X+Y+'] / (tmp['X+Y+'] + tmp['X-Y+']),
                'X|Y-':tmp['X+Y-'] / (tmp['X+Y-'] + tmp['X-Y-']),
                'Y|X+':tmp['X+Y+'] / (tmp['X+Y+'] + tmp['X+Y-']),
                'Y|X-':tmp['X-Y+'] / (tmp['X-Y+'] + tmp['X-Y-'])})
    return tmp

def adjustnonnan(pvalues, method='holm'):
    """Convenience function for doing p-value adjustment.
    Accepts any matrix shape and adjusts across the entire matrix.
    Ignores nans appropriately.

    Parameters
    ----------
    pvalues : list, pd.DataFrame, pd.Series or np.ndarray
        Contains pvalues and optionally nans for adjustment.
    method : str
        An adjustment method for sm.stats.multipletests.
        Use 'holm' for Holm-Bonferroni FWER-adj and
        'fdr_bh' for Benjamini and Hochberg FDR-adj

    Returns
    -------
    adjpvalues : same as pvalues in type and shape"""

    """Turn it into a one-dimensional vector"""

    p = np.asarray(pvalues).flatten()

    """adjpvalues intialized with p to copy nans in the right places"""
    adjpvalues = np.copy(p)
    
    nanInd = np.isnan(p)
    p = p[~nanInd]
    if len(p) == 0:
        return pvalues
        
    """Drop the nans, calculate adjpvalues, copy to adjpvalues vector"""
    rej, q, alphasidak, alphabon = sm.stats.multipletests(p, alpha=0.05, method=method)
    adjpvalues[~nanInd] = q
    
    """Reshape adjpvalues"""
    if not isinstance(pvalues, list):
        adjpvalues = adjpvalues.reshape(pvalues.shape)

    """Return same type as pvalues"""
    if isinstance(pvalues, list):
        return [pv for pv in adjpvalues]
    elif isinstance(pvalues, pd.core.frame.DataFrame):
        return pd.DataFrame(adjpvalues, columns=pvalues.columns, index=pvalues.index)
    elif isinstance(pvalues, pd.core.series.Series):
        return pd.Series(adjpvalues, name=pvalues.name, index=pvalues.index)
    else:
        return adjpvalues
