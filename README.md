## fishersapi
[![Build Status](https://travis-ci.com/agartland/fishersapi.svg?branch=master)](https://travis-ci.com/agartland/fishersapi)

A package for applying a fast implementation of Fisher's exact test to observations in a pandas DataFrame.

Contingency tables are computed based on all pairs of columns in cols and all pairs of unique values within the columns.
The results are tested against scipy.stats.fishers_exact and fallback on scipy if the faster brentp/fishers_exact_test (~1000x faster) is not installed.
The fast version of the test is computed using the package `fisher` developed by Haibao Tang and Brent Pedersen, which uses cython.
 - https://pypi.python.org/pypi/fisher/
 - https://github.com/brentp/fishers_exact_test

## Installation
The package is compatible with Python 2.7 or Python 3.6 and can be installed from PyPI or cloned and installed directly.

```bash
pip install fishersapi
```

## Example
```python
import fishersapi
n = 50
df = pd.DataFrame({'VA':np.random.choice(['TRAV14', 'TRAV12', 'TRAV3', 'TRAV23', 'TRAV11', 'TRAV6'], n),
                   'JA':np.random.choice(['TRAJ4', 'TRAJ2', 'TRAJ3','TRAJ5', 'TRAJ21', 'TRAJ13'], n),
                   'VB':np.random.choice(['TRBV14', 'TRBV12', 'TRBV3', 'TRBV23', 'TRBV11', 'TRBV6'], n),
                   'JB':np.random.choice(['TRBJ4', 'TRBJ2', 'TRBJ3','TRBJ5', 'TRBJ21', 'TRBJ13'], n)})
df = df.assign(Count=1)
df.loc[:10, 'Count'] = 15

res = fishersapi.fishers_frame(df, ['VA', 'JA', 'VB', 'JB'], count_col=None, alternative='two-sided')
```