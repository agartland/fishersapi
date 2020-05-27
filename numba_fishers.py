"""An attempt to implement a fishers exact test in numba.
Gave up after a day because some ofthe distributions required
are written in fortran in scipy.special.
Seems like soon numba-scipy may make this effort much easier.
For now will require use of cython"""

from numba import njit
import numpy as np
from scipy.special import comb

# @njit()
def binary_search(n, n1, n2, side, epsilon, pexact, mode):
    """Binary search for where to begin halves in two-sided test."""
    if side == "upper":
        minval = mode
        maxval = n
    else:
        minval = 0
        maxval = mode
    guess = -1
    while maxval - minval > 1:
        if maxval == minval + 1 and guess == minval:
            guess = maxval
        else:
            guess = (maxval + minval) // 2
        pguess = hypergeometric_pmf(guess, n1 + n2, n1, n)
        if side == "upper":
            ng = guess - 1
        else:
            ng = guess + 1
        if pguess <= pexact < hypergeometric_pmf(ng, n1 + n2, n1, n):
            break
        elif pguess < pexact:
            maxval = guess
        else:
            minval = guess
    if guess == -1:
        guess = minval
    if side == "upper":
        while guess > 0 and hypergeometric_pmf(guess, n1 + n2, n1, n) < pexact * epsilon:
            guess -= 1
        while hypergeometric_pmf(guess, n1 + n2, n1, n) > pexact / epsilon:
            guess += 1
    else:
        while hypergeometric_pmf(guess, n1 + n2, n1, n) < pexact * epsilon:
            guess += 1
        while guess > 0 and hypergeometric_pmf(guess, n1 + n2, n1, n) > pexact / epsilon:
            guess -= 1
    return guess

def fisher_exact(table, alternative='two-sided'):
    #table = [[a, b], [c, d]]
    c = np.asarray(table, dtype=np.int64)  # int32 is not enough for the algorithm
    if not c.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        return np.nan, 1.0

    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf

    n1 = c[0, 0] + c[0, 1] # a + b
    n2 = c[1, 0] + c[1, 1] # c + d
    n = c[0, 0] + c[1, 0]  # a + c


    if alternative == 'less':
        pvalue = hypergeometric_cdf(c[0, 0], n1 + n2, n1, n)
    elif alternative == 'greater':
        # Same formula as the 'less' case, but with the second column.
        pvalue = hypergeometric_cdf(c[0, 1], n1 + n2, n1, c[0, 1] + c[1, 1])
    elif alternative == 'two-sided':
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        #print(n, n1, n2)
        pexact = hypergeometric_pmf(c[0, 0], n1 + n2, n1, n)
        pmode = hypergeometric_pmf(mode, n1 + n2, n1, n)
        #print(pexact, pmode)

        epsilon = 1 - 1e-4
        if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= 1 - epsilon:
            return oddsratio, 1.

        elif c[0, 0] < mode:
            print('LT', mode, c[0, 0], n1 + n2, n1, n)
            plower = hypergeometric_cdf(c[0, 0], n1 + n2, n1, n)
            if hypergeometric_pmf(n, n1 + n2, n1, n) > pexact / epsilon:
                return oddsratio, plower

            guess = binary_search(n, n1, n2, "upper", epsilon, pexact, mode)
            pvalue = plower + hypergeometric_sf(guess - 1, n1 + n2, n1, n)
        else:
            pupper = hypergeometric_sf(c[0, 0] - 1, n1 + n2, n1, n)
            if hypergeometric_pmf(0, n1 + n2, n1, n) > pexact / epsilon:
                return oddsratio, pupper

            guess = binary_search(n, n1, n2, "lower", epsilon, pexact, mode)
            pvalue = pupper + hypergeometric_cdf(guess, n1 + n2, n1, n)
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)

    pvalue = min(pvalue, 1.0)

    return oddsratio, pvalue

# @njit(types.intp(types.intp, types.intp), cache=True)
def comb_jit(N, k):
    """
    Numba jitted function that computes N choose k. Return `0` if the
    outcome exceeds the maximum value of `np.intp` or if N < 0, k < 0,
    or k > N.

    Parameters
    ----------
    N : scalar(int)

    k : scalar(int)

    Returns
    -------
    val : scalar(int)

    """
    # From scipy.special._comb_int_long
    # github.com/scipy/scipy/blob/v1.0.0/scipy/special/_comb.pyx
    
    INTP_MAX = np.iinfo(np.intp).max
    if N < 0 or k < 0 or k > N:
        return 0
    if k == 0:
        return 1
    if k == 1:
        return N
    if N == INTP_MAX:
        val = 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        if val > (INTP_MAX // (M - j)):
            val = 0
            break

        val *= M - j
        val //= j
    
    if val != 0:
        return val
    
    M = N + 1
    nterms = min(k, N - k)

    numerator = 1
    denominator = 1
    for j in range(1, nterms + 1):
        numerator *= M - j
        denominator *= j

    val = numerator // denominator
    if val == 0 and comb(N, k) != 0:
        print('comb0_Nk', N, k, '\n\t', numerator, denominator)
        raise ValueError
    return val

def hypergeometric_pmf(k, M, n, N):
    """scipy parameterization
    pmf(k, M, n, N) = choose(n, k) * choose(M - n, N - k) / choose(M, N),
                               for max(0, N - (M-n)) <= k <= min(n, N)
    """
    a = comb_jit(n, k)
    b = comb_jit(M - n, N - k)
    c = comb_jit(M, N)
    res = np.exp(np.log(a)+np.log(b)-np.log(c))
    if np.isnan(res) and not np.isnan(stats.hypergeom.pmf(k, M, n, N)):
        print('NAN', k, M, n, N)
        print('\tABC', a, b, c)
        print('\tA_NK', a, comb(n, k))
        print('\tB_NK',b, comb(M-n, N-k))
        print('\tC_NK',c, comb(M, N))
    return res 

def hypergeom_logpmf(k, M, n, N):
    tot, good = M, n
    bad = tot - good
    result = (betaln(good+1, 1) + betaln(bad+1, 1) + betaln(tot-N+1, N+1) -
              betaln(k+1, good-k+1) - betaln(N-k+1, bad-N+k+1) -
              betaln(tot+1, 1))
    return result

def hypergeom_pmf(self, k, M, n, N):
    # return comb(good, k) * comb(bad, N-k) / comb(tot, N)
    return np.exp(hypergeom_logpmf(k, M, n, N))

def hypergeometric_cdf(k, M, n, N):
    c = np.log(comb(M, N))

    tot = 0
    for kk in range(k+1):
        a = np.log(comb(n, kk))
        b = np.log(comb(M - n, N - kk))
        tot += np.exp(a+b-c)
    return tot

def hypergeometric_sf(k, M, n, N):
    tot = 0

    c = np.log(comb(M, N))
    for kk in range(k+1, N+1):
        a = np.log(comb(n, kk))
        b = np.log(comb(M - n, N - kk))
        tot += np.exp(a+b-c)
    return tot



#M, n, N = [20, 7, 12]
#x = np.arange(0, n+1)

#stats.hypergeom.pmf(x[3], M, n, N)

#hypergeometric_pmf(x[3], M, n, N)


def _betaln(p,q):
    return lgamma(p) + lgamma(q) - lgamma(p + q)
