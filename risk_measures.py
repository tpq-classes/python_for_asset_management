#
# Mean-Variance Portofolio Class
# Markowitz (1952)
#
# Python for Asset Management
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import math
import numpy as np
import pandas as pd
import scipy.stats as scs

def portfolio_return(weights, rets):
    return np.dot(weights.T, rets.mean()) * 252

def portfolio_variance(weights, rets):
    return np.dot(weights.T, np.dot(rets.cov(), weights)) * 252

def portfolio_volatility(weights, rets):
    return math.sqrt(portfolio_variance(weights, rets))

def standard_deviation_mu(weights, rets, c=1):
    return float(c * portfolio_volatility(weights, rets) -
                 portfolio_return(weights, rets))

def value_at_risk(weights, rets, percs=None):
    r = np.dot(rets, weights)
    if percs is None:
        percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
    var = scs.scoreatpercentile(r, percs)
    v = list()
    for pair in zip(percs, var):
        v.append((100 - pair[0], -pair[1]))
    df = pd.DataFrame(v, columns=['conf', 'VaR'])
    df.set_index('conf', inplace=True)
    return df

def expected_shortfall(weights, rets, percs=None):
    r = np.dot(rets, weights)
    if percs is None:
        percs = [0.01, 0.1, 1., 2.5, 5.0, 10.0]
    var = scs.scoreatpercentile(r, percs)
    v = list()
    for pair in zip(percs, var):
        es = r[r < pair[1]].mean()
        v.append((100 - pair[0], -pair[1], -es))
    df = pd.DataFrame(v, columns=['conf', 'VaR', 'ES'])
    df.set_index('conf', inplace=True)
    return df