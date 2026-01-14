#
# Black-Litterman 1992
# Portfolio Class
#
# Python for Asset Management
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import math
import cufflinks
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.io as pio
cufflinks.go_offline()
cufflinks.set_config_file(offline=True)
np.set_printoptions(suppress=True, precision=4)
pio.renderers.default = "colab"


class FinancialData:
    url = 'http://hilpisch.com/aiif_eikon_eod_data.csv'
    # url = 'http://hilpisch.com/indices_eikon_eod_data.csv'
    # url = 'http://hilpisch.com/dax_eikon_eod_data.csv'
    def __init__(self, universe=None):
        self.universe = universe
        self.retrieve_data()
        self.prepare_data()
    def retrieve_data(self):
        self.raw = pd.read_csv(self.url, index_col=0, parse_dates=True)
    def prepare_data(self):
        if self.universe is None:
            self.data = self.raw
            self.universe = self.data.columns
        else:
            self.data = self.raw[self.universe]
        self.no_assets = len(self.universe)
        self.rets = np.log(self.data / self.data.shift(1))
    def plot_data(self, cols=None):
        if cols is None:
            cols = self.universe
        self.data[cols].normalize().iplot()
    def plot_corr(self):
        self.rets.corr().iplot(kind='heatmap', colorscale='reds')
        
        
class BL92Portfolio(FinancialData):
    def __init__(self, universe, start=None, end=None, views=None, tau=None):
        super().__init__(universe)
        self.equal_weights = self.no_assets * [1 / self.no_assets]
        self.start = start
        self.end = end
        if views is None:
            self.views = list()
            self.tau = 0.000001
        else:
            self.views = views
            if tau is None:
                self.tau = 1
            else:
                self.tau = tau
        self.generate_bl_objects()
        self.generate_bl_statistics()
    def add_view(self, view):
        self.views.append(view)
        self.generate_bl_objects()
        self.generate_bl_statistics()
    def remove_view(self, index):
        self.views.pop(index)
        self.generate_bl_objects()
        self.generate_bl_statistics()
    def generate_bl_objects(self):
        v = len(self.views)
        self.P = pd.DataFrame(np.zeros((v, self.no_assets)),
                              columns=self.universe)
        self.q = np.zeros(v)
        self.omega = np.zeros((v, v))
        for i, view in enumerate(self.views):
            for key, value in view[0].items():
                self.P.loc[i, key] = value
            self.q[i] = view[1]
            self.omega[i, i] = view[2]
    def generate_bl_statistics(self):
        if self.start is None:
            self.start = self.raw.index[0]
        if self.end is None:
            self.end = self.raw.index[-1]
        self.mu = self.rets.loc[self.start:self.end].mean() * 252
        self.cov = self.rets.loc[self.start:self.end].cov() * 252
        C = self.tau * self.cov
        m1 = np.dot(self.P.T, np.dot(np.linalg.inv(self.omega), self.P))
        m1 += np.linalg.inv(C)
        m2 = np.dot(self.P.T, np.dot(np.linalg.inv(self.omega), self.q))
        m2 += np.dot(np.linalg.inv(C), self.mu)
        self.mu_ = np.dot(np.linalg.inv(m1), m2)
        self.cov_ = np.linalg.inv(m1) + self.cov
    def portfolio_return(self, weights):
        return np.dot(self.mu_, weights)
    def portfolio_variance(self, weights):
        return np.dot(weights, np.dot(self.cov_, weights))
    def portfolio_volatility(self, weights):
        return math.sqrt(self.portfolio_variance(weights))
    def portfolio_sharpe(self, weights):
        sharpe = self.portfolio_return(weights) / self.portfolio_volatility(weights)
        return sharpe
    def _set_bounds_constraints(self, bnds, cons):
        if bnds is None:
            self.bnds = self.no_assets * [(0, 1)]
        else:
            self.bnds = bnds
        if cons is None:
            self.cons = {'type': 'eq', 'fun': lambda weights: weights.sum() - 1}
        else:
            self.cons = cons
    def _get_results(self, opt, kind):
        ret = self.portfolio_return(opt['x'])
        vol = self.portfolio_volatility(opt['x'])
        sharpe = self.portfolio_sharpe(opt['x'])
        weights = pd.DataFrame(opt['x'], index=self.universe, columns=['weights',])
        res = {'kind': kind, 'weights': weights.round(7), 'return': ret,
               'volatility': vol, 'sharpe': sharpe}
        return res
    def minimum_volatility_portfolio(self, bnds=None, cons=None):
        self._set_bounds_constraints(bnds, cons)
        opt = minimize(self.portfolio_volatility, self.equal_weights,
                      bounds=self.bnds, constraints=self.cons)
        self.results = self._get_results(opt, 'Minimum Volatility')
        return self.results
    def maximum_sharpe_portfolio(self, bnds=None, cons=None):
        self.generate_bl_objects()
        self.generate_bl_statistics()
        self._set_bounds_constraints(bnds, cons)
        tf = lambda weights: -self.portfolio_sharpe(weights)
        opt = minimize(tf, self.equal_weights, bounds=self.bnds,
                       constraints=self.cons)
        self.results = self._get_results(opt, 'Maximum Sharpe')
        return self.results
    def plot_weights(self, kind='pie'):
        if kind == 'pie':
            nonzero = self.results['weights'] > 0
            to_plot = self.results['weights'][nonzero['weights']].copy()
            to_plot['names'] = to_plot.index
            to_plot.iplot(kind='pie', values='weights',
                          labels='names', colorscale='rdylbu',
                          title='Optimal Weights | ' + self.results['kind'])
        else:
            self.results['weights'].iplot(kind='bar',
                    title='Optimal Weights | ' + self.results['kind'])
    def plot_performance(self):
        perf = (self.results['return'], self.results['volatility'], self.results['sharpe'])
        index = ['return', 'volatility', 'sharpe']
        to_plot = pd.DataFrame(perf, index=index, columns=['metrics',])
        to_plot.iplot(kind='bar', title='Performance Metrics  | ' + self.results['kind'])
