#
# Mean-Variance Portofolio Class
# Markowitz (1952)
#
# Python for Asset Management
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import math
import cufflinks
import eikon as ek
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.io as pio
cufflinks.go_offline()
cufflinks.set_config_file(offline=True)
pio.renderers.default = "colab"
np.set_printoptions(suppress=True, precision=4)


class FinancialData:
    url = 'http://hilpisch.com/dax_eikon_eod_data.csv'
    url_ = 'http://hilpisch.com/dax_eikon_mc_data.csv'
    def __init__(self, universe=None):
        self.universe = universe
        self.retrieve_data()
        self.prepare_data()
    def retrieve_data(self):
        self.raw = pd.read_csv(self.url, index_col=0, parse_dates=True)
        self.raw_ = pd.read_csv(self.url_, index_col=0)
    def prepare_data(self):
        if self.universe is None:
            self.data = self.raw
            self.universe = self.data.columns
        else:
            self.data = self.raw[self.universe]
        self.no_assets = len(self.universe)
        self.rets = np.log(self.data / self.data.shift(1))
        self.mc = (self.raw_.T[self.universe]).T
        self.mc['MC%'] = self.mc['MC'].apply(lambda x: x / self.mc['MC'].sum())
    def plot_data(self, cols=None):
        if cols is None:
            cols = self.universe
        self.data[cols].normalize().iplot()
    def plot_mc(self):
        self.mc.sort_values('MC').iplot(kind='pie',
                values='MC', labels='NAME', colorscale='rdylbu')
    def plot_corr(self):
        self.rets.corr().iplot(kind='heatmap', colorscale='reds')
        
        
class MVPPortfolio(FinancialData):
    def __init__(self, universe):
        super().__init__(universe)
        self.equal_weights = np.array(self.no_assets * [1 / self.no_assets])
        self.mc_weights = self.mc['MC%'].values
    def portfolio_return(self, weights, days=252):
        return np.dot(self.rets.mean(), weights) * days
    def portfolio_variance(self, weights, days=252):
        return np.dot(weights, np.dot(self.rets.cov(), weights)) * days
    def portfolio_volatility(self, weights, days=252):
        return math.sqrt(self.portfolio_variance(weights, days))
    def portfolio_sharpe(self, weights, days=252):
        sharpe = (self.portfolio_return(weights, days) /
                  self.portfolio_volatility(weights, days))
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
