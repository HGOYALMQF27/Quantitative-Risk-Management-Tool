# risk_management.py

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize

def compute_var(returns, confidence_level=0.95):
    portfolio_returns = np.sum(returns, axis=1)
    var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
    return var

def compute_es(returns, confidence_level=0.95):
    portfolio_returns = np.sum(returns, axis=1)
    var = compute_var(returns, confidence_level)
    exceedances = portfolio_returns[portfolio_returns < var]
    es = np.mean(exceedances)
    return es

def compute_cdar(returns, lookback_period=100, confidence_level=0.95):
    drawdowns = np.cumsum(returns, axis=0)
    max_drawdowns = np.maximum.accumulate(drawdowns, axis=0)
    drawdown_periods = np.argmax(drawdowns - max_drawdowns, axis=0)
    exceedances = drawdowns[drawdown_periods <= lookback_period, :]
    cdar = np.percentile(exceedances, (1 - confidence_level) * 100, axis=0)
    return cdar

def mean_variance_optimization(returns, target_return):
    num_assets = returns.shape[1]

    def objective(weights):
        portfolio_return = np.sum(returns.mean(axis=0) * weights)
        portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
        return portfolio_variance

    def constraint(weights):
        return np.sum(returns.mean(axis=0) * weights) - target_return

    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': constraint})
    result = minimize(objective, initial_weights, constraints=constraints)
    
    return result.x

def cvar_optimization(returns, confidence_level=0.95):
    def cvar_objective(weights):
        portfolio_returns = np.dot(returns, weights)
        cvar = np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, (1 - confidence_level) * 100)])
        return cvar

    initial_weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    result = minimize(cvar_objective, initial_weights, constraints=constraints, bounds=bounds)
    
    return result.x
