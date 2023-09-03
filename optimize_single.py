import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew, jarque_bera


def portfolio_optimization(returns, loss_func:str, cov_estimation:str='historical'):
    # return the portfolio weight for 1 rebalance period 
    def constraint_basic(weights):
        return np.sum(weights) - 1
    
    def constraint_long_only(weights):
        return weights    # make sure weights are positive
    
    def objective_variance(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return portfolio_volatility
    
    def objective_sharpe(weights):
        portfolio_return = np.dot(mean_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return - portfolio_return / portfolio_volatility
    
    def objective_markowitz3(weights):
        a = 1000 # hyperparameter tuning through grid search
        b = 1
        portfolio_return = np.dot(mean_returns, weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = portfolio_return / portfolio_volatility
        entropy = np.exp(-np.sum(weights * np.log(weights)))
        return -(a * sharpe + b * entropy) / (a + b)
    
    def objective_divers_ratio(weights):
        portfolio_variance = 0
        weighted_sum = 0
        for i in range(cov_matrix.shape[0]):
            portfolio_variance += weights[i]**2*cov_matrix[i][i]
            weighted_sum += weights[i]**2*cov_matrix[i][i]
            for j in range(cov_matrix.shape[0]):
                if i>j:
                    portfolio_variance+= 2*weights[i]*weights[j]*cov_matrix[i][j]
        DR = np.sqrt(weighted_sum/portfolio_variance)
        return - DR

    def objective_marginal_risk_contribution(weights):
        portfolio_TR = 0
        portfolio_variance = 0
        for i in range(cov_matrix.shape[0]):
            portfolio_variance += weights[i]**2*cov_matrix[i][i]
            RC_i = 2*weights[i]**2*cov_matrix[i][i]
            for j in range(cov_matrix.shape[0]):
                if j!=i:
                    portfolio_variance+= weights[i]*weights[j]*cov_matrix[i][j]
                    RC_i += weights[i]*weights[j]*cov_matrix[i][j]
            portfolio_TR+=RC_i
        return portfolio_TR/np.sqrt(portfolio_variance)
    
    # basic feature
    tickers = returns.columns
    date = returns.index[-1]
    num_assets = returns.shape[1]
    mean_returns = np.mean(returns, axis=0)
    annualized_vol = returns.std() * np.sqrt(252 / 66)
    
    # Covariance Matrix:
    if cov_estimation == 'historical':
        cov_matrix = np.cov(returns.T, ddof=1)
    if cov_estimation == 'regularized':
        c = 0.00005
        cov_matrix = np.cov(returns.T, ddof=1) 
        cov_matrix += c * np.eye(cov_matrix.shape[0])
    if cov_estimation == 'DCC_GARCH':
        vol = mgarch.mgarch()
        vol.fit(returns)
        ndays = 1
        cov_matrix = vol.predict(ndays)['cov']
    
    # calculate weight by minimizing objective, subject to constraint
    constraints = [{'type': 'eq', 'fun': constraint_basic}, {'type': 'ineq', 'fun': constraint_long_only}]
    init_weights = np.array([1 / num_assets] * num_assets)
    if loss_func == 'equal_weight':
        return pd.DataFrame({'Date':date, 
                             'Ticker':tickers, 
                             'Weight':init_weights, 
                             'Annualized_vol':annualized_vol
                            })
    objective = locals()['objective_' + loss_func] # call the objective function associated with input
    weights = minimize(objective, init_weights, method='SLSQP', constraints=constraints).x
    
    return pd.DataFrame({'Date':date, 
                         'Ticker':tickers, 
                         'Weight':weights, 
                         'Annualized_vol':annualized_vol
                        })