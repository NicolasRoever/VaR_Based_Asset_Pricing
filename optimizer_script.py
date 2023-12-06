from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest



def objective_function_only_var(w, returns, initial_wealth):
    return - (w[0]* returns[0] + w[1] *returns[1]) * initial_wealth

def objective_function_markowitz(w, returns, vcv_matrix, initial_wealth, risk_aversion):
    expected_value_portfolio_number = expected_value_portfolio(w, returns, initial_wealth)
    variance = initial_wealth**2 * variance_portfolio(w, vcv_matrix)
    return -(expected_value_portfolio_number - risk_aversion/2 * variance)


def weight_sum_constraint(w):
    e = np.ones(len(w))
    return w @ e.T - 1

def expected_value_portfolio(w, returns, initial_wealth):
    return (w @ returns.T) * initial_wealth

def variance_portfolio(w, vcv_matrix):
    return w.T @ vcv_matrix @ w

def calculate_var(w, quantile, vcv_matrix, returns, initial_wealth):
    variance = variance_portfolio(w, vcv_matrix)
    expected_value_portfolio_number = expected_value_portfolio(w, returns, initial_wealth)
    return np.sqrt(variance) * norm.ppf(quantile) + expected_value_portfolio_number


def var_risk_constraint(w, quantile, vcv_matrix, returns, initial_wealth):
    variance = variance_portfolio(w, vcv_matrix)
    var = calculate_var(w, quantile, vcv_matrix, returns, initial_wealth)
    return  var/norm.ppf(quantile) - variance  



def calculate_optimal_weights(returns, vcv_matrix, initial_wealth, var_quantile): 
    """ This function calculates the optimal weights for an investor maximizing return 
    under a Value-at-Risk constraint. The market has only 2 risky assets

    Args:

    returns (np.array): A numpy array of expected returns for each asset
    vcv_matrix (np.array): A numpy array of the variance-covariance matrix for the returns od the assets!
    initial_wealth (float): The initial wealth of the investor
    var_quantile (float): The quantile of Value-at-Risk constraint

    Returns:
    optimal_weights (np.array): A numpy array of the optimal weights for each asset

    """


    # Initial guess for the optimization
    initial_guess = [0.5, 0.5]
    
    #Constraint Dictionary
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                {'type': 'ineq', 'fun': var_risk_constraint, 'args': (var_quantile, vcv_matrix, returns, initial_wealth)})

    #Perform optimization

    result = minimize(objective_function_only_var, initial_guess, constraints=constraints, args=(returns, initial_wealth))

    return result.x

def plot_variance_grid_asset_1(returns, vcv_matrix, initial_wealth, var_quantile, risk_aversion):
    variance_grid = np.linspace(vcv_matrix[0,0], 1.5 * vcv_matrix[1,1], 100)
    results_var_constraint = []
    results_markowitz = []

    
    for variance in variance_grid:
        vcv_matrix_update = vcv_matrix.copy()
        vcv_matrix_update[0,0] = variance

        w_1 = calculate_optimal_weights(returns, vcv_matrix_update, initial_wealth, var_quantile)
        results_var_constraint.append(w_1[0])

        w_1_markowitz =  calculate_optimal_weights_markowitz(returns, vcv_matrix_update, initial_wealth, risk_aversion)
        results_markowitz.append(w_1_markowitz[0])

    fig = go.Figure()

    # First trace
    fig.add_trace(go.Scatter(x=variance_grid, y= results_var_constraint, mode='markers', name='Optimal Weight VaR Optimization'))

    # Adding a second y-variable
    results2 = [0.1, 0.15, 0.2, 0.25]
    fig.add_trace(go.Scatter(x=variance_grid, y=results_markowitz, mode='markers', name='Optimal Weight Markowitz'))

    # Update layout if necessary
    fig.update_layout(
        title='Markowitz Portfolio Optimization',
        xaxis=dict(title='Variance'),
        yaxis=dict(title='Return'),
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )


    return fig



def calculate_optimal_weights_markowitz(returns, vcv_matrix, initial_wealth, risk_aversion): 
    """ THis function calculates the optimal weights for a Markowitz optimizing agent. 

    Args: 

    returns (np.array): A numpy array of expected returns for each asset
    vcv_matrix (np.array): A numpy array of the variance-covariance matrix for the returns od the assets!
    initial_wealth (float): The initial wealth of the investor
    var_quantile (float): The quantile of Value-at-Risk constraint
    risk_aversion (float): Risk aversion coefficient of the agent

    Returns:
    optimal_weights (np.array): A numpy array of the optimal weights for each asset

    
    
    """

    # Initial guess for the optimization
    initial_guess = [0.5, 0.5]
    
    #Constraint Dictionary
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint})

    #Perform optimization

    result = minimize(objective_function_markowitz, initial_guess, constraints=constraints, args=(returns, vcv_matrix, initial_wealth, risk_aversion))

    return result.x



def calculate_optimal_weights_markowitz_var(returns, vcv_matrix, initial_wealth, risk_aversion, var_quantile): 
    """ THis function calculates the optimal weights for a Markowitz optimizing agent. 

    Args: 

    returns (np.array): A numpy array of expected returns for each asset
    vcv_matrix (np.array): A numpy array of the variance-covariance matrix for the returns od the assets!
    initial_wealth (float): The initial wealth of the investor
    var_quantile (float): The quantile of Value-at-Risk constraint
    risk_aversion (float): Risk aversion coefficient of the agent

    Returns:
    optimal_weights (np.array): A numpy array of the optimal weights for each asset

    
    
    """





    



# Check if the script is being run as the main program
if __name__ == "__main__":

    returns = np.array([1.45, 1.5])
    vcv_matrix = np.array([
    [0.06, 0.025],  # Covariance between Asset A and Asset B
    [0.025, 0.10]   # Variance of Asset B
    ])
    initial_wealth = 100
    var_quantile = 0.999
    risk_aversion = 2


    fig = plot_variance_grid_asset_1(returns, vcv_matrix, initial_wealth, var_quantile, risk_aversion)

    fig.show()
