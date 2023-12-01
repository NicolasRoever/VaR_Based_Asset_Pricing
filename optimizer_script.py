from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np
import pytest 



def objective_function(w, returns, initial_wealth):
        return - (w[0]* returns[0] + w[1] *returns[1]) * initial_wealth





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

    #Define objective function

    def objective_function(w, returns, initial_wealth):
        return - (w[0]* returns[0] + w[1] *returns[1]) * initial_wealth


    def weight_sum_constraint(w):
        return w[0] + w[1] - 1.0


    def var_risk_constraint(w, quantile, vcv_matrix, returns, initial_wealth):
        return  ((np.sqrt(w.T @ vcv_matrix @ w) * norm.ppf(quantile) + (w[0] *returns[0] + w[1] *returns[1] * initial_wealth)))^2/norm.ppf(quantile) - (w.T @ vcv_matrix @ w)

    # Initial guess for the optimization
    initial_guess = [0.5, 0.5]
    
    #Constraint Dictionary
    constraints = ({'type': 'eq', 'fun': weight_sum_constraint},
                {'type': 'ineq', 'fun': var_risk_constraint, 'args': (var_quantile, vcv_matrix, returns, initial_wealth)})

    #Perform optimization

    result = minimize(objective_function, initial_guess, constraints=constraints)

    return result.x



# Check if the script is being run as the main program
if __name__ == "__main__":

    returns = np.array([1.1, 1.2])
    vcv_matrix = np.array([[1.1, 0.5], [0.5, 1.5]])
    initial_wealth = 100
    var_quantile = 0.99

    optimal_weights = calculate_optimal_weights(returns, vcv_matrix, initial_wealth, var_quantile)

    print(optimal_weights)