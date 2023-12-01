

def calculate_optimal_weights(returns, vcv_matrix, initial_wealth, var_quantile): 
""" This function calculates the optimal weights for an investor maximizing return 
under a Value-at-Risk constraint. The market has only 2 risky assets

Args:

returns (np.array): A numpy array of expected returns for each asset
vcv_matrix (np.array): A numpy array of the variance-covariance matrix for the assets
initial_wealth (float): The initial wealth of the investor
var_quantile (float): The quantile of Value-at-Risk constraint

Returns:
optimal_weights (np.array): A numpy array of the optimal weights for each asset

"""

