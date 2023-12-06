from optimizer_script import objective_function_only_var, weight_sum_constraint, variance_portfolio, var_risk_constraint, calculate_var, expected_value_portfolio, calculate_optimal_weights, calculate_optimal_weights_markowitz, calculate_optimal_weights_markowitz_var
import numpy as np
import pytest


def test_objective_function_var_only_expected_result():
     
     w = np.array([0.5, 0.5])
     returns = np.array([1.5, 2])
     initial_wealth = 100

     actual_result = objective_function_only_var(w, returns, initial_wealth)
     expected_result = -175

     np.testing.assert_allclose(actual_result, expected_result)


def test_weight_constraint_expected_result():
     w = np.array([0.5, 0.5])
     actual_result = weight_sum_constraint(w)
     expected_result = 0

     np.testing.assert_allclose(actual_result, expected_result)


def test_variance_portfolio_expected_result():
     w = np.array([0.4, 0.6])
     vcv_matrix = np.array([[1.1, 0.5], [0.5, 1.5]])
     actual_result = variance_portfolio(w, vcv_matrix)
     expected_result = 0.956

     np.testing.assert_allclose(actual_result, expected_result)

def test_expected_value_portfolio_expected_result():
        w = np.array([0.4, 0.6])
        returns = np.array([1.5, 2])
        initial_wealth = 100
        actual_result = expected_value_portfolio(w, returns, initial_wealth)
        expected_result = 180
    
        np.testing.assert_allclose(actual_result, expected_result)

def test_calculate_var_expected_result():
     w = np.array([0.4, 0.6])
     vcv_matrix = np.array([[1.1, 0.5], [0.5, 1.5]])
     returns = np.array([1.5, 2])
     initial_wealth = 100
     quantile = 0.999
     actual_result = calculate_var(w, quantile, vcv_matrix, returns, initial_wealth)
     expected_result = 183.02
     np.testing.assert_allclose(actual_result, expected_result, atol=0.2)
     

def test_var_risk_constraint_expected_result():
     w = np.array([0.4, 0.6])
     vcv_matrix = np.array([[1.1, 0.5], [0.5, 1.5]])
     returns = np.array([1.5, 2])
     initial_wealth = 100
     quantile = 0.999

     actual_result = var_risk_constraint(w, quantile, vcv_matrix, returns, initial_wealth)
     expected_result = 58.27

     np.testing.assert_allclose(actual_result, expected_result, atol=0.2)


def test_calculate_optimal_weights_expected_result():
    returns = np.array([1.1, 1.2])
    vcv_matrix = np.array([[1.1, 0.5], [0.5, 1.5]])
    initial_wealth = 100
    var_quantile = 0.99

    # Set seed for NumPy
    np.random.seed(42)

    optimal_weights = calculate_optimal_weights(returns, vcv_matrix, initial_wealth, var_quantile)

    expected_result = np.array([-6.86379747,   7.86379747])

    np.testing.assert_allclose(optimal_weights, expected_result, atol=0.2)


def test_calculate_optimal_weights_markowitz_expected_result():
     returns = np.array([1.1, 1.15])
     vcv_matrix = np.array([
     [0.06, 0.025],  # Covariance between Asset A and Asset B
     [0.025, 0.10]   # Variance of Asset B
     ])
     initial_wealth = 100
     var_quantile = 0.999
     risk_aversion = 2.0
     actual = calculate_optimal_weights_markowitz(returns, vcv_matrix, initial_wealth, risk_aversion)
     expected = np.array([0.67954546, 0.32045454])

     assert np.allclose(actual, expected)


def test_calculate_optimal_weights_markowitz_var_expected_result():

     returns = np.array([1.1, 1.15])
     vcv_matrix = np.array([
     [0.06, 0.025],  # Covariance between Asset A and Asset B
     [0.025, 0.10]   # Variance of Asset B
     ])
     initial_wealth = 100
     var_quantile = 0.999
     risk_aversion = 2.0
     actual = calculate_optimal_weights_markowitz_var(returns, vcv_matrix, initial_wealth, risk_aversion, var_quantile)
     expected = np.array([0.67954546, 0.32045454])

     assert np.allclose(actual, expected)







     
