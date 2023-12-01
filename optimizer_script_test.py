from optimizer_script import objective_function
import numpy as np
import pytest




def test_objective_function_expected_result():
     
     w = np.array([0.5, 0.5])
     returns = np.array([1.5, 2])
     initial_wealth = 100

     actual_result = objective_function(w, returns, initial_wealth)
     expected_result = -175

     np.testing.assert_allclose(actual_result, expected_result)