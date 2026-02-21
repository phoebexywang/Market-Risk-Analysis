import numpy as np
from scipy.stats import chi2

def kupiec_test(historical_returns, weights, initial_val, var_estimate, alpha=0.99):
    hist_returns = np.dot(weights, historical_returns)
    actual_losses = -(hist_returns * initial_val)

    n = len(actual_losses)
    x = np.sum(actual_losses > var_estimate)
    p = 1-alpha

    if x == 0: return 1.0

    term1 = (1-p)**(n-x) * (p**x)
    term2 = (1-(x/n))**(n-x) * ((x/n)**x)
    lr =-2 * np.log(term1/term2)

    p_value = 1 - chi2.cdf(lr, df=1)
    return p_value #if greater than 0.05, the model passes

