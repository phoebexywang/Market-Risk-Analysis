import numpy as np
from scipy.stats import t

def run_monte_carlo(mu, cov_matrix, weights, initial_val, n_simulations=10000, days =1, dist = 'normal'):
    n_assets = len(mu)
    L = np.linalg.cholesky(cov_matrix)

    if dist=='normal':
        z = np.random.normal(size=(n_assets, n_simulations))
    else:
        z = t.rvs(df = 5, size=(n_assets, n_simulations))

    sim_returns = mu.values.reshape(-1,1) + np.dot(L, z)

    portfolio_sim_returns = np.dot(weights, sim_returns)
    portfolio_value = initial_val * (1+portfolio_sim_returns)

    return initial_val - portfolio_value

def calculate_metrics(losses, alpha = .99):
    var = np.percentile(losses, alpha*100)
    expected_shortfall = losses[losses >var ].mean()
    return var, expected_shortfall

