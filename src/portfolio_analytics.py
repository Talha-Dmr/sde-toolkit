import numpy as np
from scipy.optimize import minimize


def calc_portfolio_returns(asset_paths, weights):
    """
    Calculate portfolio returns from asset price paths.
    Args:
        asset_paths: np.ndarray, shape (num_assets, num_steps+1)
            Simulated asset price paths (each row: one asset)
        weights: array-like, shape (num_assets,)
            Portfolio weights, sum to 1
    Returns:
        port_returns: np.ndarray, shape (num_steps,)
            Portfolio return series
    """
    # Calculate arithmetic returns for each asset
    asset_returns = (asset_paths[:, 1:] - asset_paths[:, :-1]) / asset_paths[:, :-1]
    # Weighted sum across assets
    port_returns = np.dot(weights, asset_returns)
    return port_returns

def calc_portfolio_value(port_returns, initial_value=1.0):
    """
    Calculate the portfolio value path given returns.
    Args:
        port_returns: np.ndarray, shape (num_steps,)
            Portfolio return series
        initial_value: float
            Initial portfolio value
    Returns:
        value_path: np.ndarray, shape (num_steps+1,)
            Portfolio value at each step
    """
    value_path = np.empty(len(port_returns) + 1)
    value_path[0] = initial_value
    for t in range(1, len(value_path)):
        value_path[t] = value_path[t-1] * (1 + port_returns[t-1])
    return value_path


def value_at_risk(returns, alpha=0.01):
    """
    Calculate Value at Risk (VaR) at given alpha level.
    Args:
        returns: np.ndarray
            Portfolio returns
        alpha: float
            Significance level (e.g. 0.01 for 1%)
    Returns:
        var: float
            Value at Risk
    """
    return np.percentile(returns, 100 * alpha)

def conditional_value_at_risk(returns, alpha=0.01):
    """
    Calculate Conditional Value at Risk (CVaR, Expected Shortfall) at given alpha level.
    Args:
        returns: np.ndarray
            Portfolio returns
        alpha: float
            Significance level
    Returns:
        cvar: float
            Conditional Value at Risk
    """
    var = value_at_risk(returns, alpha)
    return returns[returns <= var].mean()


def min_variance_portfolio(cov_matrix, allow_short=False):
    """
    Find portfolio weights with minimum variance (risk).
    Args:
        cov_matrix: np.ndarray, shape (n_assets, n_assets)
            Covariance matrix of asset returns
        allow_short: bool
            If False, weights >= 0 (no short selling)
    Returns:
        weights: np.ndarray, shape (n_assets,)
            Optimal weights (sum to 1)
    """
    n = cov_matrix.shape[0]
    init_w = np.ones(n) / n

    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # Bounds: no short sell if not allowed
    bounds = None if allow_short else tuple((0, 1) for _ in range(n))

    def portfolio_variance(w):
        return w.T @ cov_matrix @ w

    result = minimize(portfolio_variance, init_w, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x
