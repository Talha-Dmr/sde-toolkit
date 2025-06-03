import numpy as np

def simulate_gbm_euler(S0, mu, sigma, T, N, random_seed=None):
    """
    Geometric Brownian Motion path simulation (Euler-Maruyama method)
    
    Args:
        S0: initial value
        mu: drift coefficient
        sigma: volatility coefficient
        T: total time
        N: number of steps
        random_seed: (optional) for reproducibility
        
    Returns:
        t: time vector
        S: simulated path
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))
        S[i] = S[i-1] + mu * S[i-1] * dt + sigma * S[i-1] * dW
    return t, S



def simulate_vasicek_euler(r0, a, b, sigma, T, N, random_seed=None):
    """
    Simulate a path of the Vasicek model using the Euler-Maruyama method.
    Args:
        r0: initial value
        a: mean reversion speed
        b: long-term mean
        sigma: volatility
        T: total time
        N: number of steps
        random_seed: (optional) for reproducibility
    Returns:
        t: time vector
        r: simulated path
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    r = np.zeros(N+1)
    r[0] = r0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        r[i] = r[i-1] + a * (b - r[i-1]) * dt + sigma * dW  # Vasicek update
    return t, r



def simulate_cir_euler(r0, a, b, sigma, T, N, random_seed=None):
    """
    Simulate a path of the CIR model using the Euler-Maruyama method.
    Args:
        r0: initial value
        a: mean reversion speed
        b: long-term mean
        sigma: volatility
        T: total time
        N: number of steps
        random_seed: (optional) for reproducibility
    Returns:
        t: time vector
        r: simulated path
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    r = np.zeros(N+1)
    r[0] = r0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian increment
        # Protect against negative values for the square root
        r[i] = r[i-1] + a * (b - r[i-1]) * dt + sigma * np.sqrt(max(r[i-1], 0)) * dW
    return t, r



def simulate_multi_gbm_euler(
    S0_vec, mu_vec, sigma_vec, corr_matrix, T, N, random_seed=None
):
    """
    Simulate correlated paths for multiple assets using GBM (Euler-Maruyama method).

    Args:
        S0_vec: array-like, shape (num_assets,)
            Initial prices for each asset
        mu_vec: array-like, shape (num_assets,)
            Drift for each asset
        sigma_vec: array-like, shape (num_assets,)
            Volatility for each asset
        corr_matrix: array-like, shape (num_assets, num_assets)
            Correlation matrix between the Brownian motions
        T: float
            Total time
        N: int
            Number of time steps
        random_seed: int or None
            For reproducibility

    Returns:
        t: array, shape (N+1,)
            Time vector
        S_paths: array, shape (num_assets, N+1)
            Simulated asset paths (each row: one asset)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    num_assets = len(S0_vec)
    dt = T / N
    t = np.linspace(0, T, N+1)
    S_paths = np.zeros((num_assets, N+1))
    S_paths[:, 0] = S0_vec

    # Cholesky decomposition for correlated Brownian motions
    L = np.linalg.cholesky(corr_matrix)

    for i in range(1, N+1):
        # Generate N(0,1) random variables for each asset
        Z = np.random.normal(0, 1, num_assets)
        # Create correlated increments
        dW = L @ Z * np.sqrt(dt)
        # Euler-Maruyama update for each asset
        S_paths[:, i] = (
            S_paths[:, i-1]
            + mu_vec * S_paths[:, i-1] * dt
            + sigma_vec * S_paths[:, i-1] * dW
        )

    return t, S_paths