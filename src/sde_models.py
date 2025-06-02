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
