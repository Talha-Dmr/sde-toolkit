import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from src.sde_models import (
    simulate_gbm_euler,
    simulate_vasicek_euler,
    simulate_cir_euler,
    simulate_multi_gbm_euler
)
import numpy as np

# Simulate GBM path (single asset)
t, s_gbm = simulate_gbm_euler(
    S0=0.03,      # Initial value (matching the others for fair comparison)
    mu=0.7,       # Drift
    sigma=0.1,    # Volatility
    T=1.0,        # Total time (years)
    N=252,        # Number of steps
    random_seed=42
)
plt.plot(t, s_gbm, label="GBM (single asset)")

# Simulate Vasicek path
t, r_vasicek = simulate_vasicek_euler(
    r0=0.03,
    a=0.7,
    b=0.05,
    sigma=0.1,
    T=1.0,
    N=252,
    random_seed=42
)
plt.plot(t, r_vasicek, label="Vasicek")

# Simulate CIR path
t, r_cir = simulate_cir_euler(
    r0=0.03,
    a=0.7,
    b=0.05,
    sigma=0.1,
    T=1.0,
    N=252,
    random_seed=42
)
plt.plot(t, r_cir, label="CIR")

# Multi-asset GBM example: 3 assets, plot first asset's path
S0_vec = np.array([0.03, 0.04, 0.05])
mu_vec = np.array([0.7, 0.6, 0.5])
sigma_vec = np.array([0.1, 0.08, 0.12])
corr_matrix = np.array([
    [1.0, 0.7, 0.2],
    [0.7, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])
t_multi, S_paths = simulate_multi_gbm_euler(
    S0_vec, mu_vec, sigma_vec, corr_matrix, T=1.0, N=252, random_seed=42
)
plt.plot(t_multi, S_paths[0], label="GBM (multi-asset, Asset 1)", linestyle="--")

plt.title("GBM, Vasicek, CIR, and Multi-Asset GBM Paths")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
