import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sde_models import (
    simulate_gbm_euler,
    simulate_vasicek_euler,
    simulate_cir_euler,
    simulate_multi_gbm_euler
)
from src.portfolio_analytics import (
    calc_portfolio_returns,
    calc_portfolio_value,
    value_at_risk,
    conditional_value_at_risk,
    min_variance_portfolio
)

# ---- 1. SDE Simulations ----

t, s_gbm = simulate_gbm_euler(100, 0.05, 0.2, 1.0, 252, random_seed=42)
t, r_vas = simulate_vasicek_euler(0.03, 0.7, 0.05, 0.1, 1.0, 252, random_seed=42)
t, r_cir = simulate_cir_euler(0.03, 0.7, 0.05, 0.1, 1.0, 252, random_seed=42)
S0_vec = np.array([100, 90, 80])
mu_vec = np.array([0.05, 0.03, 0.07])
sigma_vec = np.array([0.2, 0.15, 0.25])
corr_matrix = np.array([
    [1.0, 0.8, 0.2],
    [0.8, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])
t, S_paths = simulate_multi_gbm_euler(S0_vec, mu_vec, sigma_vec, corr_matrix, 1.0, 252, random_seed=42)

# ---- 2. Portfolio Analytics (Original/Manual Weights) ----

weights = np.array([0.4, 0.4, 0.2])
port_returns = calc_portfolio_returns(S_paths, weights)
port_value = calc_portfolio_value(port_returns, initial_value=1.0)
var = value_at_risk(port_returns, alpha=0.01)
cvar = conditional_value_at_risk(port_returns, alpha=0.01)

# ---- 3. Minimum Variance (Optimal) Portfolio ----

returns = (S_paths[:, 1:] - S_paths[:, :-1]) / S_paths[:, :-1]
cov_matrix = np.cov(returns)
opt_weights = min_variance_portfolio(cov_matrix, allow_short=False)
opt_port_returns = calc_portfolio_returns(S_paths, opt_weights)
opt_port_value = calc_portfolio_value(opt_port_returns, initial_value=1.0)
opt_var = value_at_risk(opt_port_returns, alpha=0.01)
opt_cvar = conditional_value_at_risk(opt_port_returns, alpha=0.01)

# ---- 4. Console Outputs ----

print(f"GBM (single) last value: {s_gbm[-1]:.4f}")
print(f"Vasicek last value: {r_vas[-1]:.4f}")
print(f"CIR last value: {r_cir[-1]:.4f}")
print(f"Multi-asset GBM last values: {np.round(S_paths[:, -1], 4)}")
print(f"\nManual Portfolio value (final): {port_value[-1]:.4f}")
print(f"Manual Portfolio 1% VaR: {var:.4f} | 1% CVaR: {cvar:.4f}")

print(f"\nOptimal (minimum variance) weights: {np.round(opt_weights, 4)}")
print(f"Optimal weights sum: {np.sum(opt_weights):.4f}")
print(f"Optimal Portfolio value (final): {opt_port_value[-1]:.4f}")
print(f"Optimal Portfolio 1% VaR: {opt_var:.4f}")
print(f"Optimal Portfolio 1% CVaR: {opt_cvar:.4f}")

# ---- 5. Plotting ----

# 1. Single asset SDE paths
plt.figure(figsize=(10, 4))
plt.plot(t, s_gbm, label='GBM (single)')
plt.plot(t, r_vas, label='Vasicek')
plt.plot(t, r_cir, label='CIR')
plt.title('Single Asset SDE Simulated Paths')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Multi-asset GBM paths
plt.figure(figsize=(10, 4))
for i in range(S_paths.shape[0]):
    plt.plot(t, S_paths[i], label=f'GBM Asset {i+1}')
plt.title('Multi-Asset GBM Simulated Paths')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Portfolio value path comparison
plt.figure(figsize=(10, 4))
plt.plot(port_value, color='black', linestyle='--', label='Manual Portfolio')
plt.plot(opt_port_value, color='green', label='Optimal (Min Variance) Portfolio')
plt.title('Portfolio Value: Manual vs. Optimal (Min Variance)')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Manual portfolio return distribution
plt.figure(figsize=(8, 4))
plt.hist(port_returns, bins=30, alpha=0.7)
plt.axvline(var, color="red", linestyle="--", label="1% VaR")
plt.axvline(cvar, color="orange", linestyle="--", label="1% CVaR")
plt.title("Manual Portfolio Return Distribution")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Optimal portfolio return distribution
plt.figure(figsize=(8, 4))
plt.hist(opt_port_returns, bins=30, alpha=0.7)
plt.axvline(opt_var, color="red", linestyle="--", label="1% VaR")
plt.axvline(opt_cvar, color="orange", linestyle="--", label="1% CVaR")
plt.title("Optimal Portfolio Return Distribution")
plt.xlabel("Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
