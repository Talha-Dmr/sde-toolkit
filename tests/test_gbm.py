import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sde_models import simulate_gbm_euler

import matplotlib.pyplot as plt

t, S = simulate_gbm_euler(S0=100, mu=0.05, sigma=0.2, T=1.0, N=252, random_seed=42)

plt.plot(t, S)
plt.title("GBM Path (Euler-Maruyama)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
