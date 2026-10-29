"""Nearest-neighbor forecast error vs. forecast horizon.

Find, for each query state, its single nearest neighbor in the
library (same reconstructed state space (I_t, I_{t-L}, I_{t-2L}) as
os_attractor.py) -- then, WITHOUT re-querying, just let both
trajectories (query's true future and neighbor's true future) run
forward and track how far apart they drift as the horizon h grows.
Two trajectories that start close together on a chaotic attractor
diverge --- on average, roughly exponentially at first, then
saturating once they're as far apart as two random points on the
attractor. RMSE(h) over many query points should therefore increase
(on average) monotonically with h -- this is the same sensitive
dependence on initial conditions that limits how far ahead any
forecast (nearest-neighbor or otherwise) can see.

Output: visualizations/nn_prediction.pdf"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from attractor import simulate
from reconstruction import lagged_coords, LAG

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

H = 180  # max forecast horizon, in days -- past ~200 days the shared
         # annual forcing partially resynchronizes the two trajectories
         # (both feel the exact same periodic beta_t), breaking monotonicity;
         # within 180 days divergence grows cleanly

df = simulate(years_total=650.0, years_burnin=350.0, dt=1.0 / 365.0)
I = df["I"].to_numpy()
logI = df["logI"].to_numpy()  # error measured in log10(I): I spans ~9 orders
                               # of magnitude, so raw-scale RMSE is dominated
                               # by rare huge-outbreak mismatches and isn't
                               # monotonic -- log-scale gives the typical,
                               # smoothly-growing divergence instead.

x, y, z = lagged_coords(I, LAG)  # state(k) = (I(k), I(k+L), I(k+2L)), "now" = k+2L
n = len(x)
states = np.column_stack([x, y, z])
now = np.arange(n) + 2 * LAG  # index into I of each state's "now" (I_t)

n_lib = int(0.7 * n)
lib_states, pred_states = states[:n_lib], states[n_lib:]
lib_now, pred_now = now[:n_lib], now[n_lib:]

tree = cKDTree(lib_states)
_, nn = tree.query(pred_states, k=1)
neighbor_now = lib_now[nn]

# drop query points too close to the end of the series to test the full horizon H
valid = (pred_now + H < len(I)) & (neighbor_now + H < len(I))
pred_now, neighbor_now = pred_now[valid], neighbor_now[valid]

horizons = np.arange(1, H + 1)
rmse = np.empty(H)
for i, h in enumerate(horizons):
    err = logI[pred_now + h] - logI[neighbor_now + h]
    rmse[i] = np.sqrt(np.mean(err ** 2))

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(horizons, rmse)
ax.set_xlabel("forecast horizon (days)")
ax.set_ylabel(r"RMSE ($\log_{10} I$)")
ax.set_title("Divergence of initially-nearby trajectories")
fig.tight_layout()
fig.savefig(VIZ_DIR / "nn_prediction.pdf")
