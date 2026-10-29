"""Correlation dimension (Grassberger-Procaccia) and Levina-Bickel MLE
dimension estimate, on the OS reconstructed attractor.

Both use log10(I) lagged coordinates, not raw I: dimension estimators
depend on local neighbor spacings at many scales, and I(t)'s ~9
orders of magnitude dynamic range (fine for *visualizing* the raw
attractor, see attractor.py) makes raw-scale pairwise distances wildly
nonuniform, swamping the small-scale structure the estimators need.
log I(t) gives a much cleaner scaling plateau in practice; the
dimension itself, as an r->0 limit, is coordinate-invariant under this
smooth a reparametrization -- only the finite-sample estimate's
quality changes.

Output: visualizations/correlation_dimension.pdf, printed D_C and D_MLE."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from attractor import simulate
from reconstruction import lagged_coords, LAG

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"
rng = np.random.default_rng(0)

df = simulate(years_total=650.0, years_burnin=350.0, dt=1.0 / 365.0)
x, y, z = lagged_coords(df["logI"].to_numpy(), LAG)
state = np.column_stack([x, y, z])
idx = rng.choice(len(state), size=3000, replace=False)
pts = state[idx]

# --- correlation dimension ---
d = pdist(pts)
rs = np.logspace(np.log10(d.min()), np.log10(d.max()), 50)
C = np.array([(d < r).mean() for r in rs])

mask = (C > 0.005) & (C < 0.05)  # clean scaling plateau, chosen by inspection
slope, intercept = np.polyfit(np.log(rs[mask]), np.log(C[mask]), 1)
print(f"correlation dimension D_C = {slope:.2f}")

fig, ax = plt.subplots(figsize=(8, 7))
ax.loglog(rs, C, "o", ms=3)
ax.loglog(rs[mask], np.exp(intercept) * rs[mask] ** slope, "k-", lw=1.5,
          label=f"slope = {slope:.2f}")
ax.set_xlabel("r")
ax.set_ylabel("C(r)")
ax.legend()
fig.tight_layout()
fig.savefig(VIZ_DIR / "correlation_dimension.pdf")

# --- Levina-Bickel MLE ---
k1, k2 = 10, 20
tree = cKDTree(pts)
dists, _ = tree.query(pts, k=k2 + 1)
dists = dists[:, 1:]  # drop self

m_k = np.zeros((len(pts), k2 - k1 + 1))
for i, k in enumerate(range(k1, k2 + 1)):
    Tk = dists[:, k - 1][:, None]
    Tj = dists[:, : k - 1]
    m_k[:, i] = 1.0 / np.log(Tk / Tj).mean(axis=1)

D_mle = m_k.mean()
print(f"Levina-Bickel MLE dimension (k={k1}..{k2}) = {D_mle:.2f}")
