"""(phi, S, I) with phi = (2*pi*t) mod 2*pi -- phase wrapped, not the
2-to-1 cos(phi)=beta. Breaks at the phi seam each year to avoid
spurious wrap-around chords. Output: visualizations/phi_S_I.pdf"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from attractor import simulate

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

df = simulate(years_total=390.0, years_burnin=350.0, dt=1.0 / 365.0)
S, I, t = df["S"].to_numpy(), df["I"].to_numpy(), df["t"].to_numpy()
phi = (2 * np.pi * t) % (2 * np.pi)

breaks = np.where(np.diff(phi) < 0)[0] + 1
runs = np.split(np.arange(len(t)), breaks)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
for r in runs:
    ax.plot(phi[r], S[r], I[r], lw=0.2, color="black")
ax.set_xlabel("phi")
ax.set_ylabel("S")
ax.set_zlabel("I")
ax.view_init(elev=18, azim=-40)
fig.tight_layout()
fig.savefig(VIZ_DIR / "phi_S_I.pdf")
