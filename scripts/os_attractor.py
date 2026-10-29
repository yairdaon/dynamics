"""Standalone Olsen-Schaffer reconstructed attractor: (I_t,I_{t-L},I_{t-2L}),
L=3mo. Output: visualizations/os_attractor.pdf"""
from pathlib import Path

import matplotlib.pyplot as plt
from attractor import simulate
from reconstruction import lagged_coords, LAG

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

df = simulate(years_total=500.0, years_burnin=350.0, dt=1.0 / 365.0)
x, y, z = lagged_coords(df["I"].to_numpy(), LAG)

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, lw=0.2, color="black")
ax.set_xlabel("$I_{t-2L}$")
ax.set_ylabel("$I_{t-L}$")
ax.set_zlabel("$I_t$")
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(elev=15, azim=30)
fig.tight_layout()
fig.savefig(VIZ_DIR / "os_attractor.pdf")
