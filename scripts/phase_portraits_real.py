"""Phase portraits of 2D diagonal linear systems with real eigenvalues:
stable node, saddle, unstable node. Output: visualizations/phase_portraits_real.pdf"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20
VIZ_DIR = Path(__file__).resolve().parent.parent / "visualizations"

cases = [
    ("stable node", -1.0, -2.0),
    ("saddle", -1.0, 1.0),
    ("unstable node", 1.0, 2.0),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
grid = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(grid, grid)

for ax, (title, l1, l2) in zip(axes, cases):
    U, V = l1 * X, l2 * Y
    ax.streamplot(X, Y, U, V, color="black", density=0.9, linewidth=0.8, arrowsize=1.2)
    ax.plot(0, 0, "ko", ms=6)
    ax.set_title(f"{title}\n$\\lambda_1={l1:g},\\ \\lambda_2={l2:g}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(VIZ_DIR / "phase_portraits_real.pdf")
