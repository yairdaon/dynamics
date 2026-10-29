"""Phase portraits of 2D linear systems with complex eigenvalues a+-ib:
spiral sink, center, spiral source. Output: visualizations/phase_portraits_complex.pdf"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20
VIZ_DIR = Path(__file__).resolve().parent.parent / "visualizations"

cases = [
    ("spiral sink", -0.3, 1.0),
    ("center", 0.0, 1.0),
    ("spiral source", 0.3, 1.0),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
grid = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(grid, grid)

for ax, (title, a, b) in zip(axes, cases):
    U = a * X - b * Y
    V = b * X + a * Y
    ax.streamplot(X, Y, U, V, color="black", density=0.9, linewidth=0.8, arrowsize=1.2)
    ax.plot(0, 0, "ko", ms=6)
    ax.set_title(f"{title}\n$a={a:g},\\ b={b:g}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")

fig.tight_layout()
fig.savefig(VIZ_DIR / "phase_portraits_complex.pdf")
