"""Why FTC works: telescoping increments of f equal a Riemann sum of f'.
Output: visualizations/ftc_riemann.pdf"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20
VIZ_DIR = Path(__file__).resolve().parent.parent / "visualizations"

f = lambda x: x**2 / 8
fp = lambda x: x / 4

a, b, n = 0.0, 6.0, 6
xs = np.linspace(a, b, 400)
nodes = np.linspace(a, b, n + 1)
dx = (b - a) / n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# left: f(x) with staircase of increments f(x_{i+1}) - f(x_i)
ax1.plot(xs, f(xs), color="black", lw=2)
for i in range(n):
    x0, x1 = nodes[i], nodes[i + 1]
    y0, y1 = f(x0), f(x1)
    ax1.plot([x0, x1], [y0, y0], color="tab:blue", lw=1.5)
    ax1.plot([x1, x1], [y0, y1], color="tab:blue", lw=1.5)
ax1.scatter(nodes, f(nodes), color="black", zorder=3, s=25)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$f(x)$")
ax1.set_title("steps sum to $f(b)-f(a)$")

# right: f'(x) with Riemann rectangles of the same widths
ax2.plot(xs, fp(xs), color="black", lw=2)
ax2.bar(nodes[:-1], fp(nodes[:-1]), width=dx, align="edge",
        color="tab:blue", alpha=0.4, edgecolor="tab:blue")
ax2.set_xlabel("$x$")
ax2.set_ylabel("$f'(x)$")
ax2.set_title("same area under $f'$")

fig.tight_layout()
fig.savefig(VIZ_DIR / "ftc_riemann.pdf")
