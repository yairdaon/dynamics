"""Periodic driven SIR: I(t) time series over a few decades, colored by
outbreak size (red = large, blue = small). Integrated in log-space (see
Log-transformations section) -- I's deep fadeouts otherwise send a
linear-space solver's step size to zero.
Output: visualizations/driven_sir.pdf"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

mu, gamma = 0.02, 365.0 / 13.0
R0 = 17.0
beta0, beta1 = R0 * (gamma + mu), 0.185


def beta(t):
    return beta0 * (1.0 + beta1 * np.cos(2.0 * np.pi * t))


def rhs(t, y):
    logS, logI = y
    S = np.exp(logS)
    b = beta(t)
    return [mu * (1 - S) / S - b * np.exp(logI), b * S - (gamma + mu)]


sol = solve_ivp(rhs, [0, 300], [np.log(0.1), np.log(1e-3)], method="LSODA",
                 rtol=1e-9, atol=1e-11, dense_output=True)
# Compute over a buffer 2 years wider than the display window so the
# trailing (and leading) peak is fully captured before classification --
# otherwise the last partial peak gets judged by an incomplete rise/fall
# and miscolored.
t0, t1 = 250, 262
t = np.linspace(t0 - 1, t1 + 1, 18000)
logS, logI = sol.sol(t)
I = np.exp(logI)

peaks, _ = find_peaks(I)
troughs = [0]
for a, b_ in zip(peaks[:-1], peaks[1:]):
    troughs.append(a + np.argmin(I[a:b_]))
troughs.append(len(I) - 1)

is_large = I[peaks] > np.median(I[peaks])

fig, ax = plt.subplots(figsize=(12, 5.5))
for k in range(len(troughs) - 1):
    lo, hi = troughs[k], troughs[k + 1] + 1
    seg = slice(lo, hi)
    mask = (t[seg] >= t0) & (t[seg] <= t1)  # clip to the display window
    if not mask.any():
        continue
    color = "tab:red" if is_large[k] else "tab:blue"
    ax.semilogy(t[seg][mask] - t0, I[seg][mask], color=color)
ax.set_xlabel("t (years)")
ax.set_ylabel("I(t)")
fig.tight_layout()
fig.savefig(VIZ_DIR / "driven_sir.pdf")
