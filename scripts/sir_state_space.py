"""(S, I, phi) state space of the driven SIR model (same parameters as
driven_sir.py's biennial example), to illustrate the idea of a state
space before Takens/reconstruction. Output: visualizations/sir_state_space.pdf"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

mu, gamma = 0.02, 365.0 / 13.0
R0 = 17.0
beta0, beta1 = R0 * (gamma + mu), 0.185


def rhs(t, y):
    logS, logI = y
    S = np.exp(logS)
    b = beta0 * (1.0 + beta1 * np.cos(2.0 * np.pi * t))
    return [mu * (1 - S) / S - b * np.exp(logI), b * S - (gamma + mu)]


sol = solve_ivp(rhs, [0, 600], [np.log(0.1), np.log(1e-3)], method="LSODA",
                 rtol=1e-9, atol=1e-11, dense_output=True)
# phi wraps at half-integer t (phi(t)=0 at integer t); start/end there so
# every run is a complete cycle -- no truncated run at either end.
t = np.linspace(300.5, 599.5, 100000)  # long burn-in, then settled trajectory only
S, I = np.exp(sol.sol(t))
logI = np.log10(I)
phi = ((2 * np.pi * t + np.pi) % (2 * np.pi)) - np.pi

breaks = np.where(np.diff(phi) < 0)[0] + 1
runs = np.split(np.arange(len(t)), breaks)
run_max = np.array([I[r].max() for r in runs])
is_large = run_max > np.median(run_max)

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")
for r, large in zip(runs, is_large):
    color = "tab:red" if large else "tab:blue"
    ax.plot(S[r], logI[r], phi[r], lw=1.0, color=color)

# mark the phi=-pi/+pi seam: the two (S, log I) points where the
# trajectory wraps (continuous in state, discontinuous on this cut axis)
for r in runs[0], runs[1]:
    ax.plot([S[r][0]] * 2, [logI[r][0]] * 2, [-np.pi, np.pi],
            "k--", lw=1.0)

ax.set_xlabel("S")
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.set_ylabel("I")
ax.set_yticks([-4, -3])
ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$"])
ax.set_zlabel("$\\phi$")
ax.view_init(elev=18, azim=-50)
fig.tight_layout()
fig.savefig(VIZ_DIR / "sir_state_space.pdf")
