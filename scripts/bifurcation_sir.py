from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20

VIZ_DIR = Path(__file__).resolve().parent.parent / "visualizations"

mu = 0.02
gamma = 365.0 / 13.0
R0 = 17.0
beta0 = R0 * (gamma + mu)

def peaks_for(beta1, t_total=400, t_burn=300):
    def rhs(t, y):
        logS, logI = y
        S = np.exp(logS)
        b = beta0 * (1 + beta1 * np.cos(2 * np.pi * t))
        return [mu * (1 - S) / S - b * np.exp(logI), b * S - (gamma + mu)]
    sol = solve_ivp(rhs, [0, t_total], [np.log(0.1), np.log(1e-3)], method="LSODA",
                     rtol=1e-9, atol=1e-11, dense_output=True)
    t = np.linspace(t_burn, t_total, 60000)
    I = np.exp(sol.sol(t)[1])
    pk, _ = find_peaks(I)
    return I[pk]

b1s = np.linspace(0.05, 0.95, 90)
xs, ys = [], []
for b1 in b1s:
    h = peaks_for(b1)
    xs.extend([b1] * len(h))
    ys.extend(h)
    print(b1, len(h))

fig, ax = plt.subplots(figsize=(10, 7))
ax.semilogy(xs, ys, ",", color="black", alpha=0.5, ms=1)
ax.set_xlabel("beta1")
ax.set_ylabel("I peak heights")
fig.tight_layout()
fig.savefig(VIZ_DIR / "bifurcation_sir.pdf")
