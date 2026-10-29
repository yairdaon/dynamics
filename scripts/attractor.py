"""
Native-coordinate attractor of the seasonally forced SEIR model, using
the measles parameters Aron & Schwartz (1984) used, as reported by
Olsen & Schaffer (1990), "Chaos Versus Noisy Periodicity", Science 249.

Olsen & Schaffer reconstruct the attractor from a *single* observed
variable, I(t), via lagged coordinates (I(t), I(t+T), I(t+2T)) with
T = 3 months (their Fig. 2A). Here we instead plot the attractor in
*native* coordinates -- no lag, no reconstruction, just quantities the
model already gives us.

Two things matter to get a clean-looking attractor instead of a tangled
mess, and are worth stating explicitly since they are easy to get wrong:

1. Stiffness. 1/g ~ 3.65 days and 1/a ~ 10.2 days are both much shorter
   than the 1-year forcing period, so this is a numerically stiff
   system. A fixed-step RK4 with dt = 1 day is *not* fine enough
   relative to those timescales and quietly injects numerical noise
   that looks like extra roughness on the attractor. We use an
   adaptive stiff solver (LSODA) with tight tolerances instead.

2. Dynamic range. I(t) ranges over roughly nine orders of magnitude
   (deep near-fadeouts between epidemics up to ~0.5% prevalence at a
   peak). Plotted on a linear scale this huge range collapses nearly
   all of the attractor's structure into a sliver near I=0 while a
   handful of outbreak peaks dominate the visible range -- another
   source of apparent "noise" that is really just bad scaling. We plot
   log10(I).

A third, more interesting, point: beta(t) itself is a *bad* third
coordinate, even though it is perfectly deterministic. Because
beta(t) = beta0(1 + beta1 cos(2 pi t)) is 2-to-1 per year (it passes
through the same value once on the way up and once on the way down),
plotting (S, I, beta) literally folds each year's loop back onto
itself in the beta direction -- a self-crossing that has nothing to do
with chaos, just a non-injective choice of coordinate. The fix is to
use something that actually parametrizes the forcing injectively, such
as elapsed time t itself (what we use below, giving a climbing
spiral/helix of annual loops) or the (cos, sin) pair of the phase.

Outputs (written to ../visualizations/):
  state_space.pdf   -- static 3D render, (S, log10 I, t), ~25 years
  state_space.html  -- interactive plotly version, full simulated window
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

# --- Aron & Schwartz measles parameters (per year), via Olsen & Schaffer 1990 ---
m = 0.02      # per-capita birth/death rate
a = 35.84     # 1/latency period
g = 100.0     # 1/infectious period
beta0 = 1800.0
beta1 = 0.28


def beta(t):
    return beta0 * (1.0 + beta1 * np.cos(2.0 * np.pi * t))


def rhs(t, y):
    S, E, I = y
    b = beta(t)
    dS = m * (1.0 - S) - b * S * I
    dE = b * S * I - (m + a) * E
    dI = a * E - (m + g) * I
    return [dS, dE, dI]


def simulate(years_total=400.0, years_burnin=350.0, dt=1.0 / 365.0):
    y0 = [0.06, 0.0, 1e-4]  # rough near-DFE start; transient is discarded
    sol = solve_ivp(
        rhs, [0.0, years_total], y0,
        method="LSODA", rtol=1e-10, atol=1e-13, dense_output=True,
    )
    t_eval = np.arange(years_burnin, years_total, dt)
    S, E, I = sol.sol(t_eval)
    df = pd.DataFrame({"t": t_eval, "S": S, "E": E, "I": I})
    df["beta"] = beta(df["t"].to_numpy())
    df["logI"] = np.log10(df["I"])
    df["logE"] = np.log10(df["E"])
    return df.reset_index(drop=True)


def plot(df, window_years=25.0):
    t0 = df["t"].iloc[0]
    window = df[df["t"] < t0 + window_years]

    fig = go.Figure(
        data=go.Scatter3d(
            x=df["S"], y=df["logI"], z=df["t"],
            mode="lines",
            line=dict(color=df["logI"], colorscale="Viridis", width=2),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="S(t)",
            yaxis_title="log10 I(t)",
            zaxis_title="t (years)",
        ),
        title="SEIR attractor in native coordinates (S, log10 I, t), measles parameters",
    )
    fig.write_html(VIZ_DIR / "state_space.html")

    mfig = plt.figure(figsize=(8, 7))
    ax = mfig.add_subplot(111, projection="3d")
    ax.plot(window["S"], window["logI"], window["t"] - t0, lw=0.5, color="C0")
    ax.set_xlabel("S(t)")
    ax.set_ylabel(r"$\log_{10} I(t)$")
    ax.set_zlabel("t (years)")
    ax.view_init(elev=18, azim=-100)
    mfig.tight_layout()
    mfig.savefig(VIZ_DIR / "state_space.pdf")


if __name__ == "__main__":
    df = simulate()
    plot(df)
    df.to_csv(OUT_DIR / "scripts" / "seir_trajectory.csv", index=False)
