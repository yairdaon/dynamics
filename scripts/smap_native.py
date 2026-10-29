"""
S-map on the *native* (non-reconstructed) state space of the seasonally
forced SEIR model: columns S(t), log10 I(t), beta(t), no time-delay
embedding.

This is the classic Sugihara (1990) nonlinearity diagnostic: fit a
sequence of S-maps with increasing localization theta and track
out-of-sample prediction skill (rho). theta=0 is a single global linear
map; theta>0 progressively localizes the linear fit to nearby points in
state space. If skill increases with theta, the dynamics are state-
dependent (nonlinear) -- as expected for a chaotic attractor, and as a
warm-up for state-space *reconstruction* (lagged coordinates) in a
later lecture.

One subtlety worth keeping: at a short forecast horizon (Tp of a few
weeks), log I(t+Tp) is so strongly autocorrelated with log I(t) that
even theta=0 (a single global linear regression) already gets rho close
to 1 -- there is no room left for localization to help, and the test is
uninformative. (S, log I, beta) is close to the full, noise-free,
autonomous state of this system -- recall E is fast and essentially
slaved to S, I, beta -- so short-horizon prediction here is close to
trivial regardless of theta. The nonlinearity only becomes visible to
this test at a forecast horizon long enough that naive persistence
fails, which is why we predict half a year ahead rather than one week.

Requires scripts/attractor.py to have been run first (reads
scripts/seir_trajectory.csv).

Output: visualizations/smap_theta.pdf / visualizations/smap_theta.html (rho vs theta).
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyEDM import SMap

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

THETAS = [0, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 7, 8]


def load_weekly():
    # I(t) spans ~9 orders of magnitude (near-fadeouts between outbreaks);
    # work in log10(I), same reasoning as in simulation.tex's
    # "Log-transformations" section, else a handful of outbreak peaks
    # dominate every distance calculation.
    df = pd.read_csv(OUT_DIR / "scripts" / "seir_trajectory.csv")
    weekly = df.iloc[::7].reset_index(drop=True)
    weekly["Time"] = weekly.index
    return weekly[["Time", "S", "logI", "beta"]]


def theta_sweep(df):
    n = len(df)
    n_lib = int(0.7 * n)
    lib = f"1 {n_lib}"
    pred = f"{n_lib + 1} {n}"

    rhos = []
    for theta in THETAS:
        res = SMap(
            dataFrame=df,
            columns="S logI beta",
            target="logI",
            lib=lib,
            pred=pred,
            E=3,
            Tp=26,  # half a year ahead (weekly-resampled data)
            theta=theta,
            embedded=True,
            showPlot=False,
        )
        pred_df = res["predictions"].dropna()
        rho = pred_df["Observations"].corr(pred_df["Predictions"])
        rhos.append(rho)
        print(f"theta={theta:>4}  rho={rho:.4f}")
    return pd.DataFrame({"theta": THETAS, "rho": rhos})


def plot(sweep):
    fig = go.Figure(data=go.Scatter(x=sweep["theta"], y=sweep["rho"], mode="lines+markers"))
    fig.update_layout(
        xaxis_title="theta (localization)",
        yaxis_title="rho (out-of-sample prediction skill)",
        title="S-map nonlinearity test, native coordinates (S, log10 I, beta), Tp=26 weeks",
    )
    fig.write_html(VIZ_DIR / "smap_theta.html")

    mfig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(sweep["theta"], sweep["rho"], "o-")
    ax.set_xlabel(r"$\theta$ (localization)")
    ax.set_ylabel(r"$\rho$ (out-of-sample skill)")
    mfig.tight_layout()
    mfig.savefig(VIZ_DIR / "smap_theta.pdf")


if __name__ == "__main__":
    weekly = load_weekly()
    sweep = theta_sweep(weekly)
    plot(sweep)
