"""
Side-by-side comparison of two 3D embeddings of the *same* attractor
(seasonally forced SEIR model, measles parameters, see attractor.py):

  Left:  native coordinates  (S(t), E(t), I(t))
         -- the model's own state variables, no reconstruction.
  Right: Olsen & Schaffer's (1990, Science 249) reconstructed
         coordinates (I(t), I(t+T), I(t+2T)), T = 3 months
         -- a single observed variable, lagged against itself, exactly
         their Fig. 2A.

Takens' theorem is the reason these can look alike: generically, a
delay-coordinate map built from one observable of a (sufficiently
smooth, finite-dimensional) attractor is a diffeomorphism onto the
true attractor, provided enough lags are used. So the reconstruction
on the right, built from I(t) alone, should reproduce the same
attractor the native coordinates on the left already show directly.

No log transform here, on purpose (contrast with an earlier version of
this script). I(t) really does span ~9 orders of magnitude between
deep fadeouts and outbreak peaks -- but on *linear* axes that just
means nearly the whole attractor is a sharply tapering cone that
collapses onto a single point (the DFE corner) while the handful of
genuinely large excursions stick out, with fine nested rings near the
tip where successively-deep fadeouts all pile up close to the origin.
That is not visual noise, it is the correct geometry -- the system
repeatedly makes a close pass by the saddle-like disease-free
equilibrium before the next outbreak -- and it is exactly the "witch's
hat" shape in Olsen & Schaffer's own Fig. 2A. (A fixed-step,
insufficiently fine integrator *does* turn this into a genuine tangle;
see attractor.py for that pitfall and why we integrate with an
adaptive stiff solver instead.)

We also use a longer simulated time series than before (300 years past
the initial transient, daily resolution) so the cone's nested structure
is filled in densely, the way repeated magnification in Olsen &
Schaffer's own description reveals finer and finer self-similar layers.

Requires attractor.py to be importable (same directory).

Outputs (in ../visualizations/):
  reconstruction.pdf    -- static side-by-side 3D render
  reconstruction.html   -- interactive plotly version (full series)
"""
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from attractor import simulate

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

DAY = 1.0 / 365.0
LAG = round(0.25 * 365)  # 3 months, in days


def lagged_coords(I, lag):
    x = I[: -2 * lag]
    y = I[lag:-lag]
    z = I[2 * lag:]
    return x, y, z


def main():
    df = simulate(years_total=650.0, years_burnin=350.0, dt=DAY)
    I = df["I"].to_numpy()
    x, y, z = lagged_coords(I, LAG)

    # --- interactive (subsampled 4x so the HTML stays a reasonable size;
    # the static PNG below uses the full daily resolution) ---
    sub = slice(None, None, 4)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Native: (S, E, I)",
                         "Reconstructed: (I(t), I(t+T), I(t+2T))"),
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["S"][sub], y=df["E"][sub], z=df["I"][sub],
            mode="lines", line=dict(color=df["I"][sub], colorscale="Viridis", width=2),
            name="native",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=x[sub], y=y[sub], z=z[sub],
            mode="lines", line=dict(color=y[sub], colorscale="Viridis", width=2),
            name="reconstructed",
        ),
        row=1, col=2,
    )
    fig.update_scenes(dict(xaxis_title="S", yaxis_title="E", zaxis_title="I"), row=1, col=1)
    fig.update_scenes(dict(xaxis_title="I(t)", yaxis_title="I(t+T)", zaxis_title="I(t+2T)"), row=1, col=2)
    fig.update_layout(title="Native vs. reconstructed coordinates, same attractor (T = 3 months)")
    fig.write_html(VIZ_DIR / "reconstruction.html")

    # --- static ---
    mfig = plt.figure(figsize=(19, 10))

    ax1 = mfig.add_subplot(121, projection="3d")
    ax1.plot(df["S"], df["E"], df["I"], lw=0.2, color="C0")
    ax1.set_xlabel("S")
    ax1.set_ylabel("E")
    ax1.set_zlabel("I")
    ax1.set_title("Native: (S, E, I)", pad=20)
    ax1.view_init(elev=20, azim=-40)

    ax2 = mfig.add_subplot(122, projection="3d")
    ax2.plot(x, y, z, lw=0.2, color="C1")
    ax2.set_xlabel("I(t)")
    ax2.set_ylabel("I(t+T)")
    ax2.set_zlabel("I(t+2T)")
    ax2.set_title("Reconstructed: lagged I(t), T=3mo", pad=20)
    ax2.view_init(elev=15, azim=30)

    mfig.suptitle("Same attractor, two embeddings (no log transform)")
    mfig.subplots_adjust(left=0.05, right=0.92, wspace=0.35, top=0.88, bottom=0.05)
    mfig.savefig(VIZ_DIR / "reconstruction.pdf")


if __name__ == "__main__":
    main()
