"""
Three random linear projections of the instantaneous state
(S, E, I, R, beta) onto single numbers, used as 3D coordinates.

For k = 1,2,3, draw a random weight vector
    v^k = (v^k_S, v^k_E, v^k_I, v^k_R, v^k_beta),  sum(v^k) = 1,
and plot the trajectory of
    coord_k(t) = v^k . (S(t), E(t), I(t), R(t), beta(t)).

This generalizes all_combinations.py: picking 3 of the 5 raw
coordinates is the special case where each v^k is a standard basis
vector (one entry 1, the rest 0, which trivially sums to 1). Generic
random weights are a much larger, "fuller-measure" sample of possible
linear projections -- the kind of thing genericity arguments are
actually about -- and the prediction from the reasoning in
all_combinations.py is that this doesn't help either, for the same two
reasons:

1. R = 1-S-E-I is exactly redundant, so any linear combination
   involving S,E,I,R is really just some affine function of (S,E,I) --
   three real numbers' worth of information, however the weights are
   split across four symbols.
2. Any nonzero weight on beta drags in beta's two-to-one-per-year fold.
   A linear combination cannot repair a non-injective ingredient:  if
   two different times t1 != t2 give the same beta but (generically)
   different S,E,I, a *specific* linear combination could in principle
   separate them -- but a *random* one has no reason to, and we are
   effectively back to the same fold just rotated/rescaled.

Practical note: S,E,I,R live on [0,1] while beta lives on
[beta0(1-beta1), beta0(1+beta1)] approx [1300,2300] -- a completely
different scale. A weight vector with, say, v_beta=0.01 would still
let beta dominate the sum by three orders of magnitude. To make "sum
of weights = 1" a meaningful statement about how much each variable
*actually* contributes, we standardize each variable (zero mean, unit
variance) before applying the weights.

Requires attractor.py to be importable (same directory).

Output: visualizations/random_projections.pdf
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from attractor import simulate, beta

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"

NAMES = ["S", "E", "I", "R", "beta"]


def random_weights(rng):
    v = rng.standard_normal(5)
    v += (1.0 - v.sum()) / 5.0  # shift onto the sum=1 hyperplane
    return v


def main():
    df = simulate(years_total=650.0, years_burnin=350.0, dt=1.0 / 365.0)
    S, E, I, t = df["S"].to_numpy(), df["E"].to_numpy(), df["I"].to_numpy(), df["t"].to_numpy()
    R = 1.0 - S - E - I
    b = beta(t)

    cols = np.column_stack([S, E, I, R, b])
    standardized = (cols - cols.mean(axis=0)) / cols.std(axis=0)

    rng = np.random.default_rng(0)
    vs = [random_weights(rng) for _ in range(3)]
    coords = [standardized @ v for v in vs]

    for k, v in enumerate(vs, start=1):
        weights = ", ".join(f"{n}={w:.2f}" for n, w in zip(NAMES, v))
        print(f"v^{k}: {weights}  (sum={v.sum():.2f})")

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[0], coords[1], coords[2], lw=0.2, color="black")
    ax.set_xlabel("$v^1 \\cdot$ state")
    ax.set_ylabel("$v^2 \\cdot$ state")
    ax.set_zlabel("$v^3 \\cdot$ state")
    ax.set_title("Three random sum-to-one projections of (S,E,I,R,beta)")
    ax.view_init(elev=18, azim=-40)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "random_projections.pdf")


if __name__ == "__main__":
    main()
