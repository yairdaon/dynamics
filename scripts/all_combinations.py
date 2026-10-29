"""
All C(5,3) = 10 ways to pick three of the model's instantaneous
quantities {S, E, I, R, beta} as native (unlagged) 3D coordinates,
plotted side by side.

This is the systematic version of the (S,E,I), (S,I,beta), etc.
experiments done by hand elsewhere in this project. None of the 10
escapes the two structural problems identified there:

1. R = 1 - S - E - I is exactly redundant. Any triple that pairs R
   with two of {S,E,I} is therefore an exact affine (linear, invertible)
   transform of the corresponding all-compartment triple -- not
   independent evidence, just the same degenerate shape viewed in a
   sheared coordinate frame. That accounts for the family
   (S,E,I), (S,E,R), (S,I,R), (E,I,R): four panels, one underlying
   shape.
2. beta(t) = beta0(1 + beta1 cos(2 pi t)) is two-to-one every year (the
   forcing's phase is topologically a circle, and a circle has no
   injective map into a single real axis). Any triple that includes
   beta therefore folds, regardless of which two compartments it is
   paired with -- the family (S,E,beta), (S,I,beta), (E,I,beta),
   (S,R,beta), (E,R,beta), (I,R,beta): six panels, same pathology.
   (S,R,beta) is visibly the most collapsed of these, because
   R = 1-S-E-I approx 1-S whenever E,I are small (most of the time),
   so S and R are themselves nearly affinely dependent.

None of these 10 instantaneous snapshots is a valid reconstruction of
the attractor; see reconstruction.py for delay-coordinate embeddings
(of a single observable, lagged against itself), which do not have
either problem and do reproduce a clean manifold.

Requires attractor.py to be importable (same directory).

Output: visualizations/all_combinations.pdf
"""
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
from attractor import simulate, beta

plt.rcParams["font.size"] = 20

OUT_DIR = Path(__file__).resolve().parent.parent
VIZ_DIR = OUT_DIR / "visualizations"


def main():
    df = simulate(years_total=650.0, years_burnin=350.0, dt=1.0 / 365.0)
    S, E, I, t = df["S"].to_numpy(), df["E"].to_numpy(), df["I"].to_numpy(), df["t"].to_numpy()
    R = 1.0 - S - E - I
    b = beta(t)
    data = {"S": S, "E": E, "I": I, "R": R, "beta": b}

    combos = list(itertools.combinations(["S", "E", "I", "R", "beta"], 3))
    assert len(combos) == 10

    fig = plt.figure(figsize=(36, 16))
    for i, combo in enumerate(combos):
        ax = fig.add_subplot(2, 5, i + 1, projection="3d")
        x, y, z = data[combo[0]], data[combo[1]], data[combo[2]]
        ax.plot(x, y, z, lw=0.15, color="black")
        ax.set_xlabel(combo[0], fontsize=16)
        ax.set_ylabel(combo[1], fontsize=16)
        ax.set_zlabel(combo[2], fontsize=16)
        ax.set_title(f"({combo[0]},{combo[1]},{combo[2]})", fontsize=18)
        ax.view_init(elev=18, azim=-40)
        ax.tick_params(labelsize=12)

    fig.suptitle("All 10 instantaneous 3-coordinate choices from {S, E, I, R, beta} -- none is a valid reconstruction")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "all_combinations.pdf")


if __name__ == "__main__":
    main()
