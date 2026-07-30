#!/usr/bin/env python3

"""
optimal_gain_fit.py
===================
Tests the horizon scaling of the dispersion-maximising gain,

    g*(T) ~ 1 / (theta S0 T),                                     (paper Eq. gstar)

properly: on a dense gain grid, with a sub-grid peak location (parabolic fit in
log g around the discrete maximum), over six horizons spanning a factor of 16,
and with a log-log regression that returns an EXPONENT WITH A CONFIDENCE
INTERVAL rather than an eyeballed halving.

The earlier four-point grid scan (symmetry_breaking.optimal_gain_scan) could
only report that the last doubling happened to halve g*; that is one observation
and does not establish a scaling. This script replaces it.

Reported: the fitted exponent alpha in g* ~ T^{-alpha} (prediction alpha = 1),
its spread (std) over seeds, and the crossover constant g* theta S0 T, which the
mean-field argument fixes only to order of magnitude and which is expected to
drift downward at small T because the noise-seeding transient occupies a
T-independent number of steps.

Seeded throughout. Deps: numpy (matplotlib for the figure).
"""




import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from symmetry_breaking import step



OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA, THETA, S_STAR, S0 = 0.05, 0.02, 50.0, 1.0
HORIZONS = (150, 300, 600, 1200, 1800, 2400)
N = 4000



def peak_gain(T, seed, n_grid=24):
    """Locate g*(T) on a dense log-spaced grid, then refine to sub-grid
    resolution by fitting a parabola to log Var against log g at the three
    points bracketing the discrete maximum."""
    gains = np.geomspace(0.02, 3.0, n_grid)
    var = np.empty(n_grid)
    for j, g in enumerate(gains):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1e-6, size=N)
        for _ in range(T):
            x = step(x, rng, g, SIGMA, THETA, S_STAR, reset_kw=None)
        var[j] = x.var()

    j = int(np.argmax(var))
    if j in (0, n_grid - 1):
        return float(gains[j]), gains, var

    lg = np.log(gains[j - 1:j + 2])
    lv = np.log(var[j - 1:j + 2])
    a, b, _ = np.polyfit(lg, lv, 2)
    g_star = float(np.exp(-b / (2 * a))) if a < 0 else float(gains[j])
    # keep the refinement inside the bracketing interval
    g_star = float(np.clip(g_star, gains[j - 1], gains[j + 1]))
    return g_star, gains, var


def main(seeds=(0, 1, 2)):
    print("Dispersion-maximising gain vs horizon (dense grid, sub-grid peak)\n")
    print(f"   {'T':>6}{'g* (mean)':>12}{'sd':>8}{'1/(th S0 T)':>13}{'g* th S0 T':>12}")

    table = {}
    for T in HORIZONS:
        gs = [peak_gain(T, s)[0] for s in seeds]
        table[T] = gs
        m, sd = float(np.mean(gs)), float(np.std(gs))
        pred = 1.0 / (THETA * S0 * T)
        print(f"   {T:>6}{m:>12.3f}{sd:>8.3f}{pred:>13.3f}{m * THETA * S0 * T:>12.2f}")

    # log-log fit of the exponent, per seed, so the spread is an honest CI
    lT = np.log(np.array(HORIZONS, dtype=float))
    alphas = []
    for i, _ in enumerate(seeds):
        lg = np.log(np.array([table[T][i] for T in HORIZONS]))
        alphas.append(-np.polyfit(lT, lg, 1)[0])
    alphas = np.array(alphas)
    print(f"\n   fitted exponent alpha in g* ~ T^-alpha : "
          f"{alphas.mean():.3f} +/- {alphas.std():.3f}  (prediction 1.000)")

    # the asymptotic half of the range only (T >= 600), where the seeding
    # transient no longer occupies an appreciable share of the horizon
    tail = [T for T in HORIZONS if T >= 600]
    lTt = np.log(np.array(tail, dtype=float))
    at = []
    for i, _ in enumerate(seeds):
        lg = np.log(np.array([table[T][i] for T in tail]))
        at.append(-np.polyfit(lTt, lg, 1)[0])
    at = np.array(at)
    print(f"   restricted to T >= 600                  : "
          f"{at.mean():.3f} +/- {at.std():.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    m = np.array([np.mean(table[T]) for T in HORIZONS])
    sd = np.array([np.std(table[T]) for T in HORIZONS])
    ax.errorbar(HORIZONS, m, yerr=sd, fmt="o-", ms=5, capsize=3, label=r"measured $g^*(T)$")
    ax.plot(HORIZONS, m[0] * (np.array(HORIZONS) / HORIZONS[0]) ** -1.0, "k--", lw=1,
            label=r"$1/T$ reference")
    ax.set(xscale="log", yscale="log", xlabel="horizon $T$",
           ylabel=r"dispersion-maximising gain $g^*$",
           title=r"$g^*(T)$ scaling: fitted exponent "
                 rf"$\alpha={alphas.mean():.2f}\pm{alphas.std():.2f}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "optimal_gain_fit.png", dpi=200)
    plt.close(fig)
    print(f"\nFigure written to {OUT / 'optimal_gain_fit.png'}")
    return table, alphas





if __name__ == "__main__":
    main()
