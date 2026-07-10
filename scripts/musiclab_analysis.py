#!/usr/bin/env python3

"""
CAUSAL test of E2 (decoupling from correctness) on the real Music Lab data
(Salganik, Dodds & Watts 2006, Science 311:854).

Why this is the causal test. The same 48 songs run through many independent
"worlds", so between-world differences CANNOT come from song quality (it is held
fixed) -- only from amplified early social-signal draws. An INDEPENDENT world
(participants saw no download counts) measures each song's intrinsic quality Q,
uncontaminated by the amplifier. Social-influence worlds carry the gain.

Design in the released data (opr.princeton.edu/archive/cm/), file
`downloads_v{E}_lexorder.txt`: row = song, columns = song_id, then the social
worlds of experiment E, then the INDEPENDENT world (last column). Experiments
differ in social-signal strength (the gain):
    Experiment 1  -> WEAK gain   (songs in a grid, weak popularity signal)
    Experiment 2  -> STRONG gain (single column sorted by popularity)
Each experiment has its own independent world, so Q is defined per experiment.

PREDICTION P1 (E2): the rank correlation rho_w = Spearman(world downloads, Q) is
LOWER under strong gain (exp 2) than weak gain (exp 1).  H1: rho_weak > rho_strong
(one-sided). If instead rho does not fall, the decoupling core of E2 fails.

Run with ONE command (auto-detects the bundled data):
    python3 scripts/musiclab_analysis.py
Falls back to a synthetic self-test only if the data directory is absent.
"""
import argparse, os, csv
from math import erf, sqrt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "musiclab", "musiclab_data")

# experiment -> a-priori gain label (Salganik et al.: exp2 signal more prominent)
GAIN = {1: "weak", 2: "strong"}          # the clean two-level P1 contrast
EXTRA = {3: "exp3", 4: "exp4(inverted)"} # reported for completeness, not in the test

# ---- statistics -----------------------------------------------------------

def market_share(d):
    d = np.asarray(d, float); s = d.sum()
    return d / s if s > 0 else d

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if x.sum() == 0: return 0.0
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)

def p1_two_level(rho_weak, rho_strong):
    """One-sided Welch t-test for H1: rho_weak > rho_strong."""
    a, b = np.asarray(rho_weak), np.asarray(rho_strong)
    na, nb = len(a), len(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/na + b.var(ddof=1)/nb)
    num = (a.var(ddof=1)/na + b.var(ddof=1)/nb) ** 2
    den = (a.var(ddof=1)/na)**2/(na-1) + (b.var(ddof=1)/nb)**2/(nb-1)
    return t, num/den

# ---- load one experiment's downloads file --------------------------------

def load_experiment(E, data_dir):
    """Return (Q, worlds) for experiment E: Q = independent-world download
    shares (true quality), worlds = list of social-world download arrays."""
    path = os.path.join(data_dir, f"downloads_v{E}_lexorder.txt")
    rows = []
    with open(path) as fh:
        for r in csv.reader(fh):
            rows.append([float(x) for x in r])
    M = np.array(rows)                      # (48 songs, 1 + nworlds + 1)
    counts = M[:, 1:]                       # drop song_id
    social = counts[:, :-1]                 # all but last = social-influence worlds
    indep = counts[:, -1]                   # last column = independent world
    Q = market_share(indep)
    worlds = [market_share(social[:, w]) for w in range(social.shape[1])]

    return Q, worlds

# ---- analysis -------------------------------------------------------------

def per_experiment_report(E, label, data_dir):
    Q, worlds = load_experiment(E, data_dir)
    rho = [spearman(w, Q) for w in worlds]
    gn = [gini(w) for w in worlds]
    print(f"  exp {E} [{label:14s}] worlds={len(worlds)}  "
          f"rho(success,quality)={np.mean(rho):.3f}+/-{np.std(rho):.3f}  "
          f"Gini={np.mean(gn):.3f}")

    return np.array(rho)

def run_real(data_dir):
    print("MUSIC LAB (real data) -- causal test of E2 decoupling\n")
    print("Per-experiment rho = Spearman(world download share, independent-world quality):")
    rhos = {}
    for E, label in {**GAIN, **EXTRA}.items():
        try:
            rhos[E] = per_experiment_report(E, label, data_dir)
        except FileNotFoundError:
            print(f"  exp {E}: file missing, skipping")

    if 1 in rhos and 2 in rhos:
        weak, strong = rhos[1], rhos[2]
        t, df = p1_two_level(weak, strong)
        p = 0.5 * (1 - erf(t / sqrt(2)))               # one-sided normal approx
        print(f"\nP1 test  H1: rho_weak(exp1) > rho_strong(exp2)")
        print(f"  rho_weak  = {weak.mean():.3f}  (n={len(weak)})")
        print(f"  rho_strong= {strong.mean():.3f}  (n={len(strong)})")
        print(f"  drop = {weak.mean()-strong.mean():+.3f}   "
              f"Welch t={t:.2f}, df={df:.1f}, one-sided p={p:.4f}")
        verdict = ("DECOUPLING CONFIRMED: correlation with quality falls under "
                   "stronger gain" if (weak.mean() > strong.mean() and p < 0.05)
                   else "sign consistent but not significant at this n"
                   if weak.mean() > strong.mean()
                   else "H0 not rejected: no drop")
        print(f"  -> {verdict}")
        print("\nNote: with only ~8 worlds per condition this two-level test has "
              "limited power;\nthe sign of the drop is the primary read. A "
              "multi-level gain sweep\n(>=5 levels x 20 worlds) is the "
              "pre-registered high-power version.")

# ---- synthetic fallback ---------------------------------------------------

def run_synthetic():
    print("Data directory not found -- SYNTHETIC self-test (mimics the design).\n")
    rng = np.random.default_rng(0); Ns = 48
    q = rng.uniform(0.3, 1.0, Ns)
    def world(gain, seed):
        r = np.random.default_rng(seed); d = np.ones(Ns)
        for _ in range(700):
            share = d / d.sum(); appeal = q * (1 + gain * share * Ns)
            d[r.choice(Ns, p=appeal/appeal.sum())] += 1
        return market_share(d)

    Q = market_share(rng.poisson(200 * q / q.mean()))

    for name, g in [("weak", 1.0), ("strong", 4.0)]:
        rho = [spearman(world(g, 10 + k), Q) for k in range(8)]
        print(f"  {name:6s} rho={np.mean(rho):.3f}+/-{np.std(rho):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR,
                    help="path to musiclab_data directory (default: bundled)")
    args = ap.parse_args()
    if os.path.isdir(args.data) and os.path.exists(
            os.path.join(args.data, "downloads_v1_lexorder.txt")):
        run_real(args.data)
    else:
        run_synthetic()
