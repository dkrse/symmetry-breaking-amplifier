#!/usr/bin/env python3

"""
Path (b): synthetic validation of the early-lead-persistence discriminator.

Question the paper's revised Design 2 rests on:
  Does rho(tau) = corr(rank x(tauT), rank x(T)) actually SEPARATE the
  symmetry-breaking amplifier from plain preferential attachment (PA)?
  The claim in the trimmed paper is that on a monotone accumulating stock it does
  NOT on its own (both bow above sqrt(tau)), and the real discriminator is the
  free-token vs. toppling contrast. Here we check that empirically.

Four processes, common cohort of N entities, T steps, near-homogeneous start:
  DIFF  g=0 random walk in log-space           -> analytic null sqrt(tau)
  AMP   multiplicative amplifier, free token    -> paper's Eq (3), no ceiling
  PA    Barabasi preferential attachment        -> plain rich-get-richer
  TOP   amplifier + reverse-dominance reset      -> toppling-limited status

We report rho(tau) for each and tau_90 (smallest tau with rho>=0.9).
"""

import numpy as np
from scipy.stats import spearmanr

RNG = np.random.default_rng(20260709)
N = 3000          # entities in the cohort
T = 300           # horizon (steps)
TAUS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.375, 0.5, 0.75, 1.0])



def rho_tau(traj):
    """traj: (T+1, N) positions. Return rho(tau) vs final, per TAUS (rank corr)."""
    final = traj[-1]
    out = []
    for tau in TAUS:
        t = int(round(tau * T))
        t = min(max(t, 1), T)
        out.append(spearmanr(traj[t], final).statistic)

    return np.array(out)


def tau90(rhos):
    hit = np.where(rhos >= 0.9)[0]
    return TAUS[hit[0]] if len(hit) else np.nan



# ---- DIFF: pure diffusion (g=0), random walk in log-space -------------------
def sim_diff(sigma=0.05):
    x = np.zeros((T + 1, N))
    x[0] = RNG.normal(0, 1e-6, N)
    for t in range(T):
        x[t + 1] = x[t] + RNG.normal(0, sigma, N)

    return x


# ---- AMP: free-token amplifier, paper Eq (3) with linear (unbounded) f ------
# x_{t+1} = x_t + ln(1 + g*theta*S) + eta ,  S = exp(x)   (no per-unit ceiling)
def sim_amp(g=0.2, theta=0.02, sigma=0.05, cap=None):
    x = np.zeros((T + 1, N))
    x[0] = RNG.normal(0, 1e-6, N)
    for t in range(T):
        S = np.exp(np.clip(x[t], None, 700))
        f = theta * S
        if cap is not None:              # optional saturating ceiling
            f = f / (1 + S / cap)
        x[t + 1] = x[t] + np.log1p(g * f) + RNG.normal(0, sigma, N)

    return x



# ---- PA: plain Barabasi preferential attachment -----------------------------
# each step, M new unit-tokens are handed out with prob proportional to current
# stock (+1 smoothing so a zero-stock entity can still receive). Position = stock.
def sim_pa(M=None, seed_stock=1.0):
    if M is None:
        M = N // 3
    stock = np.full(N, seed_stock, dtype=float)
    traj = np.zeros((T + 1, N))
    traj[0] = stock + RNG.normal(0, 1e-6, N)
    for t in range(T):
        w = stock / stock.sum()
        picks = RNG.choice(N, size=M, replace=True, p=w)
        np.add.at(stock, picks, 1.0)
        traj[t + 1] = stock

    return traj



# ---- TOP: amplifier + reverse-dominance reset (Kesten renewal) --------------
# each entity independently resets to baseline with per-step prob p_reset
# (a revocable status that can be actively toppled), applied preferentially to
# the currently over-dominant so it is a REVERSE-dominance reset.
def sim_top(g=0.2, theta=0.02, sigma=0.05, p_reset=0.02):
    x = np.zeros((T + 1, N))
    x[0] = RNG.normal(0, 1e-6, N)
    for t in range(T):
        S = np.exp(np.clip(x[t], None, 700))
        f = theta * S
        xn = x[t] + np.log1p(g * f) + RNG.normal(0, sigma, N)
        # reverse-dominance: reset probability rises with rank position
        rank_frac = xn.argsort().argsort() / (N - 1)          # 0..1, top=1
        reset = RNG.random(N) < (p_reset * rank_frac)
        xn[reset] = RNG.normal(0, 1e-6, reset.sum())
        x[t + 1] = xn

    return x


models = {
    "DIFF  (g=0, null)":        sim_diff(),
    "AMP   (free-token)":       sim_amp(g=0.2),
    "PA    (Barabasi)":         sim_pa(),
    "TOP   (amp+renewal)":      sim_top(g=0.2, p_reset=0.02),
}


sqrt_law = np.sqrt(TAUS)
print(f"{'tau':>7} " + " ".join(f"{t:>6.3f}" for t in TAUS))
print(f"{'sqrt':>7} " + " ".join(f"{v:>6.2f}" for v in sqrt_law))
print("-" * 78)
results = {}
for name, traj in models.items():
    r = rho_tau(traj)
    results[name] = r
    print(f"{name:>20} " + " ".join(f"{v:>6.2f}" for v in r) + f"   tau90={tau90(r)}")

print("\n=== discriminator checks ===")

amp, pa, top, diff = (results["AMP   (free-token)"], results["PA    (Barabasi)"],
                      results["TOP   (amp+renewal)"], results["DIFF  (g=0, null)"])
i_early = np.where(TAUS <= 0.1)[0]

print(f"mean excess over sqrt(tau), tau<=0.1:  AMP={np.mean((amp-sqrt_law)[i_early]):+.3f}  "
      f"PA={np.mean((pa-sqrt_law)[i_early]):+.3f}  DIFF={np.mean((diff-sqrt_law)[i_early]):+.3f}")
print(f"mean |AMP - PA| over all tau:          {np.mean(np.abs(amp-pa)):.3f}   "
      f"(small => sqrt-null cannot separate them)")
print(f"mean (AMP - TOP) over tau<=0.2:        {np.mean((amp-top)[TAUS<=0.2]):+.3f}   "
      f"(large => free-token/toppling contrast DOES separate)")



# ---- multi-seed robustness of the tau90 claims -----------------------------
def tau90_over_seeds(simfn, seeds=range(1, 9), **kw):
    vals = []
    for s in seeds:
        global RNG
        RNG = np.random.default_rng(1000 + s)
        vals.append(tau90(rho_tau(simfn(**kw))))
    return np.array(vals, dtype=float)


print("\n=== multi-seed tau90 (8 seeds): median [min,max] ===")
for label, fn, kw in [("AMP free-token", sim_amp, dict(g=0.2)),
                      ("PA  Barabasi",   sim_pa,  dict()),
                      ("TOP amp+renewal",sim_top, dict(g=0.2, p_reset=0.02))]:
    v = tau90_over_seeds(fn, **kw)
    print(f"{label:>16}: median={np.nanmedian(v):.3f}  min={np.nanmin(v):.3f}  max={np.nanmax(v):.3f}")
