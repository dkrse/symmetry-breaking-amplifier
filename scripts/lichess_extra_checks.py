#!/usr/bin/env python3
"""
Two additional checks on the online-chess example (Section 6.4).

1. SINGLE-CHANNEL TEST.  Proposition nonident(i) rests on capability entering the
   cascade only through the seed a = k + p.  If capability also scaled the gain
   (lambda_i = lambda(k_i)) or the noise (sigma_i = sigma(k_i)), it would show up
   in the curvature or the heteroskedasticity of an entity's own trajectory.
   With k-hat available for the chess cohort, this is testable: regress, across
   players, (a) the variance of per-game increments and (b) the curvature of the
   trajectory on k-hat.  Both are computed on the running net score (wins minus
   losses, the position that forms during the month) and, for completeness, on
   the log-rating.  Games played in the month is a nuisance variable (Glicko-2
   step sizes shrink with games), so partial correlations net of log(n) are
   reported alongside the raw ones.  Curvature is the coefficient of the
   quadratic term in an orthogonal-polynomial fit of the trajectory on the
   normalised game index.

2. INVERSE-PROBABILITY WEIGHTING FOR ATTRITION.  The convergence filter keeps
   players with >= 20 games six months later.  Survival is modelled by a logistic
   regression on entry-month observables (log month-end rating, log games, net
   score, entry rating, rating change), and corr(log end, log k-hat) and R are
   recomputed with weights 1/p(survive).  Descriptive survival rates by month-end
   rating quartile are printed, and the weighted estimate is bootstrapped.

Usage:
    python scripts/lichess_extra_checks.py [--cache-dir data/lichess] [--bootstrap 1000]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lichess_worked_example import (BANDS, COHORT_MONTH, MIN_COHORT_GAMES,  # noqa: E402
                                    MIN_REF_GAMES, build_cohort, converged_elo,
                                    ensure_dump)

REF = "2013-07"


# ---------------------------------------------------------------- trajectory stats
def traj_features(games):
    """games: sorted list of (t, elo, win_flag). Returns dict of per-player stats."""
    w = np.array([g[2] for g in games], dtype=float)
    net = np.cumsum(2.0 * w - 1.0)
    lr = np.log(np.array([g[1] for g in games], dtype=float))
    n = len(w)
    x = (np.arange(n) + 0.5) / n
    # orthogonal quadratic fit: curvature = coefficient on the 2nd Legendre-like term
    X = np.vstack([np.ones(n), x - 0.5, (x - 0.5) ** 2 - 1.0 / 12.0]).T
    c_net = np.linalg.lstsq(X, net, rcond=None)[0][2]
    c_lr = np.linalg.lstsq(X, lr, rcond=None)[0][2]
    return dict(
        n=n,
        var_inc_net=float(np.var(np.diff(net))),      # = 4 p(1-p) for Bernoulli wins
        var_inc_lr=float(np.var(np.diff(lr))),
        curv_net=float(c_net) / n,                    # per-game scale
        curv_lr=float(c_lr),
        winrate=float(w.mean()),
    )


def partial_corr(y, k, z):
    """Spearman corr of y and k after residualising both on z (linear)."""
    def resid(v):
        A = np.vstack([np.ones_like(z), z]).T
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(spearmanr(resid(y), resid(k)).statistic)


def boot_ci(stat, n, rng, B):
    vals = np.empty(B)
    for b in range(B):
        vals[b] = stat(rng.integers(0, n, n))
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- logistic IRLS
def logistic_fit(X, y, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ beta))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-3 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def wpearson(u, v, w):
    w = w / w.sum()
    mu, mv = (w * u).sum(), (w * v).sum()
    cov = (w * (u - mu) * (v - mv)).sum()
    return cov / np.sqrt((w * (u - mu) ** 2).sum() * (w * (v - mv) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data/lichess")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    byp = build_cohort(ensure_dump(COHORT_MONTH, args.cache_dir))
    bands = {name: [p for p in byp
                    if len(byp[p]) >= MIN_COHORT_GAMES and lo <= byp[p][0][1] <= hi]
             for name, lo, hi in BANDS}
    everyone = set().union(*bands.values())
    conv = converged_elo(ensure_dump(REF, args.cache_dir), everyone)

    for name, lo, hi in BANDS:
        players = bands[name]
        feats = {p: traj_features(byp[p]) for p in players}
        kmap = {p: v[0] for p, v in conv.items() if v[1] >= MIN_REF_GAMES}
        keep = [p for p in players if p in kmap]
        khat = np.log(np.array([kmap[p] for p in keep]))
        logn = np.log(np.array([feats[p]["n"] for p in keep], dtype=float))

        print(f"\n=== {name} ({lo}-{hi}): N band = {len(players)}, with k-hat (>= {MIN_REF_GAMES} games at {REF}) = {len(keep)}")

        # ---------------- 1. single-channel test
        print(f"\n[1] Single-channel test: trajectory statistic vs k-hat (Spearman; partial = net of log games)")
        print(f"    {'statistic':>22} {'rho':>7} {'95% CI':>17} {'partial':>9} {'95% CI':>17}")
        for key, label in (("var_inc_net", "Var(d net score)"), ("curv_net", "curvature(net)"),
                           ("var_inc_lr", "Var(d log rating)"), ("curv_lr", "curvature(log r)"),
                           ("winrate", "win rate")):
            y = np.array([feats[p][key] for p in keep])
            r = float(spearmanr(y, khat).statistic)
            pr = partial_corr(y, khat, logn)
            n = len(keep)
            lo_r, hi_r = boot_ci(lambda i: spearmanr(y[i], khat[i]).statistic, n, rng, args.bootstrap)
            lo_p, hi_p = boot_ci(lambda i: partial_corr(y[i], khat[i], logn[i]), n, rng, args.bootstrap)
            print(f"    {label:>22} {r:>+7.2f} [{lo_r:+.2f},{hi_r:+.2f}] {pr:>+9.2f} [{lo_p:+.2f},{hi_p:+.2f}]")
        # nuisance: how strongly the statistics depend on games played at all
        print(f"    (nuisance) Spearman of Var(d log rating) with log games: "
              f"{spearmanr(np.array([feats[p]['var_inc_lr'] for p in keep]), logn).statistic:+.2f}")

        # ---------------- 2. IPW for attrition
        print(f"\n[2] Attrition: survival = has k-hat at {REF} with >= {MIN_REF_GAMES} games")
        surv = np.array([1.0 if p in kmap else 0.0 for p in players])
        end = np.array([byp[p][-1][1] for p in players], dtype=float)
        ent = np.array([byp[p][0][1] for p in players], dtype=float)
        ng = np.array([len(byp[p]) for p in players], dtype=float)
        netf = np.array([np.sum(2.0 * np.array([g[2] for g in byp[p]]) - 1.0) for p in players])
        # entry rating and rating change are collinear with month-end rating inside a
        # narrow band, so the change is omitted; entry is kept for the control band
        Z = np.vstack([np.log(end), np.log(ng), netf / ng, ent / 100.0]).T
        Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        X = np.hstack([np.ones((len(players), 1)), Zs])
        beta = logistic_fit(X, surv)
        p_surv = 1.0 / (1.0 + np.exp(-X @ beta))
        print(f"    survival rate {surv.mean():.2f}; logistic coefficients (standardised): "
              + ", ".join(f"{nm}={b:+.2f}" for nm, b in zip(
                  ["const", "log end", "log games", "net/game", "entry"], beta)))
        q = np.quantile(end, [0.25, 0.5, 0.75])
        qi = np.searchsorted(q, end)
        print("    survival by month-end rating quartile: "
              + ", ".join(f"Q{j+1}={surv[qi == j].mean():.2f}" for j in range(4)))
        # weighted estimate among survivors
        idx = np.array([i for i, p in enumerate(players) if p in kmap])
        le = np.log(end[idx]); lk = khat
        w = 1.0 / np.clip(p_surv[idx], 0.02, 1.0)
        c_unw = pearsonr(le, lk).statistic
        c_w = wpearson(le, lk, w)
        print(f"    corr(log end, log k-hat): unweighted {c_unw:.3f} -> IPW {c_w:.3f};  "
              f"R: {1-c_unw**2:.3f} -> {1-c_w**2:.3f}")
        if args.bootstrap:
            # bootstrap over the FULL band, refitting the survival model each time
            vals = []
            for _ in range(args.bootstrap):
                bi = rng.integers(0, len(players), len(players))
                sb = surv[bi]
                if sb.sum() < 25:
                    continue
                bb = logistic_fit(X[bi], sb)
                pb = 1.0 / (1.0 + np.exp(-X[bi] @ bb))
                sel = np.where(sb == 1)[0]
                pl = [players[i] for i in bi[sel]]
                le_b = np.log(end[bi[sel]]); lk_b = np.log(np.array([kmap[p] for p in pl]))
                vals.append(1 - wpearson(le_b, lk_b, 1.0 / np.clip(pb[sel], 0.02, 1.0)) ** 2)
            lo_R, hi_R = np.percentile(vals, [2.5, 97.5])
            print(f"    IPW R bootstrap 95% CI [{lo_R:.2f},{hi_R:.2f}]  (B={len(vals)})")
        print(f"    weight range {w.min():.2f}-{w.max():.2f}, effective sample size "
              f"{w.sum()**2 / (w**2).sum():.0f} of {len(idx)}")


if __name__ == "__main__":
    main()
