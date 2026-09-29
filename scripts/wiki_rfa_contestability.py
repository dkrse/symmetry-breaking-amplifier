#!/usr/bin/env python3
"""
Contestability dose-response in Wikipedia RfA: one functional, varying revocation.

The free-token / revocable contrast of wiki_rfa_toppling.py compares two different
functionals of the same vote stream (support-only vs net support). A natural
objection is that the two functionals differ mechanically, not only in
revocability: a monotone count locks in earlier simply because it never falls.
This script keeps ONE functional, the net support, and lets contestability vary
across elections instead: elections are binned by their final oppose share, and
early-lead persistence rho(tau) and tau_90 are computed within each bin.

Prediction: the more contestable the election (higher oppose share), the later
the net-support order locks in (lower rho at small tau, larger tau_90).

Two mechanical effects must be netted out, and the script reports both:
  1. Binning narrows the cross-election spread of support shares inside a bin,
     which lowers rank correlations in BOTH arms whatever the dynamics.
  2. At high oppose share the net signal (2p-1) is small relative to its
     binomial noise, which lowers rho_net even for i.i.d. votes.
Both are captured by an i.i.d.-vote placebo run per bin (same election sizes,
same support shares, no compounding), so the informative quantity is the
data-minus-placebo excess within each bin, and its trend across bins.
The support-only arm is reported in every bin as a same-data control.

Usage:
    python scripts/wiki_rfa_contestability.py [--min-votes 25] [--bins 5] [--bootstrap 1000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiki_rfa_toppling import (TAUS, ensure_data, parse_elections,  # noqa: E402
                               positions_at_taus, rho_curve, tau90)

ROOT = Path(__file__).resolve().parent.parent
I_TAU01 = int(np.where(np.isclose(TAUS, 0.1))[0][0])


def placebo_bin(ns, shares, rng):
    """i.i.d. votes given the candidate, same n and support share as the bin."""
    net_rows, supp_rows = [], []
    for n_i, p_i in zip(ns, shares):
        v = np.where(rng.random(n_i) < p_i, 1.0, -1.0)
        net_c = np.cumsum(v)
        supp_c = np.cumsum(np.clip(v, 0, None))
        idx = np.clip((np.round(TAUS * n_i) - 1).astype(int), 0, n_i - 1)
        net_rows.append(net_c[idx])
        supp_rows.append(supp_c[idx])
    return rho_curve(np.array(net_rows)), rho_curve(np.array(supp_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-votes", type=int, default=25)
    ap.add_argument("--bins", type=int, default=5, help="quantile bins of oppose share")
    ap.add_argument("--bootstrap", type=int, default=1000, metavar="B")
    ap.add_argument("--placebo-reps", type=int, default=50)
    ap.add_argument("--out-fig", default=str(ROOT / "output/figures/wiki_rfa_contestability.png"))
    args = ap.parse_args()

    ensure_data()
    elections = parse_elections(args.min_votes)
    keys = list(elections.keys())
    net_rows, supp_rows, ns, shares = [], [], [], []
    for key in keys:
        votes = elections[key]
        net, supp = positions_at_taus(votes)
        v = np.array([x[1] for x in votes], dtype=float)
        net_rows.append(net); supp_rows.append(supp)
        ns.append(len(v)); shares.append(float(np.mean(v > 0)))
    net_cols = np.array(net_rows); supp_cols = np.array(supp_rows)
    ns = np.array(ns); shares = np.clip(np.array(shares), 0.02, 0.98)
    oppose = 1.0 - shares

    edges = np.quantile(oppose, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    which = np.clip(np.searchsorted(edges, oppose, side="right") - 1, 0, args.bins - 1)

    print(f"\nWikipedia RfA: {len(keys)} elections (>= {args.min_votes} votes), "
          f"{args.bins} quantile bins of final oppose share\n")
    hdr = ("{:>4} {:>5} {:>13} | {:>9} {:>9} {:>8} | {:>9} {:>9} {:>8} | "
           "{:>9} {:>9} {:>9}")
    print(hdr.format("bin", "N", "oppose range",
                     "rho_net.1", "rho_sup.1", "gap.1",
                     "plc_net.1", "plc_sup.1", "plc_gap",
                     "t90_net", "t90_sup", "excess.1"))
    rng = np.random.default_rng(0)
    rows = []
    for b in range(args.bins):
        m = which == b
        rn = rho_curve(net_cols[m]); rs = rho_curve(supp_cols[m])
        pn = np.zeros(len(TAUS)); ps = np.zeros(len(TAUS))
        for _ in range(args.placebo_reps):
            a, c = placebo_bin(ns[m], shares[m], rng)
            pn += a; ps += c
        pn /= args.placebo_reps; ps /= args.placebo_reps
        gap = rs[I_TAU01] - rn[I_TAU01]
        pgap = ps[I_TAU01] - pn[I_TAU01]
        row = dict(b=b, n=int(m.sum()), lo=oppose[m].min(), hi=oppose[m].max(),
                   mid=float(np.median(oppose[m])),
                   rho_net=rn[I_TAU01], rho_sup=rs[I_TAU01], gap=gap,
                   plc_net=pn[I_TAU01], plc_sup=ps[I_TAU01], plc_gap=pgap,
                   t90_net=tau90(rn), t90_sup=tau90(rs),
                   excess_net=rn[I_TAU01] - pn[I_TAU01], excess_gap=gap - pgap)
        rows.append(row)
        print(hdr.format(b, row["n"], f"{row['lo']:.2f}-{row['hi']:.2f}",
                         f"{row['rho_net']:.3f}", f"{row['rho_sup']:.3f}", f"{gap:+.3f}",
                         f"{row['plc_net']:.3f}", f"{row['plc_sup']:.3f}", f"{pgap:+.3f}",
                         f"{row['t90_net']:.3f}", f"{row['t90_sup']:.3f}",
                         f"{row['excess_gap']:+.3f}"))
    print("\n  gap.1   = rho_supportonly(0.1) - rho_net(0.1) in the data")
    print("  plc_*   = same quantities under i.i.d. votes (no compounding), bin-matched")
    print("  excess.1 = gap.1 - plc_gap: what heterogeneity + mechanics cannot buy\n")

    # trend across bins (Spearman of bin midpoint vs quantity), with bootstrap over elections
    mids = np.array([r["mid"] for r in rows])
    for name in ("rho_net", "gap", "excess_gap", "t90_net"):
        vals = np.array([r[name] for r in rows])
        print(f"  trend across bins, {name:>10}: Spearman = {spearmanr(mids, vals).statistic:+.2f}")

    # continuous version: per-election contribution is not defined for a correlation,
    # so bootstrap the bin-level slope of rho_net(0.1) and of the gap on oppose share.
    if args.bootstrap:
        rngb = np.random.default_rng(1)
        n = len(keys)
        slopes_net, slopes_gap = [], []
        for _ in range(args.bootstrap):
            idx = rngb.integers(0, n, n)
            o = oppose[idx]; nc = net_cols[idx]; sc = supp_cols[idx]
            e = np.quantile(o, np.linspace(0, 1, args.bins + 1)); e[-1] += 1e-9
            w = np.clip(np.searchsorted(e, o, side="right") - 1, 0, args.bins - 1)
            xs, yn, yg = [], [], []
            for b in range(args.bins):
                m = w == b
                if m.sum() < 10:
                    continue
                rn = rho_curve(nc[m]); rs = rho_curve(sc[m])
                xs.append(np.median(o[m])); yn.append(rn[I_TAU01]); yg.append(rs[I_TAU01] - rn[I_TAU01])
            xs = np.array(xs)
            slopes_net.append(np.polyfit(xs, yn, 1)[0])
            slopes_gap.append(np.polyfit(xs, yg, 1)[0])
        for nm, s in (("rho_net(0.1)", slopes_net), ("gap(0.1)", slopes_gap)):
            lo, hi = np.percentile(s, [2.5, 97.5])
            print(f"  slope of {nm:>12} on oppose share (per unit share): "
                  f"{np.mean(s):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (B={args.bootstrap})")

    # timing diagnostic: do oppose votes arrive later than support votes?
    pos_o, pos_s, late = [], [], []
    for key in keys:
        v = np.array([x[1] for x in elections[key]]); n_i = len(v)
        t = (np.arange(n_i) + 0.5) / n_i
        if (v < 0).sum() >= 3 and (v > 0).sum() >= 3:
            pos_o.append(t[v < 0].mean()); pos_s.append(t[v > 0].mean())
            k = max(int(0.1 * n_i), 1)
            late.append((v[k:] < 0).mean() - (v[:k] < 0).mean())
    pos_o, pos_s, late = map(np.array, (pos_o, pos_s, late))
    print(f"\ntiming of oppose votes ({len(pos_o)} elections with >=3 of each kind):")
    print(f"  mean normalised position: oppose {pos_o.mean():.3f}, support {pos_s.mean():.3f} (0.5 = uniform)")
    print(f"  oppose later than support in {(pos_o > pos_s).mean():.1%} of elections")
    print(f"  oppose share after first tenth minus within it: mean {late.mean():+.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(mids, [r["rho_net"] for r in rows], "o-", color="crimson", label="net support, data")
    ax[0].plot(mids, [r["plc_net"] for r in rows], "o--", color="crimson", alpha=0.5, label="net support, i.i.d. placebo")
    ax[0].plot(mids, [r["rho_sup"] for r in rows], "s-", color="seagreen", label="support-only, data")
    ax[0].plot(mids, [r["plc_sup"] for r in rows], "s--", color="seagreen", alpha=0.5, label="support-only, placebo")
    ax[0].set_xlabel("final oppose share (bin median)"); ax[0].set_ylabel(r"$\rho(0.1)$")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(mids, [r["gap"] for r in rows], "o-", color="k", label="gap, data")
    ax[1].plot(mids, [r["plc_gap"] for r in rows], "o--", color="gray", label="gap, placebo")
    ax[1].plot(mids, [r["excess_gap"] for r in rows], "^-", color="navy", label="excess (data - placebo)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("final oppose share (bin median)"); ax[1].set_ylabel(r"$\rho_{\rm supp}(0.1)-\rho_{\rm net}(0.1)$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=300)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
