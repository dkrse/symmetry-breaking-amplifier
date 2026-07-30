#!/usr/bin/env python3

"""
lichess_worked_example.py
=========================
Reproduces Section "Worked example: manufacture versus revelation in online chess"
(Table tab:lichess) of the paper.

The point: a cohort whose ENTRY RATINGS are near-equal develops an order that
*looks* manufactured (decoupled from the entry rating, early-lead lock-in faster
than the sqrt(tau) law). Neither appearance survives inspection: the entry
decoupling is what restricting entry to a 40-point band produces whatever the
dynamics, and the lock-in is what a heterogeneous population produces at g=0 with
no amplification at all (paper Eq. rhodrift). Watch `sd end` below: the band
equalises the entry ESTIMATE, not capability, and one month of play reveals the
cohort to be as dispersed as the whole population. So the trajectory settles
neither A nor B here.

What does settle B is an exogenous, non-amplified estimate of capability k-hat,
each player's CONVERGED rating months later, which shows the order is mostly
*revealed* skill, giving a manufactured share

        R = 1 - corr^2(month-end order, k-hat)   (Eq. Rshare)

that is an UPPER bound on manufacture (measurement error in k-hat attenuates the
correlation). This is the non-identifiability theorem's discriminator on real,
public, CC0 data.

Data: Lichess open database, monthly standard-rated PGN dumps.
  - 2013-01  : entry cohort            (~17 MB compressed)
  - 2013-07  : +6-month converged k    (~42 MB)
  - 2013-12  : +12-month converged k   (~92 MB)
Downloaded once from https://database.lichess.org and cached; ~150 MB total.

Deps: numpy, scipy, and the `zstd` CLI on PATH (for streaming decompression).
No randomness -> fully deterministic.
"""



import argparse
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
COHORT_MONTH = "2013-01"
REF_MONTHS = [("2013-07", "+6mo"), ("2013-12", "+12mo")]



# Entry-Elo bands. NOTE the naming: "near-equal-rated", not "near-equal". A narrow
# band means small Var of the ENTRY ESTIMATE, NOT small Var(ln k): Lichess seeds
# newcomers near 1500 precisely because it does not know them yet, so the band
# selects players the system has not yet told apart, not players who do not
# differ. The month-end s.d. printed below (213 vs the control's 211) shows the
# two bands share the same underlying capability dispersion.
BANDS = [("near-equal-rated", 1480, 1520), ("control", 1000, 2200)]

MIN_COHORT_GAMES = 30   # activity filter in the entry month
MIN_REF_GAMES = 20      # games required later so k-hat has converged

# The convergence filter is also a SELECTION: staying active for six months is
# not independent of how the month went, so it can only be assumed harmless if
# the estimate is stable when the filter is relaxed. --survivorship reruns the
# headline row across these thresholds.
SURVIVORSHIP_GRID = (1, 5, 10, 20, 40)


# --------------------------------------------------------------------------- IO
def ensure_dump(month, cache_dir):
    path = os.path.join(cache_dir, f"{month}.pgn.zst")
    if not os.path.exists(path):
        url = BASE.format(month)
        print(f"[download] {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path



def iter_games(path_zst):
    """Stream (tag-dict) per game from a .pgn.zst dump via the `zstd` CLI."""
    proc = subprocess.Popen(["zstd", "-dc", path_zst],
                            stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    cur = {}
    for line in proc.stdout:
        if line[:1] == "[":
            m = re.match(r'\[(\w+) "(.*)"\]', line)
            if m:
                cur[m.group(1)] = m.group(2)
        elif line[:1].isdigit() and "White" in cur:
            yield cur
            cur = {}
    proc.stdout.close()
    proc.wait()



def _elo(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None




# ------------- cohort build----------------------------------------------------
def build_cohort(path_zst):
    """player -> sorted list of (t_epoch, elo, win_flag) for the entry month."""
    byp = defaultdict(list)
    for g in iter_games(path_zst):
        try:
            t = datetime.strptime(g["UTCDate"] + " " + g["UTCTime"],
                                  "%Y.%m.%d %H:%M:%S").timestamp()
        except (KeyError, ValueError):
            continue
        res = g.get("Result")
        for side in ("White", "Black"):
            p, e = g.get(side), _elo(g.get(side + "Elo"))
            if not p or e is None:
                continue
            w = 0.5
            if res == "1-0":
                w = 1.0 if side == "White" else 0.0
            elif res == "0-1":
                w = 1.0 if side == "Black" else 0.0
            byp[p].append((t, e, w))
    for p in byp:
        byp[p].sort()
    return byp




def converged_elo(path_zst, wanted):
    """player -> (mean rating, n games) in a later month = k-hat channel."""
    acc = defaultdict(lambda: [0.0, 0])
    for g in iter_games(path_zst):
        for side in ("White", "Black"):
            p = g.get(side)
            if p in wanted:
                e = _elo(g.get(side + "Elo"))
                if e is not None:
                    acc[p][0] += e
                    acc[p][1] += 1
    return {p: (s / n, n) for p, (s, n) in acc.items() if n > 0}


# ---------------------- statistics-----------------------------------------
def corr_pair(u, v):

    """Both correlations, on the scale each one belongs on.

    Eq. (Rshare), R = 1 - corr^2, is a decomposition of VARIANCE, so the corr it
    contains must be a Pearson correlation on the scale the model is written in,
    namely log-status. A Spearman correlation does not decompose variance and
    1 - rho_S^2 is not a variance share. We therefore compute R from the Pearson
    correlation of log-ratings, and report the Spearman value alongside it as the
    order-only, monotone-transform-invariant summary.
    """

    return (float(spearmanr(u, v).statistic),
            float(pearsonr(np.log(u), np.log(v)).statistic))


EARLY_TAUS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)





def within_month_signatures(byp, bands):

    """The two TRAJECTORY signatures the paper reports before admitting k-hat.

    Both are what an amplifier reading would rest on, and both are shown in the
    text to be unsafe here, so they must be measured rather than asserted:

      * corr(entry Elo, month-end Elo): near zero in a narrow band, strongly
        positive in the heterogeneous control. The paper reads the first as
        "the final order is decoupled from where players began".
      * rho(tau) on the running NET score (wins minus losses, each player on the
        fraction tau of their OWN games), against the sqrt(tau) diffusion law of
        Prop. earlylead. The excess over sqrt(tau) is the "locks in faster than
        diffusion" claim.

    The point of printing them is that Eq. (rhodrift) reproduces both at g=0 on a
    heterogeneous population, so neither identifies amplification here.
    """
    out = {}
    for name, lo, hi in BANDS:
        players = bands[name]
        entry = np.array([byp[p][0][1] for p in players], dtype=float)
        end = np.array([byp[p][-1][1] for p in players], dtype=float)
        c_entry_end = float(spearmanr(entry, end).statistic)

        # running net score per player, sampled at tau-fractions of their own games
        pos = np.empty((len(players), len(EARLY_TAUS)))
        for r, p in enumerate(players):
            w = np.array([g[2] for g in byp[p]], dtype=float)   # 1 / 0.5 / 0
            net = np.cumsum(2.0 * w - 1.0)                       # +1 win, -1 loss
            n = len(net)
            idx = np.clip((np.round(np.array(EARLY_TAUS) * n) - 1).astype(int), 0, n - 1)
            pos[r] = net[idx]
        fin = pos[:, -1]
        rho = np.array([spearmanr(pos[:, j], fin).statistic
                        for j in range(len(EARLY_TAUS))])
        out[name] = dict(c_entry_end=c_entry_end, rho=rho, n=len(players))

    print("\nWithin-month trajectory signatures (before k-hat is admitted)")
    print(f"{'cohort':18} {'n':>5} {'corr(entry,end)':>16}")
    for name, _, _ in BANDS:
        o = out[name]
        print(f"{name:18} {o['n']:>5} {o['c_entry_end']:>16.2f}")
    print(f"\nearly-lead persistence rho(tau) of the running net score")
    print("   {:>7}".format("tau")
          + "".join(f"{n[:12]:>14}" for n, _, _ in BANDS) + f"{'sqrt(tau)':>11}")
    for j, t in enumerate(EARLY_TAUS):
        row = "".join(f"{out[n]['rho'][j]:>14.3f}" for n, _, _ in BANDS)
        print(f"   {t:>7.3f}{row}{np.sqrt(t):>11.3f}")
    ne = out["near-equal-rated"]["rho"]
    exc = ne - np.sqrt(EARLY_TAUS)
    print(f"   near-equal excess over sqrt(tau): min={exc.min():+.2f} "
          f"max={exc.max():+.2f} (at tau<1: max={exc[:-1].max():+.2f})")
    print("   ^ an excess, but Eq. (rhodrift) produces one at g=0 on a "
          "heterogeneous population, so it does not identify amplification.")

    return out


# ------------------ analysis----------------------------------------------
def survivorship_check(byp, bands, conv, ref="2013-07"):

    """Rerun the headline row across convergence-filter thresholds.

    Requiring k games in the reference month keeps players who stayed active,
    and staying active correlates with both skill and how the entry month went.
    That selection can inflate corr(month-end, k-hat) and so DEFLATE R, in the
    opposite direction to the attenuation and capability-drift arguments in the
    text. The estimate is only usable if it is stable here.
    """

    print(f"\nSurvivorship sensitivity (near-equal-rated cohort, ref {ref})")
    print(f"   {'min ref games':>14}{'n':>6}{'corr_S':>9}{'corr_P(log)':>13}{'R':>8}")
    players = bands["near-equal-rated"]
    for thr in SURVIVORSHIP_GRID:
        kmap = {p: v[0] for p, v in conv[ref].items() if v[1] >= thr}
        keep = [p for p in players if p in kmap]
        if len(keep) < 25:
            print(f"   {thr:>14}{len(keep):>6}   (too few)")
            continue
        end = np.array([byp[p][-1][1] for p in keep], dtype=float)
        khat = np.array([kmap[p] for p in keep], dtype=float)
        cs, cp = corr_pair(end, khat)
        print(f"   {thr:>14}{len(keep):>6}{cs:>9.2f}{cp:>13.2f}{1 - cp ** 2:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    # default assumes cwd = scripts/ (as run_all.sh invokes it); absolute paths ok too
    ap.add_argument("--cache-dir", default="data/lichess")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="B",
                    help="bootstrap replicates (resampling players) for CIs on "
                         "corr(end,k) and R (0 = skip)")
    ap.add_argument("--survivorship", action="store_true",
                    help="rerun the headline row across convergence-filter "
                         "thresholds, to expose selection on staying active")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    cohort_path = ensure_dump(COHORT_MONTH, args.cache_dir)
    byp = build_cohort(cohort_path)

    # entry Elo = first game of the month; month-end Elo = last game
    def in_band(p, lo, hi):
        return len(byp[p]) >= MIN_COHORT_GAMES and lo <= byp[p][0][1] <= hi

    bands = {name: [p for p in byp if in_band(p, lo, hi)]
             for name, lo, hi in BANDS}

    # pre-load converged-Elo maps for the union of all cohort players
    everyone = set().union(*bands.values())
    conv = {ref: converged_elo(ensure_dump(ref, args.cache_dir), everyone)
            for ref, _ in REF_MONTHS}

    # --- diagnostic: is the "near-equal-rated" band actually near-equal? ---
    # Measured on the FULL band (before the convergence filter), as the paper
    # reports its entry s.d. If the band were near-equal in CAPABILITY, its
    # month-end spread would stay well below the heterogeneous control's. It does
    # not: it lands on it, and on the whole active population's. The symmetry was
    # in the measurement, not in the players.
    active = [p for p in byp if len(byp[p]) >= MIN_COHORT_GAMES]
    pop_end_sd = np.std([byp[p][-1][1] for p in active])
    print(f"\nEntry-band diagnostic (full bands, {len(active)} active players)")
    print(f"{'band':18} {'n':>5} {'sd entry':>9} {'sd end':>8} {'ratio':>7}")

    for name, lo, hi in BANDS:
        pl = bands[name]
        e = np.std([byp[p][0][1] for p in pl])
        f = np.std([byp[p][-1][1] for p in pl])
        print(f"{name:18} {len(pl):>5} {e:>9.1f} {f:>8.1f} {f/max(e,1e-9):>6.1f}x")

    print(f"{'whole population':18} {len(active):>5} {'':>9} {pop_end_sd:>8.1f}")
    print("  ^ the narrow band's month-end spread lands on the population's:")
    print("    the band equalised the ESTIMATE, not capability (Var(ln k) is NOT small).")

    within_month_signatures(byp, bands)

    print(f"\nLichess worked example  (cohort {COHORT_MONTH})")
    print("  correlations printed as  Spearman/Pearson(log);  "
          "R = 1 - Pearson^2 (a variance share)")
    print(f"{'cohort':18} {'band':11} {'ref':6} {'n':>4}  "
          f"{'corr(entry,k)':>13} {'corr(end,k)':>11}  {'R':>5}")
    print("-" * 84)

    rows = []
    for name, lo, hi in BANDS:
        players = bands[name]
        entry_sd = np.std([byp[p][0][1] for p in players])
        for ref, tag in REF_MONTHS:
            kmap = {p: v[0] for p, v in conv[ref].items() if v[1] >= MIN_REF_GAMES}
            keep = [p for p in players if p in kmap]
            if len(keep) < 25:
                print(f"{name:18} {lo}-{hi:<6} {tag:6} {len(keep):>4}  (too few) ")
                continue
            entry = np.array([byp[p][0][1] for p in keep], dtype=float)
            end = np.array([byp[p][-1][1] for p in keep], dtype=float)
            khat = np.array([kmap[p] for p in keep], dtype=float)
            med_gain = float(np.median(khat - end))
            cs_entry, cp_entry = corr_pair(entry, khat)
            cs_end, cp_end = corr_pair(end, khat)
            R = 1.0 - cp_end ** 2          # variance share: Pearson, log scale
            ci = ""
            if args.bootstrap:
                rng = np.random.default_rng(0)
                n = len(keep)
                cb = np.empty(args.bootstrap)
                le, lk = np.log(end), np.log(khat)
                for b in range(args.bootstrap):
                    i = rng.integers(0, n, n)
                    cb[b] = pearsonr(le[i], lk[i]).statistic
                clo, chi = np.percentile(cb, [2.5, 97.5])
                rb = 1.0 - cb ** 2
                rlo, rhi = np.percentile(rb, [2.5, 97.5])
                ci = (f"  corr CI[{clo:.2f},{chi:.2f}]"
                      f"  R CI[{rlo:.2f},{rhi:.2f}]")
            print(f"{name:18} {lo}-{hi:<6} {tag:6} {len(keep):>4}  "
                  f"{cs_entry:>6.2f}/{cp_entry:<6.2f} {cs_end:>5.2f}/{cp_end:<5.2f}  "
                  f"{R:>5.2f}{ci}  medGain={med_gain:+.0f}")
            rows.append((name, f"{lo}-{hi}", tag, len(keep),
                         round(cp_entry, 2), round(cp_end, 2), round(R, 2),
                         round(entry_sd, 1), round(cs_entry, 2), round(cs_end, 2)))

    # headline number: near-equal-rated, +6mo
    for r in rows:
        if r[0] == "near-equal-rated" and r[2] == "+6mo":
            print(f"\nHeadline: near-equal-RATED cohort, +6mo -> "
                  f"corr(entry,k)={r[4]}, corr(end,k)={r[5]}, R={r[6]} "
                  f"(entry Elo sd={r[7]}); ~{round((1-r[6])*100)}% revealed.")

    if args.survivorship:
        survivorship_check(byp, bands, conv)
    return rows



if __name__ == "__main__":
    main()
