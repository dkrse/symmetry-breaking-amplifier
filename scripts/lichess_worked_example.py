#!/usr/bin/env python3
"""
lichess_worked_example.py
=========================
Reproduces Section "Worked example: manufacture versus revelation in online chess"
(Table tab:lichess) of the paper.

The point: a near-equal online-chess cohort develops an order that looks
*manufactured* (decoupled from the entry rating, super-diffusive early-lead
lock-in -- signature A). Admitting an exogenous, non-amplified estimate of
capability k-hat -- each player's CONVERGED rating months later -- shows the
order is mostly *revealed* skill, giving a manufactured share

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
from scipy.stats import spearmanr

BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst"
COHORT_MONTH = "2013-01"
REF_MONTHS = [("2013-07", "+6mo"), ("2013-12", "+12mo")]

# entry-Elo bands: near-equal (small Var(ln k)) vs heterogeneous control
BANDS = [("near-equal", 1480, 1520), ("control", 1000, 2200)]

MIN_COHORT_GAMES = 30   # activity filter in the entry month
MIN_REF_GAMES = 20      # games required later so k-hat has converged


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


# ------------------------------------------------------------------ cohort build
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


# ----------------------------------------------------------------------- analysis
def main():
    ap = argparse.ArgumentParser()
    # default assumes cwd = scripts/ (as run_all.sh invokes it); absolute paths ok too
    ap.add_argument("--cache-dir", default="data/lichess")
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

    print(f"\nLichess worked example  (cohort {COHORT_MONTH})")
    print(f"{'cohort':12} {'band':11} {'ref':6} {'n':>4}  "
          f"{'corr(entry,k)':>13} {'corr(end,k)':>11}  {'R':>5}")
    print("-" * 70)

    rows = []
    for name, lo, hi in BANDS:
        players = bands[name]
        entry_sd = np.std([byp[p][0][1] for p in players])
        for ref, tag in REF_MONTHS:
            kmap = {p: v[0] for p, v in conv[ref].items() if v[1] >= MIN_REF_GAMES}
            keep = [p for p in players if p in kmap]
            if len(keep) < 25:
                print(f"{name:12} {lo}-{hi:<6} {tag:6} {len(keep):>4}  (too few) ")
                continue
            entry = np.array([byp[p][0][1] for p in keep])
            end = np.array([byp[p][-1][1] for p in keep])
            khat = np.array([kmap[p] for p in keep])
            c_entry = spearmanr(entry, khat).statistic
            c_end = spearmanr(end, khat).statistic
            R = 1.0 - c_end ** 2
            print(f"{name:12} {lo}-{hi:<6} {tag:6} {len(keep):>4}  "
                  f"{c_entry:>13.2f} {c_end:>11.2f}  {R:>5.2f}")
            rows.append((name, f"{lo}-{hi}", tag, len(keep),
                         round(c_entry, 2), round(c_end, 2), round(R, 2), round(entry_sd, 1)))

    # headline number: near-equal, +6mo
    for r in rows:
        if r[0] == "near-equal" and r[2] == "+6mo":
            print(f"\nHeadline: near-equal cohort, +6mo -> "
                  f"corr(entry,k)={r[4]}, corr(end,k)={r[5]}, R={r[6]} "
                  f"(entry Elo sd={r[7]}); ~{round((1-r[6])*100)}% revealed.")
    return rows


if __name__ == "__main__":
    main()
