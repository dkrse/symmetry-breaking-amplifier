#!/usr/bin/env python3

"""
Re-extract the GitHub and Wikipedia early-lead trajectories on a FINE tau grid.

The bundled output/{github,wiki_rfa}_earlylead.csv are on a coarse 10-point grid.
For the rho(tau) collapse test (real_data_collapse.py) it helps to have all real
domains on a comparable fine grid, so this script rebuilds both OFFLINE from the
raw sources already in the repo:

  * GitHub  : star timestamps cached under scripts/data/github_cache/ (all 295
              repos of the existing cohort present). Position = cumulative star
              stock; per-repo clock t0 = first cached star (a Q1-2019 cohort sharing
              an entry POINT, not a common quality), horizon 24 months. -> output/github_earlylead_fine.csv
  * Wikipedia RfA : raw scripts/data/wiki-RfA.txt.gz (bundled). Position at tau =
              running NET support (support-oppose, revocable) and SUPPORT-only
              (free token). -> output/wiki_rfa_earlylead_fine.csv

Same fine 18-point grid as startup_earlylead.py, so the three domains are
directly comparable. real_data_collapse.py prefers these *_fine.csv when present.
"""
import csv
import glob
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

import wiki_rfa_toppling as wiki   # reuse the raw RfA parser (offline, bundled gz)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "scripts" / "data" / "github_cache"
GH_COHORT_CSV = ROOT / "output" / "github_earlylead.csv"
GH_OUT = ROOT / "output" / "github_earlylead_fine.csv"
WIKI_OUT = ROOT / "output" / "wiki_rfa_earlylead_fine.csv"

FINE = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25,
                 0.3, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0])
GH_HORIZON_MONTHS = 24
MIN_VOTES = 25


# --------------------------------------------------------------------------
# GitHub: cumulative star stock on the fine grid, from the cache
# --------------------------------------------------------------------------
def _star_times_from_cache(repo):
    """All starredAt datetimes for a repo, read across cached GraphQL pages."""
    san = repo.replace("/", "_")
    times = []
    page = 0
    while True:
        p = CACHE / f"gql_{san}_{page}.json"
        if not p.exists():
            break
        data = json.loads(p.read_text())
        rep = (data.get("data") or {}).get("repository")
        if not rep:
            break
        for e in rep["stargazers"]["edges"]:
            ts = e.get("starredAt")
            if ts:
                times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
        page += 1

    return sorted(times)


def refine_github():
    repos = [r["repo"] for r in csv.DictReader(open(GH_COHORT_CSV))]
    rows = []
    horizon_days = int(GH_HORIZON_MONTHS * 30.44)
    for repo in repos:
        ts = _star_times_from_cache(repo)
        if len(ts) < 2:
            continue
        t0 = ts[0]                                   # entry clock = first star
        days = np.array([(d - t0).days for d in ts], dtype=float)
        days = days[days <= horizon_days]
        pos = [int((days <= tau * horizon_days).sum()) for tau in FINE]
        rows.append((repo, pos))

    with open(GH_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo"] + [f"tau_{t:g}" for t in FINE])
        for repo, pos in rows:
            w.writerow([repo] + pos)
    print(f"GitHub  : {len(rows)} repos on {len(FINE)}-point grid -> {GH_OUT.name}")

    return len(rows)


# --------------------------------------------------------------------------
# Wikipedia RfA: net and support-only positions on the fine grid, from raw gz
# --------------------------------------------------------------------------
def _positions_fine(votes):
    v = np.array([vote for (_, vote) in votes], dtype=float)
    n = len(v)
    net_cum = np.cumsum(v)
    supp_cum = np.cumsum(np.clip(v, 0, None))
    net, supp = [], []
    for tau in FINE:
        idx = max(int(round(tau * n)) - 1, 0)
        net.append(net_cum[idx])
        supp.append(supp_cum[idx])

    return net, supp


def refine_wiki():
    wiki.ensure_data()                               # bundled gz -> no download
    net_rows, supp_rows = [], []
    for key, votes in wiki.parse_elections(MIN_VOTES).items():
        net, supp = _positions_fine(votes)
        net_rows.append((key, net))
        supp_rows.append((key, supp))
    with open(WIKI_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "arm"] + [f"tau_{t:g}" for t in FINE])
        for key, net in net_rows:
            w.writerow([key, "net"] + list(net))
        for key, supp in supp_rows:
            w.writerow([key, "support"] + list(supp))
    print(f"Wikipedia: {len(net_rows)} elections x2 arms on {len(FINE)}-point grid "
          f"-> {WIKI_OUT.name}")

    return len(net_rows)


if __name__ == "__main__":
    refine_github()
    refine_wiki()
