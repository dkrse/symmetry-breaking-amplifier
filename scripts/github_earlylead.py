#!/usr/bin/env python3

"""
Design 2 on real GitHub data: early-lead persistence of repository star stock.

GitHub stars are a *free token* (no per-unit ceiling, no toppling), so the model
predicts STRONG early-lead persistence and no optimal-gain downturn. This script
measures it on a cohort with a COMMON STARTING POSITION (repositories created in
a fixed window, every one starting at 0 stars). Note the wording: a common start
is NOT a homogeneous population. Repositories plainly differ in quality, i.e. in
their per-step drift, which is the case of paper Eq. rhodrift: such a cohort
exceeds sqrt(tau) at g=0, with no amplification at all. The premise of the
sqrt(tau) law is therefore NOT met here, and nothing below rests on it. Reports:

  rho(tau) = corr( rank of star stock at early fraction tau of the horizon,
                   rank of star stock at the horizon )              [Eq. (rhotau)]

against three references:
  * the analytic diffusion null   sqrt(tau)   (too weak twice over: it assumes a
    homogeneous population, and it is not sufficient on a monotone stock anyway),
  * a preferential-attachment (PA) null fitted to the cohort's own growth,
  * tau_90, the smallest tau whose ordering correlates >= 0.9 with the final one.

The key methodological point (validated synthetically in earlylead_pa_null.py):
the MAGNITUDE of rho(tau) does not by itself separate this amplifier from plain
cumulative advantage -- PA already exceeds sqrt(tau), often by more. The clean
discriminator is the free-token/toppling CONTRAST: run --arm free (GitHub stars,
here) against a contestable/revocable status (--arm topple, a separate dataset).
This script provides the free-token arm and leaves a documented hook for the other.

Usage:
    export GITHUB_TOKEN=ghp_...            # a classic or fine-grained PAT
    python scripts/github_earlylead.py --created 2019-01-01..2019-03-31 \
        --min-stars 80 --max-stars 3000 --cohort 300 --horizon-months 24

Notes / limits:
  * The stargazers endpoint returns at most 40,000 stars per repo and is
    rate-limited (5000 requests/hour authenticated). Keep --max-stars well under
    40k and the cohort modest; the script caches every API response under
    --cache-dir so re-runs are free.
  * Repos created in the same window all start at 0 stars => the homogeneous
    start the model assumes; the baseline is sqrt(tau), not zero.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import spearmanr

API = "https://api.github.com"
GQL = "https://api.github.com/graphql"
SESSION = requests.Session()
ROOT = Path(__file__).resolve().parent.parent          # repo root (CWD-independent)

# Star timestamps are read via the GraphQL API, not the REST /stargazers endpoint:
# the REST activity/starring endpoint is blocked (404) behind some egress proxies,
# whereas GraphQL goes through, is cheaper (cursor pagination), and lets us order
# by STARRED_AT ascending so we can stop as soon as we pass the horizon.
STAR_QUERY = """
query($owner:String!,$name:String!,$cursor:String){
  repository(owner:$owner,name:$name){
    stargazers(first:100, after:$cursor, orderBy:{field:STARRED_AT,direction:ASC}){
      pageInfo{hasNextPage endCursor}
      edges{starredAt}
    }
  }
}"""


# --------------------------------------------------------------------------- #
# HTTP with rate-limit handling and on-disk caching
# --------------------------------------------------------------------------- #
def _headers(star_json=False):
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("ERROR: set GITHUB_TOKEN to a GitHub personal access token.")
    h = {"Authorization": f"Bearer {tok}",
         "X-GitHub-Api-Version": "2022-11-28",
         "Accept": "application/vnd.github.star+json" if star_json
                   else "application/vnd.github+json"}
    return h



def _cache_path(cache_dir, key):
    safe = key.replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return Path(cache_dir) / f"{safe}.json"



def gh_get(url, params=None, star_json=False, cache_dir=None, cache_key=None):
    """GET with retry, secondary-rate-limit backoff, and optional disk cache."""
    if cache_dir and cache_key:
        cp = _cache_path(cache_dir, cache_key)
        if cp.exists():
            return json.loads(cp.read_text())
    for attempt in range(6):
        r = SESSION.get(url, headers=_headers(star_json), params=params, timeout=30)
        # primary rate limit: wait until reset
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 1) + 2
            print(f"  rate limit hit; sleeping {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            continue
        # secondary rate limit / abuse detection
        if r.status_code in (403, 429) and "retry-after" in {k.lower() for k in r.headers}:
            wait = int(r.headers.get("Retry-After", 5))
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        if cache_dir and cache_key:
            _cache_path(cache_dir, cache_key).write_text(json.dumps(data))
        return data
    raise RuntimeError(f"giving up on {url}")



# --------------------------------------------------------------------------- #
# Cohort selection and star-timestamp retrieval
# --------------------------------------------------------------------------- #
def search_cohort(created, min_stars, max_stars, cohort, language, cache_dir):
    """Repos created in `created` window with final stars in [min,max]."""
    q = f"created:{created} stars:{min_stars}..{max_stars}"
    if language:
        q += f" language:{language}"
    repos, page = [], 1
    while len(repos) < cohort and page <= 10:            # Search API caps at 1000
        data = gh_get(f"{API}/search/repositories",
                      params={"q": q, "sort": "stars", "order": "desc",
                              "per_page": 100, "page": page},
                      cache_dir=cache_dir, cache_key=f"search_{created}_{page}")
        items = data.get("items", [])
        if not items:
            break
        for it in items:
            repos.append({"full_name": it["full_name"],
                          "created_at": it["created_at"],
                          "stars": it["stargazers_count"]})
        page += 1
        time.sleep(1)                                     # be gentle on Search API

    return repos[:cohort]


def gql_post(variables, cache_dir=None, cache_key=None):
    """POST a GraphQL query with retry, rate-limit backoff, and disk cache."""
    if cache_dir and cache_key:
        cp = _cache_path(cache_dir, cache_key)
        if cp.exists():
            return json.loads(cp.read_text())
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("ERROR: set GITHUB_TOKEN to a GitHub personal access token.")
    for attempt in range(6):
        r = SESSION.post(GQL, headers={"Authorization": f"Bearer {tok}"},
                         json={"query": STAR_QUERY, "variables": variables}, timeout=30)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(reset - time.time(), 1) + 2)
            continue
        if r.status_code in (403, 429):
            time.sleep(int(r.headers.get("Retry-After", 5)) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()

        if cache_dir and cache_key:
            _cache_path(cache_dir, cache_key).write_text(json.dumps(data))
        return data

    raise RuntimeError("giving up on graphql")


def star_times(full_name, horizon_end, cache_dir):
    """Return sorted list of starred_at datetimes up to horizon_end (GraphQL)."""
    owner, _, name = full_name.partition("/")
    times, cursor, page = [], None, 0
    while True:
        data = gql_post({"owner": owner, "name": name, "cursor": cursor},
                        cache_dir=cache_dir, cache_key=f"gql_{full_name}_{page}")
        repo = (data.get("data") or {}).get("repository")
        if not repo:                                       # renamed / missing repo
            break
        sg = repo["stargazers"]
        stop = False
        for e in sg["edges"]:
            ts = e.get("starredAt")
            if not ts:
                continue
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt <= horizon_end:
                times.append(dt)
            else:                                          # ASC order: rest are later
                stop = True
                break
        if stop or not sg["pageInfo"]["hasNextPage"] or page >= 500:
            break
        cursor = sg["pageInfo"]["endCursor"]
        page += 1

    return sorted(times)


def cumulative_at_taus(created_at, star_dts, horizon_months, taus):
    """Cumulative star count at each tau-fraction of the horizon."""
    t0 = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    horizon = t0 + timedelta(days=int(horizon_months * 30.44))
    total_days = (horizon - t0).days
    arr = np.array([(d - t0).days for d in star_dts if d <= horizon])
    return t0, horizon, np.array([int((arr <= tau * total_days).sum()) for tau in taus])


# --------------------------------------------------------------------------- #
# Metrics and PA null
# --------------------------------------------------------------------------- #
def rho_curve(counts, taus):
    """counts: (n_repos, n_taus) cumulative stock. rho(tau) vs final column."""
    final = counts[:, -1]
    out = []
    for j in range(len(taus)):
        # jitter-free spearman; ties handled by scipy
        out.append(spearmanr(counts[:, j], final).statistic)
    return np.array(out)


def tau90(rhos, taus):
    hit = np.where(rhos >= 0.9)[0]
    return float(taus[hit[0]]) if len(hit) else float("nan")


def pa_null_rho(final_counts, taus, n_steps=300, seed=0):
    """
    Preferential-attachment null fitted to the cohort: hand out the same TOTAL
    number of stars as observed, one at a time, with prob proportional to current
    stock (+1 smoothing), over n_steps rounds; measure rho(tau). This is the
    'plain cumulative advantage' reference the free-token amplifier must beat on
    the CONTRAST test, not on magnitude (see module docstring).
    """
    rng = np.random.default_rng(seed)
    n = len(final_counts)
    total = int(final_counts.sum())
    per_step = max(total // n_steps, 1)
    stock = np.ones(n, dtype=float)
    snap_idx = [int(round(t * n_steps)) for t in taus]
    snaps = {}
    for step in range(n_steps + 1):
        if step in snap_idx:
            snaps[step] = stock.copy()
        if step == n_steps:
            break
        w = stock / stock.sum()
        picks = rng.choice(n, size=per_step, replace=True, p=w)
        np.add.at(stock, picks, 1.0)

    cols = np.column_stack([snaps[i] for i in snap_idx])
    return rho_curve(cols, taus)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--created", default="2019-01-01..2019-03-31",
                    help="repo creation window (GitHub search syntax)")
    ap.add_argument("--min-stars", type=int, default=80)
    ap.add_argument("--max-stars", type=int, default=3000)
    ap.add_argument("--cohort", type=int, default=300)
    ap.add_argument("--horizon-months", type=float, default=24)
    ap.add_argument("--language", default=None)
    ap.add_argument("--arm", choices=["free", "topple"], default="free",
                    help="'free' = GitHub stars (implemented). 'topple' = a "
                         "contestable/revocable status; needs a different source "
                         "(see hook below) and is intentionally not implemented.")
    ap.add_argument("--cache-dir", default=str(ROOT / "scripts/data/github_cache"))
    ap.add_argument("--out-fig", default=str(ROOT / "output/figures/early_lead_github.png"))
    ap.add_argument("--out-csv", default=str(ROOT / "output/github_earlylead.csv"))
    args = ap.parse_args()

    if args.arm == "topple":
        sys.exit(
            "The 'topple' arm needs a contestable, revocable-status dataset "
            "(e.g. Wikipedia RfA adminship with de-adminship, Hacker News "
            "front-page rank, or a moderator roster with removals), not GitHub "
            "stars. Point this at such a source and reuse rho_curve()/tau90(); "
            "the model predicts tau_90 -> 1 (persistence collapses) there, versus "
            "the strong persistence measured on the free-token GitHub arm.")

    taus = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Selecting cohort: created:{args.created} "
          f"stars:{args.min_stars}..{args.max_stars} lang:{args.language}")
    cohort = search_cohort(args.created, args.min_stars, args.max_stars,
                           args.cohort, args.language, args.cache_dir)
    print(f"  {len(cohort)} repos")

    rows, kept = [], []
    for i, repo in enumerate(cohort, 1):
        try:
            t0 = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00"))
            horizon_end = t0 + timedelta(days=int(args.horizon_months * 30.44))
            dts = star_times(repo["full_name"], horizon_end, args.cache_dir)
            _, _, counts = cumulative_at_taus(repo["created_at"], dts,
                                              args.horizon_months, taus)
            if counts[-1] < 20:               # too little signal within horizon
                continue
            rows.append(counts)
            kept.append(repo["full_name"])
        except Exception as e:                # noqa: BLE001 -- skip flaky repos
            print(f"  [skip] {repo['full_name']}: {e}", file=sys.stderr)
        if i % 25 == 0:
            print(f"  processed {i}/{len(cohort)} (kept {len(rows)})")

    if len(rows) < 30:
        sys.exit(f"Only {len(rows)} usable repos; widen the cohort or the window.")

    counts = np.array(rows, dtype=float)
    obs = rho_curve(counts, taus)
    pa = pa_null_rho(counts[:, -1], taus)
    sq = np.sqrt(taus)

    print("\n{:>7} {:>8} {:>8} {:>8}".format("tau", "obs", "PA-null", "sqrt"))

    for t, o, p, s in zip(taus, obs, pa, sq):
        print(f"{t:>7.3f} {o:>8.3f} {p:>8.3f} {s:>8.3f}")

    print(f"\nN repos = {len(rows)}   horizon = {args.horizon_months:.0f} months")
    print(f"tau_90:  observed={tau90(obs, taus):.3f}   "
          f"PA-null={tau90(pa, taus):.3f}   diffusion=1.000")
    print("Reminder: obs > sqrt(tau) confirms nothing on its own. That null "
          "assumes a HOMOGENEOUS population; these repos merely share a start, "
          "and unequal quality alone clears it at g=0 (paper Eq. rhodrift). "
          "Nor does beating the PA null identify this mechanism specifically. "
          "The identifying test is the free-token vs. toppling contrast "
          "(wiki_rfa_toppling.py), which heterogeneity inflates on both arms "
          "alike and so cannot manufacture.")

    # save CSV
    header = "repo," + ",".join(f"tau_{t:g}" for t in taus)
    with open(args.out_csv, "w") as fh:
        fh.write(header + "\n")
        for name, c in zip(kept, counts):
            fh.write(name + "," + ",".join(str(int(v)) for v in c) + "\n")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(taus, obs, "o-", color="crimson", label="GitHub stars (observed)")
    ax.plot(taus, pa, "s--", color="steelblue", label="preferential-attachment null")
    ax.plot(taus, sq, "k:", label=r"diffusion null $\sqrt{\tau}$")
    ax.set_xlabel(r"early fraction of horizon $\tau$")
    ax.set_ylabel(r"rank correlation $\rho(\tau)$ with final order")
    ax.set_title(f"Early-lead persistence, GitHub stars (N={len(rows)})")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)

    print(f"\nwrote {args.out_fig}\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
