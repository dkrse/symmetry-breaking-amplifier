#!/usr/bin/env python3

"""
Design 2, toppling arm: early-lead persistence in Wikipedia Requests for
Adminship (RfA), a contested, revocable status.

The model predicts that early-lead persistence collapses when the position can be
actively contested (reverse dominance: a coalition piles opposition onto the
front-runner), in contrast to the strong persistence of a free-token accumulator
(GitHub stars; github_earlylead.py). The Stanford SNAP wiki-RfA dataset lets us
measure BOTH arms inside ONE dataset, which is the cleanest possible contrast:

  * TOPPLE arm  -- running NET support (support minus oppose). Oppose votes push
    the tally DOWN, so an early front-runner can be toppled; this is the
    revocable, contested position.
  * FREE-TOKEN reference -- running SUPPORT-ONLY cumulative count, which can only
    rise, exactly the monotone free token GitHub stars are.

For each election (candidate x year) we order its votes in time and, at each early
fraction tau of the vote sequence, record the position. Treating elections as the
cohort, rho(tau) = corr(rank position at tau, rank final position). The prediction
is rho_net(tau) << rho_supportonly(tau) at small tau, and tau_90(net) closer to 1:
the SAME community process is far less early-determined when opposition can topple
a lead than when only positive endorsements accumulate.

Data: https://snap.stanford.edu/data/wiki-RfA.html  (wiki-RfA.txt.gz, ~2 MB).
Downloaded once to scripts/data/ and cached. Network needed only for that fetch.

Usage:
    python scripts/wiki_rfa_toppling.py [--min-votes 25]
"""
import argparse
import gzip
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent          # repo root (CWD-independent)
DATA_URL = "https://snap.stanford.edu/data/wiki-RfA.txt.gz"
DATA_PATH = Path(__file__).resolve().parent / "data" / "wiki-RfA.txt.gz"
TAUS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])


def ensure_data():
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f">> downloading {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    except Exception as e:                                   # noqa: BLE001
        sys.exit(f"ERROR downloading wiki-RfA data: {e}\n"
                 f"Fetch it manually to {DATA_PATH} from {DATA_URL}")


def parse_elections(min_votes):
    """Yield (key, votes) where votes = list of (datetime, vote_int) sorted."""
    elections = {}
    rec = {}
    opener = gzip.open(DATA_PATH, "rt", encoding="latin-1")
    with opener as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "" and rec:
                _add_record(elections, rec)
                rec = {}
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                rec[k.strip()] = v.strip()
        if rec:
            _add_record(elections, rec)
    out = {}
    for key, votes in elections.items():
        votes = [v for v in votes if v[0] is not None]
        if len(votes) >= min_votes:
            votes.sort(key=lambda z: z[0])
            out[key] = votes
    return out


def _add_record(elections, rec):
    try:
        tgt, yea = rec.get("TGT", ""), rec.get("YEA", "")
        vot = int(rec.get("VOT", "0"))
        dat = _parse_date(rec.get("DAT", ""))
    except (ValueError, KeyError):
        return
    if not tgt or not yea:
        return
    elections.setdefault((tgt, yea), []).append((dat, vot))


def _parse_date(s):
    # SNAP format e.g. "19:53, 25 January 2013"
    for fmt in ("%H:%M, %d %B %Y", "%H:%M, %d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def positions_at_taus(votes):
    """Return (net_at_tau, supportonly_at_tau) arrays over TAUS for one election."""
    v = np.array([x[1] for x in votes], dtype=float)
    n = len(v)
    net_cum = np.cumsum(v)                        # can go down (opposes)
    supp_cum = np.cumsum(np.clip(v, 0, None))     # monotone free token
    net, supp = [], []
    for tau in TAUS:
        idx = max(int(round(tau * n)) - 1, 0)
        idx = min(idx, n - 1)
        net.append(net_cum[idx])
        supp.append(supp_cum[idx])
    return np.array(net), np.array(supp)


def rho_curve(cols):
    final = cols[:, -1]
    return np.array([spearmanr(cols[:, j], final).statistic
                     for j in range(cols.shape[1])])


def tau90(rhos):
    hit = np.where(rhos >= 0.9)[0]
    return float(TAUS[hit[0]]) if len(hit) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-votes", type=int, default=25,
                    help="minimum votes for an election to enter the cohort")
    ap.add_argument("--out-fig", default=str(ROOT / "output/figures/early_lead_wiki_rfa.png"))
    ap.add_argument("--out-csv", default=str(ROOT / "output/wiki_rfa_earlylead.csv"))
    args = ap.parse_args()

    ensure_data()
    elections = parse_elections(args.min_votes)
    if len(elections) < 30:
        sys.exit(f"Only {len(elections)} elections with >= {args.min_votes} "
                 f"votes; lower --min-votes.")

    net_rows, supp_rows, keys = [], [], []
    for key, votes in elections.items():
        net, supp = positions_at_taus(votes)
        net_rows.append(net)
        supp_rows.append(supp)
        keys.append(key)
    net_cols = np.array(net_rows, dtype=float)
    supp_cols = np.array(supp_rows, dtype=float)

    rho_net = rho_curve(net_cols)
    rho_supp = rho_curve(supp_cols)
    sq = np.sqrt(TAUS)

    print(f"\nWikipedia RfA cohort: {len(keys)} elections "
          f"(>= {args.min_votes} votes each)\n")
    print("{:>7} {:>10} {:>12} {:>8}".format("tau", "NET(topple)", "SUPP(free)", "sqrt"))
    for t, rn, rs, s in zip(TAUS, rho_net, rho_supp, sq):
        print(f"{t:>7.3f} {rn:>10.3f} {rs:>12.3f} {s:>8.3f}")
    print(f"\ntau_90:  net(toppling)={tau90(rho_net):.3f}   "
          f"support-only(free-token)={tau90(rho_supp):.3f}")
    i_early = TAUS <= 0.15
    print(f"mean early (tau<=0.15) persistence:  "
          f"net={rho_net[i_early].mean():.3f}   support-only={rho_supp[i_early].mean():.3f}")
    print("Prediction: net << support-only early, and tau_90(net) larger -- the "
          "same community process is far less early-determined when a lead can be "
          "toppled by opposition than when only endorsements accumulate. This is "
          "the free-token/toppling contrast the amplifier predicts, measured "
          "inside one dataset and with no capability variable.")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as fh:
        fh.write("candidate,year,arm," + ",".join(f"tau_{t:g}" for t in TAUS) + "\n")
        for (tgt, yea), row in zip(keys, net_cols):
            fh.write(f"{tgt},{yea},net," + ",".join(f"{v:g}" for v in row) + "\n")
        for (tgt, yea), row in zip(keys, supp_cols):
            fh.write(f"{tgt},{yea},support," + ",".join(f"{v:g}" for v in row) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(TAUS, rho_supp, "s-", color="seagreen",
            label="support-only (free-token reference)")
    ax.plot(TAUS, rho_net, "o-", color="crimson",
            label="net support (revocable / toppling)")
    ax.plot(TAUS, sq, "k:", label=r"diffusion null $\sqrt{\tau}$")
    ax.set_xlabel(r"early fraction of the vote sequence $\tau$")
    ax.set_ylabel(r"rank correlation $\rho(\tau)$ with final order")
    ax.set_title(f"Wikipedia RfA early-lead persistence (N={len(keys)} elections)")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)

    print(f"\nwrote {args.out_fig}\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
