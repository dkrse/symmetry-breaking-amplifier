#!/usr/bin/env python3

"""
Third real domain for Test B: startup funding trajectories (early-lead persistence).

Free, open data: the Crunchbase December-2015 export mirrored at
https://github.com/notpeter/crunchbase-data (rounds.csv), downloaded once to
scripts/data/crunchbase_rounds.csv. Columns used:
  company_permalink, funded_at (YYYY-MM-DD), raised_amount_usd, funding_round_type.

Mapping to the model. A startup's position is its CUMULATIVE capital raised -- a
self-reinforcing, essentially free-token stock (funding is not revoked, so this is
the free-token pole, like GitHub stars). Entrants start near-equal at their first
(seed/early) round. We take the cohort of companies whose FIRST round falls in one
year and that raise at least twice, clock each company from its own first round
over a fixed horizon, and ask the capability-free early-lead question (Eq. rhotau):

    rho(tau) = corr( rank cumulative-capital at fraction tau of the horizon,
                     rank cumulative-capital at the horizon ).

This is measured on a FINE tau grid (18 points), finer than the bundled
GitHub/Wikipedia curves, to resolve the small-tau lock-in.

Prediction: as a free token, startups should lock in early (rho(tau) well above
sqrt(tau), tau_90 small-to-moderate) and, after tau_90 rescaling, sit near the
amplifier master and the other free-token systems (see real_data_collapse.py).
No capability/quality variable is used.
"""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "data" / "crunchbase_rounds.csv"
OUTCSV = ROOT / "output" / "startup_earlylead.csv"

# fine tau grid (finer than the 10-point GitHub/Wiki grids)
TAUS = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25,
                 0.3, 0.375, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0])

COHORT_YEAR = "2010"        # first-round year -> near-equal entry cohort
HORIZON_DAYS = 60 * 30.44   # ~60 months of runway, data extends to end-2015
MIN_ROUNDS = 2              # need a trajectory, not a single point


def _ordinal(d):
    """Parse YYYY-MM-DD to a day ordinal; None if unparseable."""
    if len(d) < 10 or d[4] != "-" or d[7] != "-":
        return None
    try:
        y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
        # cheap ordinal: days since year 0 (month-length approx is fine for deltas
        # within a company, but use a real calendar to be exact)
        import datetime
        return datetime.date(y, m, dd).toordinal()
    except ValueError:
        return None


def load_company_rounds():
    """company_permalink -> list of (day_ordinal, amount_usd)."""
    comp = {}
    with open(DATA, newline="") as f:
        for r in csv.DictReader(f):
            o = _ordinal(r["funded_at"])
            if o is None:
                continue
            amt = r["raised_amount_usd"].strip()
            amt = float(amt) if amt not in ("", "-") else 0.0
            comp.setdefault(r["company_permalink"], []).append((o, amt))

    return comp


def build_cohort(comp):
    """Companies whose first round is in COHORT_YEAR with >= MIN_ROUNDS rounds.
    Return positions matrix (n_companies, n_taus): cumulative capital at each tau
    of the per-company horizon."""
    rows = []
    for c, rounds in comp.items():
        rounds.sort()

        if len(rounds) < MIN_ROUNDS:
            continue

        first_o = rounds[0][0]
        import datetime

        if datetime.date.fromordinal(first_o).year != int(COHORT_YEAR):
            continue

        days = np.array([o - first_o for o, _ in rounds], dtype=float)
        amts = np.array([a for _, a in rounds], dtype=float)
        cum = np.cumsum(amts)
        # cumulative capital raised by time first + tau*horizon
        pos = []

        for tau in TAUS:
            cutoff = tau * HORIZON_DAYS
            idx = np.searchsorted(days, cutoff, side="right") - 1
            pos.append(cum[idx] if idx >= 0 else 0.0)
        rows.append(pos)

    return np.array(rows, dtype=float)


def _rank(a):
    return np.asarray(a).argsort().argsort().astype(float)


def rho_tau(P):
    rf = _rank(P[:, -1])
    return np.array([np.corrcoef(_rank(P[:, j]), rf)[0, 1] for j in range(P.shape[1])])


def tau90(rhos):
    for i in range(len(TAUS)):
        if rhos[i] >= 0.9:
            if i == 0:
                return float(TAUS[0])
            r0, r1, t0, t1 = rhos[i - 1], rhos[i], TAUS[i - 1], TAUS[i]
            return float(t0 if r1 == r0 else t0 + (0.9 - r0) * (t1 - t0) / (r1 - r0))

    return 1.0


def run():
    if not DATA.exists():
        raise SystemExit(
            f"missing {DATA}\n  download once (open/free, no login):\n"
            "  curl -sL -o scripts/data/crunchbase_rounds.csv \\\n"
            "    https://raw.githubusercontent.com/notpeter/crunchbase-data/master/rounds.csv")

    comp = load_company_rounds()
    P = build_cohort(comp)
    r = rho_tau(P)

    print("=== Startup funding early-lead persistence (Crunchbase 2015, open) ===")
    print(f"cohort: first round in {COHORT_YEAR}, >= {MIN_ROUNDS} rounds "
          f"-> {P.shape[0]} companies; horizon ~60 months; free-token (cumulative capital)\n")
    print(f"{'tau':>7} {'rho':>7} {'sqrt':>7}")

    for t, rr in zip(TAUS, r):
        print(f"{t:>7.3f} {rr:>7.3f} {np.sqrt(t):>7.3f}")
    print(f"\ntau_90 = {tau90(r):.3f}   (sqrt-null would need tau_90={0.9**2:.2f})")
    print(f"rho(0.1) = {np.interp(0.1, TAUS, r):.3f}   vs sqrt(0.1)={np.sqrt(0.1):.3f}")

    # save positions CSV in the same schema family as the other real curves
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTCSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["company_idx"] + [f"tau_{t:g}" for t in TAUS])
        for i, row in enumerate(P):
            w.writerow([i] + [f"{v:.1f}" for v in row])
    print(f"\npositions -> {OUTCSV}")

    return dict(taus=TAUS, rho=r, tau90=tau90(r), n=P.shape[0])


if __name__ == "__main__":
    run()
