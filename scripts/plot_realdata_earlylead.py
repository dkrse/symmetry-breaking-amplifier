#!/usr/bin/env python3

"""
Combined real-data early-lead-persistence figure (Design 2, both poles).

Reads the CSVs written by the two --online arms and plots rho(tau) for all three
curves on one axis against the sqrt(tau) diffusion null:

  * GitHub stars            -- free-token pole (github_earlylead.py)
  * Wikipedia RfA net       -- revocable / toppling position (wiki_rfa_toppling.py)
  * Wikipedia RfA support   -- free-token reference within the same RfA dataset

The scientifically valid contrast is the WITHIN-Wikipedia pair (net vs support),
where revocation is toggled inside one dataset; GitHub is shown as an independent
free-token pole, not as a controlled comparison against Wikipedia (their tau axes
and position definitions are not commensurable). The figure is annotated to say so.

Usage: python scripts/plot_realdata_earlylead.py   (needs both --online CSVs)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TAUS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])
GH_CSV = ROOT / "output" / "github_earlylead.csv"
WIKI_CSV = ROOT / "output" / "wiki_rfa_earlylead.csv"
OUT = ROOT / "output" / "figures" / "early_lead_realdata.png"



def _rows_last10(path, arm=None, arm_field=-11):
    """Parse rows; take the last 10 comma-fields as the tau positions.
    Robust to commas inside entity names. If arm is given, keep only rows whose
    field at arm_field equals it."""
    out = []
    with open(path) as fh:
        next(fh)                                   # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 11:
                continue
            if arm is not None and parts[arm_field].strip() != arm:
                continue
            try:
                out.append([float(x) for x in parts[-10:]])
            except ValueError:
                continue

    return np.array(out, dtype=float)


def rho_curve(cols):
    final = cols[:, -1]
    return np.array([spearmanr(cols[:, j], final).statistic
                     for j in range(cols.shape[1])])


def main():
    for p in (GH_CSV, WIKI_CSV):
        if not p.exists():
            sys.exit(f"missing {p}; run the --online arms first "
                     f"(github_earlylead.py / wiki_rfa_toppling.py).")

    gh = _rows_last10(GH_CSV)
    wiki_net = _rows_last10(WIKI_CSV, arm="net")
    wiki_sup = _rows_last10(WIKI_CSV, arm="support")

    r_gh = rho_curve(gh)
    r_net = rho_curve(wiki_net)
    r_sup = rho_curve(wiki_sup)
    sq = np.sqrt(TAUS)

    print(f"GitHub free-token   (N={len(gh)}):   rho(0.1)={r_gh[2]:.3f}  tau90-ish")
    print(f"Wiki support-only   (N={len(wiki_sup)}):   rho(0.1)={r_sup[2]:.3f}")
    print(f"Wiki net (revocable)(N={len(wiki_net)}):   rho(0.1)={r_net[2]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(TAUS, r_sup, "s-", color="seagreen",
            label=f"Wikipedia RfA, support-only — free-token ref (N={len(wiki_sup)})")
    ax.plot(TAUS, r_net, "o-", color="crimson",
            label=f"Wikipedia RfA, net support — revocable/toppling (N={len(wiki_net)})")
    ax.plot(TAUS, r_gh, "^-", color="steelblue",
            label=f"GitHub stars — free-token pole (N={len(gh)})")
    ax.plot(TAUS, sq, "k:", label=r"diffusion null $\sqrt{\tau}$")
    ax.set_xlabel(r"early fraction of the horizon $\tau$")
    ax.set_ylabel(r"rank correlation $\rho(\tau)$ with final order")
    ax.set_title("Early-lead persistence on real data (Design 2, capability-free)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(0.335, 0.37,
            "Controlled contrast is WITHIN Wikipedia\n"
            "(support-only vs net support). GitHub is an\n"
            "independent free-token pole, not a like-for-\n"
            "like comparison (tau axes / position\n"
            "definitions differ).",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            color="0.4")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
