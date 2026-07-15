#!/usr/bin/env python3

"""
Test B on REAL DATA: do real systems' early-lead curves collapse onto one shape?

The simulation (data_collapse.py) shows that IF several systems are the same
multiplicative normal form, their rho(tau) curves collapse under tau_90 rescaling
onto one master, while diffusion and preferential attachment fall off it. That is
a property of the equation. This script runs the collapse on THREE independent
real free-token domains plus a revocable arm, all free/open data:

  * Startup funding  -- cumulative capital raised, first-round-2010 cohort, 2325
                        companies, FINE 18-point tau grid
                        (Crunchbase 2015 open export; startup_earlylead.py)
  * GitHub stars     -- cumulative star stock, 295 repos
                        (output/github_earlylead.csv)
  * Wikipedia RfA    -- support-only cumulative count = free token
                        (output/wiki_rfa_earlylead.csv, arm=support)
  * Wikipedia RfA    -- NET support (support-oppose) = revocable/toppling position
                        (same CSV, arm=net)

Each system stores per-entity positions at fractions tau of its own horizon, so
the tau axes are already internally normalised; the tau grids need NOT match
because the collapse rescales each curve by its own tau_90 and interpolates the
rescaled shape onto a common axis. That rescaling is exactly what makes the
otherwise incommensurable real axes (months of star stock vs a days-long vote
sequence vs years of funding) comparable -- the comparison the paper (Sec. 'First
real-data evidence') declined to make on raw tau_90.

Two tests:
  (1) collapse onto the SIMULATED amplifier master (calibration-dependent);
  (2) mutual collapse of the three real FREE-TOKEN domains onto a data-derived
      master, vs the sqrt(tau) null and the revocable arm.
The verdict is left to the numbers; nothing is forced to pass.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_collapse import sim_amp, FAMILY

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures"


def _prefer_fine(name):
    """Use the fine-grid re-extraction (refine_grids.py) when present, else the
    bundled coarse CSV."""
    fine = ROOT / "output" / f"{name}_fine.csv"
    return fine if fine.exists() else ROOT / "output" / f"{name}.csv"


GH_CSV = _prefer_fine("github_earlylead")
WIKI_CSV = _prefer_fine("wiki_rfa_earlylead")
STARTUP_CSV = ROOT / "output" / "startup_earlylead.csv"
T_SIM = 600
UGRID = np.linspace(0.05, 1.0, 40)


# --------------------------------------------------------------------------
def _rank(a):
    return np.asarray(a).argsort().argsort().astype(float)


def load_positions(path, filt=None):
    """Read a positions CSV. Return (P, taus): P is (n_rows, n_taus) over the
    tau_* columns, taus the parsed grid. Keep rows where filt(row) is True."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        tau_cols = [c for c in rd.fieldnames if c.startswith("tau_")]
        taus = np.array([float(c[4:]) for c in tau_cols])
        rows = []
        for row in rd:
            if filt is not None and not filt(row):
                continue
            rows.append([float(row[c]) for c in tau_cols])

    return np.array(rows, dtype=float), taus


def rho_of(P):
    rf = _rank(P[:, -1])
    return np.array([np.corrcoef(_rank(P[:, j]), rf)[0, 1] for j in range(P.shape[1])])


def tau90(rhos, taus):
    for i in range(len(taus)):
        if rhos[i] >= 0.9:
            if i == 0:
                return float(taus[0])
            r0, r1, t0, t1 = rhos[i - 1], rhos[i], taus[i - 1], taus[i]
            return float(t0 if r1 == r0 else t0 + (0.9 - r0) * (t1 - t0) / (r1 - r0))

    return 1.0


def rescale(rhos, taus):
    t90 = tau90(rhos, taus)
    u = taus / t90
    o = np.argsort(u)
    return np.interp(UGRID, u[o], rhos[o]), t90


def het_master(seed=20260709):
    """Mean rescaled curve of the g=0 heterogeneity null (paper Eq. rhodrift).

    Entities start together but differ in drift, so persistence exceeds sqrt(tau)
    with no amplification at all. In simulation this null does NOT collapse
    (data_collapse.py), so shape should exclude it. On the real data it is not
    excluded -- reported here rather than omitted, since it counts against the
    amplifier reading.
    """
    from data_collapse import sim_het, HET_VAR_A

    rng = np.random.default_rng(seed)
    taus = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])
    resc = []
    for va in HET_VAR_A:
        tr = sim_het(rng, var_a=va)
        rf = _rank(tr[-1])
        r = np.array([np.corrcoef(_rank(tr[min(max(int(round(t * T_SIM)), 1), T_SIM)]),
                                  rf)[0, 1] for t in taus])
        resc.append(rescale(r, taus)[0])
    resc = np.array(resc)
    m = resc.mean(axis=0)
    return m, float(np.sqrt(((resc - m) ** 2).mean()))


def sim_master(seed=20260709):
    rng = np.random.default_rng(seed)
    taus = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])
    resc = []
    for cfg in FAMILY:
        tr = sim_amp(rng, g=cfg["g"], sigma=cfg["sigma"], cap=cfg["cap"])
        rf = _rank(tr[-1])
        r = np.array([np.corrcoef(_rank(tr[min(max(int(round(t * T_SIM)), 1), T_SIM)]),
                                  rf)[0, 1] for t in taus])
        resc.append(rescale(r, taus)[0])
    resc = np.array(resc)
    m = resc.mean(axis=0)
    return m, float(np.sqrt(((resc - m) ** 2).mean())), resc


def _rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def run():
    # --- load the real domains (each on its own native tau grid) ---
    sources = {}
    if STARTUP_CSV.exists():
        P, t = load_positions(STARTUP_CSV)
        sources["Startups (free)"] = (rho_of(P), t, "free")
    else:
        print(">> startup_earlylead.csv missing; run startup_earlylead.py first "
              "(downloads the open Crunchbase rounds.csv). Skipping startups.")
    Pg, tg = load_positions(GH_CSV)
    sources["GitHub stars (free)"] = (rho_of(Pg), tg, "free")
    Pw, tw = load_positions(WIKI_CSV, filt=lambda r: r["arm"] == "support")
    sources["Wiki support (free)"] = (rho_of(Pw), tw, "free")
    Pn, tn = load_positions(WIKI_CSV, filt=lambda r: r["arm"] == "net")
    sources["Wiki net (revocable)"] = (rho_of(Pn), tn, "revoc")

    master, internal, fam_resc = sim_master()
    sqrt_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    sqrt_resc, _ = rescale(np.sqrt(sqrt_grid), sqrt_grid)

    print("=== Test B on REAL DATA (3 free-token domains + revocable) ===\n")
    print(f"simulated amplifier-family internal spread : {internal:.4f}\n")
    print(f"{'domain':<24}{'tau90':>7}{'->sim master':>14}{'  xInt':>8}")
    resc = {}
    for name, (r, t, kind) in sources.items():
        rr, t90 = rescale(r, t)
        resc[name] = (rr, kind)
        d = _rms(rr, master)
        print(f"{name:<24}{t90:>7.3f}{d:>14.4f}{d/internal:>7.1f}x")
    d_sqrt = _rms(sqrt_resc, master)
    print(f"{'sqrt(tau) null':<24}{'0.810':>7}{d_sqrt:>14.4f}{d_sqrt/internal:>7.1f}x")

    # --- data-derived master over the real FREE-TOKEN domains ---
    free = [rr for name, (rr, k) in resc.items() if k == "free"]
    free_master = np.mean(free, axis=0)
    free_int = float(np.sqrt(((np.array(free) - free_master) ** 2).mean()))
    d_net = _rms(resc["Wiki net (revocable)"][0], free_master)
    d_sq2 = _rms(sqrt_resc, free_master)
    print(f"\n--- data-derived master over the {len(free)} real free-token domains ---")
    print(f"  mutual spread of the free-token curves : {free_int:.4f}")
    print(f"  sqrt(tau) null distance to it          : {d_sq2:.4f}"
          f"   ({d_sq2/free_int:.1f}x mutual spread)")
    print(f"  revocable (Wiki net) distance to it    : {d_net:.4f}"
          f"   ({d_net/free_int:.1f}x mutual spread)")
    hm, h_int = het_master()
    d_het = _rms(hm, free_master)
    print(f"  heterogeneity null (g=0) distance to it: {d_het:.4f}"
          f"   ({d_het/free_int:.1f}x mutual spread)")
    if d_het < free_int:
        print("    ^ NEARER than the domains are to each other: on these data the")
        print("      collapse does NOT discriminate against mere heterogeneity.")
        print("      (that null carries a free parameter the amplifier family does not,")
        print("       so this is not a like-for-like defeat -- but it is not excluded)")
    print("\nreading: three INDEPENDENT free-token domains (startups, stars, Wiki")
    print("  support) collapsing onto each other, and far above the sqrt null, is")
    print("  the universality signal; the sim master is calibration-dependent so the")
    print("  data-derived collapse is the honest test. The revocable arm is expected")
    print("  further out, but on mild Wikipedia toppling that separation is weak.")

    # ---- figure ----
    style = {"Startups (free)": ("D-", "darkorange"),
             "GitHub stars (free)": ("o-", "steelblue"),
             "Wiki support (free)": ("s-", "seagreen"),
             "Wiki net (revocable)": ("^--", "crimson")}
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for name, (r, t, _) in sources.items():
        st, c = style[name]
        ax[0].plot(t, r, st, c=c, lw=1.5, ms=4, label=name)

    fine = np.linspace(0.01, 1, 100)
    ax[0].plot(fine, np.sqrt(fine), "k--", lw=1.3, label=r"$\sqrt{\tau}$ null")
    ax[0].axhline(0.9, ls=":", c="grey", lw=0.8)
    ax[0].set(xlabel=r"early fraction $\tau$", ylabel=r"$\rho(\tau)$",
              title="Raw: real early-lead curves"); ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=7, loc="lower right")


    for rr in fam_resc:
        ax[1].plot(UGRID, rr, "-", c="lightsteelblue", lw=0.9, alpha=0.6)
    ax[1].plot(UGRID, free_master, "-", c="black", lw=2.4, label="free-token master (data)")
    ax[1].plot(UGRID, sqrt_resc, "k--", lw=1.5, label=r"$\sqrt{\tau}$ null")

    for name, (rr, _) in resc.items():
        st, c = style[name]
        ax[1].plot(UGRID, rr, st, c=c, lw=1.5, ms=3, markevery=5, label=name)
    ax[1].axhline(0.9, ls=":", c="grey", lw=0.8)
    ax[1].set(xlabel=r"rescaled fraction $\tau/\tau_{90}$", ylabel=r"$\rho$",
              title="Rescaled: do the free-token domains collapse?")
    ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "real_data_collapse.png", dpi=200)
    plt.close(fig)
    print(f"\nfigure -> {OUT / 'real_data_collapse.png'}")


if __name__ == "__main__":
    run()
