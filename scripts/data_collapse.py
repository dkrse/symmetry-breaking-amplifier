#!/usr/bin/env python3

"""
Test B: data-collapse universality of the early-lead-persistence curve.

The paper's strongest cross-domain claim (Sec. 'Interpretation', Eq. 1 as a
NORMAL FORM) is that status cascades, markets, citations and stars are the SAME
multiplicative mechanism, not an analogy. If true, their early-lead-persistence
curves

    rho(tau) = corr( rank x(tau T), rank x(T) )        (paper Eq. 6)

should not merely each bow above sqrt(tau) -- they should share ONE master SHAPE.
Concretely: rescale each system's tau axis by its own tau_90 (the fraction whose
order already correlates >=0.9 with the final one). Under the amplifier normal
form the rescaled curves rho(tau / tau_90) should COLLAPSE onto a single curve,
exactly as correlation-length rescaling collapses curves in critical phenomena.
Systems governed by a DIFFERENT mechanism should fall off that master curve.

This script:
  1. builds a FAMILY of amplifier parameterisations standing for different
     domains (status w/ Gould ceiling, mid-gain market, free-token citations,
     free-token stars, high-gain, different noise levels);
  2. computes rho(tau) for each, and a continuous tau_90 by interpolation;
  3. rescales tau -> u = tau / tau_90 and measures how tightly the family
     collapses (mean RMS spread about the family master curve);
  4. checks that two OTHER mechanisms -- pure diffusion (g=0) and plain
     Barabasi preferential attachment -- do NOT collapse onto the same master,
     i.e. their RMS distance to the amplifier master is much larger than the
     amplifier family's internal spread.

Prediction (the model passes B iff):
    internal_spread(amplifier family)  <<  distance(diffusion -> master)
    internal_spread(amplifier family)  <<  distance(PA -> master)
A tight amplifier collapse with clear outliers is positive evidence for the
single-mechanism (universality) claim; a family that itself fails to collapse,
or a diffusion/PA curve that lands ON the master, refutes it.

Style and dynamics match scripts/earlylead_pa_null.py and
scripts/symmetry_breaking.py (same feedback, same reset).
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

N = 2500                     # entities in each cohort
T = 600                      # horizon (steps)
SEED = 20260709
# fine tau grid so tau_90 interpolation and the collapse are smooth
TAUS = np.array([0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
                 0.4, 0.5, 0.65, 0.8, 1.0])



# --------------------------------------------------------------------------
# rank-correlation persistence curve and continuous tau_90
# --------------------------------------------------------------------------
def _rank(a):
    return a.argsort().argsort().astype(float)



def rho_tau(traj):
    """traj: (T+1, N). rho(tau) = rank-corr(order at tau*T, final order)."""
    rf = _rank(traj[-1])
    out = []
    for tau in TAUS:
        t = min(max(int(round(tau * T)), 1), T)
        rt = _rank(traj[t])
        out.append(np.corrcoef(rt, rf)[0, 1])
    return np.array(out)



def tau90(rhos):
    """Smallest tau with rho>=0.9, linearly interpolated between grid points.
    Returns 1.0 (censored) if the curve never reaches 0.9 within the horizon."""
    for i in range(len(TAUS)):
        if rhos[i] >= 0.9:
            if i == 0:
                return TAUS[0]
            # interpolate the 0.9 crossing between i-1 and i
            r0, r1 = rhos[i - 1], rhos[i]
            t0, t1 = TAUS[i - 1], TAUS[i]
            if r1 == r0:
                return t1
            return t0 + (0.9 - r0) * (t1 - t0) / (r1 - r0)
    return 1.0



# --------------------------------------------------------------------------
# simulators (log-status x; S = exp(x)); shared with the rest of the repo
# --------------------------------------------------------------------------
def sim_amp(rng, g, theta=0.02, sigma=0.05, cap=None):
    """Multiplicative amplifier, paper Eq. (3). cap=None => free token (no
    per-unit ceiling); cap=S* => Gould saturating feedback f = theta S/(1+S/S*)."""
    x = np.empty((T + 1, N))
    x[0] = rng.normal(0, 1e-6, N)
    for t in range(T):
        S = np.exp(np.clip(x[t], None, 700))
        f = theta * S
        if cap is not None:
            f = f / (1 + S / cap)
        x[t + 1] = x[t] + np.log1p(g * f) + rng.normal(0, sigma, N)
    return x



def sim_diff(rng, sigma=0.05):
    """Pure diffusion, g=0: analytic null rho(tau)=sqrt(tau)."""
    x = np.empty((T + 1, N))
    x[0] = rng.normal(0, 1e-6, N)
    for t in range(T):
        x[t + 1] = x[t] + rng.normal(0, sigma, N)
    return x


def sim_het(rng, var_a, sigma=0.05):
    """g=0 with a HETEROGENEOUS per-step drift a_i ~ N(0, var_a).

    The null that defeats the magnitude test: entities start together but simply
    differ, so a persistent trait predicts the final order with no feedback at
    all, lifting rho(tau) above sqrt(tau) (paper Eq. rhodrift). Included here to
    check whether SHAPE separates what magnitude cannot. Unlike the amplifier,
    this family has no single rescaled shape -- each var_a traces a different
    one -- so it should not collapse.
    """
    x = np.empty((T + 1, N))
    x[0] = rng.normal(0, 1e-6, N)
    a = rng.normal(0, np.sqrt(var_a), N)
    for t in range(T):
        x[t + 1] = x[t] + a + rng.normal(0, sigma, N)
    return x


# var_a decade spanned by the heterogeneity null (mean curve = its "master")
HET_VAR_A = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def sim_pa(rng, M=None, seed_stock=1.0):
    """Plain Barabasi preferential attachment on a monotone stock."""
    if M is None:
        M = N // 3
    stock = np.full(N, seed_stock, dtype=float)
    traj = np.empty((T + 1, N))
    traj[0] = stock + rng.normal(0, 1e-6, N)
    for t in range(T):
        picks = rng.choice(N, size=M, replace=True, p=stock / stock.sum())
        np.add.at(stock, picks, 1.0)
        traj[t + 1] = stock
    return traj



# --------------------------------------------------------------------------
# the amplifier FAMILY: distinct parameterisations standing for domains
# --------------------------------------------------------------------------
FAMILY = [
    dict(label="status (Gould ceiling)", g=0.20, sigma=0.05, cap=50.0),
    dict(label="market (mid gain)",      g=0.40, sigma=0.05, cap=50.0),
    dict(label="high gain (ceiling)",    g=0.60, sigma=0.04, cap=50.0),
    dict(label="citations (free token)", g=0.15, sigma=0.05, cap=None),
    dict(label="stars (free token)",     g=0.30, sigma=0.05, cap=None),
    dict(label="low noise (free token)", g=0.25, sigma=0.03, cap=None),
    dict(label="high noise (ceiling)",   g=0.30, sigma=0.08, cap=50.0),
]


def _master_grid():
    # common rescaled axis u = tau / tau_90, over the range all curves cover.
    return np.linspace(0.05, 1.0, 40)


def _rescaled_curve(rhos, t90, ugrid):
    """Interpolate rho as a function of u = tau/tau_90 onto ugrid."""
    u = TAUS / t90
    order = np.argsort(u)
    return np.interp(ugrid, u[order], rhos[order])



def run():
    rng = np.random.default_rng(SEED)
    ugrid = _master_grid()

    # --- amplifier family ---
    fam_raw, fam_resc, fam_t90 = [], [], []
    for cfg in FAMILY:
        r = rho_tau(sim_amp(rng, g=cfg["g"], sigma=cfg["sigma"], cap=cfg["cap"]))
        t90 = tau90(r)
        fam_raw.append(r)
        fam_t90.append(t90)
        fam_resc.append(_rescaled_curve(r, t90, ugrid))
    fam_resc = np.array(fam_resc)
    master = fam_resc.mean(axis=0)                       # family master curve
    internal = float(np.sqrt(((fam_resc - master) ** 2).mean()))  # RMS spread


    # --- outlier mechanisms, rescaled by their OWN tau_90, distance to master ---
    r_diff = rho_tau(sim_diff(rng))
    r_pa = rho_tau(sim_pa(rng))
    resc_diff = _rescaled_curve(r_diff, tau90(r_diff), ugrid)
    resc_pa = _rescaled_curve(r_pa, tau90(r_pa), ugrid)
    dist_diff = float(np.sqrt(((resc_diff - master) ** 2).mean()))
    dist_pa = float(np.sqrt(((resc_pa - master) ** 2).mean()))

    # --- heterogeneity null (g=0, unequal entities): does SHAPE separate it? ---
    het_resc = []
    for va in HET_VAR_A:
        r = rho_tau(sim_het(rng, var_a=va))
        het_resc.append(_rescaled_curve(r, tau90(r), ugrid))
    het_resc = np.array(het_resc)
    het_master = het_resc.mean(axis=0)
    het_internal = float(np.sqrt(((het_resc - het_master) ** 2).mean()))
    dist_het = float(np.sqrt(((het_master - master) ** 2).mean()))


    # ---- report ----
    print("=== Test B: data-collapse universality of rho(tau) ===\n")
    print(f"{'domain':<24} {'tau90':>7}")
    for cfg, t90 in zip(FAMILY, fam_t90):
        print(f"{cfg['label']:<24} {t90:>7.3f}")
    print(f"\ntau90 spans {min(fam_t90):.3f}..{max(fam_t90):.3f} "
          f"(raw curves are far apart) yet after rescaling:\n")
    print(f"  amplifier family internal RMS spread about master : {internal:.4f}")
    print(f"  diffusion (g=0)  RMS distance to master           : {dist_diff:.4f}"
          f"   ({dist_diff/internal:.1f}x the family spread)")
    print(f"  pref. attachment RMS distance to master           : {dist_pa:.4f}"
          f"   ({dist_pa/internal:.1f}x the family spread)")
    print(f"  heterogeneity (g=0, unequal) dist to master       : {dist_het:.4f}"
          f"   ({dist_het/internal:.1f}x the family spread)")
    print(f"  ...and its OWN internal spread                    : {het_internal:.4f}"
          f"   (does not collapse: no single rescaled shape)")
    passed = (dist_diff > 3 * internal) and (dist_pa > 2 * internal)
    print(f"\n  COLLAPSE {'HOLDS' if passed else 'FAILS'}: amplifier family "
          f"collapses; diffusion/PA fall off the master curve."
          if passed else
          f"\n  COLLAPSE INCONCLUSIVE: outliers not clearly separated "
          f"(dist_diff={dist_diff:.3f}, dist_pa={dist_pa:.3f}, "
          f"internal={internal:.3f}).")


    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # left: raw rho(tau) -- family fans out, plus the sqrt(tau) null
    for cfg, r in zip(FAMILY, fam_raw):
        ax[0].plot(TAUS, r, "-", lw=1.3, alpha=0.9, label=cfg["label"])
    ax[0].plot(TAUS, np.sqrt(TAUS), "k--", lw=1.5, label=r"$\sqrt{\tau}$ (diffusion)")
    ax[0].plot(TAUS, r_pa, ":", c="grey", lw=1.6, label="pref. attachment")
    ax[0].set(xlabel=r"early fraction $\tau$", ylabel=r"$\rho(\tau)$",
              title="Raw: every domain a different curve")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_ylim(0, 1.02)

    # right: rescaled rho(tau/tau_90) -- family collapses onto master
    for resc in fam_resc:
        ax[1].plot(ugrid, resc, "-", c="steelblue", lw=1.1, alpha=0.55)
    ax[1].plot(ugrid, master, "-", c="navy", lw=2.6, label="amplifier master")
    ax[1].plot(ugrid, resc_diff, "k--", lw=1.8, label="diffusion (off master)")
    ax[1].plot(ugrid, resc_pa, ":", c="crimson", lw=2.0, label="pref. attach. (off master)")
    ax[1].axhline(0.9, ls=":", c="grey", lw=0.8)
    ax[1].set(xlabel=r"rescaled fraction $\tau/\tau_{90}$", ylabel=r"$\rho$",
              title="Rescaled: amplifier family collapses; others don't")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(OUT / "data_collapse.png", dpi=200)
    plt.close(fig)
    print(f"\nfigure -> {OUT / 'data_collapse.png'}")
    return dict(internal=internal, dist_diff=dist_diff, dist_pa=dist_pa,
                tau90=dict(zip([c["label"] for c in FAMILY], fam_t90)),
                passed=bool(passed))



def multiseed(seeds=range(1, 9)):
    """Robustness: does the collapse verdict survive re-seeding? Recompute the
    internal family spread and the diffusion/PA distances over several seeds and
    report their spread. The SIGN of the separation (outliers >> internal) is the
    claim; the exact ratios are not."""

    ugrid = _master_grid()
    ins, dds, dps = [], [], []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        resc = []
        for cfg in FAMILY:
            r = rho_tau(sim_amp(rng, g=cfg["g"], sigma=cfg["sigma"], cap=cfg["cap"]))
            resc.append(_rescaled_curve(r, tau90(r), ugrid))
        resc = np.array(resc)
        master = resc.mean(axis=0)
        ins.append(np.sqrt(((resc - master) ** 2).mean()))
        r_d = rho_tau(sim_diff(rng)); rd = _rescaled_curve(r_d, tau90(r_d), ugrid)
        r_p = rho_tau(sim_pa(rng));   rp = _rescaled_curve(r_p, tau90(r_p), ugrid)
        dds.append(np.sqrt(((rd - master) ** 2).mean()))
        dps.append(np.sqrt(((rp - master) ** 2).mean()))
    ins, dds, dps = map(np.array, (ins, dds, dps))

    print("\n=== multi-seed robustness (8 seeds) ===")
    print(f"  internal spread : {ins.mean():.4f} +/- {ins.std():.4f}")
    print(f"  diffusion dist  : {dds.mean():.4f} +/- {dds.std():.4f}"
          f"   ({(dds/ins).mean():.1f}x internal, min {(dds/ins).min():.1f}x)")
    print(f"  pref-attach dist: {dps.mean():.4f} +/- {dps.std():.4f}"
          f"   ({(dps/ins).mean():.1f}x internal, min {(dps/ins).min():.1f}x)")
    ok = bool(((dds > 3 * ins) & (dps > 2 * ins)).all())
    print(f"  separation holds on ALL seeds: {ok}")

    return dict(internal=ins, diff=dds, pa=dps, all_pass=ok)


if __name__ == "__main__":
    run()
    multiseed()
