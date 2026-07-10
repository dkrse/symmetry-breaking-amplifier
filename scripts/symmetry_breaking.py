#!/usr/bin/env python3

"""
Symmetry-breaking simulation for the companion paper
("Breaking Symmetry, Not Choosing Direction").

Model (paper Eq. 1), written in log-status x = ln S:

    x_i(t+1) = x_i(t) + ln(1 + g f(S_i)) + eta,   eta ~ N(0, sigma^2)

with Gould saturating feedback (paper Eq. 6):

    f(S) = theta S / (1 + S/S_star)

and a Kesten reverse-dominance reset (paper Sec. 6): the top is toppled
with a state-dependent hazard and sent back near the floor.

Three experiments, each answering one claim of the companion paper:

  E1  Symmetry breaking. A homogeneous start (all x_i = 0 + infinitesimal
      noise) is an UNSTABLE state when g>0, sigma>0: dispersion grows even at
      g=0, but only DIFFUSIVELY (Var ~ sigma^2 t). Hierarchy (g>0) manufactures
      it MULTIPLICATIVELY on top. The gain-controlled quantity is the
      amplification factor A(g) = Var_g(T)/Var_0(T), the analogue of the
      "manufactured by hierarchy" term d*sigma_a^2 of paper Eq. 2. A(g) is
      non-monotone: it peaks at an optimal gain g* ~ 1/(theta*S0*T) and then
      falls, because the saturating feedback that stabilises the amplifier also
      caps and destroys the dispersion at high gain (see optimal_gain_scan()).

  E2  Direction is independent of correctness (rho_eff _|_ k). Give every
      entity a tiny TRUE quality advantage q_i. Measure the rank correlation
      between the final winner ordering and q, as amplification (g) grows.
      Prediction: correlation -> 0; who wins decouples from who is "right".

  E3  Ergodic decomposition + dead ends. The ensemble spreads (the population
      "moves"), while the typical single trajectory does not track the mean
      (non-ergodic multiplicative growth). Count the fraction of trajectories
      that end below their start ("stuck / dead end"). The spread is bought
      with those dead ends.

No Date.now / RNG-without-seed: everything uses np.random.default_rng(seed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- shared model ---------------------------------------------------------

def feedback(S, theta, S_star):
    """Gould saturating feedback, Eq. 6: linear at small S, ceiling at large S."""
    return theta * S / (1.0 + S / S_star)

def feedback_x(x, theta, S_star):
    """Same feedback in log-status x=ln S, numerically stable for large x:
    f = theta e^x/(1+e^x/S*) = theta/(e^{-x} + 1/S*)  ->  theta S* as x->inf."""
    return theta / (np.exp(-x) + 1.0 / S_star)

def reset_hazard(S, p_hi, S_crit, w):
    """Reverse-dominance reset hazard, Sec. 6: rises with dominance (logistic in ln S)."""
    return p_hi / (1.0 + np.exp(-(np.log(S) - np.log(S_crit)) / w))

def step(x, rng, g, sigma, theta, S_star, reset_kw):
    """One update of the log-status vector x (stable feedback in x)."""
    x = x + np.log1p(g * feedback_x(x, theta, S_star)) + rng.normal(0.0, sigma, size=x.shape)

    # Kesten reset: topple the over-dominant back near the floor.
    if reset_kw is not None:
        # hazard rises with dominance: logistic in (x - ln S_crit)
        hz = reset_hazard(np.exp(np.minimum(x, 700.0)), **reset_kw)
        toppled = rng.random(x.shape) < hz
        x = np.where(toppled, rng.normal(0.0, sigma), x)

    return x

# ---- E1: symmetry-breaking time ------------------------------------------

def e1_symmetry_breaking(seed=0):
    """From a homogeneous start, dispersion grows even at g=0 -- but only
    DIFFUSIVELY (Var ~ sigma^2 t, from noise). Hierarchy (g>0) grows it
    MULTIPLICATIVELY on top of that. The clean, gain-controlled quantity is
    therefore the amplification factor

        A(g) = Var_g(T) / Var_{g=0}(T),

    the direct simulated analogue of the "manufactured by hierarchy" term
    d*sigma_a^2 in paper Eq. 2: how much dispersion structure adds beyond the
    diffusion a flat (non-hierarchical) population would produce anyway.
    A(0)=1; it rises with gain and plateaus at the saturating-feedback ceiling.
    """
    rng = np.random.default_rng(seed)
    N, T = 4000, 300
    sigma, theta, S_star = 0.05, 0.02, 50.0
    gains = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]

    curves = {}
    for g in gains:
        x = rng.normal(0.0, 1e-6, size=N)   # homogeneous start + infinitesimal noise
        var_t = np.empty(T)
        for t in range(T):
            x = step(x, rng, g, sigma, theta, S_star, reset_kw=None)
            var_t[t] = x.var()
        curves[g] = var_t

    baseline = curves[0.0][-1]
    A = {g: curves[g][-1] / baseline for g in gains}

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for g in gains:
        ax[0].plot(curves[g], label=f"g={g}")
    ax[0].set(xlabel="time step", ylabel="Var(ln S)  (dispersion)",
              title="g=0 diffuses (linear); g>0 amplifies (multiplicative)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7, ncol=2)

    gs = np.array(gains)
    As = np.array([A[g] for g in gains])
    ax[1].semilogy(gs, As, "o-")
    ax[1].axhline(1, ls=":", c="grey", label="A=1 (pure diffusion)")
    ax[1].set(xlabel="gain g", ylabel=r"amplification factor $A(g)$",
              title="Dispersion manufactured by hierarchy (Eq. 2)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "e1_symmetry_breaking.png", dpi=200)
    plt.close(fig)

    return A

# ---- E2: direction vs correctness ----------------------------------------

def e2_direction_vs_correctness(seed=1):
    """Every entity has a tiny TRUE quality q; does the winner track q as g grows?"""
    from numpy.random import default_rng

    N, T = 2000, 300
    sigma, theta, S_star = 0.05, 0.02, 50.0
    q_advantage = 0.02  # deterministic per-step drift proportional to true quality
    gains = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8])
    reset_kw = dict(p_hi=0.01, S_crit=200.0, w=0.5)

    corrs = []
    for g in gains:
        rng = default_rng(seed)
        q = rng.normal(0.0, 1.0, size=N)          # true, fixed quality k
        x = rng.normal(0.0, 1e-6, size=N)
        for t in range(T):
            # quality enters ONCE-per-step as a small honest drift (best case for merit)
            x = x + q_advantage * q
            x = step(x, rng, g, sigma, theta, S_star, reset_kw)
        # Spearman-like: correlation of final rank with true-quality rank
        rf = np.argsort(np.argsort(x))
        rq = np.argsort(np.argsort(q))
        c = np.corrcoef(rf, rq)[0, 1]
        corrs.append(c)
    corrs = np.array(corrs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gains, corrs, "o-")
    ax.axhline(0, ls=":", c="grey")
    ax.set(xlabel="amplification gain g",
           ylabel="corr(final rank, true quality)",
           title="Who wins decouples from who is 'right' as g grows")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(OUT / "e2_direction_vs_correctness.png", dpi=200)
    plt.close(fig)

    return dict(zip(gains.tolist(), corrs.tolist()))

# ---- E3: ergodic decomposition + dead ends -------------------------------

def e3_ergodicity_deadends(seed=2):
    rng = np.random.default_rng(seed)
    M, T = 5000, 400            # M independent single-entity trajectories
    sigma, theta, S_star = 0.05, 0.02, 50.0
    g = 0.2
    reset_kw = dict(p_hi=0.01, S_crit=200.0, w=0.5)

    x = rng.normal(0.0, 1e-6, size=M)
    ens_mean = np.empty(T)      # ensemble-average of x  (population "moves")
    ens_med = np.empty(T)       # typical trajectory

    for t in range(T):
        x = step(x, rng, g, sigma, theta, S_star, reset_kw)
        ens_mean[t] = x.mean()
        ens_med[t] = np.median(x)

    frac_stuck = float(np.mean(x <= 0.0))          # ended at/below start
    growth_ensemble = ens_mean[-1] / T             # per-step ensemble growth
    growth_typical = ens_med[-1] / T               # per-step typical growth

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ens_mean, label="ensemble mean  (population moves)")
    ax[0].plot(ens_med, label="median trajectory  (typical entity)")
    ax[0].set(xlabel="time step", ylabel="ln S",
              title="Non-ergodic: mean ≠ typical")
    ax[0].legend(fontsize=8)

    ax[1].hist(x, bins=60, color="steelblue")
    ax[1].axvline(0, ls=":", c="crimson", label="start / dead-end line")
    ax[1].set(xlabel="final ln S", ylabel="count",
              title=f"Dead ends: {frac_stuck:.0%} end at/below start")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "e3_ergodicity_deadends.png", dpi=200)
    plt.close(fig)

    return dict(frac_stuck=frac_stuck,
                growth_ensemble=growth_ensemble,
                growth_typical=growth_typical)

# ---- E4: amplification is NECESSARY for development (vs mere diffusion) ----

def e4_development(seed=3):
    """Give entities DISTINCT initial attributes and ask whether they DEVELOP
    into structure or merely DIFFUSE. Three signatures separate g=0 from g>0:
      (i)  Var growth: linear (diffusion) vs exponential (multiplicative);
      (ii) excess kurtosis of the final distribution: ~0 (Gaussian, no new
           levels) vs >0 (heavy tail, structure) -- strongest at moderate g,
           truncated at high g by the ceiling (the optimal-gain story of E1);
      (iii) persistence corr(x0, x(T)): decays under diffusion, locks in under
            amplification.
    At g=0 the process is a martingale: E[x_i(t)]=x_i(0) (stays in original
    conditions IN EXPECTATION) and the CLT keeps it Gaussian -- no development.
    """
    N, T = 4000, 300
    sigma, theta, S_star = 0.05, 0.02, 50.0
    gains = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

    def kurt(x):
        z = (x - x.mean()) / x.std()
        return float((z ** 4).mean() - 3.0)

    var_curves, stats = {}, {}
    finals = {}
    for g in gains:
        r = np.random.default_rng(seed)
        x0 = r.normal(0.0, 0.3, size=N)          # distinct initial attributes
        x = x0.copy()
        vt = np.empty(T)
        for t in range(T):
            x = step(x, r, g, sigma, theta, S_star, reset_kw=None)
            vt[t] = x.var()
        var_curves[g] = vt
        finals[g] = x
        stats[g] = dict(persist=float(np.corrcoef(x0, x)[0, 1]),
                        var_ratio=float(vt[-1] / vt[30]),
                        kurt=kurt(x))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for g in gains:
        ax[0].plot(var_curves[g], label=f"g={g}")
    ax[0].set(xlabel="time step", ylabel="Var(x)",
              title="Diffusion (g=0, linear) vs development (g>0)")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7, ncol=2)

    for g in (0.0, 0.1):
        xf = finals[g]; xf = (xf - xf.mean()) / xf.std()
        ax[1].hist(xf, bins=80, density=True, histtype="step", label=f"g={g}")
    ax[1].set(xlabel="standardised final x", ylabel="density", xlim=(-4, 8),
              title="g=0 Gaussian; g>0 heavy tail (new level)")
    ax[1].legend(fontsize=8)

    ps = [stats[g]["persist"] for g in gains]
    ax[2].plot(gains, ps, "o-")
    ax[2].set(xlabel="gain g", ylabel=r"persistence corr$(x_0, x_T)$",
              title="Attributes wash out at g=0, lock in as g grows")
    fig.tight_layout(); fig.savefig(OUT / "e4_development.png", dpi=200)
    plt.close(fig)

    return stats

def e2_multiplicative(seed=1, c=1.0):
    """Robustness for E2: quality enters MULTIPLICATIVELY (scales each entity's
    gain) instead of as an additive drift. Better entities amplify faster, yet
    the amplifier cannot distinguish a quality-driven increment from a
    noise-driven one, so the correlation with quality stays LOW (~0.1) -- the
    decoupling strengthens, not weakens. c sets the quality-advantage spread;
    large c (very different entities) would recover competence (cf. scale_scan).
    """

    N, T = 2000, 300
    sigma, theta, S_star = 0.05, 0.02, 50.0
    gains = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    out = {}
    for g in gains:
        r = np.random.default_rng(seed)
        q = r.normal(0, 1, N)
        qpos = 1 + c * (q - q.min()) / (q.max() - q.min())   # gain multiplier in [1,1+c]
        x = r.normal(0, 1e-6, N)

        for _ in range(T):
            S = np.exp(np.minimum(x, 700.0))
            x = x + np.log1p(g * qpos * (theta * S / (1 + S / S_star))) + r.normal(0, sigma, N)
            hz = 0.01 / (1 + np.exp(-(np.minimum(x, 700.0) - np.log(200.0)) / 0.5))
            x = np.where(r.random(N) < hz, r.normal(0, sigma, N), x)
        out[g] = float(np.corrcoef(np.argsort(np.argsort(x)),
                                   np.argsort(np.argsort(q)))[0, 1])

    return out

def robustness_scan(seed=0):
    """Do the three headline signatures survive changes in the calibration?
    Reports, over a sigma x theta grid: the peak amplification factor A_peak
    (E1), the E2 correlation drop between g=0 and g=0.6, and the dead-end
    fraction (E3). The DIRECTIONS are invariant; the magnitudes (esp. dead-end
    fraction, 1%-40%) are not -- so the paper's point estimates are one point in
    a family, and only the signs are claimed robust.
    """
    S_star = 50.0
    rk = dict(p_hi=0.01, S_crit=200.0, w=0.5)
    def run(g, sigma, theta, reset, N=3000, T=300):
        r = np.random.default_rng(seed)
        x = r.normal(0, 1e-6, N)
        for _ in range(T):
            x = step(x, r, g, sigma, theta, S_star, rk if reset else None)

        return x

    def e2corr(g, sigma, theta):
        r = np.random.default_rng(1); q = r.normal(0, 1, 1500); x = r.normal(0, 1e-6, 1500)
        for _ in range(250):
            x = x + 0.02 * q; x = step(x, r, g, sigma, theta, S_star, rk)

        return np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(q)))[0, 1]

    rows = []
    for sigma in (0.02, 0.05, 0.1):
        for theta in (0.01, 0.02, 0.04):
            v0 = run(0.0, sigma, theta, False).var()
            A = max(run(g, sigma, theta, False).var() / v0 for g in (0.1, 0.2, 0.4))
            drop = e2corr(0.0, sigma, theta) - e2corr(0.6, sigma, theta)
            dead = float(np.mean(run(0.2, sigma, theta, True) <= 0.0))
            rows.append((sigma, theta, A, drop, dead))

    return rows

def commitment_scan(seed=0, thresh=0.25):
    """Time for SOME path to commit: first step at which the top entity's share
    exceeds `thresh`. Diffusion (g=0) commits slowly and reversibly; amplification
    commits fast and locks in, with a U-shaped time in g (minimum at the same
    optimal gain as A(g); too-high gain hits the ceiling and no single winner
    forms). This is the 'so it does not take too long' claim, made quantitative.
    Returns {g: (commit_time_or_None, final_top_share)}.
    """
    N, T = 2000, 2000
    sigma, theta, S_star = 0.05, 0.02, 50.0
    out = {}
    for g in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):

        r = np.random.default_rng(seed)
        x = r.normal(0.0, 1e-3, size=N)
        hit = None

        for t in range(T):
            x = step(x, r, g, sigma, theta, S_star, reset_kw=None)
            S = np.exp(x - x.max()); share = S / S.sum()
            if hit is None and share.max() > thresh:
                hit = t
        S = np.exp(x - x.max())
        out[g] = (hit, float((S / S.sum()).max()))

    return out

def scale_scan(seed=0):
    """How the amplifier's role depends on the SIZE of initial differences.
    Sweep the initial-attribute spread s0 at fixed gain and measure how much of
    the final ordering the amplifier MANUFACTURES vs merely REFLECTS:
      persist   = corr(x0, x_T)              (does final order track initial?)
      manuf     = resid.var / x_T.var        (share of final variance NOT
                  explained by initial attributes = manufactured by the cascade)
    Small s0 (similar entities): persist~0, manuf~1 -> hierarchy is manufactured
      and decoupled from attributes (R >> 1, the series' thesis regime).
    Large s0 (order-of-magnitude differences): persist~1, manuf~0 -> hierarchy
      reflects real attributes (R -> 1); the amplifier only magnifies a real
      signal. The 'position not competence' phenomenon is thus a claim about
      SIMILAR entities.
    """
    N, T = 4000, 300
    sigma, theta, S_star, g = 0.05, 0.02, 50.0, 0.2
    out = {}
    for s0 in (0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 6.0):
        r = np.random.default_rng(seed)
        x0 = r.normal(0.0, s0, size=N); x = x0.copy()
        for _ in range(T):
            x = step(x, r, g, sigma, theta, S_star, reset_kw=None)
        b = np.polyfit(x0, x, 1); resid = x - np.polyval(b, x0)
        out[s0] = dict(persist=float(np.corrcoef(x0, x)[0, 1]),
                       manuf=float(resid.var() / x.var()))
    return out

def early_lead_persistence(seed=7):
    """Capability-free signature (paper Sec. 'Early-lead persistence').

    How much of the FINAL ordering is already fixed by the first tau-fraction of
    the trajectory? Start near-homogeneous (x_i(0)~N(0,1e-6)); inject NO quality.
    For each early fraction tau, measure the rank correlation between the ordering
    at time tau*T and the ordering at T:

        rho(tau) = corr( rank x(tau T), rank x(T) ).

    Diffusion baseline (g=0): a pure random walk has x_T = x_{tau T} + (later
    increments), independent, so corr(x_{tau T}, x_T) = sqrt(Var x_{tau T}/Var x_T)
    = sqrt(tau). The first 1% of time then predicts only sqrt(0.01)=0.10 of the
    final order. Amplification (g>0): because the earliest amplified fluctuation
    dominates the final variance (the same 'earliest input wins' mechanism as
    Prop. 3), rho(tau) -> 1 for tau far below 1 -- the order freezes early. The
    GAP between the measured rho(tau) and the sqrt(tau) diffusion baseline is a
    signature of amplification that needs no measurement of capability.

    Reported: rho(tau) curves per gain, and tau90 = smallest early fraction whose
    ordering already correlates >=0.9 with the final one. reset_kw=None isolates
    the pure amplifier; a Kesten reset (toppling) weakens but does not remove the
    early lock-in, reported separately as tau90 under reset.
    """
    N, T = 3000, 600
    sigma, theta, S_star = 0.05, 0.02, 50.0
    taus = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    t_idx = np.unique((taus * T).astype(int).clip(1, T))
    gains = [0.0, 0.05, 0.1, 0.2, 0.4]
    reset_kw = dict(p_hi=0.01, S_crit=200.0, w=0.5)

    def run(g, reset):
        r = np.random.default_rng(seed)
        x = r.normal(0.0, 1e-6, size=N)
        snaps = {}
        for t in range(1, T + 1):
            x = step(x, r, g, sigma, theta, S_star, reset if reset else None)
            if t in t_idx:
                snaps[t] = x.copy()
        rf = np.argsort(np.argsort(snaps[T]))
        rho = {}
        for t in t_idx:
            re = np.argsort(np.argsort(snaps[t]))
            rho[t / T] = float(np.corrcoef(re, rf)[0, 1])

        return rho

    def tau90(rho):
        for tau in sorted(rho):
            if rho[tau] >= 0.9:
                return tau

        return None

    curves = {g: run(g, None) for g in gains}
    curves_reset = {g: run(g, reset_kw) for g in (0.0, 0.2)}
    base = {float(t / T): float(np.sqrt(t / T)) for t in t_idx}   # sqrt(tau) diffusion law

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = sorted(base)
    ax.plot(xs, [base[t] for t in xs], "k--", lw=1.3, label=r"diffusion law $\sqrt{\tau}$")

    for g in gains:
        c = curves[g]
        ax.plot(sorted(c), [c[t] for t in sorted(c)], "o-", ms=3, label=f"g={g}")

    ax.set(xlabel=r"early fraction of horizon $\tau$",
           ylabel=r"corr(rank at $\tau T$, final rank)",
           title="Early-lead persistence: order freezes early under amplification")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.9, ls=":", c="grey", lw=0.8)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "early_lead_persistence.png", dpi=200)
    plt.close(fig)

    return dict(curves=curves,
                tau90={g: tau90(curves[g]) for g in gains},
                tau90_reset={g: tau90(curves_reset[g]) for g in curves_reset},
                base=base)


def optimal_gain_scan(seed=0):
    """Locate the peak gain g*(T) of A(g) for several horizons T, to test the
    mean-field crossover prediction g* ~ 1/(theta S0 T). Asymptotically (large
    T) g* halves when T doubles (1/T scaling); the O(1) crossover constant
    g* theta S0 T drifts at small T because the noise-seeding transient occupies
    a T-independent number of steps.
    """
    sigma, theta, S_star, S0 = 0.05, 0.02, 50.0, 1.0
    gains = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4])
    rows = []
    for T in (150, 300, 600, 1200):
        var = []
        for g in gains:
            rng = np.random.default_rng(seed)
            x = rng.normal(0.0, 1e-6, size=4000)
            for _ in range(T):
                x = step(x, rng, g, sigma, theta, S_star, reset_kw=None)
            var.append(x.var())
        gstar = float(gains[int(np.argmax(var))])
        rows.append((T, gstar, 1.0 / (theta * S0 * T), gstar * theta * S0 * T))

    return rows

if __name__ == "__main__":
    print("E1 amplification factor A(g) = Var_g(T)/Var_0(T) (dispersion vs pure diffusion):")
    for g, a in e1_symmetry_breaking().items():
        print(f"   g={g:<5} A={a:8.2f}")

    print("\nE2 corr(final rank, true quality) by gain g:")
    for g, c in e2_direction_vs_correctness().items():
        print(f"   g={g:<5} corr={c:+.3f}")

    print("\nE3 ergodic decomposition / dead ends:")
    for k, v in e3_ergodicity_deadends().items():
        print(f"   {k:18s} {v:.4f}")

    print("\nE4 development vs diffusion (distinct initial attributes):")
    print("   {:>5} {:>10} {:>12} {:>10}".format("g", "persist", "Var ratio", "kurt"))

    for g, s in e4_development().items():
        print(f"   {g:>5} {s['persist']:>10.3f} {s['var_ratio']:>12.1f} {s['kurt']:>10.2f}")

    print("\nE4b scale scan: amplifier manufactures vs reflects (by initial spread s0):")
    print("   {:>6} {:>10} {:>12}".format("s0", "persist", "manuf.share"))

    for s0, s in scale_scan().items():
        print(f"   {s0:>6} {s['persist']:>10.3f} {s['manuf']:>12.3f}")

    print("\nE2 robustness: quality entering MULTIPLICATIVELY (corr with quality):")

    for g, c in e2_multiplicative().items():
        print(f"   g={g:<5} corr={c:+.3f}")

    print("\nRobustness scan (sigma, theta -> A_peak, E2 corr drop, dead-end frac):")
    print("   {:>6}{:>7}{:>9}{:>10}{:>10}".format("sigma", "theta", "A_peak", "corrDrop", "dead%"))

    for sigma, theta, A, drop, dead in robustness_scan():
        print(f"   {sigma:>6}{theta:>7}{A:>9.0f}{drop:>10.3f}{100*dead:>10.1f}")

    print("\nCorollary: commitment time (steps to 25% dominant share) by gain:")
    for g, (hit, sh) in commitment_scan().items():
        print(f"   g={g:<5} commit={('never' if hit is None else hit):>6}  final top share={sh:.3f}")

    print("\nEarly-lead persistence: corr(rank at tau*T, final rank), capability-free:")
    elp = early_lead_persistence()
    print("   {:>7}".format("tau") + "".join(f"{('g='+str(g)):>9}" for g in elp["curves"]) + f"{'sqrt(tau)':>11}")
    for tau in sorted(elp["base"]):
        row = "".join(f"{elp['curves'][g][tau]:>9.3f}" for g in elp["curves"])
        print(f"   {tau:>7.3f}{row}{elp['base'][tau]:>11.3f}")
    print("   tau90 (smallest early fraction with rho>=0.9), no reset:")

    for g, t in elp["tau90"].items():
        print(f"      g={g:<5} tau90={('none' if t is None else f'{t:.3f}')}")

    print("   tau90 with Kesten reset (toppling):")

    for g, t in elp["tau90_reset"].items():
        print(f"      g={g:<5} tau90={('none' if t is None else f'{t:.3f}')}")

    print("\nOptimal-gain scaling  g*(T)  vs prediction 1/(theta S0 T):")
    print("   {:>6} {:>8} {:>12} {:>10}".format("T", "g*", "1/(t S0 T)", "g* t S0 T"))

    for T, gstar, pred, ratio in optimal_gain_scan():
        print(f"   {T:6d} {gstar:8.2f} {pred:12.3f} {ratio:10.2f}")

    print(f"\nFigures written to {OUT}")
