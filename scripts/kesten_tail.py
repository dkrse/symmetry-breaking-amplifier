#!/usr/bin/env python3


"""
kesten_tail.py
==============
Estimates the power-law tail index of the reset-plus-drift (Kesten) dynamics of
the paper, and checks the analytic prediction against a Hill estimate from the
simulated stationary distribution.

Why this exists
---------------
Section "amplification is necessary for development" asserts that the
reset-plus-multiplicative-drift dynamics produce a power-law tail "by the
standard Kesten result", without pinning the index. For a Physica A readership
the index is the natural observable, and it is available in closed form here.

The prediction
--------------
In the saturated regime the feedback is flat, f(S) -> theta S*, so a surviving
entity's status is multiplied each step by

    M = exp(c + eta),   c = ln(1 + g theta S*),   eta ~ N(0, sigma^2),

while the reverse-dominance hazard saturates at p_hi and sends the entity back
to the floor. This is a multiplicative random walk with geometric killing and
renewal: the Kesten/Champernowne mechanism. The stationary law has a Pareto
tail P(S > s) ~ s^{-mu} with mu the root of the Cramer condition

    (1 - p_hi) E[M^mu] = 1,

i.e. (1-p_hi) exp(mu c + mu^2 sigma^2 / 2) = 1, whose positive root is

    mu = ( -c + sqrt(c^2 - 2 sigma^2 ln(1 - p_hi)) ) / sigma^2.          (*)

For sigma^2 << c (the paper's calibration) this reduces to the transparent
mu ~ -ln(1-p_hi)/c: the tail is thin when toppling is frequent relative to the
per-step amplification, and heavy when it is not.

What the check shows
--------------------
(a) Across a grid of toppling rates and gains, (*) matches a Hill estimate from
    the simulated stationary tail of the FULL nonlinear dynamics (not the tail
    linearisation), so the Kesten reading of the model is quantitative, not
    merely qualitative.
(b) At the paper's illustrative calibration (g=0.2, p_hi=0.01) the same formula
    gives mu ~ 0.05: an exponent far below 1, i.e. a tail so heavy that no
    stationary mean exists and no practical horizon reaches the stationary law.
    The heavy right shoulder reported in Sec. 5 is therefore a pre-asymptotic
    transient, and we say so rather than claim a fitted power law there.
(c) mu crosses 1 (finite mean) at p_hi = 1 - exp(-c), which for the paper's
    calibration is p_hi ~ 0.167: a falsifiable statement about how contestable
    a status must be for its distribution to have a mean at all.

All randomness is seeded. Deps: numpy only (matplotlib for the figure).
"""



import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from symmetry_breaking import reset_hazard, step

OUT = Path(__file__).resolve().parent.parent / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SIGMA, THETA, S_STAR = 0.05, 0.02, 50.0
S_TOP, W = 200.0, 0.5




def mu_analytic(g, p_hi, sigma=SIGMA, theta=THETA, S_star=S_STAR):
    """Positive root of the Cramer condition (*), the predicted Pareto index."""
    c = np.log1p(g * theta * S_star)
    disc = c * c - 2.0 * sigma * sigma * np.log1p(-p_hi)
    return (-c + np.sqrt(disc)) / (sigma * sigma)



def hill(x_tail_sorted, k):
    """Hill estimator of the tail index from the k largest order statistics
    (input sorted ascending, in LOG status: ln S). For a Pareto tail in S,
    the log-excesses over the (k+1)-th largest are Exp(1/mu)."""
    top = x_tail_sorted[-(k + 1):]
    excess = top[1:] - top[0]
    return 1.0 / excess.mean()


def stationary_sample(g, p_hi, n=200_000, T=6000, burn=3000, seed=0):

    """Run the FULL dynamics (Gould feedback + logistic reverse-dominance
    hazard) to stationarity and pool post-burn-in states across time, so the
    tail sample is large without needing an impractical number of walkers."""

    rng = np.random.default_rng(seed)
    reset_kw = dict(p_hi=p_hi, S_top=S_TOP, w=W)
    x = rng.normal(0.0, SIGMA, size=n)
    keep = []
    for t in range(T):
        x = step(x, rng, g, SIGMA, THETA, S_STAR, reset_kw)
        if t >= burn and (t - burn) % 500 == 0:
            keep.append(x.copy())
    return np.concatenate(keep)



def scan(seed=0):
    """Grid over toppling rate and gain; compare (*) with the Hill estimate."""

    rows = []
    for g in (0.1, 0.2, 0.4):
        for p_hi in (0.10, 0.167, 0.20, 0.30, 0.40, 0.50):
            mu_pred = mu_analytic(g, p_hi)
            x = stationary_sample(g, p_hi, seed=seed)
            xs = np.sort(x)
            # Hill over the top 2% and top 0.5%, both above the saturation knee
            k2 = max(50, int(0.02 * xs.size))
            k05 = max(50, int(0.005 * xs.size))
            rows.append((g, p_hi, mu_pred, hill(xs, k2), hill(xs, k05)))
    return rows


def figure(rows):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    pred = np.array([r[2] for r in rows])
    h2 = np.array([r[3] for r in rows])
    h05 = np.array([r[4] for r in rows])
    lim = [0, max(pred.max(), h2.max(), h05.max()) * 1.1]
    ax[0].plot(lim, lim, "k--", lw=1, label="identity")
    ax[0].plot(pred, h2, "o", ms=5, label="Hill, top 2%")
    ax[0].plot(pred, h05, "s", ms=4, mfc="none", label="Hill, top 0.5%")
    ax[0].set(xlabel=r"predicted $\mu$  (Cramer condition)",
              ylabel=r"measured $\mu$  (Hill)",
              title="Kesten index: prediction vs simulation", xlim=lim, ylim=lim)
    ax[0].legend(fontsize=8)

    ps = np.linspace(0.005, 0.6, 300)
    for g in (0.1, 0.2, 0.4):
        ax[1].plot(ps, mu_analytic(g, ps), label=f"g={g}")
    ax[1].axhline(1.0, ls=":", c="crimson", label=r"$\mu=1$ (mean exists)")
    ax[1].set(xlabel=r"toppling rate $p_{\rm hi}$", ylabel=r"tail index $\mu$",
              title=r"Contestability sets the tail: $\mu\simeq-\ln(1-p_{\rm hi})/c$",
              yscale="log")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "kesten_tail.png", dpi=200)
    plt.close(fig)



if __name__ == "__main__":
    print("Kesten tail index: analytic Cramer root vs Hill estimate\n")
    print(f"   {'g':>5}{'p_hi':>8}{'mu pred':>10}{'Hill 2%':>10}{'Hill 0.5%':>11}"
          f"{'rel.err':>9}")
    rows = scan()
    for g, p, mp, h2, h05 in rows:
        print(f"   {g:>5}{p:>8.3f}{mp:>10.3f}{h2:>10.3f}{h05:>11.3f}"
              f"{abs(h05 - mp) / mp:>8.1%}")
    figure(rows)

    print("\nAt the paper's illustrative calibration:")
    for g in (0.1, 0.2, 0.4):
        print(f"   g={g:<4} p_hi=0.01 -> mu = {mu_analytic(g, 0.01):.4f}"
              "   (mu << 1: no stationary mean, no practical convergence)")


    # Why substituting g=0 into (*) does NOT contradict the necessity theorem.
    # (*) assumes the SATURATED hazard p_hi. The model's killing is
    # dominance-triggered, so at g=0 nothing ever climbs to the toppling region
    # and the effective hazard is orders below p_hi: the process is the plain
    # martingale, not a Kesten process with a power-law tail.
    print("\nThe g=0 corner (state-dependent vs constant-rate killing):")
    print(f"   formally mu(c=0, p_hi=0.01) = {mu_analytic(0.0, 0.01):.3f}"
          "   <- would be a power law IF the hazard were constant")
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1e-6, size=4000)
    for _ in range(300):
        x = x + rng.normal(0.0, SIGMA, size=4000)
    s_max = float(np.exp(x.max()))
    hz = float(reset_hazard(np.array([s_max]), p_hi=0.01, S_top=S_TOP, w=W)[0])

    print(f"   but at g=0, T=300, N=4000: max S reached = {s_max:.1f} "
          f"vs S_top = {S_TOP:.0f}")
    print(f"   hazard there = {hz:.2e}, i.e. {0.01 / hz:.0f}x below p_hi:"
          " renewals essentially never fire,")
    print("   so the dynamics reduce to the martingale of the necessity theorem.")

    print("\nThreshold for a finite stationary mean (mu = 1):")

    for g in (0.1, 0.2, 0.4):
        c = np.log1p(g * THETA * S_STAR)
        # mu=1 root of (*): sigma^2/2 + c + ln(1-p) = 0
        p1 = 1.0 - np.exp(-(c + SIGMA ** 2 / 2.0))
        print(f"   g={g:<4} c={c:.4f} -> p_hi* = {p1:.4f}")

    print(f"\nFigure written to {OUT / 'kesten_tail.png'}")
