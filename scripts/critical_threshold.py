#!/usr/bin/env python3
"""
Critical threshold / buffer asymmetry: "same mistake, different consequence".

A complementary mechanism to the amplifier (K. Sestak's note): a LOWER absorbing
threshold S_crit. Above it an entity can use its competence to grow; once a shock
pushes it below, it drops into survival mode -- competence stops mattering and it
treads water near the floor. The buffer (distance above S_crit) means the SAME
shock is a scratch for a high entity and fatal for a low one.

The amplifier explains the ORIGIN of a gap (symmetry-breaking from noise); this
threshold explains its PERSISTENCE and asymmetry (why it does not revert). This
script asks whether the threshold, on its own, DECOUPLES outcome from competence
-- i.e. whether "position (buffer), not competence" emerges from it too.

Three demonstrations:
  1. Twin test: two entities, IDENTICAL competence and IDENTICAL shock sequence,
     differing only in starting buffer -> one absorbs an early shock and thrives,
     the other is pushed below S_crit by the same shock and is trapped.
  2. Decoupling: corr(final wealth, true competence) with the floor OFF vs ON.
  3. Distribution: the floor turns the tail of dead ends into an absorbing trap
     (bimodal), and outcome tracks starting buffer more than competence.
"""


from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "figures" / "critical_threshold.png"

N, T = 3000, 300
SIGMA = 0.06          # shock (mistake/luck) size, log-return sd -- SAME for all
MU0 = 0.010           # base log-growth above the floor
BETA = 0.010          # competence -> extra log-growth above the floor
MU_SURVIVE = -0.008   # below the floor: slowly sinking, NO competence (a real trap)
S_CRIT = 1.0          # critical threshold: fall below -> survival mode
S_REC = 1.30          # recovery band: must climb back to here to regain growth mode
S0_SPREAD = 0.5       #  sd of ln(S0 / S_crit): heterogeneous starting buffers



def run(q, S0, shocks, floor=True):
    """Vectorised wealth dynamics with hysteresis. q,(N); S0,(N); shocks,(T,N).
    Growth mode uses competence; once a shock drops S below S_CRIT the entity
    loses growth mode and must climb back to S_REC (> S_CRIT) to regain it -- a
    sticky poverty trap, not a line you bounce over."""
    S = S0.copy()
    growth = np.ones(N, bool) if not floor else (S0 >= S_CRIT)

    for t in range(T):
        if floor:
            growth = np.where(S < S_CRIT, False, np.where(S >= S_REC, True, growth))
        drift = np.where(growth, MU0 + BETA * q, MU_SURVIVE)
        S = S * np.exp(drift + shocks[t])
    return S



def main():
    rng = np.random.default_rng(20260709)
    q = rng.normal(0, 1, N)                                  # true competence
    S0 = S_CRIT * np.exp(rng.normal(0.35, S0_SPREAD, N))     # starting buffer (mostly above floor)
    shocks = rng.normal(0, SIGMA, (T, N))                    # the "mistakes/luck"

    S_floor = run(q, S0, shocks, floor=True)
    S_nofloor = run(q, S0, shocks, floor=False)


    # --- decoupling: does outcome track competence, or starting buffer? ---
    def rc(a, b):
        return spearmanr(a, b).statistic
    print("Decoupling (Spearman rank corr with final wealth):")
    print(f"  floor OFF: corr(outcome, competence)={rc(S_nofloor,q):+.3f}   "
          f"corr(outcome, start)={rc(S_nofloor,S0):+.3f}")
    print(f"  floor ON : corr(outcome, competence)={rc(S_floor,q):+.3f}   "
          f"corr(outcome, start)={rc(S_floor,S0):+.3f}")
    trapped = np.mean(S_floor < S_CRIT)
    print(f"  trapped below S_crit (floor ON): {trapped*100:.1f}%")

    # --- twin test: same competence, same shocks, only buffer differs ---
    # craft a shock sequence with an early bad shock both twins receive.
    tw_shocks = rng.normal(0, SIGMA, T)
    tw_shocks[3] = -0.30                                     # the identical "mistake"
    q_twin = 0.0                                             # identical competence
    S_hi, S_lo = [], []
    s_hi, s_lo = 1.8 * S_CRIT, 1.03 * S_CRIT                 # large vs thin buffer
    g_hi = g_lo = True

    for t in range(T):
        for s_list, s in ((S_hi, s_hi), (S_lo, s_lo)):
            s_list.append(s)
        g_hi = False if s_hi < S_CRIT else (True if s_hi >= S_REC else g_hi)
        g_lo = False if s_lo < S_CRIT else (True if s_lo >= S_REC else g_lo)
        s_hi = s_hi * np.exp((MU0 + BETA * q_twin if g_hi else MU_SURVIVE) + tw_shocks[t])
        s_lo = s_lo * np.exp((MU0 + BETA * q_twin if g_lo else MU_SURVIVE) + tw_shocks[t])

    print(f"\nTwin test (identical competence & shocks, buffer differs):")
    print(f"  large buffer S0={S_hi[0]:.2f}: final={S_hi[-1]:.2f}")
    print(f"  thin  buffer S0={S_lo[0]:.2f}: final={S_lo[-1]:.2f}  "
          f"(same early shock {tw_shocks[3]*100:.0f}% at t=3)")


    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    # panel 1: twins
    ax[0].axhline(S_CRIT, color="0.6", ls="--", lw=1, label="critical threshold")
    ax[0].plot(S_hi, color="seagreen", label="large buffer: absorbs the shock")
    ax[0].plot(S_lo, color="crimson", label="thin buffer: falls into survival mode")
    ax[0].axvline(3, color="0.8", lw=1)
    ax[0].annotate(f"same mistake\n({tw_shocks[3]*100:.0f}%)", (5, 0.55),
                   fontsize=7, color="0.4")
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("wealth S")
    ax[0].set_title("Same competence, same shocks,\ndifferent buffer")
    ax[0].legend(frameon=False, fontsize=7.5)

    # panel 2: decoupling bars
    labels = ["outcome↔\ncompetence", "outcome↔\nstart buffer"]
    off = [rc(S_nofloor, q), rc(S_nofloor, S0)]
    on = [rc(S_floor, q), rc(S_floor, S0)]
    x = np.arange(2); w = 0.35
    ax[1].bar(x - w/2, off, w, color="steelblue", label="floor OFF")
    ax[1].bar(x + w/2, on, w, color="indianred", label="floor ON")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("rank correlation"); ax[1].set_ylim(0, 1)
    ax[1].set_title("The floor decouples outcome\nfrom competence")
    ax[1].legend(frameon=False, fontsize=8)

    # panel 3: distributions
    bins = np.linspace(-2, 4, 60)
    ax[2].hist(np.log(S_nofloor), bins=bins, color="steelblue", alpha=0.55,
               label="floor OFF", density=True)
    ax[2].hist(np.log(S_floor), bins=bins, color="indianred", alpha=0.55,
               label="floor ON (trap)", density=True)
    ax[2].axvline(np.log(S_CRIT), color="0.5", ls="--", lw=1)
    ax[2].set_xlabel("ln final wealth"); ax[2].set_ylabel("density")
    ax[2].set_title("Floor creates an absorbing\ntrap (bimodal)")
    ax[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
