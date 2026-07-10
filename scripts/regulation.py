#!/usr/bin/env python3

"""
Why the ceiling exists (self-regulation of the amplifier) -- parts 1-3.

PART 1 (analytic).  Gould's saturating feedback f(S)=theta*S/(1+S/S*) is NOT a free
parameter: S* is DERIVED from a finite reciprocation budget. An agent of status S
has ~S deferrers and reciprocates from a fixed budget D, so reciprocation per
deferrer is D/S; a deferrer's willingness scales with that, r(S)=D/(D+kappa*S).
Hence
    f(S) = theta*S*r(S) = theta*S/(1 + kappa*S/D) = theta*S/(1+S/S*),   S* = D/kappa.
Finite budget D => finite ceiling. D->inf (a free token needing no reciprocation)
=> S*->inf, the ceiling vanishes. The hard version: if status only counts when
mutual, S_j = sum_i min(a_ij,a_ji) <= sum_i a_ji = D, so S* = D exactly.

PART 2 (simulation, this file).  Agents each hold a giving-budget D, allocated
preferentially toward high status. Status is either MUTUAL (counts only if
reciprocated) or a FREE token (non-mutual). No per-agent ceiling is imposed by
hand; we measure whether one emerges and how it scales with D.
    Result: mutual -> max status = D exactly (ceiling emerges, S*/D=1 for all D);
    free token -> winner-take-all, unbounded (max ~ N times the budget).

PART 3 (prediction).  A domain self-regulates iff its currency is relational /
reciprocated (status, deference, attention, political power granted by
subordinates -> Boehm's reverse dominance). A domain with a free, self-replicating
token (download counts, followers, money-as-number) has no built-in ceiling and
runs away until some EXTERNAL limit binds -- which is exactly why the optimal-gain
downturn failed to appear in the download-market test (paper Section 'empirical').
"""
import numpy as np

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)

def run(N=300, D=1.0, g=4.0, T=250, mutual=True, seed=0):
    """Each agent gives budget D, preferring high-status recipients (gain g).
    mutual: status counts only if reciprocated, S_j = sum_i min(a_ij, a_ji).
    free:   S_j = sum_i a_ij (a self-replicating token)."""
    r = np.random.default_rng(seed)
    S = np.ones(N) + r.normal(0, 1e-2, N)
    a = np.full((N, N), D / N); np.fill_diagonal(a, 0)

    for _ in range(T):
        W = S[None, :] ** g * np.ones((N, 1)); np.fill_diagonal(W, 0)
        a = D * W / W.sum(1, keepdims=True)             # each row (giver) sums to D
        S = (np.minimum(a, a.T) if mutual else a).sum(0)
        S = np.maximum(S, 1e-9)

    return S

if __name__ == "__main__":
    print("PART 2 -- does a per-agent ceiling EMERGE? (D=1, strong preferential g=4)")
    for label, mut in [("mutual (reciprocated status)", True), ("free token", False)]:
        S = run(mutual=mut)
        print(f"  {label:28s}: max/D={S.max():.2f}  max/mean={S.max()/S.mean():6.1f}  Gini={gini(S):.3f}")

    print("\n  ceiling S* vs budget D (mutual):")

    for D in (0.5, 1, 2, 4):
        S = run(D=D, mutual=True)
        print(f"    D={D:<4} S*={S.max():.3f}  S*/D={S.max()/D:.3f}")

    print("  -> S* = D exactly: the ceiling is the reciprocation budget, derived not assumed.")
    print("\n  ceiling is set by upkeep, not ambition -- S*/D vs acquisition gain g:")

    for g in (1, 2, 4, 8, 16):
        S = run(g=g, D=1.0, mutual=True)
        print(f"    g={g:<3} S*/D={S.max():.3f}")
