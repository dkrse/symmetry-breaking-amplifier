# Algorithms

This document describes the algorithms behind every script in `scripts/`. It is
prose + pseudocode; for flowcharts see [`ALGORITHMS_DIAGRAMS.md`](ALGORITHMS_DIAGRAMS.md).

All randomness uses a seeded `numpy.random.default_rng`, so every result is
deterministic.

---

## 1. The shared amplifier model (`symmetry_breaking.py`)

Everything rests on one update rule, the **position amplifier** written in
log-status `x = ln S`:

```
x_i(t+1) = x_i(t) + ln(1 + g * f(S_i)) + eta_i,     eta_i ~ N(0, sigma^2)
```

with the **Gould saturating feedback**

```
f(S) = theta * S / (1 + S / S_star)
```

which is linear at small `S` (the heavy-tail engine) and approaches a soft
ceiling `theta * S_star` at large `S`.

### Building blocks

- **`feedback_x(x, theta, S_star)`**: computes `f` directly in log-status, in the
  numerically stable form `theta / (exp(-x) + 1/S_star)`, so it never overflows
  for large `x` and tends to `theta*S_star` as `x -> inf`.
- **`reset_hazard(S, p_hi, S_crit, w)`**: the reverse-dominance / Kesten reset
  probability. A logistic in `ln S`: near zero for small `S`, rising toward
  `p_hi` once `S` exceeds `S_crit`. Models a coalition toppling the over-dominant.
- **`step(x, rng, g, sigma, theta, S_star, reset_kw)`**: one update of the whole
  log-status vector:
  1. `x <- x + log1p(g * feedback_x(x)) + normal(0, sigma)`
  2. if a reset is active: draw a hazard per entity, and with that probability
     send the entity back near the floor (`x <- normal(0, sigma)`).

Calibration used throughout: `sigma = 0.05`, `theta = 0.02`, `S_star = 50`.

### E1: symmetry breaking and the amplification factor `A(g)`

Start homogeneous (`x_i(0) ~ N(0, 1e-6)`), iterate `T` steps for a range of gains,
and record the cross-sectional variance each step.

```
A(g) = Var_g(T) / Var_0(T)
```

is the dispersion **manufactured by hierarchy** on top of the diffusion a flat
population (`g=0`) would show anyway. `A(0)=1`; `A(g)` peaks (~400x) then falls,
because at high gain every entity saturates at the ceiling and the spread
collapses.

### Optimal gain, derived and scanned (`optimal_gain_scan`)

Mean-field: `dS/dt = g*theta*S^2` gives finite-time blow-up at
`t* = 1/(g*theta*S0)`. Dispersion is made only while the map stretches, so it is
maximal when the blow-up meets the horizon, `t*(g*) = T`, i.e.

```
g* ~ 1 / (theta * S0 * T)
```

`optimal_gain_scan` locates the peak of `A(g)` at horizons `T = 150,300,600,1200`
and checks the predicted `1/T` scaling (`g* = 0.60,0.40,0.30,0.15`; the last
doubling halves `g*` exactly). The robust claim is the non-monotonicity itself,
not any single optimum.

### E2: direction vs correctness

Give each entity a fixed true quality `q ~ N(0,1)` entering as an honest,
noise-free per-step drift (the best case for merit). Measure the rank correlation
between the final ordering and `q` as `g` rises; it falls monotonically
(0.94 -> 0.66), i.e. the winner **decouples** from quality as amplification grows.
`e2_multiplicative` re-runs it with quality scaling each entity's gain instead;
the correlation is even lower (`<= 0.1`).

### E3: ergodic decomposition and dead ends

Run `M = 5000` independent single-entity trajectories with the Kesten reset at
`g = 0.2`. Track the **ensemble mean** (`<x>`) against the **median trajectory**.
The mean climbs while the median peaks and declines: non-ergodic. Count the
**dead-end fraction** = share of trajectories ending at or below their start
(~20% at this calibration).

### E4: amplification is necessary for development (`e4_development`)

Give entities distinct initial attributes `x0 ~ N(0, 0.3)`. At `g=0` variance
grows linearly (diffusion), the final law is Gaussian, and the initial ordering
`corr(x0, xT)` washes out. At `g>0` variance explodes, a heavy tail forms, and the
ordering locks in. This is the simulation face of the **necessity theorem**: at
`g=0` the process is a martingale, so no structure forms without amplification.

### Scope (`scale_scan`)

Sweep the initial-attribute spread `s0`. For each, fit `xT ~ x0` and report the
**manufactured share** = residual variance / total variance. Small `s0` -> share
~1 (order is manufactured noise); large `s0` -> share ~0 (order reflects real
attributes). Shows the whole effect is a claim about *similar* entities.

### Commitment time (`commitment_scan`)

First step at which the top entity's share of total status exceeds a threshold.
Amplification commits fast and locks in; diffusion commits slowly and reversibly.

### Robustness (`robustness_scan`)

Sweep `sigma x theta` on a 3x3 grid and report, for each cell: peak `A`, the E2
correlation drop, and the dead-end fraction. The **signs/directions** are
invariant across the grid; the magnitudes (especially the dead-end fraction,
~1%-40%) are not. This is the source of the paper's robustness table.

### Early-lead persistence: capability-free (`early_lead_persistence`)

The signature that needs **no quality measure**. From a homogeneous start with no
quality injected, record snapshots at early fractions `tau` of the horizon and
compute

```
rho(tau) = corr( rank x(tau*T), rank x(T) )
```

- **Diffusion null (`g=0`)**: increments are i.i.d., so `rho(tau) = sqrt(tau)`.
  The first 1% of history predicts only `sqrt(0.01) = 10%` of the final order.
- **Amplification (`g>0`)**: the earliest amplified fluctuation dominates the
  final variance, so `rho(tau) -> 1` for small `tau`; the excess over `sqrt(tau)`
  grows with gain.

Reported: `rho(tau)` per gain, and `tau90` = smallest early fraction with
`rho >= 0.9`. A Kesten reset (toppling) erases the early lead (`tau90 -> 1`), a
discriminating sub-signature separating free-token from actively-contested
amplifiers.

---

## 2. PA-null and the toppling discriminator (`earlylead_pa_null.py`)

A synthetic check that disciplines Design 2: on a **monotone** accumulating stock,
does early-lead persistence `rho(tau)` actually separate the amplifier from plain
cumulative advantage? Four processes on a common cohort from a near-homogeneous
start, each yielding `rho(tau)` and `tau90`:

- **DIFF** (`g=0` random walk): lands on the analytic null `sqrt(tau)` (`tau90=1`).
- **AMP** (free-token amplifier, no ceiling): strong persistence (`tau90 ~ 0.375`).
- **PA** (Barabasi preferential attachment): persistence **as strong or stronger**
  than the amplifier (`tau90 ~ 0.1`).
- **TOP** (amplifier + reverse-dominance Kesten renewal): persistence **collapses**
  back to the diffusion regime (`tau90 -> 1`).

Conclusion, carried into the paper: the *magnitude* of `rho(tau)` does not
identify this mechanism (PA already exceeds the amplifier), so an excess over
`sqrt(tau)`, or even over a PA null, is not diagnostic. The clean discriminator
is the **free-token/toppling contrast**: the same amplifier loses its early-lead
persistence under revocation, something plain cumulative advantage cannot mimic.
This motivates the two real-data arms of sections 8 and 9.

---

## 3. Lower threshold: buffer asymmetry (`critical_threshold.py`)

The complementary mechanism at the **bottom** of the distribution. Where the
amplifier explains the *origin* of a gap, a lower critical threshold explains why
a gap, once opened, does not revert. Symmetric to the upper carrying capacity
(section 4), it posits a floor `S_crit` below which an entity's flow is consumed by
subsistence and none is left to grow: below it competence is switched off
(survival mode), and the entity recovers only after clearing a **recovery band**
`S_rec > S_crit`. The hysteresis makes the floor a sticky trap, not a line one
bounces over.

- **Twin test**: two entities with **identical** competence and the **identical**
  shock sequence, differing only in starting buffer. A single early shock the
  large buffer absorbs drops the thin one into survival mode; it ends several
  times lower for no difference in competence or luck, only in position.
- **Population test**: with vs without the floor, the rank correlation between
  final wealth and true competence falls (`0.93 -> 0.56`) while the correlation
  with the starting buffer rises to match it (`0.13 -> 0.56`); the dead-end tail of
  E3 hardens into an **absorbing** trap and the final distribution goes bimodal.

Only the signs are claimed; magnitudes depend on `S_crit` and the survival rate.
Reaches the paper's "position, not competence" conclusion from the opposite end of
the distribution to the amplifier.

---

## 4. Emergent ceiling (`regulation.py`)

Tests Proposition "the ceiling is a maintenance carrying capacity". Run a
preferential-giving rule with **no imposed cap**: status is acquired at a rate
increasing in `S`, while upkeep draws a bounded shared flow `Phi` and a unit is
retained only while per-unit service `Phi/S >= rho`. Across a factor-sixteen range
of the acquisition gain the emergent ceiling holds `S_star / D = 1`, the cap is
set by upkeep, not ambition. A free token (no upkeep, `rho=0`) has no ceiling and
runs away, which is why the optimal-gain downturn is absent in free-token markets.

---

## 5. Joint dose-response fingerprint (`dose_response.py`)

The identifying signature is not any single curve but that **several signatures
move together with the one gain knob**. Sweep the preferential-attachment cultural
market across gain and, at each gain, measure three quantities on the same run:

- **decoupling** `corr(success, quality)`,
- **concentration** (Gini of final outcomes),
- **early-lead lock-in** `rho(0.1)`.

As gain rises, decoupling **falls** while concentration and lock-in both **rise** —
a coherent joint response no rival reproduces (meritocracy predicts no gain effect;
plain cumulative advantage has no single tunable gain that moves all three at
once). The two real Music Lab points (weak/strong signal, `0.765` and `0.651`) are
overlaid on the decoupling curve as a two-point real anchor.

---

## 6. Design 1 real data: the causal decoupling (`musiclab_analysis.py`)

Design 1 on real data, on the archived Salganik-Dodds-Watts Music Lab (bundled,
offline). The same 48 songs run through many independent "worlds"; an
**independent** condition (no visible counts) measures each song's intrinsic
quality `Q`, and two experiments differ in social-signal strength (weak vs strong =
low vs high gain). Because the songs are identical across worlds, between-world
differences **cannot** come from quality — they are the amplifier.

- Per experiment: `rho = Spearman(world download share, independent-world quality Q)`,
  plus the Gini of download shares.
- Test: `rho_weak` (exp 1) `>` `rho_strong` (exp 2), a one-sided Welch `t`-test.

Result: `rho` falls `0.765 -> 0.651` (eight worlds each; Welch `t = 4.19`, one-sided
`p < 1e-3`) while concentration rises (Gini `0.34 -> 0.50`), landing almost exactly
on the simulated Design 1 figure (`0.76 -> 0.65`). Caveat: with only two gain levels
and eight worlds each the two-level test is under-powered; the sign is the primary
read, and the pre-registered high-power version is a multi-level gain sweep (whose
power is estimated in section 7).

---

## 7. Monte-Carlo power analysis (`power_analysis.py`, offline, slow)

- **`market(g)`**: a preferential-attachment cultural market: 48 songs with
  hidden appeal `q`, download counts `d`; each trial picks a song with probability
  proportional to `q * (1 + g * share)`, incrementing its count. Returns `(q, d)`.
- **Design 1 power**: regress the per-world quality/outcome rank correlation on
  gain across many synthetic worlds; estimate the power to detect a negative slope
  at `W = 10, 20, 30` worlds per level (power `>= 0.96` at `W = 10`).
- **Cohort-concentration power (`cohort_gini`, the withdrawn cohort design)**:
  simulate cohorts at a low vs high gain, score each by the Gini of final outcomes,
  and estimate the power of a high-vs-low `t`-test at `n = 10, 20, 30` cohorts per
  domain.
- **Download-market Gini sweep**: with no per-unit ceiling the Gini rises
  monotonically (`0.20 -> 0.52` across `g in [0,8]`), no downturn — the free-token
  case where the section-1 optimal-gain downturn is withdrawn.

---

## 8. Design 2 real data, toppling arm (`wiki_rfa_toppling.py`, `--online`)

Wikipedia Requests for Adminship (RfA): a contested, **revocable** status. The
Stanford SNAP `wiki-RfA` dataset (downloaded once) gives timestamped, signed votes
per election (candidate x year). Both arms are measured **inside one dataset**:

- **TOPPLE arm** = running **net support** (support minus oppose): oppose votes push
  the tally down, so an early front-runner can be toppled (the revocable position).
- **FREE-TOKEN reference** = running **support-only** cumulative count, which can
  only rise, exactly the monotone free token GitHub stars are.

For each election, order votes in time; at each early fraction `tau` of the vote
sequence record both positions. Treating elections as the cohort,
`rho(tau) = corr(rank position at tau, rank final position)`. Prediction:
`rho_net(tau) << rho_supportonly(tau)` early, and `tau90(net)` closer to 1
(`tau90 = 0.375` net vs `0.10` support-only). The same community process is far less
early-determined when a lead can be toppled.

## 9. Design 2 real data, free-token arm (`github_earlylead.py`, `--online`)

GitHub repository stars: a **free token** (no per-unit ceiling, no toppling).
Entity = repo; cohort = repos created in a fixed window (homogeneous start, every
repo at 0 stars).

- **Cohort**: Search API for repos `created:<window> stars:<min>..<max>`.
- **Trajectory**: paginate `stargazers` with `starred_at` timestamps
  (`application/vnd.github.star+json`); cumulative star stock at each `tau`.
- **`rho_curve` / `tau90`**: tie-averaged Spearman via `scipy.stats.spearmanr`.
- **`pa_null_rho`**: a preferential-attachment null fitted to the cohort's own
  total growth, plotted alongside, the reference the free-token arm must be read
  against (magnitude alone is not diagnostic; see section 2).

Needs a `GITHUB_TOKEN` (rate limit); every API response is cached under
`scripts/data/github_cache/`. The model predicts strong early-lead persistence
here, in contrast to the collapse on the toppling arm (section 8).

---

## 10. Test B: the `rho(tau)` universality collapse

The paper's strongest cross-domain claim (Eq. 1 as a *normal form*, not an
analogy) predicts more than "each early-lead curve bows above `sqrt(tau)`":
rescaling each system's horizon by its own `tau90`, the curves `rho(tau/tau90)`
should **collapse** onto one master shape, as correlation-length rescaling
collapses curves at a critical point. A different mechanism should fall off it.

### 10a. Simulation (`data_collapse.py`)

Build a **family** of amplifier parameterisations standing for different domains
(varying gain, noise, and ceiling-vs-free-token), each from a near-homogeneous
start. For each: `rho(tau)`, a continuous `tau90` (linear interpolation of the
0.9 crossing), and the rescaled curve `rho(tau/tau90)` on a common `u`-grid. The
family **master** is their mean; the **internal spread** is the RMS deviation
about it. Compare two other mechanisms rescaled the same way:

- **diffusion** (`g=0`, the `sqrt(tau)` shape) and
- **plain preferential attachment**.

Result: the amplifier family collapses tightly (internal RMS `~0.01`) while
diffusion and PA sit `7`-`15x` that spread off the master, on **every seed**
(`multiseed`, 8 seeds). The collapse separates the amplifier from PA by
**shape** even though raw magnitude cannot (section 2).

### 10b. Three real free-token domains (`real_data_collapse.py`)

Run the same collapse on real trajectories, all free/open data. Each stores
per-entity positions at fractions `tau` of its own horizon, so the `tau` grids
need **not** match: the collapse rescales each by its own `tau90` and
interpolates the rescaled shape onto a common axis. This rescaling is exactly what
makes the otherwise **incommensurable** real clocks (years of startup capital vs
months of stars vs a days-long vote sequence) comparable.

- **Startups** (`startup_earlylead.py`): position = cumulative capital raised;
  cohort = companies whose first round is in 2010 with >=2 rounds (near-equal
  entry, 2325 companies); horizon ~60 months. Free token (funding not revoked).
  Open Crunchbase 2015 export (`scripts/data/crunchbase_rounds.csv`).
- **GitHub stars** and **Wikipedia support-only** counts, re-extracted on a common
  fine 18-point grid by `refine_grids.py` (below).
- plus the **revocable** Wikipedia net-support arm, for the free/revocable contrast.

Two tests: (1) collapse onto the *simulated* amplifier master, which **fails**,
because the real curves lock in *harder* than the calibrated family, so that
master is calibration-dependent; (2) a **data-derived** master over the three real
free-token domains, the honest object. Finding: the three collapse onto a common
shape with mutual RMS spread `~0.07`, about `3x` closer to one another than the
`sqrt(tau)` null sits to them. This is a shared shape distinct from diffusion, but a
**loose** one. Two honest limits: refining the grid (section 10c) *weakens* the
collapse (finer resolution exposes real small-`tau` shape differences that coarse
binning hides), and the revocable arm is **not** resolved from the free-token
master on Wikipedia's mild toppling. With only three domains one cannot yet tell a
power problem (too few, too heterogeneously clocked) from a substantive limit (the
"one exact curve" claim being too strong); only adding domains separates them.

### 10c. Fine-grid re-extraction (`refine_grids.py`)

The bundled GitHub/Wikipedia early-lead CSVs are on a coarse 10-point grid. This
script rebuilds both **offline** on the same fine 18-point grid as the startups,
so all three domains are directly comparable:

- **GitHub**: cumulative star stock from the cached `starred_at` timestamps
  (`scripts/data/github_cache/`, all 295 cohort repos present); per-repo clock
  `t0` = first cached star; horizon 24 months.
- **Wikipedia**: net and support-only positions from the raw bundled
  `wiki-RfA.txt.gz`, on the fine grid, both arms.

`real_data_collapse.py` prefers these `*_fine.csv` when present.

### The combined real-data figure (`plot_realdata_earlylead.py`)

A plotting-only step: reads the GitHub and Wikipedia (net + support-only)
early-lead CSVs and draws the combined Design 2 figure, all three real curves
against the `sqrt(tau)` null. No new computation; it exists so the figure
regenerates from the saved positions without re-running the `--online` passes.

### Data provenance and integrity

Every dataset is public and free; `scripts/data/DATA_SOURCES.md` records each
source URL, its exact download command, and a SHA-256 hash, with machine-checkable
checksums in `scripts/data/SHA256SUMS` (verified by `run_all.sh` before running).
The only scripts that touch the network are the `--online` real-data passes
(sections 8, 9) and the one-off Crunchbase fetch for 10b.
