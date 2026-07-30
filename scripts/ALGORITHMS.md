# Algorithms

This document describes the algorithms behind every script in `scripts/`. It is
prose + pseudocode; for flowcharts see [`ALGORITHMS_DIAGRAMS.md`](ALGORITHMS_DIAGRAMS.md).

All randomness uses a seeded `numpy.random.default_rng`, so every result is
deterministic.

---

## 1. The shared amplifier model (`symmetry_breaking.py`, `optimal_gain_fit.py`, `kesten_tail.py`)

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
- **`reset_hazard(S, p_hi, S_top, w)`**: the reverse-dominance / Kesten reset
  probability. A logistic in `ln S`: near zero for small `S`, rising toward
  `p_hi` once `S` exceeds the dominance threshold `S_top`. Models a coalition
  toppling the over-dominant. Note this is **state-dependent** killing, not a
  constant rate: at `g=0` entities never climb to `S_top`, the hazard stays some
  two orders below `p_hi`, and the process reduces to the plain martingale of the
  necessity theorem. (`S_top` is deliberately distinct from the `S_crit` of
  `critical_threshold.py`, which is the *lower*, poverty-trap threshold.)
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
population (`g=0`) would show anyway. `A(0)=1`; `A(g)` peaks (~400x at `g=0.4`)
then falls, because at high gain every entity saturates at the ceiling and the
spread collapses (`A = 169, 45, 14` at `g = 0.8, 1.6, 3.2`).

Note the low-gain end: `A(0.05) = 1.9`, i.e. barely above pure diffusion, and that
curve has not turned over within `T=300`. This is not a failure of the model but a
consequence of the blow-up time `t* ~ 1/(g theta S0)`, which at `g=0.05` exceeds
the horizon. The turnover is horizon-dependent, so "amplification separates from
diffusion" is a statement about gains whose `t*` fits inside the observation
window.

### Optimal gain, derived and scanned (`optimal_gain_scan`)

Mean-field: `dS/dt = g*theta*S^2` gives finite-time blow-up at
`t* = 1/(g*theta*S0)`. Dispersion is made only while the map stretches, so it is
maximal when the blow-up meets the horizon, `t*(g*) = T`, i.e.

```
g* ~ 1 / (theta * S0 * T)
```

`optimal_gain_scan` locates the peak of `A(g)` on a coarse grid at horizons
`T = 150,300,600,1200`, giving `g* = 0.60,0.40,0.30,0.15`.

**That scan is superseded, and its apparent confirmation of `1/T` was a grid
artifact.** `optimal_gain_fit.py` redoes it properly: a 24-point log-spaced gain
grid, the maximum refined to sub-grid resolution by a parabolic fit in `log g`,
six horizons up to `T = 2400`, and three seeds. It returns

```
g* = 0.62, 0.37, 0.24, 0.17, 0.14, 0.13   at T = 150, 300, 600, 1200, 1800, 2400
fitted exponent  g* ~ T^-alpha  with alpha = 0.56 +/- 0.02
restricted to T >= 600:                     alpha = 0.44 +/- 0.04
```

so the exponent is nearer `1/2` than `1`, and restricting to the asymptotic half
of the range moves it *further* from 1, not closer. The crossover constant
`g* theta S0 T` drifts monotonically from 1.9 to 6.4 instead of settling. The
mean-field argument evidently omits the noise-seeding stage that sets the
effective `S0`. The direction (`g*` falls with `T`, vanishing asymptotically) is
right; the exponent is not. The robust claim is the non-monotonicity itself, not
any single optimum or scaling.

### Kesten tail index (`kesten_tail.py`)

In the saturated regime `f -> theta*S_star`, so a surviving entity is multiplied
each step by `M = exp(c + eta)` with `c = ln(1 + g*theta*S_star)` and
`eta ~ N(0, sigma^2)`, while the hazard saturates at `p_hi` and renews it at the
floor. This is multiplicative growth with geometric killing, whose stationary law
is Pareto with index solving the Cramer condition `(1 - p_hi) E[M^mu] = 1`:

```
mu = ( -c + sqrt(c^2 - 2 sigma^2 ln(1 - p_hi)) ) / sigma^2
   -> -ln(1 - p_hi) / c            for sigma^2 << c
```

i.e. the tail index is the toppling rate over the per-step amplification:
**contestability sets the tail**. `kesten_tail.py` runs the FULL nonlinear
dynamics to stationarity over 3 gains x 6 toppling rates and compares a Hill
estimate against this root. Agreement is good where the tail is resolvable
(twelve cells with `mu <~ 2`: mean error 3.1%, worst 8.9%); above `mu ~ 2` the
Hill estimate is biased low, and moving the threshold from the top 2% to the top
0.5% moves it back toward the prediction in every such cell, which is the
signature of a non-asymptotic threshold rather than a wrong exponent.

Two consequences worth noting:

- At the illustrative calibration (`p_hi = 0.01`) the formula gives
  `mu = 0.105, 0.055, 0.030` at `g = 0.1, 0.2, 0.4` — far below 1, so no
  stationary mean and no convergence at any feasible horizon. The heavy right
  shoulder in the E4 histogram is a pre-asymptotic transient, which is why no
  fitted exponent is reported there.
- Setting `mu = 1` gives `p_hi* = 1 - exp(-(c + sigma^2/2))`
  (`= 0.092, 0.168, 0.287` at those gains): the contestability a status needs for
  its distribution to have a mean at all.

Substituting `g = 0` (hence `c = 0`) into the formula returns a finite
`mu ~ 2.84`, which does **not** contradict the necessity theorem: the formula
assumes the saturated hazard `p_hi`, whereas the model's killing is
dominance-triggered, and at `g = 0` nothing ever reaches the toppling region (see
`reset_hazard` above). The script prints the diagnostic: at `g = 0`, `T = 300`,
`N = 4000` the most extreme walker reaches `S = 17.7` against `S_top = 200`, where
the hazard is `7.8e-05`, some 128x below `p_hi`, so renewals essentially never
fire. A Kesten process with *constant-rate* killing would indeed have a power-law
tail at zero drift; this one does not, and the difference is testable.

### E2: direction vs correctness

Give each entity a fixed true quality `q ~ N(0,1)` entering as an honest,
noise-free per-step drift (the best case for merit). Measure the rank correlation
between the final ordering and `q` as `g` rises; it falls monotonically
(0.94 -> 0.66), i.e. the winner **decouples** from quality as amplification grows.
`e2_multiplicative` re-runs it with quality scaling each entity's gain instead;
the correlation is even lower (`<= 0.1`), which is the point: merit that must flow
*through* position is more decoupled, not less, so the additive design of the main
figure is the one most favourable to competence. Read that comparison only at
`g > 0` — at `g = 0` a quality that acts by scaling the gain has no channel at all,
so its correlation is trivially ~0 (measured `0.02`) rather than informatively
low. The substantive contrast is at `g = 0.8`: `0.10` multiplicative vs `0.66`
additive.

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

**The null's premise, and why it matters everywhere below.** `rho(tau) = sqrt(tau)`
needs the increments to be i.i.d. **across entities**, i.e. a homogeneous
*population*, not merely a common starting point. Either kind of persistent
difference breaks it, at `g=0` and with no feedback at all:

```
dispersed start:      rho(tau) = sqrt( (Var_x0 + sigma^2*tau*T) / (Var_x0 + sigma^2*T) )
heterogeneous drift:  rho(tau) = (tau*T^2*Var_a + sigma^2*tau*T)
                                 / sqrt( (tau^2*T^2*Var_a + sigma^2*tau*T)
                                        * (T^2*Var_a + sigma^2*T) )
```

Both exceed `sqrt(tau)` whenever the respective variance is positive, because a
persistent trait predicts the final order on its own. So an excess over `sqrt(tau)`
identifies nothing in real data, where entities differ; only a **contrast** does,
since heterogeneity inflates both arms of a within-domain comparison alike. This
simulation is exempt because it injects no quality and starts homogeneous by
construction, which is exactly the premise real cohorts fail.

Reported: `rho(tau)` per gain, and `tau90` = smallest early fraction with
`rho >= 0.9`. A Kesten reset (toppling) erases the early lead (`tau90 -> 1`), a
discriminating sub-signature separating free-token from actively-contested
amplifiers.

**Two nulls, and which one the measurement actually has.** The proposition derives
`sqrt(tau)` for the **Pearson** correlation of `x`, but `rho(tau)` is measured as
a **rank** correlation (scale-invariant, hence the right choice for data). At
`g=0` the pair `(x(tau T), x(T))` is bivariate normal, so the two are related
exactly by `rho_S = (6/pi) arcsin(r/2)` and the rank-scale null is

```
rho_S(tau) = (6/pi) * arcsin(sqrt(tau)/2)   <   sqrt(tau)
```

below `sqrt(tau)` by ~0.013 at `tau ~ 0.1` (at most ~0.018 near
`tau ~ 0.35`). The script prints both
columns and the `g=0` residual against each: **mean `-0.0088` against
`sqrt(tau)`** (negative at almost every grid point — a bias, not scatter) versus
**mean `+0.0007` against the rank null** (sign-changing — scatter only). So the
diffusive baseline is verified more exactly than the `sqrt(tau)` comparison
suggests, and using `sqrt(tau)` as the null on real data is conservative, since
the true rank null is lower and clearing `sqrt(tau)` is the harder test.

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
`sqrt(tau)`, or even over a PA null, is not diagnostic. It is in fact worse than
that: a `g=0` population whose entities merely differ in quality also clears the
`sqrt(tau)` null (section 1), so magnitude fails to separate amplification even
from **no amplification at all**. The clean discriminator is the
**free-token/toppling contrast**: the same amplifier loses its early-lead
persistence under revocation, something plain cumulative advantage cannot mimic,
and heterogeneity inflates both arms alike so it cannot manufacture the gap.
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
on the simulated Design 1 figure (`0.76 -> 0.65`).

**Why this design succeeds where the chess example (section 11) cannot.** Identical
songs across worlds mean the heterogeneity that defeats the trajectory signatures
elsewhere cannot operate: with quality held fixed, a between-world difference has
nowhere to come from but the amplifier. This is not an assumption to be doubted, it
is a property of the design.

**`attenuation_check`: the drop is conservative.** `Q` is estimated from ONE
independent world per experiment, so measurement error in `Q` attenuates `rho`. But
exp2's independent world is the LARGER (`2192` vs `1578` downloads), so its `Q` is
the *less* attenuated and its `rho` should be the *higher* on that count. It is the
lower. Resampling exp2's `Q` down to exp1's precision drops it further, to
`0.573 +/- 0.038`, widening the gap from `0.114` to `0.192`. The gain effect thus
overcomes a measurement advantage working against it.

Caveat: with only two gain levels and eight worlds each the two-level test is
under-powered; the sign is the primary read, and the pre-registered high-power
version is a multi-level gain sweep (whose power is estimated in section 7).

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

**Inference (`--bootstrap B`).** Resampling elections with replacement puts a
95% CI on the contrast: the free-token-minus-revocable gap at `tau = 0.1` is
`0.11 [0.09, 0.13]`, positive with zero excluded at every `tau < 1`, and the
`tau90` difference is `0.275` in the full sample; its bootstrap mean is
`0.245 [0.20, 0.28]` (`B = 2000`, seeded).

**Heterogeneity placebo (`heterogeneity_placebo`, always printed).** The standing
worry about any early-lead magnitude is that a persistent trait produces it at
`g = 0` (Eq. `rhodrift`). Here it can be quantified rather than argued, because
the two arms are the same votes read two ways. The placebo keeps each election's
real vote count, gives candidates the empirical spread of final support shares,
and draws votes i.i.d. given the candidate: heterogeneity is maximal, nothing
compounds. To first order the gap must vanish, since at an early fraction the
support-only count has signal `tau*n` against noise `sqrt(tau*n*p*(1-p))` while
the net count doubles both and the factor cancels. Measured, a small residual
survives — `rho` is a *rank* correlation and each arm is referred to its own final
ordering, which the two arms do not share — so the placebo is a calibration, not a
proof of zero:

```
placebo:   mean early gap = +0.033,  tau90  0.10 (net) vs 0.05 (support)
real data: mean early gap = +0.115,  tau90  0.375      vs 0.10
```

Heterogeneity buys just under a third of the observed `rho` gap and about a fifth
of the `tau90` separation, so it cannot account for the contrast.

## 9. Design 2 real data, free-token arm (`github_earlylead.py`, `--online`)

GitHub repository stars: a **free token** (no per-unit ceiling, no toppling).
Entity = repo; cohort = repos created in a fixed window. Every repo starts at 0
stars, which is a common starting POINT, not a homogeneous population: repos
plainly differ in quality, so this is the heterogeneous-drift case of section 1
and the `sqrt(tau)` null is cleared here at `g=0` with no amplification. Nothing
rests on beating it; the arm exists for the contrast with section 8.

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

- **diffusion** (`g=0`, the `sqrt(tau)` shape),
- **plain preferential attachment**, and
- **heterogeneity** (`sim_het`, `g=0` with an unequal per-step drift), the null
  that defeats the magnitude test (sections 1-2).

Result: the amplifier family collapses tightly (internal RMS `~0.01`) while
diffusion (`7x`), PA (`12x`) and heterogeneity (`10x`) all sit far off the master,
on **every seed** (`multiseed`, 8 seeds). The heterogeneity null does not collapse
at all (its own internal spread `~0.13` across a decade of `Var_a`), since each
`Var_a` traces a different shape: persistent heterogeneity, unlike amplification,
has no single rescaled shape. So the collapse separates the amplifier from PA
**and from mere inequality** by shape, even though raw magnitude cannot (section 2).

### 10b. Three real free-token domains (`real_data_collapse.py`)

Run the same collapse on real trajectories, all free/open data. Each stores
per-entity positions at fractions `tau` of its own horizon, so the `tau` grids
need **not** match: the collapse rescales each by its own `tau90` and
interpolates the rescaled shape onto a common axis. This rescaling is exactly what
makes the otherwise **incommensurable** real clocks (years of startup capital vs
months of stars vs a days-long vote sequence) comparable.

- **Startups** (`startup_earlylead.py`): position = cumulative capital raised;
  cohort = companies whose first round is in 2010 with >=2 rounds (a common entry
  POINT, not equal quality; 2325 companies); horizon ~60 months. Free token (funding not revoked).
  Open Crunchbase 2015 export (`scripts/data/crunchbase_rounds.csv`).
- **GitHub stars** and **Wikipedia support-only** counts, re-extracted on a common
  fine 18-point grid by `refine_grids.py` (below).
- plus the **revocable** Wikipedia net-support arm, for the free/revocable contrast.

Two tests: (1) collapse onto the *simulated* amplifier master, which **fails**,
because that master is calibration-dependent; (2) a **data-derived** master over
the three real free-token domains, the honest object. Finding: the three collapse
onto a common shape with mutual RMS spread `~0.07`, about `3x` closer to one
another than the `sqrt(tau)` null sits to them. This is a shared shape distinct
from diffusion, but a **loose** one.

Four honest limits, all reported by the script rather than suppressed:

1. Refining the grid (section 10c) *weakens* the collapse, since finer resolution
   exposes real small-`tau` shape differences that coarse binning hides.
2. The revocable arm is **not** resolved from the free-token master on Wikipedia's
   mild toppling (`0.9x` the mutual spread).
3. On the calibration-dependent arm the real domains sit `7.7`-`9.8x` the internal
   spread off the simulated master, while the `sqrt(tau)` null sits **closer**, at
   `4.1x`. The direction is unfavourable, which is why the data-derived collapse is
   the honest test and the one reported, but reporting only the mutual spread would
   conceal it.
4. The heterogeneity null, which the simulation (10a) separates cleanly, is **not**
   separated by the real data: the real master sits `~0.03` from it, nearer than the
   three domains sit to one another. That null carries a free parameter the
   amplifier family does not, so this is not a like-for-like defeat, but it is an
   alternative this test does not exclude.

With only three domains one cannot yet tell a power problem (too few, too
heterogeneously clocked) from a substantive limit (the "one exact curve" claim
being too strong); only adding domains separates them.

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

## 11. Non-identifiability worked example (`lichess_worked_example.py`, `--lichess`)

Demonstrates, on one large public dataset, the boundary between the two questions
the paper separates:

- **A (the map)**, whether the process is amplifying, is identifiable from a
  trajectory (sections 1, 10).
- **B (the seed)**, whether what got inflated was capability or position, skill or
  amplified noise, is **not**. The non-identifiability theorem says the map
  carries capability `k` and position `p` only through their sum `a = k + p`, and
  carries an early fluctuation and an inherited (latent) head start through one
  identical term, so no trajectory statistic recovers the split. Recovering B needs an exogenous,
  non-amplified handle on capability, returning a **manufactured share**
  `R = 1 - corr^2(final order, k-hat)`.

**Data & channel.** Online chess supplies a start that *looks* homogeneous
(entrants seeded near a common provisional rating), a plausible amplifier (rating
amplified and revocable game by game), and an exogenous `k-hat` (each player's
*converged* rating months later, a different sampling that averages the early
fluctuation away). The first two are less than they appear; the third is what
makes the lesson visible. Three monthly Lichess dumps are streamed via `zstd`:
`2013-01` (entry cohort), `2013-07`, `2013-12` (+6/+12-month `k-hat`).

**Algorithm.**
```
for each player in 2013-01 with >= 30 games:
    entry_elo   = rating at first game        # near-equal ESTIMATE, not near-equal skill
    monthend_elo = rating at last game        # the order that "forms" during the month
for the +6 and +12 month dumps:
    k_hat[player] = mean rating that month, for players with >= 20 games there
for band in {near-equal-rated 1480-1520, control 1000-2200}:
    sd(entry_elo), sd(monthend_elo)   # entry-band diagnostic: was the band ever equal?
    corr(entry_elo,    k_hat)   # does the seed predict true skill?
    corr(monthend_elo, k_hat)   # does the formed order predict true skill?
    R = 1 - pearson(log monthend_elo, log k_hat)^2
```

`R` is a share of **variance**, so the correlation in it must be a Pearson
correlation on the scale the model is written in (log-status). `1 - rho_S^2` is
not a variance share, so the Spearman values are printed alongside as the
order-only summary but never used for `R`; the two agree to within `0.05` in
every cell.

Two further routines guard the reading, both printed by the same script:

- `within_month_signatures` measures the two TRAJECTORY signatures the paper
  reports before `k-hat` is admitted, so they are computed rather than asserted:
  `corr(entry, month-end)` (`-0.05` near-equal vs `+0.60` control) and `rho(tau)`
  on each player's running net score against the `sqrt(tau)` law (excess positive
  everywhere, peaking at `+0.28`). Both are then shown to be reproducible at
  `g = 0` by Eq. `rhodrift`, so neither identifies amplification here.
- `survivorship_check` (`--survivorship`) reruns the headline row across
  convergence-filter thresholds of 1, 5, 10, 20, 40 later games. Staying active
  correlates with skill, so the filter could inflate `corr(month-end, k-hat)` and
  deflate `R` — the opposite direction to the attenuation argument. It returns
  `R = 0.45, 0.35, 0.35, 0.38, 0.33` with no monotone trend, and the loosest
  filter (almost no selection) gives the *highest* `R`, so the filter bounds the
  precision of `k-hat` rather than manufacturing the correlation.

**The band is not what its name suggests.** A 40-point band makes the entry
*estimate* narrow, not capability: Lichess seeds newcomers near 1500 precisely
because it does not know them yet, so the band selects players the system has not
yet told apart. The `Entry-band diagnostic` prints the evidence: the band's
month-end s.d. is `213.1` against the control's `211.4` and the whole active
population's `212.1`, i.e. the two bands share the same underlying `Var(ln k)` and
differ only in how much of it was visible at entry. Hence the naming
`near-equal-rated`, not `near-equal`.

**Result (Table `tab:lichess`).** The month-end order is decoupled from the entry
seed (`corr = -0.06`) and locks in faster than `sqrt(tau)`, so it *looks*
manufactured, yet it predicts converged skill at `0.79`, so `R ~ 0.38`. Because
measurement error in `k-hat` attenuates the correlation, that `0.38` is an **upper
bound** on manufacture, so at least about `62%` is revealed skill. The
heterogeneous control is skill-dominated already at entry
(`corr(entry, k-hat) = 0.74`). The example does not show chess hierarchies are
manufactured. It shows the opposite, and that is the methodological point, twice
over: the trajectory was silent about the seed (B), as it must be, and **not
entitled to its verdict on the dynamics (A) either**, since a 40-point band
produces the entry decoupling whatever the dynamics and an unequal population
produces the lock-in at `g=0` (section 1, `rho` with a heterogeneous drift). One
exogenous channel corrects both errors. The headline analysis has no randomness
and is deterministic given the dumps; `--bootstrap B` adds seeded
player-resampling CIs (`corr(month-end, k-hat) = 0.79 [0.70, 0.86]`,
`R = 0.38 [0.26, 0.51]` at `B = 2000`).

### Data provenance and integrity

Every dataset is public and free; `scripts/data/DATA_SOURCES.md` records each
source URL, its exact download command, and a SHA-256 hash, with machine-checkable
checksums in `scripts/data/SHA256SUMS` (verified by `run_all.sh` before running).
The only scripts that touch the network are the `--online` real-data passes
(sections 8, 9), the one-off Crunchbase fetch for 10b, and the `--lichess` worked
example (section 11, which fetches ~150 MB of Lichess dumps once, cached under
`scripts/data/lichess/` and not bundled).
