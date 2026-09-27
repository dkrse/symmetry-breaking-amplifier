# Multiplicative amplification of position in social hierarchies

Identifiable dynamics and an unidentifiable seed: what a hierarchy's trajectory
can, and cannot, reveal about its own origin.

This repository contains all analysis code
(`scripts/`), the bundled Music Lab dataset, and a single-command runner that
reproduces every figure and headline number.

## What the paper claims

This is a **corrective** paper, not a discovery one, and it is worth being plain
about that up front. It claims no new mechanism: the multiplicative cascade is
standard and the propositions are elementary on purpose. What it does is show that
several signatures in current use **do not identify what they are taken to
identify**, and give the designs that do.

**The inference under attack.** Trajectories of positions are widely used to argue
that a hierarchy was built by amplification: the early ordering predicts the final
one far better than diffusion allows, and that excess is read as evidence of
compounding. The excess is real. It is also **not diagnostic** — entities that
merely differ in a persistent trait produce it at zero gain, with no feedback
compounding anything, and because that null carries a free parameter it has *no
characteristic lock-in speed at all*: sweeping `Var(a)` over four decades moves
`tau90` from `0.79` to `0.005`, straddling the whole range an amplifier occupies
and the preferential-attachment null with it. Magnitude therefore separates
nothing. What survives is a **contrast internal to one system**.

The question is then asked inside a falsifiable **normal form** for one route to
hierarchy: position amplifying itself from noise, a *symmetry-breaking amplifier
of position*. That is the vehicle, not the thesis — the paper does **not** claim
real hierarchies are position amplifiers, and the one case it examines with an
exogenous channel turns out to be mostly earned. The answer splits in two.

**A. The dynamics are identifiable.** Whether the process amplifies or merely
diffuses is legible from a trajectory, fixed by four results, each with an
explicit failure condition:

1. **Necessity theorem.** Without the multiplicative feedback the process is a
   martingale, so amplification (not mere multiplicativity) is what any structured,
   heavy-tailed development requires.
2. **Non-monotone, dispersion-maximising gain.** A per-unit ceiling makes
   manufactured dispersion peak and then fall with gain; plain cumulative
   advantage, with no ceiling, shows no such downturn. The peak is
   horizon-dependent and vanishes as `T -> inf`. The mean-field estimate
   `g* ~ 1/T` gets that direction but **not** the exponent: on a dense gain grid
   over six horizons, `optimal_gain_fit.py` measures `g* ~ T^-0.56 +/- 0.02`
   (and `-0.44` restricted to `T >= 600`, i.e. further from 1, not closer). The
   robust claim is the non-monotonicity, not any particular optimum or scaling.
3. **Closed-form Kesten tail index.** In the saturated regime the stationary law
   is Pareto, `P(S > s) ~ s^-mu`, with
   `mu = (-c + sqrt(c^2 - 2 sigma^2 ln(1-p_hi))) / sigma^2` and
   `c = ln(1 + g theta S*)`. For `mu sigma^2 << c` this is `mu ~ -ln(1-p_hi)/c`:
   **contestability sets the tail**. `kesten_tail.py` confirms it against a Hill
   estimate (mean error 3.1% over the twelve grid cells with `mu <~ 2`). Scope: the
   derivation assumes the saturated regime, so it holds for *maintained,
   revocable* statuses and **not** for ceiling-free tokens, which lie outside it
   rather than at its `p_hi -> 0` limit.
4. **Capability-free `sqrt(tau)` early-lead law.** An early-lead-persistence law
   (`rho(tau)`, the `sqrt(tau)` diffusion baseline, and its collapse under
   revocation) that tells amplification from diffusion without measuring capability.
   The `sqrt(tau)` baseline holds only for a homogeneous *population*, not merely a
   common start: unequal entities clear it at zero gain, so in data the identifying
   test is the revocation **contrast**, never the size of the excess. Note also
   that `sqrt(tau)` is the null for the *Pearson* correlation, while `rho(tau)` is
   measured as a *rank* correlation; at `g=0` the pair is bivariate normal, so the
   rank-scale null is exactly `(6/pi) arcsin(sqrt(tau)/2)`, slightly **below**
   `sqrt(tau)`. That accounts for the whole systematic residual of the simulated
   `g=0` curve (mean `-0.0088` against `sqrt(tau)`, `+0.0007` against the rank
   null), and it makes `sqrt(tau)` the conservative choice on real data.

**B. The seed is *not* identifiable.** The sharpest defensible form of the result
is this: **the decomposition of the amplified input is not a function of the
trajectory.** Every clause matters, and each rules out a misreading. (The paper is
equally plain about what is *not* deep here: given that the amplifier receives
capability and position through one channel, their non-separability is close to
definitional. Its content is that the single channel is forced by the normal form
and is refutable — a capability-dependent gain or noise would break it and would
leave trajectory signatures.)

- **Decomposition**, not everything. The persistent input `a = k + p` *is*
  recovered, from the drift of the trajectory, and so are the dynamics (A). What
  is lost is only the *split* — capability against position, and within position a
  recurrent advantage against an amplified transient.
- **Not a function**, rather than badly estimated. Two parameter families induce
  the **same distribution over observables**, so there is no consistent estimator
  to be had — not a noisy one. Longer horizons, more entities, denser sampling:
  none of it bears on the question, because the likelihood is flat along that
  direction. The signature is not hard to read, it is absent.
- **Of the trajectory**, not of the world. The split is *not* unknowable; it is
  unknowable **from this observable**. A different kind of observation settles it
  immediately — quality fixed by construction, a randomised seed, or a convergent
  later estimate.
- And what such a channel returns is a **proportion, not a verdict**: the
  manufactured share `R = 1 - corr^2`, somewhere between pure symmetry-breaking
  and pure revelation rather than a decision between them.

**The boundary has teeth, and the central exhibit is one the paper loses**
(`--lichess`). An online-chess cohort with near-equal *entry ratings* displays
*every* signature the amplifier reading predicts — a near-equal start, an order
decoupled from it, lock-in faster than the `sqrt(tau)` law. A trajectory-only
analysis would therefore classify it as manufactured order. Admit a convergent
skill estimate and it is about `62%` **revealed** skill (`R ~ 0.38`). The reading
is wrong. The trajectory is not silent throughout — the cohort's own dispersion
betrays that its near-equal entry was an artefact of measurement, so the *dynamics*
reading (A) was never entitled — but between manufacture and revelation (B) it says
nothing at all, and that is the half no amount of data repairs. The boundary is not
a caveat to note and move past, it is the difference between a right and a wrong
answer.

Each result is stated with an explicit failure condition; the empirical passes are
suggestive rather than confirmatory, and the paper says so. In particular the
Music Lab two-point dose-response carries a presentation confound.

**Removed in revision 1.** The original submission also reported a cross-domain
`rho(tau/tau90)` curve-collapse test (simulation plus three real free-token
domains, including a Crunchbase startup cohort). Its real-data half was
inconclusive and supported no conclusion, so it was removed from the revised
manuscript. The code and data are kept here for completeness; see the
**Paper-to-repository map** below for what the revised paper does and does not
use.

## Reproduce

```bash
./run_all.sh            # fast offline reproduction: figures + every headline number
./run_all.sh --power    # + power analysis, g*(T) fit, Kesten tail index
./run_all.sh --online   # + Design 2 real-data passes (Wikipedia RfA + GitHub)
./run_all.sh --lichess  # + non-identifiability worked example (Lichess)
./run_all.sh --power --online --lichess   # everything
./run_all.sh --help
```

### How long it takes

Measured wall-clock on one ordinary workstation (8-core, no GPU). Every run also
prints its own per-script timing, so these are checkable rather than promises.

| Tier | Wall-clock | Dominated by |
|---|---|---|
| default (offline) | **≈30 s** | nothing; the ten scripts are under 10 s each |
| `--power` | **≈22 min** | `kesten_tail.py` 15 min, `power_analysis.py` 5.5 min |
| `--online` | **≈30 s** | Wikipedia RfA 28 s (+ GitHub API, see below) |
| `--lichess` | **≈20 s** | once the dumps are cached; see the caveat |
| **everything** | **≈23 min** | |

Two first-run costs are not in the table. The one-off `pip install` into `.venv`
takes a minute or two, and `--lichess` streams ≈150 MB of Lichess dumps on its
first invocation (cached afterwards under `scripts/data/lichess/`), which is
bandwidth-bound. The `--online` figure assumes the bundled GitHub star cache; with
`GITHUB_TOKEN` set and a cold cache the GitHub arm re-queries the API and is
rate-limited rather than compute-limited.

`run_all.sh` bootstraps a local virtual environment (`.venv`) from
`requirements.txt` (numpy, scipy, matplotlib, requests) and runs the analyses.
The default and `--power` tiers are fully offline (pure simulation or bundled
data); the network is touched only for the one-off `pip install`. The `--power`
tier adds three heavy offline passes: the Monte-Carlo power analysis (≈5.5 min),
the dense-grid `g*(T)` fit (≈1 min) and the Kesten tail-index check (≈15 min). The
`--online` tier runs the two real-data Design 2 passes. The `--lichess` tier runs the
non-identifiability worked example, streaming three monthly Lichess PGN dumps
(≈150 MB, cached under `scripts/data/lichess/`); it needs the `zstd` CLI on PATH
(`apt install zstd`) and is skipped gracefully if absent.

## What each script does

| Script | Result | Network |
|---|---|---|
| `symmetry_breaking.py` | E1 to E4, robustness, scale scan, early-lead persistence (simulation); its coarse `optimal_gain_scan` is superseded by `optimal_gain_fit.py` | offline |
| `optimal_gain_fit.py` | dense-grid, sub-grid-resolved fit of the g\*(T) horizon scaling (simulation; `--power`, ≈1 min) | offline |
| `kesten_tail.py` | Kesten tail index: closed-form Cramer root vs Hill estimate (simulation; `--power`, ≈15 min) | offline |
| `earlylead_pa_null.py` | preferential-attachment null + free-token/toppling discriminator (simulation) | offline |
| `data_collapse.py` | heterogeneous-drift sweep of `tau90` over four decades of `Var(a)` (paper Sec. 5.2); also the `rho(tau/tau90)` collapse test, **not in the revised paper** | offline |
| `critical_threshold.py` | lower absorbing barrier: buffer asymmetry decouples outcome from competence | offline |
| `dose_response.py` | joint dose-response fingerprint (three signatures move with one gain), Music Lab anchor | offline |
| `regulation.py` | emergent maintenance ceiling S\*/D = 1 across a factor-16 gain range | offline |
| `musiclab_analysis.py` | Design 1: causal decoupling on the bundled Music Lab data | offline |
| `startup_earlylead.py` | startup funding (open Crunchbase 2010 cohort); **not in the revised paper**, kept for completeness | offline\* |
| `refine_grids.py` | re-extract GitHub + Wikipedia early-lead curves on a common 18-point grid; **not in the revised paper** | offline |
| `real_data_collapse.py` | real-data `rho(tau/tau90)` collapse of three free-token domains; **not in the revised paper** (inconclusive, removed in revision 1) | offline |
| `power_analysis.py` | Design 1 power + cohort-concentration power + download-market Gini sweep (`--power`) | offline |
| `wiki_rfa_toppling.py` | Design 2 toppling arm: early-lead persistence on Wikipedia RfA (`--online`) | SNAP download |
| `github_earlylead.py` | Design 2 free-token arm: early-lead persistence on GitHub stars (`--online`) | GitHub API |
| `plot_realdata_earlylead.py` | combined real-data early-lead figure (GitHub + Wikipedia arms) | offline |
| `lichess_worked_example.py` | non-identifiability worked example: manufactured share `R` and its convergent-`k` discriminator on online chess (Table `tab:lichess`) | Lichess dumps |

\* `startup_earlylead.py` runs offline once `run_all.sh` has fetched the open
Crunchbase `rounds.csv` (a one-off ≈19 MB download); it is skipped with no network.

All simulation randomness is seeded (`numpy.random.default_rng`); figures
regenerate deterministically into `output/figures/`. Every input dataset is
SHA-256 hashed in `scripts/data/SHA256SUMS`, with source URLs and download
commands in `scripts/data/DATA_SOURCES.md`.

## Paper-to-repository map (revision 1)

Section numbers refer to the revised manuscript.

| Paper | Result | Script | Figure / table |
|---|---|---|---|
| 2.3, Prop. 1 | maintenance ceiling `S*/D = 1` | `regulation.py` | Table 2 |
| 3.1 | dispersion non-monotone in gain, `A(g)` | `symmetry_breaking.py` (E1) | Fig. 1 |
| 3.1 | `g*(T)` horizon scaling | `optimal_gain_fit.py` (`--power`) | Fig. 2 |
| 3.1 | free-token Gini rises monotonically | `power_analysis.py` (`--power`) | text |
| 3.2, Prop. 2 | amplification necessary; scope scan | `symmetry_breaking.py` (E4, `scale_scan`) | Fig. 3 |
| 3.3 | closed-form tail index vs Hill | `kesten_tail.py` (`--power`) | Fig. 4, Table 2 |
| 3.4 | non-ergodicity, dead ends | `symmetry_breaking.py` (E3) | Fig. 5 |
| 3.5 | parameter sensitivity | `symmetry_breaking.py` (`robustness_scan`) | Table 3 |
| 4.3 | gain decouples outcome from quality | `symmetry_breaking.py` (E2) | Fig. 6 |
| 5.1 | `sqrt(tau)` law and rank null | `symmetry_breaking.py` (`early_lead_persistence`) | Fig. 7 |
| 5.2 | heterogeneity sweep of `tau90` | `data_collapse.py` | text |
| 5.3 | PA null, toppling discriminator | `earlylead_pa_null.py` | text |
| 6.1 | Wikipedia revocation contrast, bootstrap, placebo | `wiki_rfa_toppling.py` (`--online`) | Fig. 8 |
| 6.2 | GitHub stars free-token arm | `github_earlylead.py` (`--online`), `plot_realdata_earlylead.py` | Fig. 8 |
| 6.3 | Music Lab decoupling, dose-response, power | `musiclab_analysis.py`, `dose_response.py`, `power_analysis.py` | Fig. 9 |
| 6.4 | manufactured share in online chess | `lichess_worked_example.py` (`--lichess`) | Table 4 |
| Appendix A | lower threshold, buffer asymmetry | `critical_threshold.py` | Fig. A.1 |

Not used by the revised paper: `startup_earlylead.py`, `refine_grids.py`,
`real_data_collapse.py`, the collapse half of `data_collapse.py`, and the figures
`data_collapse.png` and `real_data_collapse.png`. `run_all.sh` still produces
them.

## The two Design 2 arms (`--online`)

Early-lead persistence `rho(tau)` on a monotone accumulating stock does **not** by
itself distinguish this amplifier from anything. Plain preferential attachment
already exceeds the `sqrt(tau)` diffusion null (shown by `earlylead_pa_null.py`),
and so does a population at **zero gain** whose entities merely differ in quality,
since a persistent trait predicts the final order with no feedback at all. The
identifying test is therefore the **free-token vs. toppling contrast**, which
heterogeneity inflates on both arms alike and so cannot manufacture:

- **Free-token arm** — `github_earlylead.py` on GitHub stars (no per-unit ceiling,
  no revocation): the model predicts *strong* early-lead persistence. Read only
  against the toppling arm, never as a magnitude in its own right.
- **Toppling arm** — `wiki_rfa_toppling.py` on Wikipedia Requests for Adminship, a
  contested, revocable status: the model predicts persistence *collapses* toward
  the diffusion regime, because opposition can topple an early front-runner. The
  script measures both arms inside the one RfA dataset (net support = revocable;
  support-only cumulative = free-token reference). `run_all.sh --online` passes
  `--bootstrap 2000`, resampling elections to put a 95% CI on the contrast (gap at
  `tau=0.1`: `0.11 [0.09, 0.13]`; `tau90` difference `0.275` in the full sample,
  bootstrap mean `0.245 [0.20, 0.28]`; zero excluded at every `tau < 1`). It also
  runs a **heterogeneity placebo**: with i.i.d. votes at
  zero gain and candidates carrying the empirical spread of final support shares,
  the same contrast returns a mean early gap of only `+0.033` against `+0.115` in
  the data, so candidate heterogeneity buys just under a third of the observed gap
  and cannot account for it.

### GitHub token

`github_earlylead.py` reads public star timestamps through the GitHub API, which
is rate-limited to 60 requests/hour unauthenticated and 5000/hour with a token.
The script therefore needs a **personal access token** (never committed):

1. Go to <https://github.com/settings/tokens>.
2. Create either a **fine-grained token** with *Repository access → Public
   repositories (read-only)*, or a **classic token** with **no scopes ticked**
   (an unscoped token already reads public data at 5000 req/h).
3. Export it before running:
   ```bash
   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
   ./run_all.sh --online          # runs the GitHub arm; skipped if the var is unset
   ```

Every API response is cached under `scripts/data/github_cache/`, so re-runs are
free. The Wikipedia RfA arm needs no token (it downloads a public SNAP file once).

## The non-identifiability worked example (`--lichess`)

`lichess_worked_example.py` demonstrates the boundary between A and B on one large
public dataset. Online chess supplies a start that *looks* homogeneous (every
entrant seeded near a common provisional rating), a plausible amplifier (the
rating is amplified and revocable game by game), and an exogenous channel to
capability (each player's *converged* rating months later, once hundreds of games
have averaged the early fluctuation away, which is the `k-hat` of `R = 1 - corr^2`).
The first two turn out to be less than they appear; the third is what makes the
lesson visible.

The script streams three monthly [Lichess open-database](https://database.lichess.org)
dumps (2013-01 as the entry cohort; 2013-07 and 2013-12 as the +6- and +12-month
convergent skill estimates), forms a near-equal-**rated** entry cohort (1480–1520)
and a heterogeneous control (1000–2200), and prints `Table tab:lichess`:

- **The order looks manufactured.** The order that forms is decoupled from the
  entry rating (`corr(entry, month-end) = -0.05`, against `+0.60` in the
  heterogeneous control) and locks in faster than the `sqrt(tau)` law (the excess
  is positive at every `tau` and peaks at `+0.28`). From the trajectory alone, it
  looks manufactured.
- **Neither appearance survives.** The band equalised the entry *estimate*, not
  capability: Lichess seeds newcomers near 1500 precisely because it does not know
  them yet. The script's `Entry-band diagnostic` makes this visible, as the
  cohort's month-end spread (s.d. `213`) lands on the whole population's (`212`).
  So the entry decoupling is what a 40-point band produces whatever the dynamics,
  and the lock-in is what an unequal population produces at zero gain. Neither
  signature identifies A here.
- **The convergent channel settles B.** The month-end order predicts converged
  skill at `0.79` (Pearson on log-ratings, 95% bootstrap CI `[0.70, 0.86]` over
  players; `run_all.sh --lichess` passes `--bootstrap 2000 --survivorship`), so
  `R = 1 - 0.79^2 ~ 0.38` (CI `[0.26, 0.51]`), an *upper* bound on manufacture
  (it charges the amplified trajectory noise to manufacture, and measurement error
  in `k-hat` attenuates the correlation on top of that), so at least about
  `62%` is revealed skill. `R` is computed from a **Pearson** correlation because
  it is a variance share (`1 - rho_Spearman^2` is not); the rank correlations are
  printed alongside and agree to within `0.05`. `--survivorship` checks that the
  convergence filter is not doing the work: sweeping its threshold over 1, 5, 10,
  20, 40 later games returns `R = 0.45, 0.35, 0.35, 0.38, 0.33`, with no monotone
  trend. The example does not show chess hierarchies are
  manufactured. It shows the opposite, and that is the point: what looked like an
  amplifier acting on noise was shown to be neither, and one exogenous channel
  corrects both errors.

## Build the paper

The paper reads its figures from `output/figures/`, so run `./run_all.sh` first to
generate them, then compile from the repository root (the `\graphicspath` is
relative to it):

```bash
pdflatex -output-directory=paper paper/paper.tex   # run twice for references
```

## License

Released under the MIT License. See `LICENSE`.

## Author
krse
