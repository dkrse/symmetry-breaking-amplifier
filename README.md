# Breaking Symmetry, Not Choosing Direction

A Model of Hierarchy as a Position Amplifier, and What It Can and Cannot Recover.

This repository contains  all analysis code
(`scripts/`), the bundled Music Lab dataset, and a single-command runner that
reproduces every figure and headline number.

## What the paper claims

Hierarchies form even among near-equals. Alongside the usual readings (meritocracy
sorting on ability, coordination, or cumulative advantage), the paper gives a
falsifiable **normal form** for a further possibility: position amplifying itself
from noise, a *symmetry-breaking amplifier of position*. On top of the normal form
it draws the line the title names, through what such a hierarchy's trajectory can
and cannot recover about its own origin. The line has two sides.

**A. The dynamics are identifiable.** Whether the process amplifies or merely
diffuses is legible from a trajectory, fixed by three results, each with an
explicit failure condition:

1. **Necessity theorem.** Without the multiplicative feedback the process is a
   martingale, so amplification (not mere multiplicativity) is what any structured,
   heavy-tailed development requires.
2. **Non-monotone, dispersion-maximising gain.** A per-unit ceiling makes
   manufactured dispersion peak and then fall with gain (horizon-dependent,
   `g* ~ 1/T`); plain cumulative advantage, with no ceiling, shows no such downturn.
3. **Capability-free `sqrt(tau)` early-lead law.** An early-lead-persistence law
   (`rho(tau)`, the `sqrt(tau)` diffusion baseline, and its collapse under
   revocation) that tells amplification from diffusion without measuring capability.

**B. The seed is *not* identifiable.** Whether what got inflated was capability
or position, real skill or amplified noise, is **not** recoverable from a
trajectory. This is the central new result, a **non-identifiability theorem**. The
map carries capability and position only through their *sum*, and weights a lucky
early draw almost like an inherited head start, so no statistic of the motion
recovers the split. The statement that the amplifier keeps no signature of what it
amplified is therefore a theorem rather than a metaphor, and cannot be refuted by
data. Recovering B needs an **exogenous, non-amplified handle on capability**
(quality fixed by construction, randomised, or a convergent later estimate), which
returns not a verdict but a **manufactured share** `R = 1 - corr^2`.

A single large public dataset shows both sides at once (`--lichess`). A near-equal
online-chess cohort whose order *looks* manufactured, in that it is decoupled from
entry and locks in faster than diffusion, turns out about `60%` **revealed** skill
(`R ~ 0.4`) once a convergent skill estimate is admitted.

Each result is stated with an explicit failure condition; the empirical passes are
suggestive rather than confirmatory, and the paper says so.

## Reproduce

```bash
./run_all.sh            # fast offline reproduction (figures + numbers, ~1-2 min)
./run_all.sh --power    # + Monte-Carlo power analysis (offline, several minutes)
./run_all.sh --online   # + Design 2 real-data passes (Wikipedia RfA + GitHub)
./run_all.sh --lichess  # + non-identifiability worked example (Lichess, ~150MB)
./run_all.sh --help
```

`run_all.sh` bootstraps a local virtual environment (`.venv`) from
`requirements.txt` (numpy, scipy, matplotlib, requests) and runs the analyses.
The default and `--power` tiers are fully offline (pure simulation or bundled
data); the network is touched only for the one-off `pip install`. The `--online`
tier runs the two real-data Design 2 passes. The `--lichess` tier runs the
non-identifiability worked example, streaming three monthly Lichess PGN dumps
(~150 MB, cached under `scripts/data/lichess/`); it needs the `zstd` CLI on PATH
(`apt install zstd`) and is skipped gracefully if absent.

## What each script does

| Script | Result | Network |
|---|---|---|
| `symmetry_breaking.py` | E1 to E4, optimal gain g\*, robustness, scale scan, early-lead persistence (simulation) | offline |
| `earlylead_pa_null.py` | preferential-attachment null + free-token/toppling discriminator (simulation) | offline |
| `data_collapse.py` | universality test: `rho(tau/tau90)` collapse, amplifier vs diffusion/PA (simulation, multi-seed) | offline |
| `critical_threshold.py` | lower absorbing barrier: buffer asymmetry decouples outcome from competence | offline |
| `dose_response.py` | joint dose-response fingerprint (three signatures move with one gain), Music Lab anchor | offline |
| `regulation.py` | emergent maintenance ceiling S\*/D = 1 across a factor-16 gain range | offline |
| `musiclab_analysis.py` | Design 1: causal decoupling on the bundled Music Lab data | offline |
| `startup_earlylead.py` | third free-token domain: startup funding (open Crunchbase 2010 cohort) | offline\* |
| `refine_grids.py` | re-extract GitHub + Wikipedia early-lead curves on a common 18-point grid | offline |
| `real_data_collapse.py` | Test B on real data: collapse of three free-token domains + revocable arm | offline |
| `power_analysis.py` | Design 1 power + cohort-concentration power + download-market Gini sweep (`--power`) | offline |
| `wiki_rfa_toppling.py` | Design 2 toppling arm: early-lead persistence on Wikipedia RfA (`--online`) | SNAP download |
| `github_earlylead.py` | Design 2 free-token arm: early-lead persistence on GitHub stars (`--online`) | GitHub API |
| `plot_realdata_earlylead.py` | combined real-data early-lead figure (GitHub + Wikipedia arms) | offline |
| `lichess_worked_example.py` | non-identifiability worked example: manufactured share `R` and its convergent-`k` discriminator on online chess (Table `tab:lichess`) | Lichess dumps |

\* `startup_earlylead.py` runs offline once `run_all.sh` has fetched the open
Crunchbase `rounds.csv` (a one-off ~19 MB download); it is skipped with no network.

All simulation randomness is seeded (`numpy.random.default_rng`); figures
regenerate deterministically into `output/figures/`. Every input dataset is
SHA-256 hashed in `scripts/data/SHA256SUMS`, with source URLs and download
commands in `scripts/data/DATA_SOURCES.md`.

## The two Design 2 arms (`--online`)

Early-lead persistence `rho(tau)` on a monotone accumulating stock does **not** by
itself distinguish this amplifier from ordinary cumulative advantage: plain
preferential attachment already exceeds the `sqrt(tau)` diffusion null (shown by
`earlylead_pa_null.py`). The identifying test is the **free-token vs. toppling
contrast**:

- **Free-token arm** — `github_earlylead.py` on GitHub stars (no per-unit ceiling,
  no revocation): the model predicts *strong* early-lead persistence.
- **Toppling arm** — `wiki_rfa_toppling.py` on Wikipedia Requests for Adminship, a
  contested, revocable status: the model predicts persistence *collapses* toward
  the diffusion regime, because opposition can topple an early front-runner. The
  script measures both arms inside the one RfA dataset (net support = revocable;
  support-only cumulative = free-token reference).

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
public dataset. Online chess supplies three things together: a near-homogeneous
start (every entrant seeded near a common provisional rating), a genuine amplifier
(the rating is amplified and revocable game by game), and an exogenous channel to
capability (each player's *converged* rating months later, once hundreds of games
have averaged the early fluctuation away, which is the `k-hat` of `R = 1 - corr^2`).

The script streams three monthly [Lichess open-database](https://database.lichess.org)
dumps (2013-01 as the entry cohort; 2013-07 and 2013-12 as the +6- and +12-month
convergent skill estimates), forms a near-equal entry-Elo cohort (1480–1520) and a
heterogeneous control (1000–2200), and prints `Table tab:lichess`:

- **The order looks manufactured.** In the near-equal cohort the order that forms
  is decoupled from the entry rating (`corr(entry, month-end) = -0.03`) with
  super-diffusive early-lead lock-in. From the trajectory alone, it looks
  manufactured.
- **The convergent channel refutes it.** The convergent `k-hat` overturns that
  reading. The month-end order predicts converged skill at `0.77`, so
  `R = 1 - 0.77^2 ~ 0.40`, an *upper* bound on manufacture (measurement error in
  `k-hat` attenuates the correlation), so at least about `60%` is revealed skill.
  The example does not show chess hierarchies are manufactured. It shows the
  opposite, and that is the point. Signature A was correct about the dynamics and
  silent, as it must be, about the seed.

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
