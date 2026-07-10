# Breaking Symmetry, Not Choosing Direction

A Model of Hierarchy as a Position Amplifier, and What It Can and Cannot Recover.

This repository contains  all analysis code
(`scripts/`), the bundled Music Lab dataset, and a single-command runner that
reproduces every figure and headline number.

## What the paper claims

Hierarchies form even among near-equals. Alongside the usual readings (meritocracy
sorting on ability, coordination, or cumulative advantage), the paper gives a
falsifiable **normal form** for a further possibility: position amplifying itself
from noise, a *symmetry-breaking amplifier of position*. The mechanism is not new;
the contribution is to make it precise and to locate it against its rivals, in
three results:

1. **Necessity theorem** — without the multiplicative feedback the process is a
   martingale, so amplification (not mere multiplicativity) is what any structured,
   heavy-tailed development requires.
2. **Non-monotone, dispersion-maximising gain** — a per-unit ceiling makes
   manufactured dispersion peak and then fall with gain (horizon-dependent,
   `g* ~ 1/T`); plain cumulative advantage, with no ceiling, shows no such downturn.
3. **Capability-free signatures** — an early-lead-persistence law (`rho(tau)`, the
   `sqrt(tau)` diffusion baseline, and its collapse under revocation) that tells
   positional amplification apart from meritocracy, coordination, and cumulative
   advantage without ever measuring capability.

Each is stated with an explicit failure condition; the empirical passes are
suggestive rather than confirmatory, and the paper says so.

## Reproduce

```bash
./run_all.sh            # fast offline reproduction (figures + numbers, ~1-2 min)
./run_all.sh --power    # + Monte-Carlo power analysis (offline, several minutes)
./run_all.sh --online   # + Design 2 real-data passes (Wikipedia RfA + GitHub)
./run_all.sh --help
```

`run_all.sh` bootstraps a local virtual environment (`.venv`) from
`requirements.txt` (numpy, scipy, matplotlib, requests) and runs the analyses.
The default and `--power` tiers are fully offline (pure simulation or bundled
data); the network is touched only for the one-off `pip install`. The `--online`
tier runs the two real-data Design 2 passes.

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

## Build the paper

The paper reads its figures from `output/figures/`, so run `./run_all.sh` first to
generate them, then compile from the repository root (the `\graphicspath` is
relative to it):

```bash
pdflatex -output-directory=paper paper/paper.tex   # run twice for references
```

## License

## Author
krse

See `LICENSE`.
