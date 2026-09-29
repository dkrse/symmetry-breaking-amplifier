#!/usr/bin/env bash
#
# run_all.sh -- single entry point that reproduces every result in the paper.
#
# It bootstraps a local virtual environment (.venv) from requirements.txt and
# runs the analysis scripts in scripts/. By default it runs the fast OFFLINE
# reproduction (pure simulations + bundled Music Lab data), which regenerates
# every figure and prints every headline number in well under a minute.
# No analysis script touches the network; only the one-off pip bootstrap does.
# Two opt-in tiers add slower or networked work:
#   --power    also run the heavy offline passes: the Monte-Carlo power analysis,
#              the dense-grid g*(T) horizon-scaling fit, and the Kesten tail-index
#              check (all offline; ~22 min, dominated by the Kesten check)
#   --online   also run the Design 2 real-data passes: Wikipedia RfA (toppling
#              arm, downloads SNAP data) and, if GITHUB_TOKEN is set, GitHub
#              stars (free-token arm)
#   --lichess  also run the non-identifiability worked example (Section
#              "manufacture vs revelation in online chess"): downloads ~150 MB
#              of monthly Lichess PGN dumps once and prints Table tab:lichess
#
# Usage:
#   ./run_all.sh                   # fast offline reproduction (figures + numbers)
#   ./run_all.sh --power           # + power analysis, g*(T) fit, Kesten tail (slow, offline)
#   ./run_all.sh --online          # + Wikipedia RfA + GitHub stars (Design 2)
#   ./run_all.sh --lichess         # + Lichess non-identifiability worked example
#   ./run_all.sh --power --online --lichess  # everything
#   ./run_all.sh --help
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
SCRIPTS="$ROOT/scripts"

usage() { sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

ONLINE=0
POWER=0
LICHESS=0
for arg in "$@"; do
  case "$arg" in
    --online) ONLINE=1 ;;
    --power) POWER=1 ;;
    --lichess) LICHESS=1 ;;
    -h|--help) usage ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

# ---- 1. bootstrap the virtual environment ---------------------------------
if [ ! -x "$PY" ]; then
  echo ">> creating virtual environment in .venv"
  python3 -m venv "$VENV"
fi
echo ">> installing dependencies from requirements.txt"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$ROOT/requirements.txt"
echo ">> using $("$PY" --version) with $("$PY" -c 'import numpy,scipy,matplotlib;print("numpy",numpy.__version__,"scipy",scipy.__version__,"matplotlib",matplotlib.__version__)')"

# Wall-clock per script, so a run always reports how long each stage took and the
# README's estimates can be checked rather than trusted.
T0_ALL=$SECONDS
run() {
  echo
  echo "=================================================================="
  echo ">> $1"
  echo "=================================================================="
  local t0=$SECONDS
  ( cd "$SCRIPTS" && "$PY" "$@" )
  local dt=$(( SECONDS - t0 ))
  printf '<< %s finished in %dm %02ds\n' "$1" $(( dt / 60 )) $(( dt % 60 ))
}

# ---- 1b. data integrity check (provenance + hashes in scripts/data/) ------
# Verifies bundled inputs byte-for-byte against scripts/data/SHA256SUMS. Non-fatal:
# a fresh clone may not yet have the once-fetched files (Crunchbase). See
# scripts/data/DATA_SOURCES.md for every source URL and download command.
if [ -f "$ROOT/scripts/data/SHA256SUMS" ]; then
  echo
  echo ">> verifying data integrity (scripts/data/SHA256SUMS)"
  ( cd "$ROOT/scripts/data" && grep -v '^#' SHA256SUMS | grep -v 'github_cache/\[' \
      | sha256sum -c 2>/dev/null ) || echo ">> some data files absent or changed (see scripts/data/DATA_SOURCES.md)"
fi

# ---- 2. offline reproduction (simulations + bundled data) -----------------
echo
echo "### OFFLINE reproduction: simulations, figures, and bundled-data tests"
run symmetry_breaking.py     # E1-E4, optimal gain, robustness, early-lead (sim)
run earlylead_pa_null.py     # PA-null + free-token/toppling discriminator (sim)
run data_collapse.py         # Test B: rho(tau) universality collapse (sim + multi-seed)
# Startup funding domain (open Crunchbase 2015 export). Downloads rounds.csv once
# if absent; skips gracefully with no network.
if [ ! -f "$ROOT/scripts/data/crunchbase_rounds.csv" ]; then
  echo ">> fetching open Crunchbase rounds.csv (one-off, ~19MB)"
  curl -sL --max-time 120 -o "$ROOT/scripts/data/crunchbase_rounds.csv" \
    https://raw.githubusercontent.com/notpeter/crunchbase-data/master/rounds.csv \
    || echo ">> download failed (offline?); startup domain will be skipped"
fi
[ -f "$ROOT/scripts/data/crunchbase_rounds.csv" ] && run startup_earlylead.py  # 3rd real free-token domain
# Fine-grid re-extraction of GitHub (from cached star timestamps) and Wikipedia
# (from bundled raw gz), so all real domains share an 18-point grid. Offline.
if [ -f "$ROOT/output/github_earlylead.csv" ] && [ -f "$ROOT/scripts/data/wiki-RfA.txt.gz" ]; then
  run refine_grids.py
fi
# Test B on real data: collapses startups + GitHub + Wikipedia RfA. Uses whichever
# early-lead CSVs are present; needs at least GitHub + Wikipedia (bundled).
if [ -f "$ROOT/output/github_earlylead.csv" ] && [ -f "$ROOT/output/wiki_rfa_earlylead.csv" ]; then
  run real_data_collapse.py  # Test B on real trajectories (startups if available)
else
  echo ">> skipping real_data_collapse.py: needs output/github_earlylead.csv and output/wiki_rfa_earlylead.csv (run --online)"
fi
run dose_response.py         # joint dose-response fingerprint (sim; ML arms in a side panel)
run critical_threshold.py    # lower absorbing barrier: buffer decouples outcome
run regulation.py            # emergent ceiling S*/D = 1
run musiclab_analysis.py     # Design 1 (causal decoupling) on bundled Music Lab data

# ---- 3. slow offline heavy passes (Monte-Carlo + long simulations, opt-in) --
if [ "$POWER" -eq 1 ]; then
  echo
  echo "### HEAVY offline passes: Monte-Carlo and long simulations (~22 min)"
  run power_analysis.py        # Design 1 & 2 power analysis
  run optimal_gain_fit.py      # dense-grid fit of the g*(T) horizon scaling (~1 min)
  run kesten_tail.py           # Kesten tail index: Cramer root vs Hill (~15 min)
else
  echo
  echo ">> skipping heavy offline passes (power analysis, g*(T) fit, Kesten tail);"
  echo "   use --power to include them"
fi

# ---- 4. online passes (Design 2, real data) -------------------------------
if [ "$ONLINE" -eq 1 ]; then
  echo
  echo "### ONLINE passes: Design 2 early-lead persistence on real data"
  # --bootstrap reproduces the CIs on the free-token-minus-revocable gap quoted
  # in the text; the script downloads the SNAP data if it is not bundled.
  run wiki_rfa_toppling.py --bootstrap 2000
  # one functional (net support), contestability varied across elections:
  # quintiles of oppose share, bin-matched i.i.d. placebo, oppose-vote timing
  run wiki_rfa_contestability.py --bootstrap 1000
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    run github_earlylead.py    # free-token arm: GitHub stars (needs GITHUB_TOKEN)
  else
    echo ">> skipping github_earlylead.py: set GITHUB_TOKEN to run the free-token arm"
  fi
  # combined figure (needs both arms' CSVs; skips gracefully if either is missing)
  if [ -f "$ROOT/output/github_earlylead.csv" ] && [ -f "$ROOT/output/wiki_rfa_earlylead.csv" ]; then
    run plot_realdata_earlylead.py
  fi
else
  echo
  echo ">> skipping online passes (Wikipedia RfA download, GitHub API); use --online to include them"
fi

# ---- 5. Lichess non-identifiability worked example (real data, opt-in) -----
if [ "$LICHESS" -eq 1 ]; then
  echo
  echo "### LICHESS worked example: manufacture vs revelation (Table tab:lichess)"
  if ! command -v zstd >/dev/null 2>&1; then
    echo ">> skipping: the 'zstd' CLI is required to stream the PGN dumps (apt install zstd)"
  else
    # --bootstrap/--survivorship reproduce the CIs and the selection sweep in
    # Table tab:lichess and its caveat paragraph.
    run lichess_worked_example.py --bootstrap 2000 --survivorship
    # single-channel test (increment variance and curvature vs k-hat) and
    # inverse-probability weighting for attrition (Section 6.4, Robustness i, iv)
    run lichess_extra_checks.py --bootstrap 1000
  fi
else
  echo
  echo ">> skipping Lichess worked example (~150MB download); use --lichess to include it"
fi

echo
printf '>> done in %dm %02ds. Figures written to output/figures/\n' \
  $(( (SECONDS - T0_ALL) / 60 )) $(( (SECONDS - T0_ALL) % 60 ))
