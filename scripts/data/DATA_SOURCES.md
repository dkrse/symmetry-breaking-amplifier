# Data provenance, download commands, and integrity hashes

Every dataset used in the paper is public and free. This file records, per source:
where it came from, the exact command that fetches it, and a SHA-256 hash so a
reproducer can verify byte-for-byte integrity. Machine-checkable checksums are in
`SHA256SUMS` (see *Verifying* at the bottom).

All paths below are relative to this directory (`scripts/data/`).

---

## 1. Wikipedia Requests for Adminship — `wiki-RfA.txt.gz`

- **Source:** SNAP (Stanford Large Network Dataset Collection), `wiki-RfA`.
- **URL:** <https://snap.stanford.edu/data/wiki-RfA.txt.gz>
- **Citation:** West, Paskov, Leskovec & Potts (2014), *TACL* 2:297–310; SNAP (Leskovec & Krevl, 2014).
- **Fetch:**
  ```bash
  curl -sL -o wiki-RfA.txt.gz https://snap.stanford.edu/data/wiki-RfA.txt.gz
  ```
  (The analysis script `wiki_rfa_toppling.py` also auto-downloads it here if absent.)
- **Used by:** `wiki_rfa_toppling.py`, `refine_grids.py` (toppling arm of Design 2).
- **SHA-256:** `88d53196fb2564a2e20286dbba818832f718cc352bb181a2101d23d2556f0862`

## 2. Startup funding rounds — `crunchbase_rounds.csv`

- **Source:** Crunchbase December-2015 open export, mirrored (read-only archive) at
  `notpeter/crunchbase-data`. Directly downloadable, no login.
- **URL:** <https://raw.githubusercontent.com/notpeter/crunchbase-data/master/rounds.csv>
- **Fetch:**
  ```bash
  curl -sL -o crunchbase_rounds.csv \
    https://raw.githubusercontent.com/notpeter/crunchbase-data/master/rounds.csv
  ```
  (`run_all.sh` fetches it once if absent; `startup_earlylead.py` prints this command if it is missing.)
- **Columns used:** `company_permalink`, `funded_at`, `raised_amount_usd`, `funding_round_type`.
- **Used by:** `startup_earlylead.py` (third real free-token domain of Test B).
- **SHA-256:** `89f575ed0d850e2890e903cde0bca8bdf66b028cac6fa298d534a252d6517315`

## 3. GitHub star timestamps — `github_cache/` (3114 JSON files)

- **Source:** GitHub GraphQL API (`stargazers { starredAt }`), one cached response
  per repo per page. Needs a token to (re)crawl, but the cache here is complete and
  the analysis runs offline from it.
- **Cohort:** repositories created `2019-01-01..2019-03-31`, final stars 80–3000,
  ~300 repos, 24-month horizon.
- **Recrawl (only if you want to rebuild the cache; needs a token):**
  ```bash
  export GITHUB_TOKEN=ghp_...
  python github_earlylead.py --created 2019-01-01..2019-03-31 \
      --min-stars 80 --max-stars 3000 --cohort 300 --horizon-months 24
  ```
- **Used by:** `github_earlylead.py`, `refine_grids.py` (free-token arm of Design 2).
- **Aggregate SHA-256** (order-independent digest of the 3114 files; **run from
  `scripts/data/`**):
  ```bash
  find github_cache -name '*.json' | sort | xargs sha256sum | sha256sum
  # => 54556cb41f059787a9b97b2975c45f80bfce93c835e8dd988396b38f7031abdc
  ```

## 4. Music Lab (Salganik–Dodds–Watts artificial cultural market) — `musiclab/`

- **Source:** Princeton dataset release (Salganik, 2026), DOI `10.34770/y56c-ym90`.
- **URL:** <https://doi.org/10.34770/y56c-ym90>
- **Citation:** Salganik, Dodds & Watts (2006), *Science* 311:854–856.
- **Files used:** `musiclab/musiclab_data/downloads_v{1,2,3,4}_lexorder.txt`
  (per-world final download counts; `README.txt` and `data_release.pdf` document the schema).
  Note: only **final** counts are bundled, not the timestamped `dynamics` files, so
  Music Lab supports Design 1 (causal decoupling) but not an early-lead trajectory.
- **Used by:** `musiclab_analysis.py` (Design 1).
- **SHA-256:** see `SHA256SUMS` (the four `downloads_v*_lexorder.txt`, plus README and PDF).

---

## Verifying

From this directory:

```bash
sha256sum -c SHA256SUMS          # checks files 1, 2, 4 byte-for-byte
# github_cache (file 3) is verified by its aggregate digest:
find github_cache -name '*.json' | sort | xargs sha256sum | sha256sum
#   expect: 54556cb41f059787a9b97b2975c45f80bfce93c835e8dd988396b38f7031abdc
```

`SHA256SUMS` lists the aggregate cache digest as a comment (it is path-sensitive,
so it is checked by the command above rather than by `sha256sum -c`).
