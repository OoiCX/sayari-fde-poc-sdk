# Sayari: shared upstream exposure screening

1. **Clone this private repository, or choose Code → Download ZIP** and extract it.
2. **Double-click `docs/report.html` locally.** No server, installation or
   credentials are needed to read it. **Do not click the HTML file on GitHub to
   view the report: GitHub displays its source, not the rendered report.**
3. To reproduce the analysis, use Python 3.11 or newer from the repository root:

   ```console
   pip install -e .
   python -m sayari_poc run --sheet list_3 --offline
   ```

For reproducible installation, use Python 3.13 (verified on 3.13.15), run
`pip install -r requirements.lock`, then `pip install -e . --no-deps`;
`pip install -e ".[dev]"` remains the development install.

Installing dependencies needs package access; the offline analysis itself needs
neither credentials nor network access. The supplied workbook, Sayari response
cache and ontology are included.

## What the report answers

Where does retrieved upstream trade evidence converge across the supplier list?
The current run accounts for 50 input rows, resolves 46 and retains 4 unresolved
for review. Eight supplier identities connect through retrieved trade evidence to
ALLEGRO MICROSYSTEMS PHILIPPINES INC. The bounded
upstream funnel is **8,628 → 848 shared → 548 after hub exclusion → 129 flagged
for review** using Sayari's published critical/high factors. These counts are
lower bounds on retrieved evidence, not a complete map of supply relationships.

The HTML includes ranked entities, supplier evidence, unadjudicated resolution
candidates, published factor definitions and coverage limitations. JavaScript
adds filters, pagination and disclosure controls; evidence is present in the
HTML and readable without it. It was checked in Chrome at 320 to 1920 pixels,
with JavaScript on and off.

| Generated artifact | Purpose |
|---|---|
| `data/processed/findings.json` | Structured evidence and report numbers |
| `data/processed/report.html` | Self-contained interactive report |
| `data/processed/flagged_subtier_entities.csv` | Spreadsheet-safe ranked export |
| `data/processed/run_manifest.json` | Per-run audit counters and execution details |

## Run modes and API cost

| Mode | Command after `python -m sayari_poc run --sheet list_3` | API cost |
|---|---|---|
| Offline | `--offline` | Zero network requests; a missing required cache entry fails the run |
| Refresh | `--refresh` | Bypasses Sayari cache; one resolution per input row, then profile and traversal requests per distinct accepted identity; retries consume the budget |
| Live, cache first | no mode flag | Cache hits cost zero data requests; cache misses make authenticated requests and retries consume the budget |

The committed report was produced offline from the committed cache. OAuth
attempts are counted separately from the data-request budget. Actual transport
counts appear in the manifest; the run records local transport counters only and
never queries account usage. `--dry-run` prints a logical-request estimate with
zero API calls; it does not verify cache availability or execute either mode.

All settings are documented in [.env.example](.env.example). An offline reviewer
needs no `.env`. Never commit credentials or `.env`.

## Running the report on fresh Sayari data

This makes authenticated calls against your Sayari account. For `list_3` a full
refresh costs about **142 data requests** (50 resolutions, then one profile and
one upstream traversal for each of the 46 accepted identities), within the
default ceiling of 400.

1. **Credentials.** Copy `.env.example` to `.env` and set `SAYARI_CLIENT_ID` and
   `SAYARI_CLIENT_SECRET` from your Sayari account. `.env` is gitignored; never
   commit it. Leave `DECLARED_GENERATED_AT` blank so the report is stamped with
   the time of the run.
2. **Estimate first**, with no API calls:

   ```console
   python -m sayari_poc run --sheet list_3 --refresh --dry-run
   ```

3. **Run.** `--refresh` fetches every response again; with no mode flag, the run
   reuses cached responses and fetches only what is missing.

   ```console
   python -m sayari_poc run --sheet list_3 --refresh
   ```

4. **Read the result** in `data/processed/report.html`. Copy it to
   `docs/report.html` only if the new run is meant to replace the published one.

What to expect:

- **Different numbers.** Sayari's data changes over time; a capture four days
  earlier returned 6,485 upstream entities where the committed one returns 8,628.
  The byte-identical checks in [Exact byte reproduction](#exact-byte-reproduction)
  apply only to the committed cache, not to a fresh run.
- **The cache changes.** New responses are written to `data/cache/` and show up
  in `git status`. Commit them only if the new evidence should replace the old.
- **Limits are enforced for you.** The adapter paces requests to Sayari's
  published rate-limit tiers and refuses any data request beyond `CALL_BUDGET`,
  retries included. A row that fails, or that the budget stops, is recorded as a
  retrieval issue and the run continues; the command line reports an exhausted
  budget.
- **Other suppliers.** Point `ENTITY_FILE_PATH` at another workbook with the same
  sheet layout, or pass `--all-sheets`.

## Exact byte reproduction

Keep the default settings and committed evidence. Set the same declared
generation timestamp used for the committed report; this also works with an
empty output directory and a new warehouse.

PowerShell:

```powershell
$env:DECLARED_GENERATED_AT = '2026-09-24T01:52:59.468774+00:00'
python -m sayari_poc run --sheet list_3 --offline
(Get-FileHash docs/report.html -Algorithm SHA256).Hash
(Get-FileHash data/processed/report.html -Algorithm SHA256).Hash
```

POSIX shell:

```sh
DECLARED_GENERATED_AT='2026-09-24T01:52:59.468774+00:00' python -m sayari_poc run --sheet list_3 --offline
sha256sum docs/report.html data/processed/report.html
```

The two report digests must match. The regenerated `data/processed/findings.json`
and `flagged_subtier_entities.csv` hash to
`19c3032d8b59e67a458554334ea184ea0c2ec1b85508eef93b0ad778e3784c51` and
`823888c823513d781246543c51668a830770c38197a9b55b6b1fcecd485a737b`. Without the declared
timestamp, generation uses the clock or carries forward the timestamp of an
unchanged existing artifact. `run_manifest.json` intentionally changes between
runs and is excluded from byte identity. A generation date is not a source-data
freshness date.

## Design and interpretation

The official Python SDK is pinned to `sayari==0.1.43`; it owns authentication,
endpoint encoding, typing and retries. The adapter adds raw-response caching,
budget enforcement before network access, audit counters, rate-limit pacing and
domain normalization. Processing is synchronous and deterministic.

The layers are input → adapter → stages → analysis → presentation. DuckDB joins
retrieved upstream sets by entity ID; Jinja renders completed findings without
network or database access. Sayari's Projects pattern is the documented path
for production monitoring; this replayable screening PoC retains adjudication
evidence outside account-scoped project state.

Interpretation rules apply throughout:

- No data is absence of evidence, never absence of risk. Errors and requests
  not attempted have distinct states.
- Resolution score is relevance, never calibrated confidence or a percentage.
  Weak candidates are not accepted identities.
- Paths are trade-record evidence. They do not prove direct or contractual
  supply; tiers are Sayari annotations and missing paths do not prove no route.
- Bounded evidence counts are lower bounds.
  There is no composite risk score; possible identity matches remain unconfirmed.

## Verification and repository contents

CI (`.github/workflows/ci.yml`) runs every check on a clean Linux runner: the
network-blocked test suite, ruff, mypy `--strict`, the fixture scanner, and an
offline rebuild of the report that must match `docs/report.html` byte for byte.
Run the same commands locally from the repository root after
`pip install -r requirements.lock` and `pip install -e . --no-deps`.

`src/sayari_poc/` contains production code, SQL and templates; `tests/` contains
the network-blocked suite; `scripts/` contains diagnostics. Tracked evidence
lives in `data/input/`, `data/cache/` and `data/public/`. Local private files and
generated output belong in gitignored `data/private/` and `data/processed/`.

The workbook and cached vendor responses are proprietary interview material.
Delivery is through a **private GitHub repository with reviewer access**;
**GitHub Pages stays off**.
