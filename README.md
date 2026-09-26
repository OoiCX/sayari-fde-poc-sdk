# Sayari: shared upstream exposure screening

The report is a single file, `docs/report.html`. It opens in any web browser,
with no installation, sign-in or internet connection needed.

## Download

Either download the whole project:

1. On this repository's GitHub page, click the green **Code** button.
2. Choose **Download ZIP**.
3. Unzip the file.

Or download just the report:

1. Open the `docs` folder on GitHub and click `report.html`.
2. Click the **Download raw file** button (the arrow icon, top right of the file).

## Open the report

Double-click `report.html`. It opens in your default browser.

Don't open it by clicking the file on GitHub: GitHub shows the file's source code,
not the report.

## Rebuild the report yourself (no credentials needed)

The Sayari responses behind the report are saved in `data/cache/`, so you can
rebuild it offline without contacting Sayari. You need Python 3.13.

1. Open a terminal in the project folder and install:

   ```console
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.lock
   pip install -e . --no-deps
   ```

   On macOS or Linux, activate with `source .venv/bin/activate` instead.

2. Run:

   ```console
   python -m sayari_poc run --sheet list_3 --offline
   ```

3. Open the rebuilt report at `data/processed/report.html`. It shows the same
   results as `docs/report.html`.

For a byte-for-byte match with the published file, set the report's generation time
first: in PowerShell run `$env:DECLARED_GENERATED_AT="2026-09-24T01:52:59.468774+00:00"`,
or on macOS/Linux `export DECLARED_GENERATED_AT=2026-09-24T01:52:59.468774+00:00`.
This is the check the project's CI runs on every change.

## Running the analysis on live Sayari data

To rerun the analysis against Sayari's live data you need **your own Sayari API
credentials** (a client ID and client secret); none are included here.

1. Install as above.

2. Copy `.env.example` to a new file named `.env`, then fill in your credentials:

   ```text
   SAYARI_CLIENT_ID=your-client-id
   SAYARI_CLIENT_SECRET=your-client-secret
   ```

   Keep `.env` private; never share or commit it.

3. Run:

   ```console
   python -m sayari_poc run --sheet list_3 --refresh
   ```

4. Open the new report at `data/processed/report.html`.

A full run makes about 142 requests to Sayari and takes a few minutes. Sayari's data
changes over time, so your numbers can differ from the published report.

A live run replaces the saved responses in `data/cache/` with the new data. To go back
to the published data, download the project again (or, in a git clone, run
`git checkout -- data/cache`).
