# AGENTS.md

## Project Overview

This project studies differences between misinformation/fake news and real news engagement on social media, with particular interest in whether the Fake–Real engagement gap differs between socially sensitive and non-sensitive contexts.

Primary research question:

> How does the difference between misinformation and real-news diffusion/engagement on social media differ between sensitive periods and more normal periods?

The primary comparison currently uses:

- **CoAID News** — COVID-19 sensitive-period health-news context
- **FakeHealth HealthStory** — non-COVID health-information comparison context

Additional datasets:

- **COVID-19 Rumor Dataset** — secondary COVID robustness dataset
- **FakeNewsNet** — broader-domain/text-generalization robustness dataset

The project has two major analytical branches:

1. **Engagement analysis — primary**
   - observed social-post engagement
   - Fake vs Real engagement differences
   - cross-context comparison

2. **Text analysis — supporting/baseline**
   - TF-IDF and interpretable classifiers first
   - within-context classification
   - cross-context transfer/generalization
   - transformer models only later if justified

Do not redesign the research question or substitute a classifier-leaderboard objective for the current explanatory design.

---

## Current Research Status

The primary engagement-analysis stage is substantially complete.

Validated primary within-dataset estimates:

- FakeHealth HealthStory Fake/Real expected-post ratio ≈ **0.840**
- CoAID Specification A Fake/Real expected-post ratio ≈ **0.253**
- CoAID Specification B ≈ **0.240**
- CoAID Specification C ≈ **0.289**

Primary cross-context contrast:

- Ratio-of-ratios ≈ **0.301**
- 95% independent overlap-cluster bootstrap percentile interval ≈ **0.154–0.645**

Interpret this as an observational dataset-context difference.

Do NOT interpret this as:
- a causal COVID effect;
- evidence that sensitive periods inherently reduce misinformation diffusion;
- proof of a universal misinformation mechanism.

Primary engagement work is frozen unless a reproducibility or data-integrity problem is discovered.

The next major project stage is **leakage-safe text analysis and cross-context transfer**.

---

## Environment

- Project uses a local virtual environment:
  `.venv/`
- Activate on macOS/Linux:

  ```bash
  source .venv/bin/activate
  ```

## Token Efficiency

- Do not read `.ipynb` files unless explicitly asked. Prefer the corresponding reusable `.py` script in `src/` or a paired `.py` file when one exists.
- Do not repeatedly inspect raw datasets when validated processed tables, manifests, reports, or audit outputs already answer the question.
- Before rerunning an expensive analysis, check whether validated outputs already exist and whether the relevant inputs/code have changed.
- Reuse validated saved metrics, bootstrap draws, model outputs, and reports whenever fingerprints or validation metadata indicate they are still current.
- Do not rerun long bootstrap/model jobs merely to reproduce results already validated.
- For long-running jobs that are actually required, write progress to `logs/` and final outputs to `outputs/`; avoid flooding the terminal with per-iteration output.
- Read concise project summaries before reopening large datasets:
  - `AGENTS.md`
  - `notes/dataset_summary.md` if it exists and is current
  - the most recent relevant file in `outputs/reports/`
- Treat `notes/dataset_summary.md` as a convenience summary, not as a substitute for validated outputs. If it conflicts with a newer report/table, use the newer validated artifact and flag the discrepancy.
- Prefer targeted reads:
  - inspect only required columns;
  - read summary tables before raw records;
  - use `head`, schema, counts, and targeted filtering instead of printing full files;
  - search for specific functions/fields rather than reading large source files end-to-end.
- Do not read large generated CSVs or logs in full when aggregate summaries already exist.
- When a previous task already generated a reproducible script, modify or call that script rather than recreating the analysis from scratch.
- Keep terminal responses concise: report commands, key validation results, major metrics, warnings, and output paths rather than dumping full logs.

## CURRENT STATUS — 2026-10-13

- Engagement, classical text, transformer and COVID-Rumor external-validation stages are frozen; no frozen baseline changed. Do not overwrite anything under `outputs/tables/`, `outputs/models/`, `outputs/manuscript/review_package_v2/`, or `data/processed/` without an explicit request.
- extension_v1 rounds 1–4 are complete and registered: round 1 (hurdle component reporting, top-term reporting, FakeNewsNet title-only LR external scoring), round 2 (CoAID encoding-artifact sensitivity under the locked protocol; FakeNewsNet engagement under the frozen estimand), round 3 (FakeNewsNet dominant-component exclusion, multi-membership and domain diagnostics), round 4 (reviewer checks: CoAID findability, document length as a shared cue with a length-only diagnostic baseline, bootstrap shape conditional on dominant-component inclusions, mean versus typical item, FakeHealth truncation refits, fold-to-fold coefficient reliability, rounding consistency). Protocols, tables, figures and reports are under `outputs/extension_v1/`; they are registered in Supplementary S8 and Tables S11–S23 with Figure S1 and are supplementary, descriptive additions.
- Revision v3 package built: `outputs/manuscript/review_package_v3/` (revised in place 2026-10-13 through mentor round 1 and reviewer round 1 with round-4 results registered; see `review_package_v3_report.md` sections 6–7 and `review_package_v3_validation.json`). The v3 draft, v3 supplement, verification, reference checks and migration log are under `outputs/manuscript/revision_v3/`. One author placeholder remains in the manuscript (data and code availability statement).
- Venue pending (shortlist: `outputs/manuscript/journal_shortlist.md`). Code and data availability open; authors, declarations and AI-use disclosure open. Nothing submitted.
- Engagement comparability feasibility audit (2026-10-09): judgment C; next step is a new-collection protocol, not matching on existing data.
- Current synthesis: `outputs/reports/manuscript_results_synthesis.md`; key results, claims and limitations in `outputs/tables/manuscript_*.csv`.
- Private GitHub repository `health-misinfo-context` built from `release/staging_v3/` only: initial clean commit 2026-10-13, updated the same day with the round-5 manuscript package (2 commits, 759 files, 145.2 MB, visibility PRIVATE); URL, commit SHAs and verification in `release/github_push_report.md`. Public flip, v1.0.0 tag and Zenodo DOI pending acceptance. The research folder itself is not a git repository.
- This section supersedes earlier status statements; validated findings and interpretation limits remain in force.
