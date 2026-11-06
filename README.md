# health-misinfo-context

Code and derived tables for the paper "Fake–Real gaps in observed spread and detectability differ between health misinformation datasets built in different ways: a comparison of CoAID and FakeHealth". The study compares Fake and Real items on observed social-media engagement and on how well text classifiers transfer between datasets, across CoAID, FakeHealth, the COVID-19 Rumor Dataset and two FakeNewsNet collections.

FakeHealth labels mainstream health news by expert-rated reporting quality, whereas CoAID contrasts fact-checked COVID-19 claims with material from health agencies and health-information publishers. Counting items never posted and standardizing on a common model scale, Fake items drew fewer observed posts than Real items in both datasets, but the Fake/Real ratio was 0.253 in CoAID and 0.840 in FakeHealth (ratio-of-ratios 0.301, 95% interval 0.154–0.645). In GossipCop, Fake items drew more observed posts, and PolitiFact's interval included parity. Fake and Real were highly separable within CoAID and modestly within FakeHealth, transfer was poor in both directions, and document length transferred better than words. Because period, labels, sources and collection differ together, these comparisons describe dataset contexts and cannot isolate a pandemic effect.

The repository holds the analysis, audit and figure scripts, the pre-specified extension protocols, the saved splits, locks and classical model bundles, the aggregate tables and bootstrap draws behind the numbers in the manuscript, item-level predictions, processed tables with article text and post identifiers removed, and the manuscript and supplement sources with their validation record. Built 2026-11-06 by `scripts/build_release_staging.py`; integrity in `SHA256SUMS`, file-level provenance in `DEPOSIT_MANIFEST.json`.

## Repository layout
- `src/`: analysis, audit and figure scripts with their unit tests (`src/test_*.py`). `outputs/extension_v1/scripts/`, `protocols/` and `reports/`: the extension analyses with their pre-specified protocols.
- `README.md`, `REPRODUCING.md`, `LICENSE` (MIT), `LICENSE-DERIVED-TABLES.md` (CC BY 4.0), `CITATION.cff`, `.zenodo.json`, `CHANGELOG_release.md`, `requirements.txt` (Python 3.14.2), `requirements-transformer.txt` (Python 3.12.7), `requirements-render.txt`; `README_project.md` and `AGENTS.md` are the working project notes.
- `outputs/splits/`: grouped-CV, holdout, outer-partition and transfer assignment manifests.
- `outputs/models/*/`: every selection, prediction and target lock, the transformer `locked_configurations.json`, and all classical joblib model bundles (TF-IDF LR/SVM, title-only, transfer, publisher-held-out).
- `outputs/extension_v1/round5/`: the within-length-quintile ROC-AUC re-scoring diagnostic (Supplementary Table S19.4): script, table, SHA-256 and README.
- `outputs/extension_v1/figures/`: the round-4 bootstrap-shape figures (also used as Figure S1 panels).
- `outputs/extension_v1/tables/round4_politifact_component_top_titles.csv` is the one table that carries article titles (normalized PolitiFact titles of the dominant component, a reported result); every other title, body, abstract and excerpt column is dropped.
- `outputs/tables/` and `outputs/extension_v1/tables/`: every aggregate table, including the four bootstrap-draw files (`version_aware_bootstrap_refits.csv`, `cross_context_no_dominant_bootstrap_refits.csv`, `fakenewsnet_engagement_bootstrap_refits.csv`, `fakenewsnet_round3_exclusion_bootstrap_refits.csv`); `outputs/predictions/`: item-level predictions keyed by internal UIDs; `outputs/reports/`: the report for each analysis stage.
- `data/processed/`: the processed tables **with article-text and post/tweet-ID columns removed** (see `DEPOSIT_MANIFEST.json` → `column_drops`); counts, labels, lineage, eligibility flags and split fields are retained.
- `outputs/manuscript/socarxiv_v1/`: the preprint as posted (combined PDF, manuscript-only and supplement-only PDFs, and the build record with their SHA-256).
- `outputs/manuscript/review_package_v3/`: the manuscript and supplement sources, Figures 1–4 and M1 with their scripts, the data behind the tables, and the validation record (see its README); `outputs/manuscript/revision_v3/`: verification, reference checks, migration log and supplement build inputs.

## Excluded and why
- The raw datasets are not redistributed. Obtain them from the providers: CoAID (https://github.com/cuilimeng/CoAID), FakeHealth (https://github.com/EnyanDai/FakeHealth; engagement content must be rehydrated through the X API), COVID-19 Rumor Dataset (https://github.com/MickeysClubhouse/COVID-19-rumor-dataset, MIT), FakeNewsNet (https://github.com/KaiDMML/FakeNewsNet). CoAID, FakeHealth and FakeNewsNet state no license; FakeNewsNet states that its complete dataset cannot be distributed for Twitter-policy and publisher-copyright reasons.
- Article text, tweet/post/reply identifiers and per-identifier evidence tables (6 files listed under `excluded` in the manifest): provider and platform terms unknown or restrictive.
- Transformer checkpoints (`outputs/models/distilroberta*`, 88 GB): size. The pinned checkpoint is `distilbert/distilroberta-base` at revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`; locked settings, seeds and all predictions are included, so the transformer rows can be re-run on an MPS, CUDA or CPU device (bitwise reproducibility across devices is not claimed).
- Home-directory paths in environment audit records are replaced by `[REDACTED_HOME]` (see `redactions`).
- Working and editorial documents (readiness audit, venue planning notes, mentor and reviewer round records, editorial package reports, repository audit note, rendering QA screenshots): internal records, listed under `excluded` in the manifest.

## Licensing
- **Code** (`src/`, `outputs/extension_v1/scripts/`, `scripts/`, package build scripts): MIT, see `LICENSE`.
- **Derived tables, draws, manifests, predictions and figures** produced by this project: Creative Commons Attribution 4.0 International (CC BY 4.0), see `LICENSE-DERIVED-TABLES.md`; cite the paper and this repository.
- **Source datasets**: not redistributed; CoAID, FakeHealth, the COVID-19 Rumor Dataset and FakeNewsNet remain under their providers' own terms (the COVID-19 Rumor repository is MIT; the other three state no license, and FakeNewsNet states that its complete dataset cannot be distributed). The processed tables here are derivatives shared only as counts, labels, lineage and eligibility fields.

## Checking the reported numbers
`outputs/manuscript/review_package_v3/review_package_v3_validation.json` maps each number displayed in the manuscript and supplement to the table and column it comes from, and every one of those tables is in this repository. The validation ran on the unredacted research tables; `REDACTION_MANIFEST.md` shows that redaction changed no numeric cell in any aggregate table.

## Reproducing
See `REPRODUCING.md`. Checking the reported numbers needs only this repository; re-running the pipeline needs the four source datasets, in the order given there.

## How to cite
Cite the paper and this repository. `CITATION.cff` has both, and GitHub shows it under "Cite this repository". The repository is archived on Zenodo: https://doi.org/10.5281/zenodo.24015772 (this DOI always resolves to the latest version). The paper: Zhao, J., & Valdiviezo, J. (2026). Fake–Real gaps in observed spread and detectability differ between health misinformation datasets built in different ways: a comparison of CoAID and FakeHealth. *SocArXiv*. https://doi.org/10.31235/osf.io/c83dk_v1

## Redaction
Some staged files differ from their source in the research folder: article-text, title, abstract, excerpt and tweet-ID columns are dropped, tweet/post identifiers in URL, key, evidence and file-name fields are masked as `[TWEET_ID_REDACTED]`, and home-directory paths, the author name and personal file names are replaced in a few records. `REDACTION_MANIFEST.md` lists every such file with its source and staged SHA-256, the columns dropped or masked and the number of masked cells, and confirms that no numeric cell in any aggregate table under `outputs/tables/` or `outputs/extension_v1/tables/` changed. The validation hashes and fingerprints inside `outputs/manuscript/review_package_v3/` (for example `review_package_v3_validation.json`) and in the unit tests refer to the unredacted source tables, so they will not match the staged files that the manifest lists; `SHA256SUMS` holds the staged hashes.
