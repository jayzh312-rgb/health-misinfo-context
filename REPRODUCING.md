# Reproducing the analyses

There are two levels. Checking the reported numbers needs only this repository. Re-running the pipeline needs the four source datasets, which are not redistributed here.

## 1. Check the reported numbers (repository only)

`outputs/manuscript/review_package_v3/review_package_v3_validation.json` maps each number displayed in the manuscript and supplement to the table and column it comes from. All of those tables are in `outputs/tables/` and `outputs/extension_v1/tables/`. The validation ran on the unredacted research tables, and `REDACTION_MANIFEST.md` confirms that redaction changed no numeric cell in any aggregate table. Bootstrap draws, model locks, saved classical models and item-level predictions are included, so intervals and metrics can be recomputed from the saved outputs without refitting.

## 2. Re-run the pipeline from the source datasets

### 2.1 Environments

| Environment | Python | Requirements | Used for |
|---|---|---|---|
| `.venv` | 3.14.2 | `requirements.txt` | everything except the transformer |
| `.venv-transformer` | 3.12.7 | `requirements-transformer.txt` | DistilRoBERTa (step 7) |
| any 3.12+ | | `requirements-render.txt`, plus Chrome or Chromium (`REVIEW_CHROME`) | rendering the manuscript (step 10) |

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.12 -m venv .venv-transformer && .venv-transformer/bin/pip install -r requirements-transformer.txt
```

The reported transformer results were produced on Apple Silicon (MPS). The scripts fall back to CUDA and then CPU, and bitwise reproducibility across devices is not claimed. The base checkpoint is `distilbert/distilroberta-base` at revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`.

### 2.2 Source data layout

| Dataset | Source | Place it at |
|---|---|---|
| CoAID | https://github.com/cuilimeng/CoAID | `data/raw/coaid/CoAID-master/` (the repository as downloaded, with its dated release folders) |
| FakeHealth | https://github.com/EnyanDai/FakeHealth | `data/raw/FakeHealth-master/dataset/`, with the engagement files rehydrated through the X API as its README describes |
| COVID-19 Rumor Dataset | https://github.com/MickeysClubhouse/COVID-19-rumor-dataset | `data/raw/COVID-19-rumor-dataset-master/` (the news file is `Data/news/news.csv`) |
| FakeNewsNet | https://github.com/KaiDMML/FakeNewsNet | `data/raw/fakenewsnet/` (the PolitiFact and GossipCop CSVs from its `dataset/` folder) |

Never modify files under `data/raw/`.

### 2.3 Fingerprint guards

Each analysis script records the SHA-256 of its inputs in a lock or run-metadata file and stops if an input differs on a later run. This is deliberate: it keeps a re-run from silently mixing old and new inputs. It has two consequences:

- The de-texted tables in `data/processed/` do not match those fingerprints, so the scripts below run only on tables rebuilt from the source data in steps 1 to 3.
- Source data that differs from what we used will also stop the scripts. This can happen with FakeHealth engagement in particular, because posts deleted since our collection cannot be rehydrated. Re-running on such data is a new analysis, and the locks must be regenerated deliberately rather than bypassed.

### 2.4 Run order

Run each script from the repository root with `.venv/bin/python <script>`, unless noted. Each step depends only on the steps before it.

1. **Inspect the sources:** `src/inspect_coaid.py`, `src/inspect_fakehealth.py`, `src/inspect_covid_rumor.py`.
2. **Content tables:** `src/build_coaid_content_table.py`, `src/build_fakehealth_content_table.py`, `src/validate_measurement_alignment.py`.
3. **Engagement samples:** `src/build_primary_analysis_samples.py`, `src/audit_coaid_entity_resolution.py`, `src/audit_coaid_content_version_stability.py`, `src/rebuild_coaid_version_aware_samples.py`.
4. **Engagement models and contrast:** `src/run_primary_engagement_eda.py`, `src/run_primary_engagement_models.py`, `src/audit_engagement_estimand_robustness.py`, `src/run_version_aware_engagement_analysis.py`, `src/run_cross_context_engagement_contrast.py --workers 4`.
5. **Text cohorts and audits:** `src/prepare_text_analysis_cohorts.py`, `src/audit_text_quality_and_source_cues.py`, `src/audit_publisher_heldout_feasibility.py`, `src/audit_transformer_feasibility.py`.
6. **Classical text models:** `src/run_tfidf_logistic_baseline.py`, `src/run_tfidf_logistic_cross_context_transfer.py`, `src/run_tfidf_linear_svm_replication.py`, `src/run_tfidf_logistic_title_sensitivity.py`, `src/audit_cross_context_lexical_alignment.py`, `src/run_fakehealth_publisher_heldout.py`, `src/run_covid_rumor_external_transfer.py`.
7. **Transformer (`.venv-transformer`):** `src/run_distilroberta_text_experiment.py`, `src/run_distilroberta_budget_seed_stability.py`, then `src/plot_distilroberta_results.py` and `src/plot_distilroberta_budget_seed_stability.py`.
8. **Extension analyses (Supplement S8, Tables S11–S23):** follow the protocols in `outputs/extension_v1/protocols/`, running the scripts in `outputs/extension_v1/scripts/` in this order: `task1_hurdle_components.py`, `task2_top_terms.py`, `task_a_encoding_audit.py`, `task3a_fakenewsnet_feasibility.py`, `task3c_fakenewsnet_score.py`, `task_b_fakenewsnet_engagement.py`, `task_c_fakenewsnet_round3.py`, `round4_check1_findability.py` to `round4_check6_folds.py`, then `outputs/extension_v1/round5/round5_within_length_auc.py`. `round4_check7_rounding.py` reads the synthesis and figure tables, so it runs after step 9.
9. **Synthesis and figures:** `src/build_manuscript_results_synthesis.py`, `src/create_manuscript_figures.py`, `outputs/manuscript/review_package_v3/scripts/figure_1_four_context.py`, `outputs/manuscript/review_package_v3/workflow/generate_workflow.py`, then `outputs/extension_v1/scripts/round4_check7_rounding.py`.
10. **Manuscript:** `outputs/manuscript/review_package_v3/internal/render_review_package_v3.py`, then `internal/validate_review_package_v3.py`. Do not run `internal/build_review_package_v3.py`, which rebuilds the manuscript from an earlier draft.

### 2.5 Tests

`cd src && python -m unittest discover -s . -p 'test_*.py'`. In this repository as shipped, without the source data or the transformer environment, 66 of 96 tests pass. The rest need article text, the source data or the transformer packages, or compare fingerprints of the unredacted tables. A full rebuild with both environments should pass more of them, but this has not been checked.
