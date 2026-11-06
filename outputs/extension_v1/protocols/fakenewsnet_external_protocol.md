# FakeNewsNet external transfer protocol (extension_v1)

Written 2026-10-12, before any FakeNewsNet item was scored. Frozen baseline untouched: nothing under `outputs/tables/`, `outputs/manuscript/`, `outputs/models/` or `data/processed/` is modified. All outputs go to `outputs/extension_v1/`.

## Purpose

Second independent external text target, parallel to the frozen COVID-Rumor NEWS evaluation (`src/run_covid_rumor_external_transfer.py`). Question: does the CoAID-source portability advantage observed on COVID-Rumor NEWS recur on a non-health, non-COVID target (PolitiFact political news; GossipCop entertainment news)?

## Target cohort rules (label-blind construction)

1. Source files: the local FakeNewsNet copy `data/raw/fakenewsnet/{politifact,gossipcop}_{fake,real}.csv`, columns `id, news_url, title, tweet_ids`. No article body text is present locally; no tweets are fetched.
2. Item: one row per `id` within subset; `target_uid = fakenewsnet:<subset>:<id>`. Subsets PolitiFact and GossipCop are evaluated separately and never pooled for the primary result.
2b. Items whose `id` appears in both the fake and the real file of the same subset carry conflicting inherited labels and are excluded before scoring (`label_conflict_excluded`). This gate reveals eligibility only; the labels themselves are not used until evaluation. Within-file duplicate ids do not occur.
3. Predictor: `normalize(title)` from `prepare_text_analysis_cohorts.normalize` (whitespace and obvious-null normalization; case and punctuation preserved). `news_url`, `tweet_ids`, `id` and the class file name are never concatenated into predictor text.
4. Quality screen, identical to COVID-Rumor: drop empty text; drop `quality(text,'body') == 'clear_crawl_artifact'`; drop items with fewer than two alphabetic analyzer tokens after stripping `https?://` and `www.` URLs; drop items whose alphabetic characters are less than 90% Latin letters. No keyword or outcome-driven exclusions.
5. Source-overlap audit, identical rules to COVID-Rumor, run against **all** modeling-ready CoAID and FakeHealth rows (eligible or not): exact normalized-text match against source `title_text`, `body_text`, `combined_text`; exact resource match of `url_key(news_url)` (specific URL only, homepage-only keys ignored) against source `article_url`, `canonical_resource_key`, `source_review_url`; high-similarity flags via case-preserving title bigram Jaccard >= 0.95 (>= 8 tokens) and body/combined five-gram Jaccard >= 0.90 (>= 30 tokens). Exact matches are removed from the primary target; high-similarity items are flagged, not removed.
6. Scopes: `clean_primary` = quality-eligible and no exact source overlap (primary); `all_eligible` = quality-eligible regardless of overlap (sensitivity); `clean_without_high_similarity` (sensitivity).
7. Labels: `fake` file = Fake = 1, `real` file = Real = 0. Label definitions are those documented by the FakeNewsNet repository (PolitiFact: journalist/fact-checker verdicts; GossipCop: entertainment stories scored by GossipCop, low scores treated as fake, with E! Online stories as real). These definitions are not documented in the local copy and were not verified against the current repository here. Labels are stored in a separate file and joined only at evaluation time.

## Representation decision

The local copy has titles only. The frozen primary models were fitted on `combined_text` (title + body) and are the wrong representation for a title-only target. Per Task 3b, the frozen **title-only** LR fits are used instead:

| Source | Frozen artifact | Predictor field | Train N | Locked configuration |
|---|---|---|---|---|
| CoAID | `outputs/models/tfidf_logistic_title/transfer_coaid_natural.joblib` | `title_text` (all title-only-eligible CoAID rows) | 3,571 | cfg_16: unigram+bigram, min_df 2, C 10, no class weight |
| FakeHealth | `outputs/models/tfidf_logistic_title/transfer_fakehealth_natural.joblib` | `title_text` (all title-only-eligible FakeHealth rows) | 1,645 | cfg_11: unigram, min_df 5, C 10, balanced |

These are the full-source transfer fits whose configurations were selected on source training-pool CV and locked (`selection_lock.json` in the same directory). Only the `natural` condition is scored. No frozen title-only Linear SVM fit exists, so the SVM arm of the COVID-Rumor comparison cannot be replicated; the comparison here is LR-only and is stated as such.

## Fixed scoring rules

- Positive class: Fake = 1.
- Threshold: LR probability >= 0.5 → Fake. (SVM decision >= 0 would apply if an SVM fit existed; none does.)
- Vectorizers transform only; source vocabulary and IDF are immutable. Runtime guards patch `TfidfVectorizer.fit/fit_transform` and `LogisticRegression.fit` to raise.
- No target-side fitting, calibration, threshold tuning, vocabulary extension, or label use before predictions are written and hashed.
- Validation before scoring: artifact `fit.converged`, `classes_ == [0,1]`, vectorizer `ngram_range`/`min_df` and classifier `C`/`class_weight` match the locked configuration, and `fit.train_uid_sha256` equals the digest of current `title_only_modeling_eligible` content UIDs.

## Metrics (per subset × source model × scope)

`N, fake_N, real_N, accuracy, balanced_accuracy, macro_f1, fake_precision/recall/f1, real_precision/recall/f1, roc_auc, average_precision, tn/fp/fn/tp, threshold, actual_fake_prevalence, predicted_fake_prevalence`, from `run_tfidf_logistic_baseline.metrics`, plus a vocabulary coverage diagnostic (share of target items with at least one known token; share with zero features). Metrics are item-weighted descriptives; no inferential test is attached.

## Interpretation bounds

Descriptive portability comparison on an unrelated domain with inherited labels, different label constructs, and a title-only representation. Results do not identify a COVID-specific mechanism, do not calibrate any model for FakeNewsNet, and do not alter any frozen result.
