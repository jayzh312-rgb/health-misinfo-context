# Text cohort construction and leakage-safe evaluation plans

No TF-IDF/vectorizer, Logistic Regression, SVM, transformer, classifier, feature selection, resampling/rebalancing, or threshold tuning was fitted. StratifiedGroupKFold was used only to assign groups to evaluation partitions. Engagement measurements, social-overlap features and engagement-file provenance were not carried into either text cohort.

## Coverage and eligibility

| dataset | scope | total_rows | fake_rows | real_rows | nonempty_title | nonempty_body | title_only | body_only | both | neither | artifact_only_rows | mixed_artifact_component_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | all_input_rows | 1690 | 472 | 1218 | 1690 | 1614 | 76 | 0 | 1614 | 0 | 42 | 13 |
| FakeHealth | primary_eligible | 1648 | 460 | 1188 | 1648 | 1574 | 74 | 0 | 1574 | 0 | 0 | 13 |
| CoAID | all_input_rows | 5337 | 922 | 4415 | 4903 | 4350 | 592 | 39 | 4311 | 395 | 1262 | 105 |
| CoAID | primary_eligible | 3680 | 834 | 2846 | 3674 | 3192 | 488 | 6 | 3186 | 0 | 0 | 105 |

Primary eligibility excludes rows with neither usable field and rows whose available fields are exclusively obvious crawl-error/boilerplate artifacts. This label-blind rule is a text-availability decision, not a new entity-resolution rule. Every input row and original string remains in the cohort CSV, with eligibility and exclusion-reason flags; nothing is deleted.

Candidate fields apply only whitespace normalization and obvious null-placeholder removal, preserving punctuation, URLs and case. A blank-line separator joins title and body. Quality detection inspects strings without modifying them; mixed substantive/artifact rows remain flagged, not silently cleaned. Normalized nonempty body is NOT proof of a full article: CoAID stored bodies are generally truncated near 500 characters and already normalized upstream, whereas FakeHealth often has long full article text. No stage can recover text lost upstream.

| dataset | scope | body_text_characters_median | body_text_characters_p05 | body_text_characters_p95 | body_text_whitespace_tokens_median | body_text_whitespace_tokens_p05 | body_text_whitespace_tokens_p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | all_input_rows | 3664 | 351.3 | 7843 | 591.5 | 56.65 | 1279 |
| FakeHealth | primary_eligible | 3719 | 768.4 | 7900 | 600.5 | 121.2 | 1286 |
| CoAID | all_input_rows | 478 | 79 | 497 | 73 | 13 | 87 |
| CoAID | primary_eligible | 489 | 70.55 | 498 | 77 | 12.55 | 88 |

All field-specific lengths and quantiles are in text_cohort_summary.csv, using nonempty values in the stated scope. 'Exact duplicate count' is reported both as excess rows beyond one per distinct string and as all rows participating in duplicated-string sets; empty values never count as duplicates.

| dataset | total_rows | nonempty_body_percent | title_only_percent | unflagged_body_percent |
| --- | --- | --- | --- | --- |
| FakeHealth | 1648 | 95.51 | 4.49 | 94.84 |
| CoAID | 3680 | 86.74 | 13.26 | 83.97 |

## Labels and predictor boundary

Original label/rating metadata is preserved. FakeHealth author rating <3 maps to Fake and >=3 to Real; CoAID filename Fake/Real maps to the same analysis names. Binary target is Fake=1, Real=0. This is analytical alignment for predictive comparison, NOT equivalence of ground truths. Only title_text, body_text or combined_text are candidate feature inputs; IDs, labels/ratings, source/review metadata, lineage, quality/overlap flags and assignments are NEVER text-model predictors. No engagement outcome is present in the cohort schema.

FakeHealth title provenance is checked against locally available fetched-content titles and existing review-title metadata. Editorial review-title fallbacks are flagged because evaluative wording could leak the label; they must be reviewed before training. No new fallback headline was substituted. CoAID source_title was not substituted for fetched title because the earlier audit documented source-metadata mismatches.

## Evaluation leakage groups

| dataset | total_rows | leakage_groups | largest_group_rows | title_text_exact_duplicate_excess_rows | body_text_exact_duplicate_excess_rows | cross_dataset_overlap_rows | cross_dataset_overlap_group_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | 1648 | 1635 | 11 | 2 | 11 | 0 | 0 |
| CoAID | 3680 | 2833 | 687 | 794 | 175 | 0 | 0 |

Groups are connected components of explicit evaluation barriers. CoAID uses the existing specific canonical resource, audited resource context (including separate corrected subgroups), source-member lineage, and exact/high-similarity textual duplicates. Versions/captures sharing a resource stay in one group. Old invalid overmerged parent IDs are NEVER grouping keys; release-local numeric content IDs are not global CoAID identifiers. Unresolved cross-URL contexts are conservatively kept in one evaluation group without asserting identity. FakeHealth uses original content ID, specific article URL with the prior offline URL normalization, review lineage, and exact/high-similarity text/title matches.

Exact identical titles or bodies are grouped even when they are generic templates. This is a deliberately conservative evaluation restriction protecting all three candidate input fields, NOT evidence that unrelated articles are one entity. It can create large groups and limit fold balance. No same-publisher/domain-only or semantic-topic rule joins records. Scheme/query tracking normalization handles clearly equivalent URL variants; other redirects, AMP aliases and semantic near-URLs are not guessed.

| dataset | evidence_type | documented_spanning_edges |
| --- | --- | --- |
| CoAID | exact_body_text | 1327 |
| CoAID | exact_title_text | 2020 |
| CoAID | same_audited_resource_uid | 146 |
| CoAID | same_canonical_resource_key | 205 |
| CoAID | very_high_shingle_body_text | 8 |
| CoAID | very_high_shingle_title_text | 2 |
| FakeHealth | exact_body_text | 49 |
| FakeHealth | exact_title_text | 41 |
| FakeHealth | same_canonical_resource_key | 1 |

## High-similarity and cross-dataset audit

| field | unique_strings | strings_meeting_token_minimum | verified_candidate_comparisons | shingle_width | minimum_tokens | jaccard_threshold | near_matching_string_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| title_text | 4532 | 3377 | 3 | 2 | 8 | 0.95 | 2 |
| body_text | 4588 | 4333 | 21 | 5 | 30 | 0.9 | 5 |

Exact text/title comparisons use minimally normalized, case-preserved strings. Near-duplicate screening uses case-preserving word shingles (punctuation ignored for comparison ONLY), title bigram Jaccard >=0.95 with >=8 words, and body five-gram Jaccard >=0.90 with >=30 words. A global rare-first shingle ordering and lossless prefix filter generate candidates, followed by exact Jaccard verification. No learned vocabulary weights, vectorizer, embeddings or semantic similarity model is fitted. This detects very high lexical similarity, not all paraphrases, translations, case changes or short/generic-string variants.

Cross-dataset evidence rows: 0; total flagged pair-evidence count: 0; eligible pair-evidence count: 0. Evidence rows may describe the same item pair through multiple mechanisms and are not a unique-pair count. The audit includes artifact-only/ineligible rows so exclusions do not hide overlaps. Cross-context overlaps are never automatically removed; directly affected rows and their entire within-dataset evaluation groups are flagged. Resolve flagged groups before transfer evaluation.

## Source/publisher association

| dataset | scope | source_measure | distinct_nonmissing_sources | missing_source_rows | cramers_V | majority_label_baseline | major_nearly_exclusive_sources | rows_in_major_nearly_exclusive_sources | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | primary_eligible | publisher_source | 48 | 16 | 0.3519 | 0.7209 | 0 | 0 | Descriptive source-label association, no classifier or p-value; source is audit metadata, not a predictor |
| CoAID | primary_eligible | publisher_source | 181 | 43 | 0.971 | 0.7734 | 9 | 3298 | Descriptive source-label association, no classifier or p-value; source is audit metadata, not a predictor |

The source table reports Fake/Real counts by publisher metadata and separately by article URL host, for all rows and eligible rows. Major sources have >=10 items; near-exclusive means >=95% from one class. Cramér's V is descriptive, excluding missing source identifiers, with no causal or predictive interpretation. Sparse categories and class imbalance can inflate apparent source purity. Splits are resource/text-grouped, NOT publisher-held-out: source identity can still be learned across folds and transfer results can reflect publisher, collection, period, and formatting differences rather than misinformation language. Source names are not removed.

## Fixed holdout and training-only CV

| dataset | scheme | assignment | N | groups | fake_N | real_N | fake_proportion | fraction_of_eligible | n_inner_folds | n_outer_folds | selected_outer_holdout_fold | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | fixed_holdout | test | 330 | 329 | 92 | 238 | 0.2788 | 0.2002 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | fixed_holdout | train | 1318 | 1306 | 368 | 950 | 0.2792 | 0.7998 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | -1 | 330 | 329 | 92 | 238 | 0.2788 | 0.2002 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | 0 | 264 | 263 | 74 | 190 | 0.2803 | 0.1602 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | 1 | 263 | 262 | 73 | 190 | 0.2776 | 0.1596 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | 2 | 264 | 254 | 74 | 190 | 0.2803 | 0.1602 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | 3 | 264 | 264 | 74 | 190 | 0.2803 | 0.1602 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| FakeHealth | training_pool_CV | 4 | 263 | 263 | 73 | 190 | 0.2776 | 0.1596 | 5 | 5 | 0 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | fixed_holdout | test | 707 | 667 | 167 | 540 | 0.2362 | 0.1921 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | fixed_holdout | train | 2973 | 2166 | 667 | 2306 | 0.2244 | 0.8079 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | -1 | 707 | 667 | 167 | 540 | 0.2362 | 0.1921 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | 0 | 822 | 126 | 133 | 689 | 0.1618 | 0.2234 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | 1 | 538 | 509 | 134 | 404 | 0.2491 | 0.1462 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | 2 | 538 | 514 | 134 | 404 | 0.2491 | 0.1462 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | 3 | 538 | 514 | 133 | 405 | 0.2472 | 0.1462 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |
| CoAID | training_pool_CV | 4 | 537 | 503 | 133 | 404 | 0.2477 | 0.1459 | 5 | 5 | 3 | CV -1 rows are reserved test only; folds 0..K-1 are validation allocations within training pool |

All primary-eligible rows are sorted by stable UID. Seed-42 StratifiedGroupKFold proposes up to five outer folds; class-specific group support and both-class support in every training/validation partition are checked. The outer fold minimizing the sum of absolute deviations from 20% of total rows, Fake rows and Real rows becomes fixed test (fold-index tie break). This uses counts only, not model results. Five inner grouped folds are then constructed only inside the remaining training pool when feasible. A smaller K is used only when class/group feasibility fails.

The requested *_text_grouped_cv.csv files list every eligible row exactly once: fold >=0 is its validation fold within the training pool; fold=-1 is RESERVED HOLDOUT and must never be treated as a CV fold or included in CV training. To train fold f use fold>=0 AND fold!=f; validate on fold==f. The separate *_text_holdout.csv records train/test. Outer partition files are provenance only, NOT an alternative tuning plan. No held-out test item or group is exposed to training-pool CV. Group constraints can leave unequal fold sizes and class proportions; these are reported, not fixed by oversampling or discarding rows.

Five-fold feasibility is not a promise of near-equal class balance: CoAID's largest eligible evaluation group is a collection linked by repeated/generic text, not one validated resource. Keeping it intact makes one inner fold larger and more Real-heavy. Compare fold-level counts and eventual uncertainty; never break this group merely to improve a score or class balance. Quality exclusions are label-blind, not rebalancing, but missing/artifact text is source- and class-associated and changes the eligible cohort composition.

## Cross-context transfer and model readiness

FH→CA and CA→FH plans use source-only training and whole target-dataset testing, never random dataset mixing. Source hyperparameters/thresholds must be selected using source training-pool CV only. After protocol lock and one-time source holdout evaluation, a frozen configuration may be refitted on all eligible source rows; target labels must not drive selection. Both directional protocols should be locked before inspecting either target result. Transfer tables retain all eligible overlaps with review flags; no overlap-based exclusion has been silently applied.

Recommended first candidate: combined title + body, because it retains coverage and avoids relying solely on generic scraped headings. Title-only and body-only should be separately pre-specified sensitivity inputs using these SAME evaluation barriers and field-eligibility flags. Mixed crawl artifacts, review-derived titles and text truncation require quality review before treating combined text as a clean linguistic measurement. No normalized candidate field is silently edited to resolve these problems.

Resource/text leakage safeguards and fixed evaluation plans are prepared. This is not a blanket declaration of leakage-free language learning: resolve cross-context overlap flags and review editorial-title/mixed-artifact flags before training. Publisher overlap is allowed by this resource-generalization design; a later publisher-held-out experiment would answer a different question. Use training-fold-only TF-IDF fitting, preserve natural class distributions, and reserve class weighting, threshold selection and balanced metrics for later training-only decisions. No classifier is trained now.

## Reproduction

Run `.venv/bin/python src/prepare_text_analysis_cohorts.py`; use `--keep-artifact-only` only to explicitly choose the alternative eligibility policy. Run `.venv/bin/python -m unittest discover -s src -p 'test_text_analysis_cohorts.py'`. The metadata contains the input fingerprints, normalization/quality rules, feature allowlist, split protocol and deterministic validation status. Original analytical sample files and raw data remain unchanged.
