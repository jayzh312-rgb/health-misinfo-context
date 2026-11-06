# Independent COVID-Rumor NEWS external transfer

## Design and schema

The headerless Data/news/news.csv is read with positional columns rumor_id, label, content, sentiment. One row per rumor_id; content is the item-level rumor/news statement. Labels are F=False/Fake=1, T=True/Real=0; U and U(Twitter) are unverified and excluded. The U(Twitter) marker occurs within NEWS and does not authorize use of the Twitter subset. No Twitter files were read.
The main table has no item publication date. Associated rumor####.csv files contain website URLs, repost dates and sometimes stance. Only website URLs were read for overlap screening; no repost is an independent item and no repost text is concatenated. IDs, URLs, source identities, sentiment, stance, dates and engagement are never supplied as separate predictors.

## Cohort and quality

| measure | value |
| --- | --- |
| original_NEWS_N | 4129.0000 |
| True_N | 659.0000 |
| False_N | 3041.0000 |
| Unverified_N | 429.0000 |
| unusable_or_missing_text_N | 0.0000 |
| duplicate_ID_excess_N | 0.0000 |
| otherwise_eligible_binary_N | 3700.0000 |
| exact_source_overlap_excluded_N | 33.0000 |
| final_clean_binary_N | 3667.0000 |
| clean_high_similarity_flagged_N | 22.0000 |
| all_NEWS_exact_duplicate_text_excess_N | 0.0000 |
| all_NEWS_exact_duplicate_text_involved_N | 0.0000 |
| all_eligible_exact_duplicate_text_excess_N | 0.0000 |
| all_eligible_exact_duplicate_text_involved_N | 0.0000 |
| clean_primary_exact_duplicate_text_excess_N | 0.0000 |
| clean_primary_exact_duplicate_text_involved_N | 0.0000 |
| clean_Fake_N | 3015.0000 |
| clean_Real_N | 652.0000 |
| clean_Fake_prevalence | 0.8222 |

| scope | measure | N | minimum | p05 | median | p95 | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_NEWS | normalized_character_N | 4129 | 13 | 40.0000 | 90.0000 | 249.0000 | 727 |
| all_NEWS | analyzer_token_N | 4129 | 2 | 6.0000 | 14.0000 | 40.0000 | 123 |
| all_eligible | normalized_character_N | 3700 | 13 | 40.0000 | 89.0000 | 247.0000 | 727 |
| all_eligible | analyzer_token_N | 3700 | 2 | 6.0000 | 14.0000 | 39.0000 | 123 |
| clean_primary | normalized_character_N | 3667 | 13 | 40.0000 | 89.0000 | 247.0000 | 727 |
| clean_primary | analyzer_token_N | 3667 | 2 | 6.0000 | 14.0000 | 39.0000 | 123 |

Rules were fixed before predictions: whitespace/obvious-null normalization preserving case and punctuation; remove missing text, existing obvious artifact-only patterns, fewer than two alphabetic analyzer tokens after URLs are stripped for quality inspection, or <90% Latin letters. The repository designates these as English; the script check cannot certify language or exclude every mixed-language sentence. This language limitation is retained. No keyword dictionary or outcome-driven exclusions were introduced. Normalized exact duplicates within NEWS retain distinct item IDs; this is an item-weighted descriptive evaluation, not independent observations for inferential testing.

## Source overlap

| source_dataset | overlap_type | pair_evidence_N | all_NEWS_target_N | eligible_binary_target_N | clean_primary_target_N |
| --- | --- | --- | --- | --- | --- |
| FakeHealth | exact_text | 0 | 0 | 0 | 0 |
| FakeHealth | exact_resource | 0 | 0 | 0 | 0 |
| FakeHealth | high_similarity | 0 | 0 | 0 | 0 |
| CoAID | exact_text | 4 | 2 | 2 | 0 |
| CoAID | exact_resource | 126 | 41 | 31 | 0 |
| CoAID | high_similarity | 29 | 25 | 24 | 22 |

Exact comparison uses the previous case-preserving whitespace/null normalization and checks each statement against source title, body and combined text, including source records not eligible for training. Known-resource comparisons use the prior offline URL normalizer (tracking parameters removed, path identity retained); homepage-only matches do not count. Source article URLs, canonical resource keys and review URLs are compared with associated target website URLs and URLs in statements. Shared review/repost resources are conservatively removed without claiming identical underlying claims. Redirects and unobserved resources cannot be resolved offline.
URL availability: {'items_with_website_file': 3855, 'items_with_specific_resource': 2442}. Missing website files do not exclude an item.
High-similarity screening reuses case-preserving title bigram Jaccard >=0.95 with >=8 words, and body/combined five-gram Jaccard >=0.90 with >=30 words. These matches are flagged, not automatically removed. Exact source overlaps are removed from the primary target; all-eligible and clean-without-high-similarity results are sensitivity analyses. Pair evidence and every item exclusion reason are saved separately.

## Frozen models and evaluation

All four validated natural combined-text full-source models were reused, with matching artifact/cohort/configuration fingerprints. No source or target model was fitted. Runtime guards prohibit LR, SVM and vectorizer fitting. Only the existing source vocabulary/IDF transforms target statements. LR uses probability >=0.5; SVM uses decision score >=0. Target text and UIDs were fingerprinted before scoring; all four prediction sets and their fingerprint were written before outcome labels were joined. The binary eligibility gate exposes only membership before scoring. Predictions remain label-free and contain no raw text.

| classifier | source_dataset | scope | N | fake_N | real_N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision | tn | fp | fn | tp | threshold | actual_fake_prevalence | predicted_fake_prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | CoAID | clean_primary | 3667 | 3015 | 652 | 0.8064 | 0.5529 | 0.5580 | 0.8389 | 0.9463 | 0.8893 | 0.3910 | 0.1595 | 0.2266 | 0.6995 | 0.9114 | 104 | 548 | 162 | 2853 | 0.5000 | 0.8222 | 0.9275 |
| LR | FakeHealth | clean_primary | 3667 | 3015 | 652 | 0.7565 | 0.5021 | 0.4969 | 0.8229 | 0.8968 | 0.8583 | 0.1837 | 0.1074 | 0.1355 | 0.5219 | 0.8331 | 70 | 582 | 311 | 2704 | 0.5000 | 0.8222 | 0.8961 |
| SVM | CoAID | clean_primary | 3667 | 3015 | 652 | 0.8064 | 0.5349 | 0.5314 | 0.8330 | 0.9562 | 0.8904 | 0.3592 | 0.1135 | 0.1725 | 0.6816 | 0.9045 | 74 | 578 | 132 | 2883 | 0.0000 | 0.8222 | 0.9438 |
| SVM | FakeHealth | clean_primary | 3667 | 3015 | 652 | 0.7737 | 0.5059 | 0.4973 | 0.8241 | 0.9214 | 0.8700 | 0.1993 | 0.0905 | 0.1245 | 0.5248 | 0.8345 | 59 | 593 | 237 | 2778 | 0.0000 | 0.8222 | 0.9193 |

Confusion matrix columns: tn=Real predicted Real, fp=Real predicted Fake, fn=Fake predicted Real, tp=Fake predicted Fake. Fake is the positive class. Average precision depends on the high Fake prevalence; it should be read with prevalence and ROC-AUC. No target calibration, rebalancing, adaptation or tuning occurred.

## CoAID minus FakeHealth source comparisons

| classifier | scope | metric | coaid_source | fakehealth_source | coaid_minus_fakehealth |
| --- | --- | --- | --- | --- | --- |
| LR | clean_primary | macro_f1 | 0.5580 | 0.4969 | 0.0611 |
| LR | clean_primary | balanced_accuracy | 0.5529 | 0.5021 | 0.0508 |
| LR | clean_primary | fake_f1 | 0.8893 | 0.8583 | 0.0311 |
| LR | clean_primary | roc_auc | 0.6995 | 0.5219 | 0.1776 |
| LR | clean_primary | average_precision | 0.9114 | 0.8331 | 0.0783 |
| SVM | clean_primary | macro_f1 | 0.5314 | 0.4973 | 0.0342 |
| SVM | clean_primary | balanced_accuracy | 0.5349 | 0.5059 | 0.0289 |
| SVM | clean_primary | fake_f1 | 0.8904 | 0.8700 | 0.0203 |
| SVM | clean_primary | roc_auc | 0.6816 | 0.5248 | 0.1568 |
| SVM | clean_primary | average_precision | 0.9045 | 0.8345 | 0.0700 |

Differences are descriptive, with no statistical winner designation or prespecified materiality cutoff. Effect magnitudes and agreement across metrics determine how strongly the pattern can be interpreted.

## Representation

| classifier | source_dataset | vocabulary_size | target_token_occurrence_coverage_percent | zero_feature_rows | zero_feature_percent | median_target_nonzero_features |
| --- | --- | --- | --- | --- | --- | --- |
| LR | FakeHealth | 35168 | 81.8743 | 0 | 0.0000 | 15.0000 |
| LR | CoAID | 7299 | 89.4714 | 0 | 0.0000 | 12.0000 |
| SVM | FakeHealth | 35168 | 81.8743 | 0 | 0.0000 | 15.0000 |
| SVM | CoAID | 7299 | 89.4714 | 0 | 0.0000 | 12.0000 |

Coverage counts occurrences of default analyzer unigrams found in the source vocabulary; bigrams are not double-counted. Nonzero rows can still have poor semantic or genre alignment.

## Sensitivities

| classifier | source_dataset | scope | N | fake_N | real_N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision | tn | fp | fn | tp | threshold | actual_fake_prevalence | predicted_fake_prevalence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR | CoAID | all_eligible | 3700 | 3041 | 659 | 0.8059 | 0.5527 | 0.5577 | 0.8385 | 0.9461 | 0.8891 | 0.3903 | 0.1593 | 0.2263 | 0.7001 | 0.9116 | 105 | 554 | 164 | 2877 | 0.5000 | 0.8219 | 0.9273 |
| LR | CoAID | clean_without_high_similarity | 3645 | 2994 | 651 | 0.8055 | 0.5528 | 0.5578 | 0.8381 | 0.9459 | 0.8887 | 0.3910 | 0.1598 | 0.2268 | 0.6985 | 0.9106 | 104 | 547 | 162 | 2832 | 0.5000 | 0.8214 | 0.9270 |
| LR | FakeHealth | all_eligible | 3700 | 3041 | 659 | 0.7551 | 0.5016 | 0.4964 | 0.8224 | 0.8954 | 0.8574 | 0.1825 | 0.1077 | 0.1355 | 0.5205 | 0.8320 | 71 | 588 | 318 | 2723 | 0.5000 | 0.8219 | 0.8949 |
| LR | FakeHealth | clean_without_high_similarity | 3645 | 2994 | 651 | 0.7553 | 0.5018 | 0.4966 | 0.8220 | 0.8961 | 0.8575 | 0.1837 | 0.1075 | 0.1357 | 0.5217 | 0.8324 | 70 | 581 | 311 | 2683 | 0.5000 | 0.8214 | 0.8955 |
| SVM | CoAID | all_eligible | 3700 | 3041 | 659 | 0.8059 | 0.5349 | 0.5314 | 0.8327 | 0.9559 | 0.8901 | 0.3589 | 0.1138 | 0.1728 | 0.6823 | 0.9047 | 75 | 584 | 134 | 2907 | 0.0000 | 0.8219 | 0.9435 |
| SVM | CoAID | clean_without_high_similarity | 3645 | 2994 | 651 | 0.8055 | 0.5348 | 0.5312 | 0.8322 | 0.9559 | 0.8898 | 0.3592 | 0.1137 | 0.1727 | 0.6808 | 0.9036 | 74 | 577 | 132 | 2862 | 0.0000 | 0.8214 | 0.9435 |
| SVM | FakeHealth | all_eligible | 3700 | 3041 | 659 | 0.7722 | 0.5054 | 0.4968 | 0.8236 | 0.9198 | 0.8690 | 0.1974 | 0.0910 | 0.1246 | 0.5234 | 0.8334 | 60 | 599 | 244 | 2797 | 0.0000 | 0.8219 | 0.9178 |
| SVM | FakeHealth | clean_without_high_similarity | 3645 | 2994 | 651 | 0.7726 | 0.5057 | 0.4970 | 0.8232 | 0.9208 | 0.8693 | 0.1993 | 0.0906 | 0.1246 | 0.5246 | 0.8338 | 59 | 592 | 237 | 2757 | 0.0000 | 0.8214 | 0.9188 |

## Interpretation and limitations

Both classifiers show a consistent CoAID-source advantage across the five comparison metrics. This is consistent with shared COVID-context linguistic associations across datasets.
H1 would be supported descriptively by a material, consistent CoAID advantage in both LR and SVM. H2 applies when that advantage is absent or inconsistent: the evidence does not support the simple COVID-context explanation. This experiment does not establish COVID-caused lexical change, a causal pandemic/sensitive-period effect, universal COVID misinformation vocabulary, or causal signaling words.
FakeHealth labels derive from its expert-review framework (review-rating threshold), CoAID labels derive from its dataset construction/veracity scheme, and COVID-Rumor uses True/False/Unverified annotations associated with fact-checking sources. These are not perfectly harmonized ground-truth mechanisms. Short target statements versus longer source combined texts, dataset construction, genre, labeling and source composition remain alternative explanations. Duplicate statements, imperfect language checks and incomplete overlap metadata further limit interpretation.

## Existing transfer comparators

LR
| source_dataset | target_dataset | N | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- |
| CoAID | FakeHealth | 1647 | 0.4429 | 0.6047 |
| FakeHealth | CoAID | 3577 | 0.3757 | 0.5367 |

SVM
| source_dataset | target_dataset | N | macro_f1 | roc_auc |
| --- | --- | --- | --- | --- |
| CoAID | FakeHealth | 1647 | 0.4466 | 0.6113 |
| FakeHealth | CoAID | 3577 | 0.3738 | 0.5471 |

These saved comparators use different target populations and label/prevalence distributions; cross-table score changes do not isolate context.

## Validation and files

Runtime checks passed: NEWS-only identity, unique IDs/output keys, binary labels, label-blind quality/overlap functions, frozen cohort and predictions before evaluation-label join, unchanged source vectorizers/configurations, zero fitting, fixed thresholds, identical clean UIDs and one prediction per model/item, and unchanged prior LR/SVM artifacts. Automated test results are recorded separately in covid_rumor_external_transfer_tests.json.
Tables: outputs/tables/covid_rumor_external_*.csv. Target: data/processed/covid_rumor_news_external_target.csv. Predictions: outputs/predictions/covid_rumor_external_transfer_predictions.csv. Freeze locks: outputs/models/covid_rumor_external/. Validation: outputs/reports/covid_rumor_external_transfer.json. No figures were required.
