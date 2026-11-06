# Cross-context lexical feature-alignment audit

This is a descriptive audit of the two validated natural-text within-dataset models. No model was fitted, tuned, or used to rerun transfer. Positive coefficients predict Fake=1; negative coefficients predict Real=0.

## Vocabulary and methods

- FakeHealth: 7,381 unigrams; 56.55% represented in the shared vocabulary.
- CoAID: 6,428 unigrams; 64.93% represented in the shared vocabulary.
- Shared: 4,174 unigrams. FakeHealth bigrams are excluded from comparisons, but remain part of the fitted model conditioning its unigram coefficients.
- FakeHealth: unigrams+bigrams, min_df=5, C=1. CoAID: unigrams, min_df=2, C=10. Both use balanced L2/liblinear.
- Document frequencies are recovered exactly (checked against integer bounds and existing top-feature rows) from saved smooth IDF and verified training N; no article text is read.
- Absolute ranks use minimum rank for ties; both full unigram-vocabulary ranks and shared-vocabulary ranks are saved. Sorted lexical order breaks top-N ties deterministically.
- The median cutoff uses each coefficient absolute-value median within the shared vocabulary, strictly exceeded in either dataset. Top-N uses maximum absolute coefficient. All cutoffs are fixed independently of target outcomes.
- Near zero means |beta| < 0.05. Meaningful in both means min(|beta_FH|, |beta_CA|) >= 0.05; “strong” is descriptive, not a statistical significance claim. Category rankings use this minimum.

## Alignment at every prespecified cutoff

| cutoff | N | pearson | spearman | absolute_magnitude_spearman | sign_agreement_percent | sign_reversal_percent |
| --- | --- | --- | --- | --- | --- | --- |
| all | 4174 | -0.0148 | -0.0034 | 0.1292 | 51.6052 | 48.3948 |
| above_median_either | 3052 | -0.0191 | -0.0098 | -0.1819 | 51.5072 | 48.4928 |
| top1000 | 1000 | -0.0393 | 0.0094 | 0.1369 | 51.2000 | 48.8000 |
| top500 | 500 | -0.0569 | 0.0050 | 0.1831 | 53.0000 | 47.0000 |
| top100 | 100 | -0.1451 | -0.1013 | 0.2983 | 52.0000 | 48.0000 |

Pearson and Spearman above describe signed coefficients; absolute-magnitude Spearman separately describes importance-rank stability. Quadrant counts and percentages, including exact-zero accounting, are in the summary CSV.

Strong reversals: 1,175/4,174 shared features (28.15%), or 1,175/2,421 meaningful-in-both features (48.53%).

## Top-feature transfer stability

| source_dataset | source_association | selected_N | same_direction_N | opposite_direction_N | near_zero_N |
| --- | --- | --- | --- | --- | --- |
| fakehealth | Fake | 25 | 9 | 16 | 0 |
| fakehealth | Real | 25 | 15 | 9 | 1 |
| coaid | Fake | 25 | 7 | 14 | 4 |
| coaid | Real | 25 | 12 | 9 | 4 |

The detail CSV reports every selected source feature and its target coefficient/status. Near-zero target coefficients take precedence over sign comparisons.

## Strong sign reversals

1,175 eligible features; CSV saves up to 100 ranked features. First 10:

| feature | fakehealth_coefficient | coaid_coefficient | explicit_source_identifier | any_context_flag |
| --- | --- | --- | --- | --- |
| release | 1.0228 | -0.9227 | False | False |
| and | 0.8417 | -9.2397 | False | False |
| to | 0.7763 | -2.5722 | False | False |
| says | -0.7370 | 2.9898 | False | False |
| vaccine | -0.6427 | 1.6915 | False | True |
| of | 0.6238 | -5.1850 | False | False |
| news | 0.8836 | -0.6169 | False | False |
| artificial | 0.6716 | -0.6014 | False | False |
| chronic | 0.6165 | -0.5532 | False | False |
| dr | -0.5364 | 1.1303 | False | False |

## Stable fake features

281 eligible features; CSV saves up to 100 ranked features. First 10:

| feature | fakehealth_coefficient | coaid_coefficient | explicit_source_identifier | any_context_flag |
| --- | --- | --- | --- | --- |
| the | 0.8383 | 2.7804 | False | False |
| able | 0.4979 | 0.4611 | False | False |
| surgical | 0.4292 | 0.4554 | False | False |
| genetic | 0.4045 | 0.3847 | False | False |
| alcohol | 0.3502 | 0.7055 | False | False |
| uk | 0.3329 | 0.7629 | False | False |
| oral | 0.3293 | 0.6153 | False | False |
| sexual | 0.3275 | 0.4458 | False | False |
| minute | 0.3187 | 1.3631 | False | False |
| foxnews | 0.3203 | 0.3159 | False | False |

## Stable real features

965 eligible features; CSV saves up to 100 ranked features. First 10:

| feature | fakehealth_coefficient | coaid_coefficient | explicit_source_identifier | any_context_flag |
| --- | --- | --- | --- | --- |
| there | -0.7016 | -1.0578 | False | False |
| drug | -0.6585 | -1.4984 | False | True |
| many | -0.6520 | -1.8796 | False | False |
| some | -0.5414 | -3.4809 | False | False |
| however | -0.5409 | -0.8010 | False | False |
| 000 | -0.5307 | -1.1718 | False | False |
| safe | -0.5209 | -2.1256 | False | False |
| sleep | -0.5360 | -0.4989 | False | False |
| experimental | -0.4752 | -0.8626 | False | False |
| but | -0.4641 | -0.9861 | False | False |

## Source identifiers and transparent context flags

The unchanged baseline mask dictionary is used label-blind. Full tokenized identifiers (at least three characters) are distinguished from component/fragment matches, following baseline rules. Components can be ordinary words; neither flag establishes actual publisher attribution. Dictionary coverage is incomplete and includes institutional mentions. No new source dictionary or masking is introduced.

| group | N | flagged_N | percent | difference_percentage_points |
| --- | --- | --- | --- | --- |
| all_shared | 4174 | 11 | 0.2635 | 0.0000 |
| meaningful_both | 2421 | 8 | 0.3304 | 0.0669 |
| strong_sign_reversals_all | 1175 | 3 | 0.2553 | -0.0082 |
| strong_sign_reversals_top100 | 100 | 0 | 0.0000 | -0.2635 |
| stable_fake_features_all | 281 | 0 | 0.0000 | -0.2635 |
| stable_fake_features_top100 | 100 | 0 | 0.0000 | -0.2635 |
| stable_real_features_all | 965 | 5 | 0.5181 | 0.2546 |
| stable_real_features_top100 | 100 | 0 | 0.0000 | -0.2635 |

Differences are descriptive percentage-point differences from all shared unigrams, without enrichment significance tests. The supplemental CSV also provides the meaningful-in-both comparison pool, broad fragments, each context category, and top-100 versus full-category counts.

Manual exact-token context dictionary (fixed before coefficient inspection):

- covid_specific: coronavirus, cov, covid, lockdown, pandemic, quarantine, sars.
- election_political: biden, democrat, election, president, republican, stimulus, trump.
- generic_health: cancer, disease, drug, drugs, health, infection, medical, patients, treatment, vaccine, vaccines, virus.

| group | N | flagged_N | percent |
| --- | --- | --- | --- |
| all_shared | 4174 | 13 | 0.3115 |
| meaningful_both | 2421 | 12 | 0.4957 |
| strong_sign_reversals_all | 1175 | 7 | 0.5957 |
| strong_sign_reversals_top100 | 100 | 1 | 1.0000 |
| stable_fake_features_all | 281 | 0 | 0.0000 |
| stable_fake_features_top100 | 100 | 0 | 0.0000 |
| stable_real_features_all | 965 | 5 | 0.5181 |
| stable_real_features_top100 | 100 | 5 | 5.0000 |

This small dictionary identifies examples; its low coverage cannot rule out topic/context dependence, and generic health terms do not uniquely identify a context.

## Interpretation

Coefficient directions are weakly aligned: overall agreement is 51.61%, with essentially zero signed Pearson (-0.0148) and Spearman (-0.0034) correlations. Absolute-magnitude rank correlation is weak (0.1292). Agreement by cutoff is all: 51.61%, above_median_either: 51.51%, top1000: 51.20%, top500: 53.00%, top100: 52.00%; there is no consistent improvement among more important features. These percentages are descriptive, not a test against a 50% null, because the marginal sign distributions differ.

Reversals are common among features meaningful in both models (48.53%). Only 3/1175 (0.26%) are full known source identifiers, versus 0.26% of all shared features; none of the top 100 reversals are exact identifiers. Thus known explicit identifiers do not account for much of the reversal. The stable Fake set has 0 exact identifiers and the stable Real set has 5; the full stable Real set is descriptively enriched relative to the shared pool, but the counts are small and its top 100 contain none.

The manual context flags identify 7/1175 reversals and 1/100 of the strongest reversals (vaccine). Health/topic vocabulary supplies a concrete instability example, but these narrow flags do not establish that reversals are strongly or predominantly topic-related. The strongest reversals also include function words (and, to, of) and reporting terms (release, says, news), consistent with multiple possible distributional or stylistic differences rather than a uniquely identified mechanism. COVID-specific tokens absent from the intersection are outside this audit.

Signed correlations and agreement rates above quantify whether lexical associations align. Importance-cutoff results must be read together rather than selecting a favorable cutoff. Stable lists identify associations in these two datasets only, not universal misinformation markers.

Opposing coefficients plausibly contribute to transfer degradation, but this comparison does not isolate their contribution or establish that they explain the collapse. Coefficients are conditional on different vocabularies, correlated predictors, IDF scales, document distributions, regularization strengths, and label constructions. Maximum-absolute-coefficient ranking can favor the less regularized CoAID model. Raw coefficient magnitudes are not standardized, directly comparable causal effects.

The models were trained only on their respective baseline training partitions. Comparing their fitted associations is a retrospective audit, not a feature-selection procedure for deployment. Cross-context differences do not prove a COVID-period effect, a causal sensitivity effect, or a universal misinformation mechanism.

Optional transfer-error linkage was skipped: it is not needed for the primary audit and would require additional prediction/cohort joins. Transfer fitting and evaluation were not rerun.

## Validation and outputs

Validation passed: baseline model/input/selection fingerprints, environment and protocol, training UID/group/vocabulary fingerprints, Fake=1 class orientation, unigram-only deterministic intersection, recovered document frequencies, prespecified coefficient-only cutoffs, unique table keys, and sign accounting. Runtime guards prohibit estimator fitting throughout the audit. Input fingerprints are rechecked after writing outputs.

Script: `src/audit_cross_context_lexical_alignment.py`. Tables: `outputs/tables/text_shared_unigram_coefficients.csv`, `text_coefficient_alignment_summary.csv`, `text_strong_sign_reversals.csv`, `text_stable_fake_features.csv`, `text_stable_real_features.csv`, `text_top_feature_transfer_stability.csv`; supplemental `text_top_feature_transfer_details.csv` and `text_feature_alignment_flag_summary.csv`. Figures: `outputs/figures/text_feature_alignment/`. Adjacent JSON records thresholds, fingerprints and validation.
