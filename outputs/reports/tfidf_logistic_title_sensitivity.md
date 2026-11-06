# Title-only TF-IDF + Logistic Regression sensitivity

## Protocol and sample

Only `title_text` enters the natural predictor pipeline, and only the already-audited `title_text_source_masked` field enters masking sensitivity. Eligibility is the existing `title_only_modeling_eligible` flag; no groups, labels or assignments are regenerated. Fake=1, Real=0; seed=42; threshold=0.5.

| dataset | partition | N | fake_N | real_N | leakage_groups | title_source_cue_N |
| --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | all | 1645 | 457 | 1188 | 1633 | 2 |
| FakeHealth | training_pool | 1315 | 365 | 950 | 1304 | 1 |
| FakeHealth | holdout | 330 | 92 | 238 | 329 | 1 |
| CoAID | all | 3571 | 819 | 2752 | 2822 | 240 |
| CoAID | training_pool | 2870 | 654 | 2216 | 2158 | 173 |
| CoAID | holdout | 701 | 165 | 536 | 664 | 67 |

All fold-level counts and training/validation class-support checks are saved in the cohort summary. Every eligible natural and masked title contains usable analyzer tokens. Existing UID/group assignments match both split manifests exactly. No group crosses partitions or transfer datasets.

The unchanged 24-configuration grid (unigrams/unigrams+bigrams; min_df 2/5; C 0.1/1/10; none/balanced) uses sublinear_tf=True, max_df=1.0, lowercase=True, L2/liblinear, max_iter=5000. Selection reuses the baseline implementation: mean grouped-CV macro F1 within 0.001 of best, then SD within 0.001 of lowest, then unigram, smaller C, unweighted, larger min_df, config ID. Holdout never enters selection.

| dataset | config_id | ngram_max | min_df | C | class_weight | macro_f1_mean | macro_f1_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | cfg_16 | 2 | 2 | 10.0000 | none | 0.8693 | 0.0238 |
| FakeHealth | cfg_11 | 1 | 5 | 10.0000 | balanced | 0.5493 | 0.0292 |

Both natural configurations were locked before any title holdout or transfer evaluation. Within fits use only the source training pool. Transfer refits use all eligible source titles (including the former source holdout); target titles are transformed only. Both directions and both representations have unlabeled frozen prediction bundles before evaluation labels are joined. Dataset labels necessarily serve as source labels in the reverse direction, but never enter the opposite source fit or selection. Masked titles use natural-title hyperparameters without retuning.

## Within-domain title holdout

| dataset | condition | N | macro_f1 | balanced_accuracy | fake_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | natural | 701 | 0.8579 | 0.8594 | 0.7831 | 0.9444 | 0.8514 |
| CoAID | source_masked | 701 | 0.8573 | 0.8573 | 0.7818 | 0.9400 | 0.8458 |
| FakeHealth | natural | 330 | 0.5127 | 0.5155 | 0.3251 | 0.5398 | 0.3387 |
| FakeHealth | source_masked | 330 | 0.5127 | 0.5155 | 0.3251 | 0.5398 | 0.3387 |

## Title cross-context transfer

| direction | scope | N | macro_f1 | balanced_accuracy | fake_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | 1645 | 0.4733 | 0.4960 | 0.1524 | 0.4808 | 0.2663 |
| CA_to_FH | same_target_holdout | 330 | 0.4636 | 0.4880 | 0.1385 | 0.4265 | 0.2568 |
| FH_to_CA | all_eligible | 3571 | 0.4148 | 0.4222 | 0.2162 | 0.4367 | 0.2019 |
| FH_to_CA | same_target_holdout | 701 | 0.4286 | 0.4288 | 0.2028 | 0.4742 | 0.2220 |

## Transfer minus within-target performance

These drops compare exactly the same title-eligible target holdout UIDs. Negative is degradation.

| direction | metric | within_target | cross_context | transfer_drop |
| --- | --- | --- | --- | --- |
| CA_to_FH | macro_f1 | 0.5127 | 0.4636 | -0.0491 |
| CA_to_FH | balanced_accuracy | 0.5155 | 0.4880 | -0.0275 |
| CA_to_FH | fake_f1 | 0.3251 | 0.1385 | -0.1867 |
| CA_to_FH | roc_auc | 0.5398 | 0.4265 | -0.1133 |
| CA_to_FH | average_precision | 0.3387 | 0.2568 | -0.0819 |
| FH_to_CA | macro_f1 | 0.8579 | 0.4286 | -0.4293 |
| FH_to_CA | balanced_accuracy | 0.8594 | 0.4288 | -0.4306 |
| FH_to_CA | fake_f1 | 0.7831 | 0.2028 | -0.5803 |
| FH_to_CA | roc_auc | 0.9444 | 0.4742 | -0.4703 |
| FH_to_CA | average_precision | 0.8514 | 0.2220 | -0.6293 |

## Title versus combined-text transfer

Primary differences below use saved combined-text predictions restricted to exactly the title-eligible evaluation UIDs. No combined model is refitted or re-predicted. The comparison CSV additionally preserves original published combined metrics, original cohort sizes and explicitly unpaired differences. Even with matched target rows, training cohorts differ slightly between representations, and hyperparameters are separately selected; this is a representation-protocol comparison, not an isolated manipulation of body length.

| direction | scope | metric | title_N | combined_original_N | title | combined_same_rows | title_minus_combined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | macro_f1 | 1645 | 1647 | 0.4733 | 0.4396 | 0.0336 |
| CA_to_FH | all_eligible | balanced_accuracy | 1645 | 1647 | 0.4960 | 0.4857 | 0.0103 |
| CA_to_FH | all_eligible | fake_f1 | 1645 | 1647 | 0.1524 | 0.0710 | 0.0814 |
| CA_to_FH | all_eligible | roc_auc | 1645 | 1647 | 0.4808 | 0.6032 | -0.1224 |
| CA_to_FH | all_eligible | average_precision | 1645 | 1647 | 0.2663 | 0.3223 | -0.0560 |
| CA_to_FH | same_target_holdout | macro_f1 | 330 | 330 | 0.4636 | 0.4507 | 0.0129 |
| CA_to_FH | same_target_holdout | balanced_accuracy | 330 | 330 | 0.4880 | 0.5028 | -0.0148 |
| CA_to_FH | same_target_holdout | fake_f1 | 330 | 330 | 0.1385 | 0.0762 | 0.0623 |
| CA_to_FH | same_target_holdout | roc_auc | 330 | 330 | 0.4265 | 0.6191 | -0.1925 |
| CA_to_FH | same_target_holdout | average_precision | 330 | 330 | 0.2568 | 0.3409 | -0.0842 |
| FH_to_CA | all_eligible | macro_f1 | 3571 | 3577 | 0.4148 | 0.3746 | 0.0402 |
| FH_to_CA | all_eligible | balanced_accuracy | 3571 | 3577 | 0.4222 | 0.5210 | -0.0988 |
| FH_to_CA | all_eligible | fake_f1 | 3571 | 3577 | 0.2162 | 0.3672 | -0.1510 |
| FH_to_CA | all_eligible | roc_auc | 3571 | 3577 | 0.4367 | 0.5370 | -0.1003 |
| FH_to_CA | all_eligible | average_precision | 3571 | 3577 | 0.2019 | 0.2560 | -0.0541 |
| FH_to_CA | same_target_holdout | macro_f1 | 701 | 702 | 0.4286 | 0.4281 | 0.0005 |
| FH_to_CA | same_target_holdout | balanced_accuracy | 701 | 702 | 0.4288 | 0.5618 | -0.1330 |
| FH_to_CA | same_target_holdout | fake_f1 | 701 | 702 | 0.2028 | 0.4012 | -0.1984 |
| FH_to_CA | same_target_holdout | roc_auc | 701 | 702 | 0.4742 | 0.5707 | -0.0965 |
| FH_to_CA | same_target_holdout | average_precision | 701 | 702 | 0.2220 | 0.2670 | -0.0449 |

## Prediction bias

| direction | scope | actual_fake_prevalence | predicted_fake_prevalence | mean_fake_probability | tn | fp | fn | tp | near_single_class_prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | 0.2778 | 0.1131 | 0.2324 | 1051 | 137 | 408 | 49 | False |
| CA_to_FH | same_target_holdout | 0.2788 | 0.1152 | 0.2360 | 209 | 29 | 83 | 9 | False |
| FH_to_CA | all_eligible | 0.2293 | 0.4313 | 0.4719 | 1467 | 1285 | 564 | 255 | False |
| FH_to_CA | same_target_holdout | 0.2354 | 0.3695 | 0.4256 | 320 | 216 | 122 | 43 | False |

Confusion counts treat Fake as positive: TN/FP/FN/TP. Near-single-class means predicted Fake prevalence <=5% or >=95%, matching the previous descriptive diagnostic. No calibration or threshold adjustment was made.

## Vocabulary shift

| direction | vocabulary_size | target_token_occurrence_coverage_percent | zero_feature_rows | source_analyzer_unigrams_median | target_analyzer_unigrams_median |
| --- | --- | --- | --- | --- | --- |
| FH_to_CA | 514 | 39.4474 | 386 | 8.0000 | 9.0000 |
| CA_to_FH | 5230 | 71.4472 | 3 | 9.0000 | 8.0000 |

Coverage counts occurrences of default analyzer unigram tokens, without counting bigrams twice. Vocabulary size includes bigrams when selected. Token lengths use that same analyzer. Coverage and zero-feature diagnostics use all eligible target titles.

## Source-masked title sensitivity

The existing audited mask dictionary/fields are unchanged. Masking uses cohort metadata from both datasets without labels (a declared transductive string-masking sensitivity). It can remove legitimate institutional mentions, retains the SOURCE token and URL paths, and does not eliminate publisher style or source selection. Cohorts and hyperparameters are identical to natural title fits.

| direction | scope | macro_f1 | fake_f1 | roc_auc | predicted_fake_prevalence |
| --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | 0.4719 | 0.1515 | 0.4819 | 0.1155 |
| CA_to_FH | same_target_holdout | 0.4540 | 0.1231 | 0.4267 | 0.1152 |
| FH_to_CA | all_eligible | 0.4165 | 0.2170 | 0.4378 | 0.4287 |
| FH_to_CA | same_target_holdout | 0.4323 | 0.2048 | 0.4789 | 0.3638 |

## Interpretation

FH_to_CA: title transfer macro F1 is 0.4286 on target holdout and 0.4148 on all eligible target rows. Same-row title-minus-combined changes are +0.0005 and +0.0402, respectively. On holdout, Fake F1 changes by -0.1984 and ROC-AUC by -0.0965; transfer-minus-within title macro F1 is -0.4293.

CA_to_FH: title transfer macro F1 is 0.4636 on target holdout and 0.4733 on all eligible target rows. Same-row title-minus-combined changes are +0.0129 and +0.0336, respectively. On holdout, Fake F1 changes by +0.0623 and ROC-AUC by -0.1925; transfer-minus-within title macro F1 is -0.0491.

CoAID within-domain title macro F1 is 0.8579, versus 0.9256 for combined text on identical holdout rows (-0.0677).

FakeHealth within-domain title macro F1 is 0.5127, versus 0.6752 for combined text on identical holdout rows (-0.1625).

Title-only does not materially rescue transfer in a consistent way across the prespecified metrics: holdout macro-F1 improvements are minimal, both directions have below-0.5 title ROC-AUC, and ROC-AUC declines relative to combined text in both scopes. FakeHealth-to-CoAID Fake F1 also declines, while the reverse direction improves Fake F1 but retains poor ranking. This is a descriptive judgment without a formal significance test. The smaller CoAID-to-FakeHealth within-versus-transfer gap largely reflects a substantially weaker within-FakeHealth title baseline, rather than successful transfer.

Natural transfer predictions are no longer near-single-class by the prespecified 5%/95% rule, but still overpredict Fake for FakeHealth-to-CoAID and underpredict Fake in the reverse direction. Median analyzer title lengths are now similar (8 versus 9 tokens), yet vocabulary coverage remains asymmetric (about 39% versus 71%); 386 CoAID targets have zero features under the FakeHealth title model. Thus length comparability does not remove lexical/domain mismatch. These findings remain consistent with dataset/context dependence, without identifying which underlying difference causes it.

Interpretation is based on the complete prespecified metrics and both evaluation scopes; no post-hoc significance or materiality cutoff is applied. A narrower transfer drop alone can reflect a weaker within-domain title model, so the absolute same-row transfer change must also be examined.

Representation changes remove bodies and reduce document-length disparity, but also change available content, vocabulary/IDF, selected hyperparameters and slightly the source sample. Therefore improvement cannot prove that body length alone caused prior failure. Remaining dataset/context dependence may include topical mix, periods, collection and publisher/style differences. FakeHealth review-rating-derived labels and CoAID veracity labels have different operational definitions; title-only modeling does not harmonize those labels or identify a causal COVID effect.

## Validation and artifacts

All 240 CV fits converged: True; all 8 final fits converged: True. CV failures, if any, remain documented and excluded by the unchanged selection rule. No grid expansion, threshold tuning or target fitting occurred.

Validation covers unchanged split and input fingerprints, group separation, title-only predictor allowlisting, class support, locked configurations before transfer, source-only vocabulary/IDF fitting with immutable target transforms, target-label-free prediction interfaces, threshold 0.5, full target coverage, masking row parity and unique output keys. Final hashes are recorded alongside this report. CV checkpoints, selection/prediction locks and final models support reuse without repeating fits.

Tables: `outputs/tables/tfidf_logistic_title_{cv_results,model_selection,holdout_results,transfer_results,transfer_degradation,vs_combined,vocabulary_diagnostics,cohort_summary,fit_manifest}.csv`. Predictions: `outputs/predictions/tfidf_logistic_title_predictions.csv` (no raw titles). Figures: `outputs/figures/tfidf_logistic_title/`. Models/locks: `outputs/models/tfidf_logistic_title/`. Script: `src/run_tfidf_logistic_title_sensitivity.py`. Progress: `logs/tfidf_logistic_title_sensitivity.log`.
