# Frozen TF-IDF + Logistic Regression cross-context transfer

Only FakeHealth → CoAID and CoAID → FakeHealth were run, each on natural text and the previously frozen source-masked text. No tuning, new classifier, recalibration, threshold change, target-domain fit or engagement analysis was performed.

## Protocol and scope

Source configurations match the validated baseline: FakeHealth uses word uni/bigrams, min_df=5, C=1; CoAID uses unigrams, min_df=2, C=10. Both use sublinear TF, balanced class weights, L2/liblinear, max_iter=5000, tol=1e-4, float64 TF-IDF and seed 42. The threshold is fixed at probability >=0.5.

Each source pipeline was refitted on ALL modeling-ready source rows, including its former within-domain holdout: 1,647 FakeHealth or 3,577 CoAID rows. Within-domain tuning and holdout evaluation had already ended. Source vocabulary and IDF were fitted only on source text. Target feature views exclude labels; the target only passed through transform/predict. Both directions and both conditions were locked before fitting; all four prediction sets were frozen to disk before evaluation labels were joined. A dataset’s labels are used only in its source role for fitting and its target role after prediction freezing for evaluation. Reverse-direction results never alter the other model.

Existing cohorts, masking, CV and holdout assignments are unchanged. The current cross-dataset overlap audit is empty and all eligible cross-dataset overlap flags are false. This reuses the existing exact/high-similarity/resource audit; it does not prove absence of every paraphrase or latent dependence. No raw datasets were accessed.

WITHIN-CONTEXT means FakeHealth → FakeHealth or CoAID → CoAID, using saved training-pool-to-holdout results. CROSS-CONTEXT means FakeHealth → CoAID or CoAID → FakeHealth, with a full-source refit. External results use the entire eligible target cohort; paired degradation comparisons separately use exactly the existing target holdout UIDs. Those saved target metrics and predictions were reused without retraining or reevaluating target-domain models. The target cohorts are external to the source fits, but are previously studied benchmark datasets, not newly collected blind test sets.

## External target performance

| direction | condition | scope | N | fake_N | real_N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | natural | all_eligible | 1647 | 459 | 1188 | 0.6825 | 0.4878 | 0.4429 | 0.2037 | 0.0479 | 0.0776 | 0.7160 | 0.9276 | 0.8082 | 0.6047 | 0.3246 |
| CA_to_FH | natural | same_target_holdout | 330 | 92 | 238 | 0.7061 | 0.5028 | 0.4507 | 0.3077 | 0.0435 | 0.0762 | 0.7224 | 0.9622 | 0.8252 | 0.6191 | 0.3409 |
| CA_to_FH | source_masked | all_eligible | 1647 | 459 | 1188 | 0.6831 | 0.4882 | 0.4432 | 0.2056 | 0.0479 | 0.0777 | 0.7162 | 0.9285 | 0.8087 | 0.6073 | 0.3252 |
| CA_to_FH | source_masked | same_target_holdout | 330 | 92 | 238 | 0.7061 | 0.5028 | 0.4507 | 0.3077 | 0.0435 | 0.0762 | 0.7224 | 0.9622 | 0.8252 | 0.6229 | 0.3418 |
| FH_to_CA | natural | all_eligible | 3577 | 825 | 2752 | 0.3757 | 0.5217 | 0.3757 | 0.2408 | 0.7927 | 0.3694 | 0.8014 | 0.2507 | 0.3820 | 0.5367 | 0.2569 |
| FH_to_CA | natural | same_target_holdout | 702 | 166 | 536 | 0.4302 | 0.5624 | 0.4290 | 0.2679 | 0.8133 | 0.4030 | 0.8434 | 0.3116 | 0.4550 | 0.5705 | 0.2677 |
| FH_to_CA | source_masked | all_eligible | 3577 | 825 | 2752 | 0.3897 | 0.5563 | 0.3897 | 0.2563 | 0.8655 | 0.3955 | 0.8597 | 0.2471 | 0.3839 | 0.5608 | 0.2617 |
| FH_to_CA | source_masked | same_target_holdout | 702 | 166 | 536 | 0.4430 | 0.5999 | 0.4428 | 0.2849 | 0.8976 | 0.4325 | 0.9050 | 0.3022 | 0.4531 | 0.5985 | 0.2782 |

Confusion matrices use rows=true Real/Fake, columns=predicted Real/Fake: [[TN, FP], [FN, TP]].

| direction | condition | scope | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | natural | all_eligible | 1102 | 86 | 437 | 22 |
| CA_to_FH | natural | same_target_holdout | 229 | 9 | 88 | 4 |
| CA_to_FH | source_masked | all_eligible | 1103 | 85 | 437 | 22 |
| CA_to_FH | source_masked | same_target_holdout | 229 | 9 | 88 | 4 |
| FH_to_CA | natural | all_eligible | 690 | 2062 | 171 | 654 |
| FH_to_CA | natural | same_target_holdout | 167 | 369 | 31 | 135 |
| FH_to_CA | source_masked | all_eligible | 680 | 2072 | 111 | 714 |
| FH_to_CA | source_masked | same_target_holdout | 162 | 374 | 17 | 149 |

## Transfer degradation on identical target holdout rows

transfer_drop = cross_context − within_target. Negative values mean poorer external transfer. Natural transfer is compared to the natural within-target model; masked transfer is compared to the saved masked within-target sensitivity model.

| direction | condition | metric | target_holdout_N | within_target | cross_context | transfer_drop |
| --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | natural | balanced_accuracy | 330 | 0.6677 | 0.5028 | -0.1649 |
| CA_to_FH | natural | macro_f1 | 330 | 0.6752 | 0.4507 | -0.2244 |
| CA_to_FH | natural | fake_f1 | 330 | 0.5176 | 0.0762 | -0.4415 |
| CA_to_FH | natural | roc_auc | 330 | 0.7701 | 0.6191 | -0.1510 |
| CA_to_FH | natural | average_precision | 330 | 0.5707 | 0.3409 | -0.2297 |
| CA_to_FH | source_masked | balanced_accuracy | 330 | 0.6644 | 0.5028 | -0.1615 |
| CA_to_FH | source_masked | macro_f1 | 330 | 0.6726 | 0.4507 | -0.2219 |
| CA_to_FH | source_masked | fake_f1 | 330 | 0.5119 | 0.0762 | -0.4357 |
| CA_to_FH | source_masked | roc_auc | 330 | 0.7648 | 0.6229 | -0.1419 |
| CA_to_FH | source_masked | average_precision | 330 | 0.5643 | 0.3418 | -0.2226 |
| FH_to_CA | natural | balanced_accuracy | 702 | 0.9313 | 0.5624 | -0.3689 |
| FH_to_CA | natural | macro_f1 | 702 | 0.9260 | 0.4290 | -0.4970 |
| FH_to_CA | natural | fake_f1 | 702 | 0.8876 | 0.4030 | -0.4846 |
| FH_to_CA | natural | roc_auc | 702 | 0.9827 | 0.5705 | -0.4121 |
| FH_to_CA | natural | average_precision | 702 | 0.9280 | 0.2677 | -0.6603 |
| FH_to_CA | source_masked | balanced_accuracy | 702 | 0.9322 | 0.5999 | -0.3323 |
| FH_to_CA | source_masked | macro_f1 | 702 | 0.9278 | 0.4428 | -0.4849 |
| FH_to_CA | source_masked | fake_f1 | 702 | 0.8902 | 0.4325 | -0.4577 |
| FH_to_CA | source_masked | roc_auc | 702 | 0.9818 | 0.5985 | -0.3833 |
| FH_to_CA | source_masked | average_precision | 702 | 0.9269 | 0.2782 | -0.6487 |

## Explicit source masking

| direction | scope | metric | natural | masked | masked_minus_natural |
| --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | balanced_accuracy | 0.4878 | 0.4882 | 0.0004 |
| CA_to_FH | all_eligible | macro_f1 | 0.4429 | 0.4432 | 0.0003 |
| CA_to_FH | all_eligible | fake_f1 | 0.0776 | 0.0777 | 0.0001 |
| CA_to_FH | all_eligible | roc_auc | 0.6047 | 0.6073 | 0.0027 |
| CA_to_FH | all_eligible | average_precision | 0.3246 | 0.3252 | 0.0005 |
| CA_to_FH | same_target_holdout | balanced_accuracy | 0.5028 | 0.5028 | 0.0000 |
| CA_to_FH | same_target_holdout | macro_f1 | 0.4507 | 0.4507 | 0.0000 |
| CA_to_FH | same_target_holdout | fake_f1 | 0.0762 | 0.0762 | 0.0000 |
| CA_to_FH | same_target_holdout | roc_auc | 0.6191 | 0.6229 | 0.0038 |
| CA_to_FH | same_target_holdout | average_precision | 0.3409 | 0.3418 | 0.0008 |
| FH_to_CA | all_eligible | balanced_accuracy | 0.5217 | 0.5563 | 0.0345 |
| FH_to_CA | all_eligible | macro_f1 | 0.3757 | 0.3897 | 0.0140 |
| FH_to_CA | all_eligible | fake_f1 | 0.3694 | 0.3955 | 0.0261 |
| FH_to_CA | all_eligible | roc_auc | 0.5367 | 0.5608 | 0.0241 |
| FH_to_CA | all_eligible | average_precision | 0.2569 | 0.2617 | 0.0048 |
| FH_to_CA | same_target_holdout | balanced_accuracy | 0.5624 | 0.5999 | 0.0375 |
| FH_to_CA | same_target_holdout | macro_f1 | 0.4290 | 0.4428 | 0.0138 |
| FH_to_CA | same_target_holdout | fake_f1 | 0.4030 | 0.4325 | 0.0295 |
| FH_to_CA | same_target_holdout | roc_auc | 0.5705 | 0.5985 | 0.0279 |
| FH_to_CA | same_target_holdout | average_precision | 0.2677 | 0.2782 | 0.0105 |

Masking uses the unchanged combined_text_source_masked field and the same source hyperparameters and rows. The prior dictionary used metadata from both cohorts without labels: this is a declared transductive string-masking sensitivity. It removes known explicit identifiers, not publisher style, topic, selection, URL paths, text length or other dataset cues. Legitimate institutional mentions may be removed and the SOURCE token remains. These differences are not estimates of the entire publisher-confounding effect.

## Vocabulary and document-length shift

| direction | condition | vocabulary_size | target_documents_with_known_token_percent | target_token_occurrence_coverage_percent | target_median_known_token_fraction | zero_feature_percent | target_feature_sparsity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FH_to_CA | natural | 35168 | 99.9161 | 88.1303 | 0.8810 | 0.0839 | 0.9980 |
| FH_to_CA | source_masked | 35125 | 99.9161 | 88.0817 | 0.8810 | 0.0839 | 0.9980 |
| CA_to_FH | natural | 7299 | 100.0000 | 88.2902 | 0.8880 | 0.0000 | 0.9670 |
| CA_to_FH | source_masked | 7289 | 100.0000 | 88.2962 | 0.8879 | 0.0000 | 0.9670 |

| direction | condition | source_characters_median | target_characters_median | source_analyzer_unigrams_median | target_analyzer_unigrams_median |
| --- | --- | --- | --- | --- | --- |
| FH_to_CA | natural | 3665.0000 | 537.0000 | 583.0000 | 85.0000 |
| FH_to_CA | source_masked | 3657.0000 | 536.0000 | 580.0000 | 84.0000 |
| CA_to_FH | natural | 537.0000 | 3665.0000 | 85.0000 | 583.0000 |
| CA_to_FH | source_masked | 536.0000 | 3657.0000 | 84.0000 | 580.0000 |

Token coverage counts occurrences of lowercased default word-tokenizer unigrams (length >=2) in the fitted source vocabulary, rather than mixing unigram and bigram counts. Fractions use each document’s tokenizer output; empty-token documents have fraction zero and are counted separately. Known-token presence and actual nonzero TF-IDF rows are reported separately. Sparsity is 1 − nnz/(target N × source vocabulary size); high sparsity is normal in text models. Vocabulary/IDF were not updated in response to diagnostics. Full p05/median/p95 length summaries and nonzero counts are saved in the diagnostics CSV.

## Prediction bias and errors

| direction | condition | scope | actual_fake_prevalence | predicted_fake_prevalence | mean_fake_probability | mean_probability_minus_actual | brier_score | near_single_class_prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | natural | all_eligible | 0.2787 | 0.0656 | 0.0734 | -0.2052 | 0.2886 | False |
| CA_to_FH | natural | same_target_holdout | 0.2788 | 0.0394 | 0.0511 | -0.2277 | 0.2774 | True |
| CA_to_FH | source_masked | all_eligible | 0.2787 | 0.0650 | 0.0728 | -0.2059 | 0.2895 | False |
| CA_to_FH | source_masked | same_target_holdout | 0.2788 | 0.0394 | 0.0503 | -0.2284 | 0.2782 | True |
| FH_to_CA | natural | all_eligible | 0.2306 | 0.7593 | 0.5339 | 0.3033 | 0.2691 | False |
| FH_to_CA | natural | same_target_holdout | 0.2365 | 0.7179 | 0.5311 | 0.2946 | 0.2651 | False |
| FH_to_CA | source_masked | all_eligible | 0.2306 | 0.7789 | 0.5352 | 0.3045 | 0.2687 | False |
| FH_to_CA | source_masked | same_target_holdout | 0.2365 | 0.7450 | 0.5317 | 0.2952 | 0.2639 | False |

A prespecified descriptive marginal-calibration warning flags |mean predicted Fake probability − actual Fake prevalence| >=0.10. This is not a full calibration assessment or statistical test. Near-single-class prediction means >=95% predicted as either class. No target recalibration or threshold repair was performed. Brier scores are evaluation diagnostics only; class weighting can shift probability calibration.

| direction | condition | error_category | N |
| --- | --- | --- | --- |
| CA_to_FH | natural | both_correct | 204 |
| CA_to_FH | natural | both_incorrect | 53 |
| CA_to_FH | natural | only_transfer_wrong | 44 |
| CA_to_FH | natural | only_within_wrong | 29 |
| CA_to_FH | source_masked | both_correct | 205 |
| CA_to_FH | source_masked | both_incorrect | 54 |
| CA_to_FH | source_masked | only_transfer_wrong | 43 |
| CA_to_FH | source_masked | only_within_wrong | 28 |
| FH_to_CA | natural | both_correct | 281 |
| FH_to_CA | natural | both_incorrect | 17 |
| FH_to_CA | natural | only_transfer_wrong | 383 |
| FH_to_CA | natural | only_within_wrong | 21 |
| FH_to_CA | source_masked | both_correct | 291 |
| FH_to_CA | source_masked | both_incorrect | 17 |
| FH_to_CA | source_masked | only_transfer_wrong | 374 |
| FH_to_CA | source_masked | only_within_wrong | 20 |

UIDs and the four mutually exclusive error categories are saved for later qualitative review, without article text.

## Interpretation and limitations

FH → CA: external macro F1 0.3757, balanced accuracy 0.5217, Fake F1 0.3694; identical-holdout macro-F1 change -0.4970. Predicted Fake prevalence is 75.9% versus actual 23.1%. Masked external macro-F1 change is +0.0140. Mean Fake probability differs from actual prevalence by at least 10 percentage points; descriptive marginal calibration warning, not a full calibration assessment.

CA → FH: external macro F1 0.4429, balanced accuracy 0.4878, Fake F1 0.0776; identical-holdout macro-F1 change -0.2244. Predicted Fake prevalence is 6.6% versus actual 27.9%. Masked external macro-F1 change is +0.0003. Mean Fake probability differs from actual prevalence by at least 10 percentage points; descriptive marginal calibration warning, not a full calibration assessment.

Both directions lose more than 0.10 macro F1 on identical target holdout rows. Together with class-bias and vocabulary/length diagnostics, this supports substantial dependence on dataset/context/domain in these models. This descriptive result does not identify which component caused the decline.

FakeHealth and CoAID have different ground-truth definitions: author review rating <3 versus >=3 in FakeHealth, and original dataset Fake/Real labels in CoAID. Publisher/source composition differs, and CoAID source/class association is extremely strong. Within-target performance can rely on recurring source and collection cues that do not travel across datasets.

FakeHealth generally has much longer article bodies; CoAID bodies are generally truncated near 500 characters. Topics and collection processes also differ. Transfer therefore combines period, dataset, source, label-definition, text-length and topic shifts. A degradation is not automatically a COVID-period effect, evidence about sensitive periods generally, or a causal explanation of misinformation diffusion.

The full-source transfer models and training-pool-only within-target baselines have different training populations and sample sizes; only evaluation rows are matched. Natural and masked differences are descriptive, not significance tests. No new bootstrap or model search was performed. The primary engagement results remain frozen.

## Validation and artifacts

All four source refits converged: True. Source-only vocabulary/IDF fitting, immutable target transforms, label-free prediction checkpoints, fixed threshold, full target coverage, identical holdout UIDs and unique output keys are checked. Original inputs and baseline artifacts are fingerprinted. Completed runs verify saved hashes and reuse results.

Reproduce: `.venv/bin/python src/run_tfidf_logistic_cross_context_transfer.py`. Tests: `.venv/bin/python -m unittest discover -s src -p "test_tfidf_logistic_cross_context_transfer.py"`. Progress is written to logs/tfidf_logistic_cross_context_transfer.log.

Tables: outputs/tables/tfidf_logistic_transfer_{results,degradation,vocabulary_diagnostics,source_mask_sensitivity,error_overlap,error_overlap_summary,fit_manifest}.csv. Predictions: outputs/predictions/tfidf_logistic_cross_context_predictions.csv. Figures: outputs/figures/tfidf_logistic_transfer/. Fitted source models and unlabeled prediction locks: outputs/models/tfidf_logistic_transfer/. Adjacent report JSON stores fingerprints and validation status.
