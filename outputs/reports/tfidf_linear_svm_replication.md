# TF-IDF Linear SVM model-robustness replication

Within-domain SVM macro F1 is 0.6697 for FakeHealth and 0.9128 for CoAID (difference +0.2430).

FH_to_CA: macro F1 0.4245 on target holdout and 0.3738 on all eligible target rows; holdout transfer-minus-within -0.4883. Holdout ROC-AUC/AP are 0.5827/0.2738. SVM-minus-LR holdout macro F1 is -0.0045. Predicted Fake prevalence is 76.64% versus actual 23.65%; near-single-class flag=False.

CA_to_FH: macro F1 0.4493 on target holdout and 0.4466 on all eligible target rows; holdout transfer-minus-within -0.2205. Holdout ROC-AUC/AP are 0.6313/0.3515. SVM-minus-LR holdout macro F1 is -0.0014. Predicted Fake prevalence is 4.24% versus actual 27.88%; near-single-class flag=True.

Directional differences are metric-specific: FH→CA minus CA→FH holdout macro F1 is -0.0248, while their within-to-transfer macro-F1 drops are -0.4883 and -0.2205. Read predicted-class bias, Fake F1 and ranking metrics together rather than treating a single macro-F1 difference as the entire asymmetry.

The prespecified qualitative pattern is reproduced by both classifiers: True. The CoAID-minus-FakeHealth within-domain macro-F1 gap is +0.2430 for SVM and +0.2508 for LR. LR target-holdout macro-F1 drops are FH→CA -0.4970 and CA→FH -0.2244, to compare with the SVM drops above. Reproduction here means a positive within-domain gap and negative transfer drops in both directions, not a post-hoc statistical test of effect size or superiority. The reported metric values and class-prediction biases determine how severe and asymmetric the transfer failure remains.

## Scope and unchanged cohorts

This replication changes the classical linear classifier and its prespecified tuning grid. Only natural combined_text is a predictor, on primary_modeling_eligible rows in FakeHealth HealthStory and CoAID News Specification A. Fake=1, Real=0. No source masking, title/body-only sensitivity, publisher-held-out SVM, transformers, or engagement analyses run. LR results are frozen comparators.

| dataset | partition | N | fake_N | real_N | leakage_groups | uid_sha256 |
| --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | all | 1647 | 459 | 1188 | 1634 | 70614b831ba540c2a3c4cfe73d616e85835c9fc57b2755b2a9b4e982b726d1d9 |
| FakeHealth | training_pool | 1317 | 367 | 950 | 1305 | aacf74df3b7f01b9388b1246941ece1cdf9ce1bab67eb0bba24a598d8123aeb3 |
| FakeHealth | holdout | 330 | 92 | 238 | 329 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| CoAID | all | 3577 | 825 | 2752 | 2828 | d61e991983615e761ee171812fde1597b689c88dba8121705b9d3845e8e8c1ae |
| CoAID | training_pool | 2875 | 659 | 2216 | 2163 | 5d1c28add5601c9064b556a409dbaad0cdb7df6a65c89ede977682a93b3eaab5 |
| CoAID | holdout | 702 | 166 | 536 | 665 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |

The existing grouped CV and holdout assignments are verified against original manifests. cv_fold=-1 is reserved holdout; the five training-pool folds are unchanged. Every training/validation/test partition retains both labels with zero resource-group overlap. Cohort, split and comparator fingerprints are checked before and after execution.

## Pipeline and selection

Pipeline: TfidfVectorizer → LinearSVC. TF-IDF uses sublinear_tf=True, max_df=1.0, lowercase=True and default L2 normalization/smoothed IDF. The 32 configurations cross unigram/unigram+bigram, min_df=2/5, C=0.01/0.1/1/10, and class_weight=None/balanced. LinearSVC uses L2 penalty, squared-hinge loss, dual="auto", tol=1e-4, max_iter=20000, fit_intercept=True, intercept_scaling=1 and random_state=42. No grid expansion occurs.

Selection uses source training-pool grouped CV only: mean macro F1 within 0.001 of best, then fold sample SD within 0.001 of lowest, then unigram, smaller C, class_weight=None and configuration ID. Unlike the LR baseline’s extra min_df tie priority, this follows the explicitly requested SVM configuration-ID final tie break. Holdout outcomes never enter selection.

| dataset | config_id | ngram_max | min_df | C | class_weight | macro_f1_mean | macro_f1_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | cfg_03 | 1 | 2 | 0.1000 | balanced | 0.9393 | 0.0147 |
| FakeHealth | cfg_27 | 2 | 5 | 0.1000 | balanced | 0.6628 | 0.0256 |

Both configurations are locked before final fitting/predictions. Within-domain models refit only the dataset-specific training pool. Transfer models refit all eligible source rows, including the former source holdout; target data is transform-only. All four unlabeled prediction bundles freeze before labels are joined for evaluation. Labels are inspected separately for preflight class support and used as source labels in the reverse direction; they never enter the opposite source fitting/selection interface.

Positive decision scores favor Fake, with prediction Fake iff score >=0. ROC-AUC and average precision use raw decision_function scores. Scores are not probabilities. No probability calibration, Brier score, probability averages, target threshold tuning or recalibration is performed.

## Within-domain holdout

| dataset | N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID | 702 | 0.9345 | 0.9301 | 0.9128 | 0.8226 | 0.9217 | 0.8693 | 0.9748 | 0.9384 | 0.9563 | 0.9795 | 0.9110 |
| FakeHealth | 330 | 0.7455 | 0.6635 | 0.6697 | 0.5500 | 0.4783 | 0.5116 | 0.8080 | 0.8487 | 0.8279 | 0.7695 | 0.5707 |

## Cross-context transfer

| direction | scope | N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | 1647 | 0.6806 | 0.4885 | 0.4466 | 0.2137 | 0.0545 | 0.0868 | 0.8065 | 0.6113 | 0.3288 |
| CA_to_FH | same_target_holdout | 330 | 0.7030 | 0.5007 | 0.4493 | 0.2857 | 0.0435 | 0.0755 | 0.8231 | 0.6313 | 0.3515 |
| FH_to_CA | all_eligible | 3577 | 0.3743 | 0.5493 | 0.3738 | 0.2525 | 0.8739 | 0.3918 | 0.3558 | 0.5471 | 0.2626 |
| FH_to_CA | same_target_holdout | 702 | 0.4245 | 0.5899 | 0.4245 | 0.2788 | 0.9036 | 0.4261 | 0.4229 | 0.5827 | 0.2738 |

## Confusion counts and class bias

| dataset | tn | fp | fn | tp | actual_fake_prevalence | predicted_fake_prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| CoAID | 503 | 33 | 13 | 153 | 0.2365 | 0.2650 |
| FakeHealth | 202 | 36 | 48 | 44 | 0.2788 | 0.2424 |

| direction | scope | tn | fp | fn | tp | actual_fake_prevalence | predicted_fake_prevalence | near_single_class_prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | all_eligible | 1096 | 92 | 434 | 25 | 0.2787 | 0.0710 | False |
| CA_to_FH | same_target_holdout | 228 | 10 | 88 | 4 | 0.2788 | 0.0424 | True |
| FH_to_CA | all_eligible | 618 | 2134 | 104 | 721 | 0.2306 | 0.7982 | False |
| FH_to_CA | same_target_holdout | 148 | 388 | 16 | 150 | 0.2365 | 0.7664 | False |

TN/FP/FN/TP use Fake as the positive class. The descriptive near-single-class flag means predicted Fake prevalence <=5% or >=95%; it does not modify fitting, classification thresholds or outputs.

## Transfer degradation on identical target holdout UIDs

| direction | target_dataset | metric | target_holdout_N | within_target | cross_context | transfer_drop | target_uid_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CA_to_FH | FakeHealth | macro_f1 | 330 | 0.6697 | 0.4493 | -0.2205 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| CA_to_FH | FakeHealth | balanced_accuracy | 330 | 0.6635 | 0.5007 | -0.1628 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| CA_to_FH | FakeHealth | fake_f1 | 330 | 0.5116 | 0.0755 | -0.4362 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| CA_to_FH | FakeHealth | roc_auc | 330 | 0.7695 | 0.6313 | -0.1383 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| CA_to_FH | FakeHealth | average_precision | 330 | 0.5707 | 0.3515 | -0.2192 | a4bc06df5c268ff3927d579d7cd756d7b055dd141a1e12f132a56c8f36118d1d |
| FH_to_CA | CoAID | macro_f1 | 702 | 0.9128 | 0.4245 | -0.4883 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |
| FH_to_CA | CoAID | balanced_accuracy | 702 | 0.9301 | 0.5899 | -0.3402 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |
| FH_to_CA | CoAID | fake_f1 | 702 | 0.8693 | 0.4261 | -0.4432 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |
| FH_to_CA | CoAID | roc_auc | 702 | 0.9795 | 0.5827 | -0.3967 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |
| FH_to_CA | CoAID | average_precision | 702 | 0.9110 | 0.2738 | -0.6372 | ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa |

## SVM versus frozen Logistic Regression

| evaluation | direction | scope | metric | N | logistic | linear_svm | svm_minus_logistic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| within | within_coaid | same_target_holdout | macro_f1 | 702 | 0.9260 | 0.9128 | -0.0132 |
| within | within_coaid | same_target_holdout | balanced_accuracy | 702 | 0.9313 | 0.9301 | -0.0012 |
| within | within_coaid | same_target_holdout | fake_f1 | 702 | 0.8876 | 0.8693 | -0.0183 |
| within | within_coaid | same_target_holdout | roc_auc | 702 | 0.9827 | 0.9795 | -0.0032 |
| within | within_coaid | same_target_holdout | average_precision | 702 | 0.9280 | 0.9110 | -0.0169 |
| within | within_fakehealth | same_target_holdout | macro_f1 | 330 | 0.6752 | 0.6697 | -0.0054 |
| within | within_fakehealth | same_target_holdout | balanced_accuracy | 330 | 0.6677 | 0.6635 | -0.0042 |
| within | within_fakehealth | same_target_holdout | fake_f1 | 330 | 0.5176 | 0.5116 | -0.0060 |
| within | within_fakehealth | same_target_holdout | roc_auc | 330 | 0.7701 | 0.7695 | -0.0005 |
| within | within_fakehealth | same_target_holdout | average_precision | 330 | 0.5707 | 0.5707 | 0.0000 |
| transfer | CA_to_FH | all_eligible | macro_f1 | 1647 | 0.4429 | 0.4466 | 0.0037 |
| transfer | CA_to_FH | all_eligible | balanced_accuracy | 1647 | 0.4878 | 0.4885 | 0.0007 |
| transfer | CA_to_FH | all_eligible | fake_f1 | 1647 | 0.0776 | 0.0868 | 0.0092 |
| transfer | CA_to_FH | all_eligible | roc_auc | 1647 | 0.6047 | 0.6113 | 0.0067 |
| transfer | CA_to_FH | all_eligible | average_precision | 1647 | 0.3246 | 0.3288 | 0.0041 |
| transfer | CA_to_FH | same_target_holdout | macro_f1 | 330 | 0.4507 | 0.4493 | -0.0014 |
| transfer | CA_to_FH | same_target_holdout | balanced_accuracy | 330 | 0.5028 | 0.5007 | -0.0021 |
| transfer | CA_to_FH | same_target_holdout | fake_f1 | 330 | 0.0762 | 0.0755 | -0.0007 |
| transfer | CA_to_FH | same_target_holdout | roc_auc | 330 | 0.6191 | 0.6313 | 0.0122 |
| transfer | CA_to_FH | same_target_holdout | average_precision | 330 | 0.3409 | 0.3515 | 0.0105 |
| transfer | FH_to_CA | all_eligible | macro_f1 | 3577 | 0.3757 | 0.3738 | -0.0019 |
| transfer | FH_to_CA | all_eligible | balanced_accuracy | 3577 | 0.5217 | 0.5493 | 0.0275 |
| transfer | FH_to_CA | all_eligible | fake_f1 | 3577 | 0.3694 | 0.3918 | 0.0225 |
| transfer | FH_to_CA | all_eligible | roc_auc | 3577 | 0.5367 | 0.5471 | 0.0104 |
| transfer | FH_to_CA | all_eligible | average_precision | 3577 | 0.2569 | 0.2626 | 0.0057 |
| transfer | FH_to_CA | same_target_holdout | macro_f1 | 702 | 0.4290 | 0.4245 | -0.0045 |
| transfer | FH_to_CA | same_target_holdout | balanced_accuracy | 702 | 0.5624 | 0.5899 | 0.0275 |
| transfer | FH_to_CA | same_target_holdout | fake_f1 | 702 | 0.4030 | 0.4261 | 0.0232 |
| transfer | FH_to_CA | same_target_holdout | roc_auc | 702 | 0.5705 | 0.5827 | 0.0122 |
| transfer | FH_to_CA | same_target_holdout | average_precision | 702 | 0.2677 | 0.2738 | 0.0061 |

Every comparison uses identical evaluation UIDs, verified against the saved LR result fingerprints; no LR prediction files or fitted models are reopened. Same rows enable direct descriptive comparisons but no paired significance test is performed. Both classifiers have independently selected source-only hyperparameters, and their C grids differ as prespecified. Small differences should not be presented as model superiority.

## Interpretation and limits

The tables above assess three distinct questions: the CoAID–FakeHealth within-domain gap, loss of performance when crossing contexts, and direction-specific error/ranking patterns. Transfer asymmetry includes different class-prediction biases and different drops from the relevant within-target baseline; its exact numerical ordering need not be identical across metrics or classifiers. The experiment tests robustness to classical linear classifier choice, not an optimized leaderboard objective.

Similar LR/SVM behavior does NOT prove that the pattern is model-independent in general. It only shows robustness across two classical linear text classifiers using TF-IDF. Dataset collection, topic, publisher/style and label-definition differences remain entangled; the replication does not identify a causal COVID-period effect, universal misinformation mechanism or probability calibration.

## Validation and reproducibility

320 CV fits and 4 final fits completed. All CV fits converged: True; all final fits converged: True. Inner nonconvergence is logged and disqualifies the configuration; final nonconvergence stops execution. No methodological changes are made in response to transfer results.

Checks cover frozen splits/groups, training-only TF-IDF, predictor allowlisting, label-free target transforms, source-only selection, both configurations locked before transfer, full target coverage, decision threshold zero, unique output keys and unchanged LR comparator inputs. A runtime guard prevents LogisticRegression.fit throughout the experiment. Fitted final models, prediction locks and CV checkpoints support completed-run reuse without new fitting or predictions.

Tables: `outputs/tables/tfidf_linear_svm_{cv_results,model_selection,holdout_results,transfer_results,transfer_degradation,vs_logistic,cohort_summary,fit_manifest}.csv`. Predictions: `outputs/predictions/tfidf_linear_svm_predictions.csv` (no text). Figures: `outputs/figures/tfidf_linear_svm/`. Models/locks: `outputs/models/tfidf_linear_svm/`. Progress: `logs/tfidf_linear_svm_replication.log`. Script: `src/run_tfidf_linear_svm_replication.py`. Adjacent report JSON records protocol, checksums and validation.
