# FakeHealth unseen-publisher TF-IDF + Logistic Regression

Natural outer macro F1: 0.6030, 0.6432, 0.6125; mean 0.6196 ± sample SD 0.0210; pooled 0.6209.

All three folds retain above-0.5 balanced accuracy and ROC-AUC: True. Outer macro F1 ranges 0.6030–0.6432; mean macro F1 after omitting any single fold ranges 0.6077–0.6278. All outer folds contain 549–549 rows, so no fold dominates pooled sample size. The pooled analytic always-Real macro-F1 reference is 0.4190, compared with model macro F1 0.6209. These descriptive checks distinguish surviving predictive association from a result sustained by one favorable fold; they are not significance tests.

## Cohort, grouping and split protocol

The unchanged primary modeling-ready FakeHealth cohort contains 1647 rows, 459 Fake and 1188 Real, 49 audited publisher/source families and 49 publisher/resource components. Only combined_text is used for natural prediction; Fake=1, Real=0. The saved publisher-family map and pure audited component function are reused without renormalizing sources or rerunning the feasibility audit.

Rows sort by content_uid. Outer StratifiedGroupKFold uses n_splits=3, shuffle=True, random_state=42. Publisher families, components and original leakage groups never cross train/test, and every outer test has >=20 Fake and >=20 Real. These authorized publisher folds are new files; the original resource splits remain unchanged. Families are observed website/name proxies, not a fully established corporate-ownership ontology.

| outer_fold | scope | inner_fold | inner_folds | N | fake_N | real_N | publisher_families | constraint_components | uid_sha256 | inner_choice_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | outer_train | -1 | 3 | 1098 | 306 | 792 | 34 | 34 | e2fc8553fa425ee6f817bf3bccd2304d974eb2937762526b398ddaa276e32fd4 | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | outer_test | -1 | 3 | 549 | 153 | 396 | 15 | 15 | 6921a94f21522fc462ecf98eb460c27774ed33009a016e528f6e7745c9f38dc7 | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_train | 0 | 3 | 737 | 204 | 533 | 26 | 26 | 4453ac021d2dd87c7a49b98723a17b3417922d888891208e69cc7954d4133aa2 | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_validation | 0 | 3 | 361 | 102 | 259 | 8 | 8 | d506a42c4fab3d7df52fc7c4d8943afbe3d48cb35a1379e141ac5aa9f33a9e36 | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_train | 1 | 3 | 724 | 205 | 519 | 28 | 28 | 6cad8603d86210ce84c60efaf1a285f40b57240b68a9d318032ef98625877e4a | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_validation | 1 | 3 | 374 | 101 | 273 | 6 | 6 | 61463c4a2369489878be5675673189a59613561eb2fba3e126c7443905692171 | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_train | 2 | 3 | 735 | 203 | 532 | 14 | 14 | 3fb2e29515b556e80eab4dbc47d833510596c7e155efaed0ac66181964c765fd | Three folds retain >=20 of each validation class with intact barriers. |
| 0 | inner_validation | 2 | 3 | 363 | 103 | 260 | 20 | 20 | 71ae208f55a7d1d94ec92e73ea69af7a43a3e9240dd5b26ba9005e508ed1bcc7 | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | outer_train | -1 | 3 | 1098 | 307 | 791 | 31 | 31 | 1528b2fa5bd5aad01fdb670e802471bc8a4182fbadba5f8b0ba695cc7ed049bc | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | outer_test | -1 | 3 | 549 | 152 | 397 | 18 | 18 | 9b2aa61c28192aabc7c2d0649bada1664aad8010ce6e7e4e8e401e24e119326e | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_train | 0 | 3 | 734 | 205 | 529 | 26 | 26 | 041ee12b579a8e5bb685101f03384e9eca4b4dff8bd4990133cc19b13e4c6887 | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_validation | 0 | 3 | 364 | 102 | 262 | 5 | 5 | 454597d80aa4cbc8e0b982a820006948e022ea031ff8a96d3d235b446dd34bc8 | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_train | 1 | 3 | 722 | 201 | 521 | 22 | 22 | b24af0f3bbc2f3f67a221ca4899decae6b29cb4dfd663069f8792fb1ccddf27c | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_validation | 1 | 3 | 376 | 106 | 270 | 9 | 9 | be3b76a86124300e8b5e1dcbfb17a219784c163dd1ec5f2fdb889b55ecaf13a7 | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_train | 2 | 3 | 740 | 208 | 532 | 14 | 14 | d062fa6e9df16728b02c8e1ce6b119946b475ca87f2adc3904493c3397c42b36 | Three folds retain >=20 of each validation class with intact barriers. |
| 1 | inner_validation | 2 | 3 | 358 | 99 | 259 | 17 | 17 | 54b89a61d2430622f0319e80169380f89fcd13a637bbbf0f14ba241f40e76456 | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | outer_train | -1 | 3 | 1098 | 305 | 793 | 33 | 33 | 1630102f8a1889ad1645fe82c0f5291dce525cbb7e51ff1438dd283255fabf03 | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | outer_test | -1 | 3 | 549 | 154 | 395 | 16 | 16 | a9a853d4d960e892c6d8bacd54fb8cc98bbd04f0e201d3abc741d8c592f2f012 | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_train | 0 | 3 | 738 | 205 | 533 | 21 | 21 | cecd647a954102c3f77f4047d082fedd7317c9cf6189575e9c04e0f11cdc0aff | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_validation | 0 | 3 | 360 | 100 | 260 | 12 | 12 | 8b1c448a0dc8c205b7efd8c93c0ae7fca5ad8d44d31be46d5b860bef3c7c0a96 | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_train | 1 | 3 | 726 | 198 | 528 | 25 | 25 | b572897088377717deef0ac7d6a3a308205aaff4cbb40703b08aef8483b6cd95 | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_validation | 1 | 3 | 372 | 107 | 265 | 8 | 8 | daf900a772589d4bc50872589dddb4e00178deee1ed316db53afb0146b6baddb | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_train | 2 | 3 | 732 | 207 | 525 | 20 | 20 | dfb03e57b03dca2a81ea1a5ab15412da9ed03b21b6525a988c89e8d798c9ecdc | Three folds retain >=20 of each validation class with intact barriers. |
| 2 | inner_validation | 2 | 3 | 366 | 98 | 268 | 13 | 13 | 67346eb2831dfadd8f607ac0fdfb9bf27aaaea5104f72512cb97126b5d92a882 | Three folds retain >=20 of each validation class with intact barriers. |

## Nested selection and leakage controls

Each outer-training partition gets its own publisher/component-aware inner StratifiedGroupKFold, shuffle=True, random_state=42. Three inner folds are preferred; two are permitted only if three fail class/barrier support, with the reason saved. The unchanged 24-point grid crosses unigrams/unigrams+bigrams, min_df=2/5, C=0.1/1/10, and class_weight=None/balanced. TF-IDF uses sublinear_tf=True, max_df=1.0, lowercase=True; LR uses L2/liblinear, max_iter=5000 and seed=42.

The baseline tie policy is reused exactly: candidates within 0.001 of maximum mean inner macro F1, then within 0.001 of the lowest candidate sample SD, then unigram, smaller C, unweighted, larger min_df, config ID. No outer outcome affects selection. Labels are used to stratify and validate fixed split support, as required; after partitions are fixed only outer-training labels enter fitting/selection. All three configurations lock before any outer prediction. Natural and masked models refit only outer-training rows, and predictions are frozen without labels before metrics are joined. Threshold is always 0.5.

| outer_fold | inner_folds | config_id | ngram_max | min_df | C | class_weight | macro_f1_mean | macro_f1_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | cfg_21 | 2 | 5 | 1.0000 | balanced | 0.6049 | 0.0191 |
| 1 | 3 | cfg_09 | 1 | 5 | 1.0000 | balanced | 0.5998 | 0.0438 |
| 2 | 3 | cfg_03 | 1 | 2 | 1.0000 | balanced | 0.6408 | 0.0723 |

## Outer and pooled performance

| condition | scope | outer_fold | N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | outer_fold | 0 | 549 | 0.6685 | 0.6078 | 0.6030 | 0.4162 | 0.4706 | 0.4417 | 0.7642 | 0.6772 | 0.4010 |
| natural | outer_fold | 1 | 549 | 0.7086 | 0.6462 | 0.6432 | 0.4753 | 0.5066 | 0.4904 | 0.7959 | 0.7134 | 0.5006 |
| natural | outer_fold | 2 | 549 | 0.7177 | 0.6057 | 0.6125 | 0.4954 | 0.3506 | 0.4106 | 0.8144 | 0.7078 | 0.4599 |
| natural | pooled | -1 | 1647 | 0.6982 | 0.6197 | 0.6209 | 0.4572 | 0.4423 | 0.4496 | 0.7921 | 0.6976 | 0.4467 |
| source_masked | outer_fold | 0 | 549 | 0.6794 | 0.6194 | 0.6147 | 0.4327 | 0.4837 | 0.4568 | 0.7726 | 0.6781 | 0.3975 |
| source_masked | outer_fold | 1 | 549 | 0.7013 | 0.6412 | 0.6370 | 0.4639 | 0.5066 | 0.4843 | 0.7897 | 0.7092 | 0.4982 |
| source_masked | outer_fold | 2 | 549 | 0.7086 | 0.6014 | 0.6071 | 0.4741 | 0.3571 | 0.4074 | 0.8068 | 0.7072 | 0.4554 |
| source_masked | pooled | -1 | 1647 | 0.6964 | 0.6204 | 0.6209 | 0.4547 | 0.4488 | 0.4518 | 0.7901 | 0.6966 | 0.4438 |

| condition | metric | outer_mean | outer_sample_sd | outer_min | outer_max | pooled |
| --- | --- | --- | --- | --- | --- | --- |
| natural | macro_f1 | 0.6196 | 0.0210 | 0.6030 | 0.6432 | 0.6209 |
| natural | balanced_accuracy | 0.6199 | 0.0228 | 0.6057 | 0.6462 | 0.6197 |
| natural | fake_f1 | 0.4476 | 0.0402 | 0.4106 | 0.4904 | 0.4496 |
| natural | roc_auc | 0.6995 | 0.0195 | 0.6772 | 0.7134 | 0.6976 |
| natural | average_precision | 0.4538 | 0.0500 | 0.4010 | 0.5006 | 0.4467 |
| source_masked | macro_f1 | 0.6196 | 0.0156 | 0.6071 | 0.6370 | 0.6209 |
| source_masked | balanced_accuracy | 0.6206 | 0.0200 | 0.6014 | 0.6412 | 0.6204 |
| source_masked | fake_f1 | 0.4495 | 0.0390 | 0.4074 | 0.4843 | 0.4518 |
| source_masked | roc_auc | 0.6982 | 0.0174 | 0.6781 | 0.7092 | 0.6966 |
| source_masked | average_precision | 0.4504 | 0.0505 | 0.3975 | 0.4982 | 0.4438 |

Means are unweighted across three outer folds; SD uses ddof=1. Min/max describe fold heterogeneity, not uncertainty intervals. Pooled metrics use each eligible row exactly once per condition. Pooled AUC/AP rank scores from different fitted models; fold-specific AUC/AP and their means should also be read because score scales can differ.

## Prediction bias and confusion counts

| condition | scope | outer_fold | actual_fake_prevalence | predicted_fake_prevalence | mean_fake_probability | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | outer_fold | 0 | 0.2787 | 0.3151 | 0.4517 | 295 | 101 | 81 | 72 |
| natural | outer_fold | 1 | 0.2769 | 0.2951 | 0.4426 | 312 | 85 | 75 | 77 |
| natural | outer_fold | 2 | 0.2805 | 0.1985 | 0.4213 | 340 | 55 | 100 | 54 |
| natural | pooled | -1 | 0.2787 | 0.2696 | 0.4385 | 947 | 241 | 256 | 203 |
| source_masked | outer_fold | 0 | 0.2787 | 0.3115 | 0.4495 | 299 | 97 | 79 | 74 |
| source_masked | outer_fold | 1 | 0.2769 | 0.3024 | 0.4458 | 308 | 89 | 75 | 77 |
| source_masked | outer_fold | 2 | 0.2805 | 0.2113 | 0.4233 | 334 | 61 | 99 | 55 |
| source_masked | pooled | -1 | 0.2787 | 0.2750 | 0.4395 | 941 | 247 | 253 | 206 |

TN/FP/FN/TP use Fake as the positive class. No recalibration, threshold optimization or output-based feature changes were made.

## Comparison with resource-grouped baseline

| metric | baseline_scope | publisher_aggregation | resource_grouped | publisher_heldout | publisher_minus_resource | paired_comparison |
| --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | resource_grouped_cv | outer_mean | 0.6598 | 0.6196 | -0.0402 | False |
| macro_f1 | resource_grouped_cv | pooled | 0.6598 | 0.6209 | -0.0389 | False |
| macro_f1 | resource_grouped_holdout | outer_mean | 0.6752 | 0.6196 | -0.0556 | False |
| macro_f1 | resource_grouped_holdout | pooled | 0.6752 | 0.6209 | -0.0543 | False |

| metric | baseline_scope | publisher_aggregation | resource_grouped | publisher_heldout | publisher_minus_resource | paired_comparison |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_accuracy | resource_grouped_cv | outer_mean | 0.6550 | 0.6199 | -0.0351 | False |
| balanced_accuracy | resource_grouped_cv | pooled | 0.6550 | 0.6197 | -0.0353 | False |
| balanced_accuracy | resource_grouped_holdout | outer_mean | 0.6677 | 0.6199 | -0.0478 | False |
| balanced_accuracy | resource_grouped_holdout | pooled | 0.6677 | 0.6197 | -0.0480 | False |
| fake_f1 | resource_grouped_cv | outer_mean | 0.4997 | 0.4476 | -0.0521 | False |
| fake_f1 | resource_grouped_cv | pooled | 0.4997 | 0.4496 | -0.0501 | False |
| fake_f1 | resource_grouped_holdout | outer_mean | 0.5176 | 0.4476 | -0.0700 | False |
| fake_f1 | resource_grouped_holdout | pooled | 0.5176 | 0.4496 | -0.0680 | False |
| roc_auc | resource_grouped_cv | outer_mean | 0.7309 | 0.6995 | -0.0314 | False |
| roc_auc | resource_grouped_cv | pooled | 0.7309 | 0.6976 | -0.0333 | False |
| roc_auc | resource_grouped_holdout | outer_mean | 0.7701 | 0.6995 | -0.0706 | False |
| roc_auc | resource_grouped_holdout | pooled | 0.7701 | 0.6976 | -0.0725 | False |
| average_precision | resource_grouped_cv | outer_mean | 0.5364 | 0.4538 | -0.0826 | False |
| average_precision | resource_grouped_cv | pooled | 0.5364 | 0.4467 | -0.0898 | False |
| average_precision | resource_grouped_holdout | outer_mean | 0.5707 | 0.4538 | -0.1168 | False |
| average_precision | resource_grouped_holdout | pooled | 0.5707 | 0.4467 | -0.1240 | False |

All comparator metrics are reused from validated natural-text baseline tables. This is an unpaired descriptive comparison: publisher outer CV covers the entire eligible cohort, whereas the original holdout covers its frozen subset, and grouping, training samples, tuning partitions and selected hyperparameters differ. The original selected-configuration CV estimate is not a nested outer-CV estimate. A degradation is consistent with publisher-dependent signal but does not identify a causal publisher effect.

## Source-masked sensitivity

Every masked model uses its natural counterpart’s locked hyperparameters on identical train/test rows. No masked retuning occurs. The prepared mask dictionary is unchanged and was constructed label-blind from metadata across the prepared datasets, including evaluation metadata; this is the previously declared transductive explicit-identifier sensitivity. It can remove legitimate institutional mentions and retain publisher style, topic, URL paths and SOURCE placeholders. Masked−natural changes do not estimate the entire publisher-confounding contribution.

| scope | outer_fold | metric | natural | source_masked | masked_minus_natural |
| --- | --- | --- | --- | --- | --- |
| outer_fold | 0 | macro_f1 | 0.6030 | 0.6147 | 0.0117 |
| outer_fold | 0 | balanced_accuracy | 0.6078 | 0.6194 | 0.0116 |
| outer_fold | 0 | fake_f1 | 0.4417 | 0.4568 | 0.0151 |
| outer_fold | 0 | roc_auc | 0.6772 | 0.6781 | 0.0009 |
| outer_fold | 0 | average_precision | 0.4010 | 0.3975 | -0.0035 |
| outer_fold | 1 | macro_f1 | 0.6432 | 0.6370 | -0.0062 |
| outer_fold | 1 | balanced_accuracy | 0.6462 | 0.6412 | -0.0050 |
| outer_fold | 1 | fake_f1 | 0.4904 | 0.4843 | -0.0062 |
| outer_fold | 1 | roc_auc | 0.7134 | 0.7092 | -0.0042 |
| outer_fold | 1 | average_precision | 0.5006 | 0.4982 | -0.0023 |
| outer_fold | 2 | macro_f1 | 0.6125 | 0.6071 | -0.0054 |
| outer_fold | 2 | balanced_accuracy | 0.6057 | 0.6014 | -0.0043 |
| outer_fold | 2 | fake_f1 | 0.4106 | 0.4074 | -0.0032 |
| outer_fold | 2 | roc_auc | 0.7078 | 0.7072 | -0.0006 |
| outer_fold | 2 | average_precision | 0.4599 | 0.4554 | -0.0045 |
| pooled | -1 | macro_f1 | 0.6209 | 0.6209 | 0.0000 |
| pooled | -1 | balanced_accuracy | 0.6197 | 0.6204 | 0.0007 |
| pooled | -1 | fake_f1 | 0.4496 | 0.4518 | 0.0021 |
| pooled | -1 | roc_auc | 0.6976 | 0.6966 | -0.0010 |
| pooled | -1 | average_precision | 0.4467 | 0.4438 | -0.0028 |
| outer_mean | -1 | macro_f1 | 0.6196 | 0.6196 | 0.0000 |
| outer_mean | -1 | balanced_accuracy | 0.6199 | 0.6206 | 0.0007 |
| outer_mean | -1 | fake_f1 | 0.4476 | 0.4495 | 0.0019 |
| outer_mean | -1 | roc_auc | 0.6995 | 0.6982 | -0.0013 |
| outer_mean | -1 | average_precision | 0.4538 | 0.4504 | -0.0035 |

## Publisher-level diagnostics and limitations

The diagnostic CSV contains 98 publisher/condition rows. Natural diagnostics flag 22 publishers with N<10 and 15 one-class publishers. Publisher macro F1 is reported only with both classes; Fake recall is undefined with no Fake items. Do not treat small publisher estimates as stable rankings.

All three folds retain above-0.5 balanced accuracy and ROC-AUC: True. Outer macro F1 ranges 0.6030–0.6432; mean macro F1 after omitting any single fold ranges 0.6077–0.6278. All outer folds contain 549–549 rows, so no fold dominates pooled sample size. The pooled analytic always-Real macro-F1 reference is 0.4190, compared with model macro F1 0.6209. These descriptive checks distinguish surviving predictive association from a result sustained by one favorable fold; they are not significance tests.

Surviving signal is an association under the audited unseen-source-family definition. It does not establish universal misinformation cues, unseen corporate-owner generalization, or a COVID/sensitive-period causal effect. Three outer folds provide limited evidence about variation across publishers; content selection and FakeHealth review-derived labels remain part of the dataset context.

CoAID source/class structure is too entangled for this experiment to be interpreted as a clean unseen-publisher linguistic-generalization test. No CoAID data were opened or CoAID models fitted.

## Validation and reproducibility

216 inner fits and 6 outer fits completed. All inner fits converged: True; all outer fits converged: True. Any inner convergence failures are logged and excluded by the unchanged selection policy. Final-fit failure would stop execution. Checkpoints avoid repeating completed fits or outer prediction calls.

Validation covers audited map/count identity, deterministic outer counts, publisher/resource/component separation, inner membership restricted to outer training, training-only TF-IDF, allowlisted predictors, selection locks, threshold 0.5, exactly one outer prediction per eligible row per condition, natural/masked parity, unique output keys and unchanged protected input fingerprints. Report JSON contains provenance and checksums.

Outputs: `outputs/splits/fakehealth_publisher_heldout_outer_folds.csv`; `outputs/tables/fakehealth_publisher_heldout_{inner_cv,model_selection,results,source_mask_sensitivity,summary,baseline_comparison,split_summary,fit_manifest}.csv`; `outputs/tables/fakehealth_publisher_level_diagnostics.csv`; `outputs/predictions/fakehealth_publisher_heldout_predictions.csv`; `outputs/figures/fakehealth_publisher_heldout/`. Models and locks: `outputs/models/fakehealth_publisher_heldout/`. Progress log: `logs/fakehealth_publisher_heldout.log`. Script: `src/run_fakehealth_publisher_heldout.py`.
