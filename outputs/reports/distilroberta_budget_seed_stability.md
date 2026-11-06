# FH-to-CoAID token-budget stability across training seeds

On the identical CoAID target holdout, macro F1 improves at 256 for all three seeds: +0.1816, +0.0979 and +0.0238, with mean paired change +0.1011 and sample SD 0.0790. However, ROC-AUC and AP improve only for seed 42; both decline for seeds 123 and 2026. The same sign pattern holds on all eligible targets. All seeds predict fewer Fake articles at 256, while still overpredicting Fake relative to observed prevalence. Thus the consistent thresholded macro-F1 improvement coexists with seed-sensitive ranking performance. The original large joint improvement does not support a robust substantive source-representation claim. The main transformer interpretation remains qualified by context, representation budget and training variability.

The original 256-token improvement is training-seed sensitive: paired effects reverse or disagree across seeds/metrics. It should not support a substantive claim that reducing the source representation consistently improves transfer.

This targeted check does not retune or replace the primary transformer experiment. Its main interpretation—no consistent cross-context benefit of the primary 512-token transformer—remains unchanged. Any token-budget claim must account for these paired-seed results. No other direction or model was fitted.

## Fixed protocol

Checkpoint `distilbert/distilroberta-base` revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Learning rate `2e-05`, epochs `3` read directly from the selection CSV. Seeds 42, 123 and 2026; caps 512 and 256. Seed 42 reuses both original frozen predictions and exact stored metrics. Only four new all-FakeHealth-source fits were run. No CV, within-context refit, target fitting or calibration.

{"betas": [0.9, 0.999], "checkpoint": "distilbert/distilroberta-base", "class_weight": "N_train/(2*N_train,c)", "decay_policy": "all parameters except bias and LayerNorm weights", "effective_batch_size": 16, "epsilon": 1e-08, "gradient_accumulation": 8, "gradient_clip_norm": 1.0, "microbatch": 2, "optimizer": "torch.optim.AdamW", "packing": "validated full-title pair; body prefix; dynamic padding", "precision": "float32", "revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "scheduler": "constant", "seed": [42, 123, 2026], "threshold": 0.5, "warmup_steps": 0, "weight_decay": 0.01}

The unchanged training helper supplies AdamW, constant LR, zero warmup, weighted CE normalized within each effective batch, gradient clipping and dynamic padding. Class weights use only each fit’s FakeHealth labels. All 1,647 source rows and 3,577 target rows are identical across budgets/seeds; the existing target holdout contains 702 rows. All six prediction files were verified frozen before this script read any CoAID labels.

## Target-content identity

{"holdout_uid_sha256": "ca96e0fc59899a1b35cbaafda638584850e74d87a6e8b3c344d882866c7496aa", "identical_unpadded_content": true, "maximum_unpadded_length": 190, "packed_256_sha256": "b1ee46afd34204048d7cc97f5b5452f1a0b93b43199fb54d22fa6b24ff812141", "packed_512_sha256": "b1ee46afd34204048d7cc97f5b5452f1a0b93b43199fb54d22fa6b24ff812141", "target_N": 3577, "target_uid_sha256": "d61e991983615e761ee171812fde1597b689c88dba8121705b9d3845e8e8c1ae"}

All unpadded CoAID token sequences are empirically identical at both caps. Only title and body enter packing. The source FakeHealth body prefix changes; full titles are preserved. No article-level text inspection was performed.

## Seed-level results

| seed | max_length | scope | macro_f1 | balanced_accuracy | fake_f1 | roc_auc | average_precision | actual_fake_prevalence | predicted_fake_prevalence | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 256 | all_eligible | 0.5120 | 0.5366 | 0.3351 | 0.5409 | 0.2302 | 0.2306 | 0.4068 | 1679 | 1073 | 443 | 382 |
| 42 | 256 | same_target_holdout | 0.5773 | 0.5957 | 0.3990 | 0.6460 | 0.2935 | 0.2365 | 0.3419 | 377 | 159 | 85 | 81 |
| 42 | 512 | all_eligible | 0.3226 | 0.3797 | 0.2445 | 0.3495 | 0.1688 | 0.2306 | 0.6542 | 799 | 1953 | 438 | 387 |
| 42 | 512 | same_target_holdout | 0.3956 | 0.4622 | 0.3113 | 0.4415 | 0.2009 | 0.2365 | 0.6239 | 192 | 344 | 72 | 94 |
| 123 | 256 | all_eligible | 0.3273 | 0.4670 | 0.3324 | 0.3597 | 0.1696 | 0.2306 | 0.7769 | 572 | 2180 | 226 | 599 |
| 123 | 256 | same_target_holdout | 0.3846 | 0.5284 | 0.3811 | 0.4375 | 0.1948 | 0.2365 | 0.7578 | 137 | 399 | 33 | 133 |
| 123 | 512 | all_eligible | 0.2543 | 0.4480 | 0.3314 | 0.3958 | 0.1952 | 0.2306 | 0.8728 | 284 | 2468 | 171 | 654 |
| 123 | 512 | same_target_holdout | 0.2867 | 0.4863 | 0.3629 | 0.4672 | 0.2240 | 0.2365 | 0.8704 | 66 | 470 | 25 | 141 |
| 2026 | 256 | all_eligible | 0.4656 | 0.4840 | 0.2819 | 0.4659 | 0.1989 | 0.2306 | 0.4258 | 1560 | 1192 | 494 | 331 |
| 2026 | 256 | same_target_holdout | 0.5339 | 0.5476 | 0.3406 | 0.5739 | 0.2483 | 0.2365 | 0.3490 | 361 | 175 | 96 | 70 |
| 2026 | 512 | all_eligible | 0.4389 | 0.4905 | 0.3167 | 0.4949 | 0.2358 | 0.2306 | 0.5516 | 1222 | 1530 | 382 | 443 |
| 2026 | 512 | same_target_holdout | 0.5101 | 0.5590 | 0.3782 | 0.6022 | 0.3094 | 0.2365 | 0.4943 | 286 | 250 | 69 | 97 |

## Primary paired differences: 256 minus 512

| seed | scope | metric | value_512 | value_256 | delta_256_minus_512 |
| --- | --- | --- | --- | --- | --- |
| 42 | all_eligible | macro_f1 | 0.3226 | 0.5120 | 0.1894 |
| 42 | all_eligible | balanced_accuracy | 0.3797 | 0.5366 | 0.1569 |
| 42 | all_eligible | fake_f1 | 0.2445 | 0.3351 | 0.0905 |
| 42 | all_eligible | roc_auc | 0.3495 | 0.5409 | 0.1914 |
| 42 | all_eligible | average_precision | 0.1688 | 0.2302 | 0.0614 |
| 42 | all_eligible | predicted_fake_prevalence | 0.6542 | 0.4068 | -0.2474 |
| 42 | same_target_holdout | macro_f1 | 0.3956 | 0.5773 | 0.1816 |
| 42 | same_target_holdout | balanced_accuracy | 0.4622 | 0.5957 | 0.1334 |
| 42 | same_target_holdout | fake_f1 | 0.3113 | 0.3990 | 0.0878 |
| 42 | same_target_holdout | roc_auc | 0.4415 | 0.6460 | 0.2045 |
| 42 | same_target_holdout | average_precision | 0.2009 | 0.2935 | 0.0926 |
| 42 | same_target_holdout | predicted_fake_prevalence | 0.6239 | 0.3419 | -0.2821 |
| 123 | all_eligible | macro_f1 | 0.2543 | 0.3273 | 0.0731 |
| 123 | all_eligible | balanced_accuracy | 0.4480 | 0.4670 | 0.0190 |
| 123 | all_eligible | fake_f1 | 0.3314 | 0.3324 | 0.0010 |
| 123 | all_eligible | roc_auc | 0.3958 | 0.3597 | -0.0360 |
| 123 | all_eligible | average_precision | 0.1952 | 0.1696 | -0.0256 |
| 123 | all_eligible | predicted_fake_prevalence | 0.8728 | 0.7769 | -0.0959 |
| 123 | same_target_holdout | macro_f1 | 0.2867 | 0.3846 | 0.0979 |
| 123 | same_target_holdout | balanced_accuracy | 0.4863 | 0.5284 | 0.0421 |
| 123 | same_target_holdout | fake_f1 | 0.3629 | 0.3811 | 0.0182 |
| 123 | same_target_holdout | roc_auc | 0.4672 | 0.4375 | -0.0297 |
| 123 | same_target_holdout | average_precision | 0.2240 | 0.1948 | -0.0293 |
| 123 | same_target_holdout | predicted_fake_prevalence | 0.8704 | 0.7578 | -0.1125 |
| 2026 | all_eligible | macro_f1 | 0.4389 | 0.4656 | 0.0267 |
| 2026 | all_eligible | balanced_accuracy | 0.4905 | 0.4840 | -0.0065 |
| 2026 | all_eligible | fake_f1 | 0.3167 | 0.2819 | -0.0347 |
| 2026 | all_eligible | roc_auc | 0.4949 | 0.4659 | -0.0290 |
| 2026 | all_eligible | average_precision | 0.2358 | 0.1989 | -0.0369 |
| 2026 | all_eligible | predicted_fake_prevalence | 0.5516 | 0.4258 | -0.1258 |
| 2026 | same_target_holdout | macro_f1 | 0.5101 | 0.5339 | 0.0238 |
| 2026 | same_target_holdout | balanced_accuracy | 0.5590 | 0.5476 | -0.0114 |
| 2026 | same_target_holdout | fake_f1 | 0.3782 | 0.3406 | -0.0375 |
| 2026 | same_target_holdout | roc_auc | 0.6022 | 0.5739 | -0.0283 |
| 2026 | same_target_holdout | average_precision | 0.3094 | 0.2483 | -0.0611 |
| 2026 | same_target_holdout | predicted_fake_prevalence | 0.4943 | 0.3490 | -0.1453 |

## Descriptive seed summaries

| kind | scope | max_length | metric | N_seeds | mean | sample_sd | minimum | maximum | positive_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget | all_eligible | 256.0000 | macro_f1 | 3 | 0.4350 | 0.0961 | 0.3273 | 0.5120 | — |
| budget | all_eligible | 256.0000 | roc_auc | 3 | 0.4555 | 0.0910 | 0.3597 | 0.5409 | — |
| budget | all_eligible | 256.0000 | average_precision | 3 | 0.1996 | 0.0303 | 0.1696 | 0.2302 | — |
| budget | all_eligible | 256.0000 | fake_f1 | 3 | 0.3165 | 0.0299 | 0.2819 | 0.3351 | — |
| budget | all_eligible | 512.0000 | macro_f1 | 3 | 0.3386 | 0.0933 | 0.2543 | 0.4389 | — |
| budget | all_eligible | 512.0000 | roc_auc | 3 | 0.4134 | 0.0743 | 0.3495 | 0.4949 | — |
| budget | all_eligible | 512.0000 | average_precision | 3 | 0.1999 | 0.0338 | 0.1688 | 0.2358 | — |
| budget | all_eligible | 512.0000 | fake_f1 | 3 | 0.2975 | 0.0465 | 0.2445 | 0.3314 | — |
| budget | same_target_holdout | 256.0000 | macro_f1 | 3 | 0.4986 | 0.1011 | 0.3846 | 0.5773 | — |
| budget | same_target_holdout | 256.0000 | roc_auc | 3 | 0.5525 | 0.1059 | 0.4375 | 0.6460 | — |
| budget | same_target_holdout | 256.0000 | average_precision | 3 | 0.2455 | 0.0494 | 0.1948 | 0.2935 | — |
| budget | same_target_holdout | 256.0000 | fake_f1 | 3 | 0.3736 | 0.0299 | 0.3406 | 0.3990 | — |
| budget | same_target_holdout | 512.0000 | macro_f1 | 3 | 0.3975 | 0.1117 | 0.2867 | 0.5101 | — |
| budget | same_target_holdout | 512.0000 | roc_auc | 3 | 0.5036 | 0.0863 | 0.4415 | 0.6022 | — |
| budget | same_target_holdout | 512.0000 | average_precision | 3 | 0.2448 | 0.0572 | 0.2009 | 0.3094 | — |
| budget | same_target_holdout | 512.0000 | fake_f1 | 3 | 0.3508 | 0.0351 | 0.3113 | 0.3782 | — |
| paired_delta | all_eligible | — | average_precision | 3 | -0.0003 | 0.0538 | -0.0369 | 0.0614 | 1.0000 |
| paired_delta | all_eligible | — | balanced_accuracy | 3 | 0.0565 | 0.0879 | -0.0065 | 0.1569 | 2.0000 |
| paired_delta | all_eligible | — | fake_f1 | 3 | 0.0189 | 0.0645 | -0.0347 | 0.0905 | 2.0000 |
| paired_delta | all_eligible | — | macro_f1 | 3 | 0.0964 | 0.0838 | 0.0267 | 0.1894 | 3.0000 |
| paired_delta | all_eligible | — | predicted_fake_prevalence | 3 | -0.1564 | 0.0803 | -0.2474 | -0.0959 | 0.0000 |
| paired_delta | all_eligible | — | roc_auc | 3 | 0.0421 | 0.1293 | -0.0360 | 0.1914 | 1.0000 |
| paired_delta | same_target_holdout | — | average_precision | 3 | 0.0008 | 0.0812 | -0.0611 | 0.0926 | 1.0000 |
| paired_delta | same_target_holdout | — | balanced_accuracy | 3 | 0.0547 | 0.0732 | -0.0114 | 0.1334 | 2.0000 |
| paired_delta | same_target_holdout | — | fake_f1 | 3 | 0.0228 | 0.0628 | -0.0375 | 0.0878 | 2.0000 |
| paired_delta | same_target_holdout | — | macro_f1 | 3 | 0.1011 | 0.0790 | 0.0238 | 0.1816 | 3.0000 |
| paired_delta | same_target_holdout | — | predicted_fake_prevalence | 3 | -0.1800 | 0.0899 | -0.2821 | -0.1125 | 0.0000 |
| paired_delta | same_target_holdout | — | roc_auc | 3 | 0.0489 | 0.1348 | -0.0297 | 0.2045 | 1.0000 |

Sample SD uses ddof=1. Three seeds do not support reliable population uncertainty: no significance test or confidence interval was constructed.

## Prediction agreement

| seed | scope | identical_class_pct | probability_correlation | mean_absolute_probability_difference |
| --- | --- | --- | --- | --- |
| 42 | all_eligible | 68.5491 | 0.6923 | 0.2132 |
| 42 | same_target_holdout | 66.3818 | 0.6461 | 0.2265 |
| 123 | all_eligible | 87.8390 | 0.8688 | 0.0959 |
| 123 | same_target_holdout | 86.4672 | 0.8593 | 0.1002 |
| 2026 | all_eligible | 80.8778 | 0.8171 | 0.1358 |
| 2026 | same_target_holdout | 78.0627 | 0.7762 | 0.1429 |

## Validation and limitations

{"classical_environment_unchanged": true, "cohorts_and_splits_unchanged": true, "freezes_verified_before_target_labels": true, "locked_protocol_unchanged": true, "new_epoch_checkpoints": 12, "new_fits": 4, "passed": true, "prediction_rows": 21462, "prior_outputs_unchanged": true, "reused_seed42_fits": 2, "seed42_metrics_reused_and_verified": true, "source_only_fitting": true, "target_content_identical": true, "threshold": 0.5, "unique_fit_and_prediction_fingerprints": true}

Seeds control initialization, dropout and epoch shuffling using the same helper as the original experiment. MPS is not asserted bitwise reproducible. Source-budget and stochastic optimization interact; this is not a causal COVID effect, a universal cue claim or a matched-information correction. The original 71.22%/87.86% FakeHealth truncation at 512/256 versus 0% for CoAID remains relevant.

Completed epoch timer sum: 1194.0 seconds. Elapsed wall time: 1376.2 seconds. FP32 and numerical hyperparameters remain unchanged. The existing MPS cache-release resource policy is reused, with a fresh process after each completed new fit/prediction. Completed epoch checkpoints resume without retraining completed fits.

Outputs: `outputs/tables/distilroberta_budget_seed_*.csv`, `outputs/predictions/distilroberta_budget_seed_predictions.csv`, `outputs/models/distilroberta_budget_seed_stability/`, and `logs/distilroberta_budget_seed_stability.log`.

The paired figure is saved at `outputs/figures/distilroberta/fh_to_coaid_budget_seed_paired_deltas.svg`. It was rendered from aggregate results with existing Matplotlib in the classical environment, read-only, with bytecode disabled and temporary font caches. A full environment file-size/mtime snapshot was unchanged. All fitting and experiment validation used `.venv-transformer/bin/python`.

Final automated tests: 7 run, 0 failures, 0 errors, 0 skips. Figure visually inspected; output hashes verified. Exactly four new fits and two reused seed-42 fits; 21,462 prediction rows.
