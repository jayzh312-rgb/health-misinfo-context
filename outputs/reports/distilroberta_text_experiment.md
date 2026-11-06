# DistilRoBERTa within-context and cross-context experiment

Primary 512-token transformer transfer macro F1 exceeds both frozen linear comparators in 1 of 2 directions. This is descriptive, single-seed evidence; it does not establish statistical superiority or isolate contextual pretraining from token-budget effects. Read ROC-AUC, AP, Fake F1 and prevalence alongside macro F1: changes in threshold behavior can differ from ranking quality.

ROC-AUC and AP are lower than both linear comparators in both transfer directions. Together with the absolute transfer scores, this provides no consistent evidence that contextual pretraining materially reduced cross-context failure. The modest CA-to-FH macro-F1 gain coexists with worse ranking quality; its smaller macro-F1 degradation is also partly a consequence of lower FakeHealth within-context macro F1. The original directional prevalence pattern persists: FH-to-CA overpredicts Fake and CA-to-FH underpredicts Fake.

FH_to_CA, identical target holdout: transformer macro F1 0.3956, LR 0.4290, SVM 0.4245. Transformer transfer-minus-within -0.5431; LR -0.4970; SVM -0.4883. Fake predicted prevalence 62.39% versus actual 23.65%.
CA_to_FH, identical target holdout: transformer macro F1 0.4638, LR 0.4507, SVM 0.4493. Transformer transfer-minus-within -0.1575; LR -0.2244; SVM -0.2205. Fake predicted prevalence 9.70% versus actual 27.88%.

## Protocol and memory gate

Pinned checkpoint `distilbert/distilroberta-base` at `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Phase A passed; microbatch 2, accumulation 8, effective batch 16. Both pilot tests used synthetic labels and weighted cross-entropy; parameter hashes were unchanged and no optimizer step occurred. Pilot models were released. Reported MPS memory is sampled after backward, not a transient peak and excludes optimizer state.

| microbatch | forward_succeeded | backward_succeeded | parameters_unchanged | sampled_mps_allocated_bytes | sampled_mps_driver_bytes | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | True | True | True | 663296512 | 1084915712 | 4.3639 |
| 2 | True | True | True | 678734848 | 2163965952 | 3.0877 |

FP32 MPS throughout; no mixed precision, oversampling, early stopping or threshold tuning. {"betas": [0.9, 0.999], "checkpoint": "distilbert/distilroberta-base", "class_weight": "N_train/(2*N_train,c)", "decay_policy": "all parameters except bias and LayerNorm weights", "effective_batch_size": 16, "epsilon": 1e-08, "gradient_clip_norm": 1.0, "optimizer": "torch.optim.AdamW", "packing": "validated full-title pair; body prefix; dynamic padding", "precision": "float32", "revision": "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b", "scheduler": "constant", "seed": 42, "threshold": 0.5, "warmup_steps": 0, "weight_decay": 0.01}
A constant learning-rate scheduler with zero warmup makes the epoch-2 checkpoint schedule identical to a two-epoch refit. AdamW uses decay 0.01 except biases/LayerNorm weights. Gradient norm is clipped to 1.0. Weighted CE gradients are normalized by the sum of source-training class weights within each effective batch, including a final partial batch. Class weights are recalculated from that fit’s training labels only.

## Cohorts, predictors and selection

FakeHealth HealthStory N=1,647; CoAID Specification A N=3,577. Fake=1, Real=0. Existing five grouped folds and reserved holdouts are unchanged. Only title_text and body_text enter the encoder. Full title is preserved, four official special tokens are reserved, and a body prefix fills the remaining 512-token budget. Dynamic padding is confined to each microbatch. No source, ID, engagement, rating or split metadata is a predictor.

Each learning rate was fitted once for three epochs in each fold, with validation at epochs 2 and 3. The 0.001 mean-macro-F1 tie set is ordered by lower sample SD, fewer epochs, then lower learning rate. Both configurations were locked before holdout evaluation and transfer.

| dataset | learning_rate | epochs | macro_f1_mean | macro_f1_sd |
| --- | --- | --- | --- | --- |
| FakeHealth | 0.0000 | 3 | 0.6458 | 0.0395 |
| CoAID | 0.0000 | 3 | 0.9560 | 0.0081 |

All logical CV candidates (five existing grouped folds each):

| dataset | learning_rate | epoch | macro_f1_mean | macro_f1_sample_sd |
| --- | --- | --- | --- | --- |
| CoAID | 0.0000 | 2 | 0.9478 | 0.0052 |
| CoAID | 0.0000 | 3 | 0.9560 | 0.0081 |
| CoAID | 0.0001 | 2 | 0.9386 | 0.0233 |
| CoAID | 0.0001 | 3 | 0.9445 | 0.0091 |
| FakeHealth | 0.0000 | 2 | 0.6137 | 0.0221 |
| FakeHealth | 0.0000 | 3 | 0.6458 | 0.0395 |
| FakeHealth | 0.0001 | 2 | 0.5816 | 0.0433 |
| FakeHealth | 0.0001 | 3 | 0.6376 | 0.0173 |

## Primary within-context holdout

| target_dataset | N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_f1 | roc_auc | average_precision | actual_fake_prevalence | predicted_fake_prevalence | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | 330 | 0.6333 | 0.6891 | 0.6212 | 0.4190 | 0.8152 | 0.5535 | 0.6889 | 0.7649 | 0.5916 | 0.2788 | 0.5424 | 134 | 104 | 17 | 75 |
| CoAID | 702 | 0.9573 | 0.9242 | 0.9387 | 0.9533 | 0.8614 | 0.9051 | 0.9724 | 0.9891 | 0.9685 | 0.2365 | 0.2137 | 529 | 7 | 23 | 143 |

## Primary transfer

Each direction starts from the same pretrained checkpoint and refits the locked source configuration on all eligible source rows, including its former holdout. Both unlabeled transfer prediction files are frozen before either transfer scoring join. A dataset’s labels serve training only in its source role; reverse-direction results cannot alter the locked model or its predictions. All eligible targets and the exact classical-model target holdouts are scored. Fake probability >=0.5 is fixed. No target tuning or calibration occurs.

| direction | scope | N | macro_f1 | balanced_accuracy | fake_f1 | roc_auc | average_precision | actual_fake_prevalence | predicted_fake_prevalence | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FH_to_CA | all_eligible | 3577 | 0.3226 | 0.3797 | 0.2445 | 0.3495 | 0.1688 | 0.2306 | 0.6542 | 799 | 1953 | 438 | 387 |
| FH_to_CA | same_target_holdout | 702 | 0.3956 | 0.4622 | 0.3113 | 0.4415 | 0.2009 | 0.2365 | 0.6239 | 192 | 344 | 72 | 94 |
| CA_to_FH | all_eligible | 1647 | 0.4591 | 0.4831 | 0.1339 | 0.4662 | 0.2717 | 0.2787 | 0.1202 | 1034 | 154 | 415 | 44 |
| CA_to_FH | same_target_holdout | 330 | 0.4638 | 0.4931 | 0.1290 | 0.4371 | 0.2611 | 0.2788 | 0.0970 | 214 | 24 | 84 | 8 |

## Transfer degradation and frozen linear comparators

| direction | metric | within_target | cross_context | transfer_minus_within |
| --- | --- | --- | --- | --- |
| FH_to_CA | macro_f1 | 0.9387 | 0.3956 | -0.5431 |
| FH_to_CA | balanced_accuracy | 0.9242 | 0.4622 | -0.4620 |
| FH_to_CA | fake_f1 | 0.9051 | 0.3113 | -0.5938 |
| FH_to_CA | roc_auc | 0.9891 | 0.4415 | -0.5476 |
| FH_to_CA | average_precision | 0.9685 | 0.2009 | -0.7676 |
| CA_to_FH | macro_f1 | 0.6212 | 0.4638 | -0.1575 |
| CA_to_FH | balanced_accuracy | 0.6891 | 0.4931 | -0.1961 |
| CA_to_FH | fake_f1 | 0.5535 | 0.1290 | -0.4245 |
| CA_to_FH | roc_auc | 0.7649 | 0.4371 | -0.3278 |
| CA_to_FH | average_precision | 0.5916 | 0.2611 | -0.3305 |

| evaluation | direction | scope | metric | logistic | linear_svm | distilroberta |
| --- | --- | --- | --- | --- | --- | --- |
| within | within_fakehealth | same_target_holdout | macro_f1 | 0.6752 | 0.6697 | 0.6212 |
| within | within_fakehealth | same_target_holdout | balanced_accuracy | 0.6677 | 0.6635 | 0.6891 |
| within | within_fakehealth | same_target_holdout | fake_f1 | 0.5176 | 0.5116 | 0.5535 |
| within | within_fakehealth | same_target_holdout | roc_auc | 0.7701 | 0.7695 | 0.7649 |
| within | within_fakehealth | same_target_holdout | average_precision | 0.5707 | 0.5707 | 0.5916 |
| within | within_coaid | same_target_holdout | macro_f1 | 0.9260 | 0.9128 | 0.9387 |
| within | within_coaid | same_target_holdout | balanced_accuracy | 0.9313 | 0.9301 | 0.9242 |
| within | within_coaid | same_target_holdout | fake_f1 | 0.8876 | 0.8693 | 0.9051 |
| within | within_coaid | same_target_holdout | roc_auc | 0.9827 | 0.9795 | 0.9891 |
| within | within_coaid | same_target_holdout | average_precision | 0.9280 | 0.9110 | 0.9685 |
| transfer | FH_to_CA | all_eligible | macro_f1 | 0.3757 | 0.3738 | 0.3226 |
| transfer | FH_to_CA | all_eligible | balanced_accuracy | 0.5217 | 0.5493 | 0.3797 |
| transfer | FH_to_CA | all_eligible | fake_f1 | 0.3694 | 0.3918 | 0.2445 |
| transfer | FH_to_CA | all_eligible | roc_auc | 0.5367 | 0.5471 | 0.3495 |
| transfer | FH_to_CA | all_eligible | average_precision | 0.2569 | 0.2626 | 0.1688 |
| transfer | FH_to_CA | same_target_holdout | macro_f1 | 0.4290 | 0.4245 | 0.3956 |
| transfer | FH_to_CA | same_target_holdout | balanced_accuracy | 0.5624 | 0.5899 | 0.4622 |
| transfer | FH_to_CA | same_target_holdout | fake_f1 | 0.4030 | 0.4261 | 0.3113 |
| transfer | FH_to_CA | same_target_holdout | roc_auc | 0.5705 | 0.5827 | 0.4415 |
| transfer | FH_to_CA | same_target_holdout | average_precision | 0.2677 | 0.2738 | 0.2009 |
| transfer | CA_to_FH | all_eligible | macro_f1 | 0.4429 | 0.4466 | 0.4591 |
| transfer | CA_to_FH | all_eligible | balanced_accuracy | 0.4878 | 0.4885 | 0.4831 |
| transfer | CA_to_FH | all_eligible | fake_f1 | 0.0776 | 0.0868 | 0.1339 |
| transfer | CA_to_FH | all_eligible | roc_auc | 0.6047 | 0.6113 | 0.4662 |
| transfer | CA_to_FH | all_eligible | average_precision | 0.3246 | 0.3288 | 0.2717 |
| transfer | CA_to_FH | same_target_holdout | macro_f1 | 0.4507 | 0.4493 | 0.4638 |
| transfer | CA_to_FH | same_target_holdout | balanced_accuracy | 0.5028 | 0.5007 | 0.4931 |
| transfer | CA_to_FH | same_target_holdout | fake_f1 | 0.0762 | 0.0755 | 0.1290 |
| transfer | CA_to_FH | same_target_holdout | roc_auc | 0.6191 | 0.6313 | 0.4371 |
| transfer | CA_to_FH | same_target_holdout | average_precision | 0.3409 | 0.3515 | 0.2611 |

Transfer minus within-target performance, for each model on identical target holdouts (negative means degradation):

| direction | metric | logistic | linear_svm | distilroberta |
| --- | --- | --- | --- | --- |
| FH_to_CA | macro_f1 | -0.4970 | -0.4883 | -0.5431 |
| FH_to_CA | balanced_accuracy | -0.3689 | -0.3402 | -0.4620 |
| FH_to_CA | fake_f1 | -0.4846 | -0.4432 | -0.5938 |
| FH_to_CA | roc_auc | -0.4121 | -0.3967 | -0.5476 |
| FH_to_CA | average_precision | -0.6603 | -0.6372 | -0.7676 |
| CA_to_FH | macro_f1 | -0.2244 | -0.2205 | -0.1575 |
| CA_to_FH | balanced_accuracy | -0.1649 | -0.1628 | -0.1961 |
| CA_to_FH | fake_f1 | -0.4415 | -0.4362 | -0.4245 |
| CA_to_FH | roc_auc | -0.1510 | -0.1383 | -0.3278 |
| CA_to_FH | average_precision | -0.2297 | -0.2192 | -0.3305 |

## 256-token sensitivity

FH_to_CA, identical target holdout at 256: macro F1 0.5773 (+0.1816 versus 512), ROC-AUC 0.6460, AP 0.2935.
CA_to_FH, identical target holdout at 256: macro F1 0.4556 (-0.0082 versus 512), ROC-AUC 0.4215, AP 0.2583.

Started after primary 512 fits and evaluation completed. Each source reuses its 512-selected learning rate and epoch count; identical rows/splits, weighted loss and packing. No retuning. This tests available body-prefix dependence; it does not equalize retained information.

CoAID inputs fit below both caps, so its within-context refits retain identical text. Any small CoAID within-context difference is therefore refit/numerical variability, not a truncation effect. Seeding MPS does not guarantee bitwise-identical fits. With one refit per cap, sensitivity differences cannot fully separate text-budget dependence from refit variability. In CA-to-FH transfer the target FakeHealth prefix changes; in FH-to-CA transfer the source FakeHealth prefix changes.

| evaluation | direction | scope | metric | primary_512 | sensitivity_256 | delta_256_minus_512 |
| --- | --- | --- | --- | --- | --- | --- |
| within | within_fakehealth | same_target_holdout | macro_f1 | 0.6212 | 0.6189 | -0.0023 |
| within | within_fakehealth | same_target_holdout | balanced_accuracy | 0.6891 | 0.6621 | -0.0270 |
| within | within_fakehealth | same_target_holdout | fake_f1 | 0.5535 | 0.5242 | -0.0293 |
| within | within_fakehealth | same_target_holdout | roc_auc | 0.7649 | 0.7164 | -0.0485 |
| within | within_fakehealth | same_target_holdout | average_precision | 0.5916 | 0.5108 | -0.0808 |
| within | within_coaid | same_target_holdout | macro_f1 | 0.9387 | 0.9368 | -0.0019 |
| within | within_coaid | same_target_holdout | balanced_accuracy | 0.9242 | 0.9233 | -0.0009 |
| within | within_coaid | same_target_holdout | fake_f1 | 0.9051 | 0.9022 | -0.0029 |
| within | within_coaid | same_target_holdout | roc_auc | 0.9891 | 0.9892 | 0.0001 |
| within | within_coaid | same_target_holdout | average_precision | 0.9685 | 0.9686 | 0.0002 |
| transfer | FH_to_CA | all_eligible | macro_f1 | 0.3226 | 0.5120 | 0.1894 |
| transfer | FH_to_CA | all_eligible | balanced_accuracy | 0.3797 | 0.5366 | 0.1569 |
| transfer | FH_to_CA | all_eligible | fake_f1 | 0.2445 | 0.3351 | 0.0905 |
| transfer | FH_to_CA | all_eligible | roc_auc | 0.3495 | 0.5409 | 0.1914 |
| transfer | FH_to_CA | all_eligible | average_precision | 0.1688 | 0.2302 | 0.0614 |
| transfer | FH_to_CA | same_target_holdout | macro_f1 | 0.3956 | 0.5773 | 0.1816 |
| transfer | FH_to_CA | same_target_holdout | balanced_accuracy | 0.4622 | 0.5957 | 0.1334 |
| transfer | FH_to_CA | same_target_holdout | fake_f1 | 0.3113 | 0.3990 | 0.0878 |
| transfer | FH_to_CA | same_target_holdout | roc_auc | 0.4415 | 0.6460 | 0.2045 |
| transfer | FH_to_CA | same_target_holdout | average_precision | 0.2009 | 0.2935 | 0.0926 |
| transfer | CA_to_FH | all_eligible | macro_f1 | 0.4591 | 0.4553 | -0.0037 |
| transfer | CA_to_FH | all_eligible | balanced_accuracy | 0.4831 | 0.4811 | -0.0020 |
| transfer | CA_to_FH | all_eligible | fake_f1 | 0.1339 | 0.1260 | -0.0080 |
| transfer | CA_to_FH | all_eligible | roc_auc | 0.4662 | 0.4622 | -0.0040 |
| transfer | CA_to_FH | all_eligible | average_precision | 0.2717 | 0.2667 | -0.0050 |
| transfer | CA_to_FH | same_target_holdout | macro_f1 | 0.4638 | 0.4556 | -0.0082 |
| transfer | CA_to_FH | same_target_holdout | balanced_accuracy | 0.4931 | 0.4826 | -0.0105 |
| transfer | CA_to_FH | same_target_holdout | fake_f1 | 0.1290 | 0.1240 | -0.0050 |
| transfer | CA_to_FH | same_target_holdout | roc_auc | 0.4371 | 0.4215 | -0.0155 |
| transfer | CA_to_FH | same_target_holdout | average_precision | 0.2611 | 0.2583 | -0.0028 |

## Interpretation and limitations

At 512, 71.22% of FakeHealth documents truncate, versus 0% of CoAID. At 256 the corresponding proportions are 87.86% and 0%. Comparison with full-text TF-IDF combines a model-family change with a representation-budget change. Poor transfer cannot be attributed to truncation alone: source, topic, collection and label-definition differences remain plausible contributors. Within-dataset performance and cross-context generalization answer different questions. These findings imply no causal COVID effect, universal misinformation cues or independence of the transformer from the data-generating context. Minor within-domain ranks are not the scientific focus.

A smaller transfer-minus-within deficit can also arise from weaker within-target performance. It should therefore be interpreted together with absolute transfer macro F1, Fake F1, ROC-AUC and AP, rather than treated alone as evidence that contextual pretraining solved transfer failure.

## Runtime, checkpoints and validation

Elapsed execution 2362.4 minutes; sum of epoch training/validation times 183.0 minutes. 84 completed epoch checkpoints. Each includes model, optimizer and CPU/MPS RNG state. The manifest records unique fingerprints, input/split hashes, hyperparameters, weighted losses, clipping diagnostics and metrics. Completed fits are reused; interrupted fits resume from the last completed epoch. Finite losses/gradients were enforced. Three epochs is a fixed budget, not a claim of optimizer convergence. After the first 15 CV epochs, host memory pressure (about 4.8 GiB swap) slowed epochs from roughly 75 seconds to as much as 563.9 seconds. Checkpoint resumes and worker recycling after completed fits were introduced. MPS allocator low/high watermarks were then set to 0.3/0.8; these are resource controls, with FP32, microbatch, optimizer and learning settings unchanged. An unfinished epoch may replay after interruption; After 19 saved CV epochs, unused MPS allocator cache was also released after each optimizer update. The first cleanup in each fit verifies that parameter version counters and MPS RNG state are unchanged. completed checkpoints are reused. See the [PyTorch allocator documentation](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html) and the progress log. Wall-clock runtime includes long overnight gaps between checkpoints despite much shorter epoch timer values. This suggests host sleep/suspension; it was not directly measured as a sleep event. A temporary caffeinate idle-sleep assertion was tied to the worker PID on September 22. Normal scheduling priority was restored without changing training calculations. Original `.venv` metadata, frozen cohorts/splits and all prior output file sizes/mtimes are unchanged. Reproducibility on MPS is seeded but bitwise reproducibility across devices/software is not asserted.

Automated validation results are recorded in the companion JSON and appended below.

Resume with the same fingerprints (completed fits are reused):

```bash
PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 DISTILROBERTA_RECYCLE_AFTER_FIT=1 DISTILROBERTA_RELEASE_CACHE_EACH_STEP=1 .venv-transformer/bin/python src/run_distilroberta_text_experiment.py --phase all
```

Outputs: `outputs/tables/distilroberta_*.csv`, `outputs/predictions/distilroberta_predictions.csv`, `outputs/models/distilroberta/`, `outputs/figures/distilroberta/`, `logs/distilroberta_text_experiment.log`.

Final tests: 12 run, 0 failures, 0 errors, 0 skips.

Figure rendering: the four SVG figures were rendered from saved aggregate result tables with `src/plot_distilroberta_results.py`, using the existing classical environment’s Matplotlib read-only. Bytecode writes were disabled and Matplotlib caches directed to the temporary directory. A before/after snapshot confirmed that all classical environment file sizes and modification times were unchanged. All model fitting and experiment tests used `.venv-transformer/bin/python`.
