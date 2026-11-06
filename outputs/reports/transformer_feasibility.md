# Transformer feasibility and tokenization audit

Status: **feasibility complete**. Tokenization and title-packing measurements are complete; inference pilot: completed. No training occurred.

## Environment and pinned checkpoint

Python 3.12.7; macOS 26.6.2; architecture arm64; torch 2.14.0; transformers 5.17.0. MPS built=True; available=True; selected device=MPS. Physical memory=17179869184 bytes (physical installed memory; not free memory). CPU and both imports explicitly tested. CUDA was not probed.

The four requested packages and their transitive dependencies were installed only into `.venv-transformer`, using stable PyPI resolution. All resolved packages are pinned in `requirements-transformer.txt`. The original `.venv` was neither installed into nor rewritten. Setup and detailed diagnostics are in `logs/transformer_environment_setup.log`.

Checkpoint [distilbert/distilroberta-base](https://huggingface.co/distilbert/distilroberta-base/tree/fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b), immutable revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b`. Official fast tokenizer instantiated with model_max_length=512; vocabulary=50265; special-token overhead: single=2, pair=4. Only this checkpoint was downloaded. Audit replay uses local_files_only=True and the exact revision. The model has 514 position embeddings, but the permitted total sequence length is 512.

Compatibility notes: Transformers 5 removed the former pair-building API; packing now derives the official special-token template and is checked against official pair tokenization. The base-encoder pilot intentionally omits the masked-LM head and pooler, so their unused checkpoint weights are expected load-report warnings; no missing encoder weights were reported. Sandbox-only MPS=False was resolved by the approved hardware-access run. `pip check` passed.

## Untruncated combined-text token lengths

All primary_modeling_eligible rows were used. Counts include the tokenizer’s single-input special tokens, with no truncation or padding. Every document was tokenized twice and the full token sequences compared in memory. No article text, token sequences, tensors, or model outputs were saved to audit files. Quantiles use linear interpolation.

| dataset | N | median | p05 | p25 | p75 | p95 | p99 | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | 1647 | 760.0000 | 21.3000 | 467.5000 | 1050.5000 | 1695.8000 | 2587.9600 | 5415 |
| CoAID | 3577 | 113.0000 | 18.0000 | 97.0000 | 120.0000 | 131.0000 | 139.0000 | 190 |

## Truncation and retained-token fractions

For untruncated combined-text length L and cap B, truncation is L>B; retained fraction is min(L,B)/L. This requested metric includes special tokens and counts tokens, not semantic information.

| dataset | max_length | percent_truncated | median_retained | p10_retained | mean_retained |
| --- | --- | --- | --- | --- | --- |
| FakeHealth | 128 | 91.4997 | 0.1684 | 0.0888 | 0.2664 |
| FakeHealth | 256 | 87.8567 | 0.3368 | 0.1776 | 0.4371 |
| FakeHealth | 384 | 80.2064 | 0.5053 | 0.2663 | 0.5795 |
| FakeHealth | 512 | 71.2204 | 0.6737 | 0.3551 | 0.6950 |
| CoAID | 128 | 7.5762 | 1.0000 | 1.0000 | 0.9966 |
| CoAID | 256 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| CoAID | 384 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| CoAID | 512 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

At 512, FakeHealth truncation is 71.2204% versus 0.0000% in CoAID, a 71.2204 percentage-point difference. The retained-fraction summaries quantify the asymmetry; a common cap does not make retained information equal.

## Full-title packing audit

Naive combined-text head truncation was checked using exact title character spans and official fast-tokenizer offsets. Any title-overlapping token beyond B minus the single-input special-token overhead counts as title loss. This measures actual combined-text head truncation, not paired longest-first truncation. An absent nonempty title is an error; empty titles require no preservation.

| dataset | title_maximum | empty_titles_N | titles_at_combined_prefix_N |
| --- | --- | --- | --- |
| FakeHealth | 27 | 0 | 1647 |
| CoAID | 148 | 6 | 3571 |

| dataset | max_length | naive_combined_title_loss_N | naive_combined_title_loss_percent | titles_over_usable_pair_budget_N | packed_full_title_verified_N | pair_percent_truncated |
| --- | --- | --- | --- | --- | --- | --- |
| FakeHealth | 128 | 0 | 0.0000 | 0 | 1647 | 91.4997 |
| FakeHealth | 256 | 0 | 0.0000 | 0 | 1647 | 87.8567 |
| FakeHealth | 384 | 0 | 0.0000 | 0 | 1647 | 80.2064 |
| FakeHealth | 512 | 0 | 0.0000 | 0 | 1647 | 71.2204 |
| CoAID | 128 | 1 | 0.0280 | 1 | 3576 | 7.5762 |
| CoAID | 256 | 0 | 0.0000 | 0 | 3577 | 0.0000 |
| CoAID | 384 | 0 | 0.0000 | 0 | 3577 | 0.0000 |
| CoAID | 512 | 0 | 0.0000 | 0 | 3577 | 0.0000 |

At 512, 0 documents lose title tokens under naive combined-text truncation. Across the 512 and 256 candidates, 0 title-over-budget cases require a later explicit policy. Thus special title preservation is not empirically necessary to avoid title loss at 256 or 512 on these cohorts; the explicit guard remains useful. At 128, one CoAID title cannot fit the paired budget and is flagged. The deterministic pair rule tokenizes the complete title, reserves four official special tokens, and assigns the remaining budget to a body prefix: `<s> title </s></s> body-prefix </s>`. Every feasible packed input was checked for exact title-token preservation and total length. Infeasible titles are flagged; they are never silently shortened or dropped. Pair truncation percentages are separate because pair boundaries and overhead differ from combined_text.

## Max-length recommendation

Recommend **512 primary** and **256 sensitivity**. The longer cap preserves more body text in the longer-document cohort; 256 tests dependence on the available body prefix and lowers sequence computation. The shorter cap is not an information-equalizing correction. Keep identical cohorts, splits, title-preserving packing, and source-selected settings. The recommendation concerns feasibility and representation, not predictive quality.

## Inference-only hardware pilot

{"backward_passes": 0, "device": "mps", "examples_per_dataset": {"CoAID": 8, "FakeHealth": 8}, "gradients_created": false, "optimizer_steps": 0, "parameter_versions_unchanged": true, "parameters": 81527808, "selection": "first eight eligible sorted content_uid per dataset; label blind", "status": "completed", "training_runs": 0, "unique_examples": 16}

| device | batch_size | unique_examples | timed_batches | mean_batch_seconds | examples_per_second | sampled_max_mps_allocated_bytes | sampled_max_mps_driver_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mps | 1 | 16 | 16 | 0.0201 | 49.7056 | 328568320 | 1082785792 |
| mps | 2 | 16 | 8 | 0.0287 | 69.6293 | 330141184 | 1082785792 |

At most eight eligible documents per dataset were selected by sorted UID without labels. Inputs used the 512-token title-preserving rule and fixed padding to 512. One warm-up per batch size was excluded. Timings cover synchronized forward passes with input tensors already on the device; tokenization, model loading, and data transfer are excluded. AutoModel without a classifier or pooler ran in eval and torch.inference_mode. MPS memory values are maximum post-forward samples, not reliable transient peaks. Parameters were not mutated; no gradients, backward pass, optimizer, or training epoch occurred.

Local small-scale fine-tuning appears practical on MPS; start with microbatch 1 and test 2 under backward/optimizer memory load. Inference microbatches 1 and 2 do not establish a training microbatch. Training requires additional activations, gradients, and optimizer state. No training-time extrapolation is claimed.

## Future training protocol — proposal only

Use AutoModelForSequenceClassification with two labels (Fake=1, Real=0), seed 42 and a pinned checkpoint. Retain existing five grouped CV folds and reserved holdout. Evaluate learning rates 2e-5/5e-5 and epoch checkpoints 2/3: four logical candidates per dataset, selectable from two three-epoch runs per training fold. Select mean grouped-CV macro F1 with a prespecified 0.001 tie tolerance, then lower sample SD, fewer epochs, lower learning rate. Weight decay 0.01. Do not use early stopping initially; two/three-epoch checkpoint choice uses training-pool validation only.

Tentative microbatch 2, gradient accumulation 8 (effective batch 16), pending hardware and memory validation; do not claim that an inference-only pilot establishes training memory requirements. Pin optimizer, schedule, precision and batch policy before tuning. Use FP32 initially and validate any hardware-dependent mixed precision separately. Training memory and wall-clock runtime are not certified by an inference-only pilot. No extrapolated hour estimate is supplied.

Primary loss proposal: weighted cross-entropy, with w_c=N_train/(2*N_train,c) computed separately from that fit’s source-training labels. This is a prespecified imbalance precaution, not a claim that this audit measured class effects. No target-derived weights or oversampling. Recompute source weights for the final source refit only, never from the target.

## Frozen transfer proposal

Select and lock both source configurations from their own training-pool CV before target evaluation. Refit each on its training pool and evaluate its original holdout once. Separately initialize the same pretrained checkpoint and fit the locked configuration on all eligible source rows (including the former source holdout), then freeze target predictions before joining target labels. Apply this in both directions, evaluating all eligible target rows and identical target-holdout UIDs used by LR/SVM. No target tuning, calibration, threshold adjustment or feature selection. A fixed Fake probability threshold of 0.5 is proposed. For the 256-token sensitivity reuse each source’s primary hyperparameters to avoid a second search. No classification model is trained in this audit.


## Validation

Runtime checks passed: exact previous cohort/split hashes, fixed eligible counts, original group/fold assignments, official-tokenizer determinism on every row, label-blind predictor-only packing, title preservation, and bounded inference-only invariants. Only UID, eligibility/split/group metadata and the three prepared text fields were read. No labels, source, or engagement metadata enter the representation. Unrelated output file sizes and modification times are checked without opening their contents. Original `.venv` Python/package/config fingerprint matched before and after setup. All 1321 unrelated output files retained their sizes and modification times. The nine automated tests passed with zero failures or skips, and `pip check` passed. Fingerprints and test results are recorded in the companion JSON.

Reproduce: `.venv-transformer/bin/python src/audit_transformer_feasibility.py`. Tests: `.venv-transformer/bin/python -m unittest discover -s src -p "test_transformer_feasibility.py" -q`.
