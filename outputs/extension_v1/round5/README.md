# extension_v1 round 5 — within-length-bin ROC-AUC (re-scoring diagnostic, no refit)

`round5_within_length_auc.csv` (SHA-256 `837b19a1afffbc23fbaf09102dffbc304f7dcc1fe35dd1e79b2561aff8e1708e`, also in `round5_within_length_auc.sha256`) answers one question for the supplement:
how much text signal is left once document length is held fixed. It is a **re-scoring diagnostic**: no model was refit, retrained or tuned,
no bootstrap was run, and no saved table or manuscript number changed.

Method (`round5_within_length_auc.py`): the script executes the scoring section of `outputs/extension_v1/scripts/round4_check2_length.py`
(the frozen LR and Linear SVM pipelines applied transform-only, with every `fit` patched to raise) and that script's quintile expression
(rank-based quintiles of analyzer token count per target, identical to `round4_length_bins.csv`). It stops unless all 40 reconstructed bins
match `round4_length_bins.csv` exactly (N, token range, Fake share, ROC-AUC) and the unstratified ROC-AUC equals Table S19.1 at 4 decimals.

Targets: within_CoAID_holdout, within_FakeHealth_holdout, transfer_CoAID_to_FakeHealth_all_eligible, transfer_FakeHealth_to_CoAID_all_eligible. Classifiers: LR, SVM.

Rows: one per length quintile (bin 1–5: token range, N, Fake N, Real N, same-bin (Fake, Real) pairs, ROC-AUC of the frozen score within the bin,
blank with `estimable = False` when the bin has fewer than 5 Fake or 5 Real items) and one `pooled_length_stratified` row per target × classifier:
the share of same-quintile (Fake, Real) pairs that the score orders correctly (ties 0.5) over all such pairs (`same_bin_pairs`), with the
unstratified frozen ROC-AUC (`unstratified_frozen_roc_auc`) and the Table S19.1 value it was checked against (`s19_1_frozen_roc_auc`).
No intervals are reported; the small Fake counts in CoAID's longer quintiles (3–14 items) make those bins imprecise.

Registered as Supplementary Table S19, panel S19.4 (with S19.3 rendering `round4_length_bins.csv`).
