# Extension v1, round 4 — reviewer checks

Date: 2026-10-13. Protocol written first: `protocols/round4_reviewer_checks.md`. Scripts: `scripts/round4_check1_findability.py` … `round4_check7_rounding.py` (Check 5a ran in `.venv-transformer`, everything else in `.venv`). Nothing frozen changed; no hyperparameter was reselected; every refit used the locked configuration and the saved split. All tables are under `tables/round4_*.csv`, figures under `figures/round4_*`.

## Check 1 — Findability (CoAID A, N = 5,337)

Publisher-resolution status from the frozen label-blind identity helper (`build_family_map`) applied to the prepared metadata: resolved = identified family or account proxy, unresolved = platform host without an account, missing = no source metadata.

| label | group | N | share_any_post | any_post_N |
| --- | --- | --- | --- | --- |
| Fake | missing | 40 | 0.0750 | 3 |
| Fake | resolved | 306 | 0.3301 | 101 |
| Fake | unresolved | 576 | 0.0747 | 43 |
| Real | missing | 3 | 0.3333 | 1 |
| Real | resolved | 4383 | 0.5759 | 2524 |
| Real | unresolved | 29 | 0.1724 | 5 |
| Fake | all | 922 | 0.1594 | 147 |
| Real | all | 4415 | 0.5730 | 2530 |

**Fake entities with a resolved publisher have any observed post 33.0% of the time; unresolved (platform-hosted) Fake entities 7.5%, missing-metadata Fake entities 7.5%.** Among Real entities the resolved share is 57.6%. Two thirds of Fake entities (616 of 922) are unresolved or missing, so the CoAID Fake occurrence deficit is concentrated where the publisher could not be identified; resolved Fake entities still sit well below resolved Real entities.

By first-recorded release the pattern is driven by capture timing as much as by class: Real entities first captured in the May or July 2020 releases have posts 91% and 90% of the time, those first captured in September or November 7.6% and 5.3%; Fake entities 12.8%, 26.2%, 7.1% and 3.2% respectively.

| label | group | N | share_any_post |
| --- | --- | --- | --- |
| Fake | 05-01-2020 | 572 | 0.1276 |
| Fake | 07-01-2020 | 263 | 0.2624 |
| Fake | 09-01-2020 | 56 | 0.0714 |
| Fake | 11-01-2020 | 31 | 0.0323 |
| Real | 05-01-2020 | 1585 | 0.9129 |
| Real | 07-01-2020 | 1076 | 0.9015 |
| Real | 09-01-2020 | 851 | 0.0764 |
| Real | 11-01-2020 | 903 | 0.0532 |

Fake entities by originating fact-check site (host of `fact_check_url`): PolitiFact-checked items are the largest group and the most often posted; AFP, CheckYourFact and Washington Post items are rarely posted.

| group | N | share_any_post |
| --- | --- | --- |
| politifact.com | 352 | 0.2216 |
| leadstories.com | 121 | 0.1322 |
| hoax-alert.leadstories.com | 103 | 0.1650 |
| healthfeedback.org | 100 | 0.1500 |
| factcheck.org | 89 | 0.1236 |
| factcheck.afp.com | 73 | 0.0548 |
| checkyourfact.com | 41 | 0.0976 |
| washingtonpost.com | 20 | 0.0500 |
| other sites (<10 each; 10 sites) | 23 | 0.0435 |

Title token length (scikit-learn word analyzer) is short throughout: resolved Real titles have a median of 9 tokens, resolved Fake 10, unresolved Fake 9 with a quarter at 3 tokens or fewer; 403 resolved Real entities and 31 Fake entities have an empty stored title. Within resolved entities, titles of at most five tokens are posted less often than longer titles (Fake 20% vs 37%, Real 65% vs 54%); within unresolved Fake entities almost nothing is posted at either length (17% vs 3%). Collection by title search can therefore explain part of the deficit for short-titled items but not the unresolved-platform deficit (`round4_title_length.csv`).

## Check 2 — Length as a shared cue

Spearman correlation of log(1 + analyzer tokens) with the frozen model score, and the frozen ROC-AUC, per model and target (`round4_length_spearman.csv`; per-quintile rates in `round4_length_bins.csv`):

| target | classifier | source | N | spearman_log_tokens_vs_score | frozen_roc_auc |
| --- | --- | --- | --- | --- | --- |
| within_CoAID_holdout | LR | CoAID | 702 | -0.5424 | 0.9827 |
| within_CoAID_holdout | SVM | CoAID | 702 | -0.5497 | 0.9795 |
| within_FakeHealth_holdout | LR | FakeHealth | 330 | -0.6280 | 0.7701 |
| within_FakeHealth_holdout | SVM | FakeHealth | 330 | -0.6355 | 0.7695 |
| transfer_CoAID_to_FakeHealth_all_eligible | LR | CoAID | 1647 | -0.6872 | 0.6047 |
| transfer_CoAID_to_FakeHealth_all_eligible | SVM | CoAID | 1647 | -0.6741 | 0.6113 |
| transfer_FakeHealth_to_CoAID_all_eligible | LR | FakeHealth | 3577 | -0.1185 | 0.5367 |
| transfer_FakeHealth_to_CoAID_all_eligible | SVM | FakeHealth | 3577 | -0.1383 | 0.5471 |
| external_COVIDRumor_clean | LR | CoAID | 3667 | -0.3558 | 0.6995 |
| external_COVIDRumor_clean | SVM | CoAID | 3667 | -0.3812 | 0.6816 |
| external_COVIDRumor_clean | LR | FakeHealth | 3667 | 0.0398 | 0.5219 |
| external_COVIDRumor_clean | SVM | FakeHealth | 3667 | 0.0262 | 0.5248 |
| external_FakeNewsNet_GossipCop_title_clean | LR | CoAID | 22114 | 0.0091 | 0.5323 |
| external_FakeNewsNet_GossipCop_title_clean | LR | FakeHealth | 22114 | -0.1253 | 0.4770 |
| external_FakeNewsNet_PolitiFact_title_clean | LR | CoAID | 1025 | 0.0941 | 0.6084 |
| external_FakeNewsNet_PolitiFact_title_clean | LR | FakeHealth | 1025 | -0.0528 | 0.5034 |

Within CoAID and FakeHealth, and for CoAID models applied to FakeHealth or COVID-Rumor, scores fall with length (ρ between −0.36 and −0.69): longer items are scored Real. The FakeHealth models carry almost no length signal on CoAID (−0.12) or COVID-Rumor (+0.04), and the title-only models none on FakeNewsNet. CoAID's lowest length quintile (4–36 tokens) is 93% Fake and the model labels 97% of it Fake; the remaining quintiles are 2–10% Fake. Length is thus a strong class marker inside CoAID, where stored excerpts of Fake claims are short.

**Length-only baseline (NEW DIAGNOSTIC MODEL: logistic regression on log(1 + tokens) alone, fitted on each source training pool; not a candidate classifier):**

| target | source_pool | representation | train_N | roc_auc_length_only | roc_auc_frozen_LR |
| --- | --- | --- | --- | --- | --- |
| within_CoAID_holdout | CoAID | combined_text | 2875 | 0.8818 | 0.9827 |
| within_FakeHealth_holdout | FakeHealth | combined_text | 1317 | 0.7289 | 0.7701 |
| transfer_CoAID_to_FakeHealth_all_eligible | CoAID | combined_text | 2875 | 0.6491 | 0.6047 |
| transfer_FakeHealth_to_CoAID_all_eligible | FakeHealth | combined_text | 1317 | 0.8521 | 0.5367 |
| external_COVIDRumor_clean | CoAID | combined_text | 2875 | 0.7158 | 0.6995 |
| external_COVIDRumor_clean | FakeHealth | combined_text | 1317 | 0.7158 | 0.5219 |
| external_FakeNewsNet_GossipCop_title_clean | CoAID | title_text | 2870 | 0.4806 | 0.5323 |
| external_FakeNewsNet_GossipCop_title_clean | FakeHealth | title_text | 1315 | 0.5194 | 0.4770 |
| external_FakeNewsNet_PolitiFact_title_clean | CoAID | title_text | 2870 | 0.7388 | 0.6084 |
| external_FakeNewsNet_PolitiFact_title_clean | FakeHealth | title_text | 1315 | 0.2612 | 0.5034 |

Length alone reaches ROC-AUC 0.88 within CoAID (frozen LR 0.98) and 0.73 within FakeHealth (0.77). On the transfer targets the length baseline is *better* than the frozen text models: 0.65 vs 0.60 for CoAID→FakeHealth and 0.85 vs 0.54 for FakeHealth→CoAID, and 0.72 vs 0.70/0.52 on COVID-Rumor. The cross-dataset "lesson" that does carry is therefore largely that short items are Fake, in the direction CoAID's data encode it; the text models underuse it on transfer because their vocabularies dominate. On FakeNewsNet titles the length baseline is informative for PolitiFact from the CoAID pool (0.74, where the frozen title LR reaches 0.61) and anti-informative from the FakeHealth pool (0.26). These numbers describe a confound shared by all models, not a competing method.

## Check 3 — Bootstrap shape and the dominant cluster

Saved draws only. Inclusions = number of times the dominant component entered a draw (`round4_bootstrap_conditional.csv`; histograms in `round4_bootstrap_histogram.csv`; figures `round4_bootstrap_shape_<context>.png/.svg`).

| context | inclusions | draws | share_of_draws | ratio_p025 | ratio_median | ratio_p975 |
| --- | --- | --- | --- | --- | --- | --- |
| CoAID_A | all | 1000 | 1.0000 | 0.1322 | 0.2656 | 0.5220 |
| CoAID_A | 0 | 363 | 0.3630 | 0.2659 | 0.4082 | 0.5689 |
| CoAID_A | 1 | 372 | 0.3720 | 0.1693 | 0.2486 | 0.3483 |
| CoAID_A | 2+ | 265 | 0.2650 | 0.1121 | 0.1824 | 0.2699 |
| PolitiFact | all | 1000 | 1.0000 | 0.4394 | 0.5653 | 1.8677 |
| PolitiFact | 0 | 372 | 0.3720 | 1.0548 | 1.4653 | 1.9843 |
| PolitiFact | 1 | 355 | 0.3550 | 0.4879 | 0.5543 | 0.6496 |
| PolitiFact | 2+ | 273 | 0.2730 | 0.4272 | 0.4703 | 0.5284 |
| GossipCop | all | 1000 | 1.0000 | 1.2116 | 2.0973 | 3.2904 |
| GossipCop | 0 | 377 | 0.3770 | 1.1500 | 1.3749 | 1.6176 |
| GossipCop | 1 | 356 | 0.3560 | 1.9665 | 2.1332 | 2.3386 |
| GossipCop | 2+ | 267 | 0.2670 | 2.5575 | 2.7804 | 3.6311 |

**CoAID A: the dominant component enters 63.7% of draws (once in 37.2%, two or more times in 26.5%). The ratio conditional on zero inclusions has median 0.408 (0.266–0.569); conditional on one inclusion 0.249 (0.169–0.348); on two or more 0.182 (0.112–0.270).** The unconditional interval 0.132–0.522 is a mixture of these three populations; the Gaussian-kernel density of the log ratio has two local maxima, and the histogram shows three largely separated clumps. PolitiFact behaves the same way with the direction reversed: zero inclusions give a median of 1.465 (1.055–1.984), one inclusion 0.554, two or more 0.470, so the unconditional interval (0.439–1.868) spans a bimodal mixture rather than a continuum. GossipCop's dominant component pushes the ratio up: 1.375 without it, 2.133 with one inclusion, 2.780 with two or more, three local maxima. In all three contexts the resampling multiplicity of one component, not sampling noise around a single value, generates the interval width.

PolitiFact dominant component, label-blind title audit (`round4_politifact_component_titles.csv`, `round4_politifact_component_top_titles.csv`):

| group | rows | fake_N | real_N | exact_duplicate_title_share | distinct_titles | near_duplicate_share_frozen_rule_min8 | mean_pairwise_title_bigram_jaccard | median_title_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dominant_component_84 | 84 | 5 | 79 | 0.2024 | 74 | 0.0000 | 0.0054 | 6.0000 |
| rest_of_PolitiFact | 968 | 425 | 543 | 0.0868 | 916 | 0.0000 | 0.0006 | 10.0000 |

The 84-row component has a higher exact-duplicate title share (20% vs 9%) and shorter titles (median 6 vs 10 tokens), but no near-duplicate pairs under the frozen rule or a relaxed 4-token rule, and a mean pairwise bigram Jaccard of 0.005. Its ten most frequent titles are generic page titles and transcripts ("Outlook, Office, Skype, Bing, Breaking News, and Latest Videos" ×4, "The Democratic Debate in Cleveland" ×3, "Transcripts", "LexisNexis(R) Publisher", "Statement by the President", "Patient Protection and Affordable Care Act" ×2 each). The component is therefore a set of mostly Real, generically titled or transcript pages linked by shared tweets, not a cluster of paraphrased stories.

## Check 4 — Mean versus typical item (zero-inclusive unique posts)

| context | label | N | share_zero | median | p75 | p90 | mean | prob_random_fake_exceeds_random_real_ties_half |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CoAID_A | Fake | 922 | 0.8406 | 0.0000 | 0.0000 | 3.0000 | 11.3265 | 0.2861 |
| CoAID_A | Real | 4415 | 0.4270 | 3.0000 | 19.0000 | 54.0000 | 32.1133 | 0.2861 |
| FakeHealth | Fake | 472 | 0.0021 | 113.0000 | 215.2500 | 400.6000 | 199.8771 | 0.4595 |
| FakeHealth | Real | 1218 | 0.0049 | 129.0000 | 270.0000 | 505.6000 | 237.8744 | 0.4595 |
| PolitiFact | Fake | 430 | 0.0930 | 77.5000 | 288.7500 | 752.3000 | 374.9907 | 0.6345 |
| PolitiFact | Real | 622 | 0.3457 | 7.5000 | 154.7500 | 1541.4000 | 670.5305 | 0.6345 |
| GossipCop | Fake | 5323 | 0.0353 | 12.0000 | 42.0000 | 303.0000 | 112.3988 | 0.3654 |
| GossipCop | Real | 16817 | 0.0629 | 45.0000 | 66.0000 | 91.0000 | 52.4247 | 0.3654 |

The probability that a random Fake item's count exceeds a random Real item's (ties half) is 0.29 in CoAID, 0.46 in FakeHealth, 0.63 in PolitiFact and 0.37 in GossipCop. GossipCop's mean ratio above 2 coexists with Fake items being *less* likely than Real items to have the higher count (medians 12 vs 45); the mean is carried by the Fake tail (p90 303 vs 91). PolitiFact's mean ratio below 1 coexists with Fake items more often having the higher count (median 77.5 vs 7.5, 35% of Real items at zero), the mean being carried by the Real tail (p90 1,541). Mean ratios and typical-item comparisons disagree in both non-health contexts.

## Check 5 — Truncation and the FakeHealth signal

Cut location from the frozen tokenizer and pair-packing rule (`round4_fakehealth_cut_offsets.csv`): 69.4% of FakeHealth items are truncated at 512 tokens (71.2% in the training cohort as reported), median body length 736.5 tokens, median kept 495.

| variant | macro_f1 | roc_auc | average_precision | vocabulary_size | empty_training_texts |
| --- | --- | --- | --- | --- | --- |
| frozen_saved | 0.6752 | 0.7701 | 0.5707 | 28514 | 0 |
| natural_refit_check | 0.6752 | 0.7701 | 0.5707 | 28514 | 0 |
| before_512_cut | 0.6509 | 0.7182 | 0.5324 | 16982 | 0 |
| after_512_cut | 0.6525 | 0.7656 | 0.4771 | 14280 | 392 |

**Refitting the locked FakeHealth LR on the text before the cut gives holdout macro F1 0.651**, ROC-AUC 0.718, AP 0.532, against the frozen 0.675 / 0.770 / 0.571; the text after the cut alone gives 0.653 / 0.766 / 0.477 (392 training items have nothing after the cut and enter as empty strings). The signal is spread through the article: the transformer saw only the first part, and that part alone carries less ranking signal than the remainder. Of the frozen model's 50 largest-magnitude terms, 40.9% of their training-pool occurrences fall after the cut (occurrence-weighted 44.5%); the press-release cues ("news release" 8.8%, "in news" 5.1%, "release" 24.7%) sit early, whereas the Real-leaning discourse terms ("still" 60%, "says" 61%, "there" 56%) sit mostly after it (`round4_fakehealth_top50_terms_after_cut.csv`).

## Check 6 — Lexical signal reliability

Locked LR refitted on each of the five saved CV training folds; pairwise Spearman of coefficients over the 4,174 shared unigrams (terms absent from a fold's vocabulary excluded pairwise; `round4_fold_coefficient_pairs.csv`):

| dataset | pairs | spearman_mean | spearman_min | spearman_max | cross_dataset_spearman_frozen |
| --- | --- | --- | --- | --- | --- |
| FakeHealth | 10 | 0.7799 | 0.7695 | 0.7941 | -0.0034 |
| CoAID | 10 | 0.8208 | 0.8003 | 0.8355 | -0.0034 |

**Within-dataset fold-to-fold coefficient correlations are 0.78 (FakeHealth, range 0.77–0.79) and 0.82 (CoAID, 0.80–0.84), against −0.003 across datasets.** The near-zero cross-dataset alignment is therefore not an artefact of coefficient noise: coefficients are stable across training folds within each dataset and unrelated between datasets.

## Check 7 — Rounding consistency

250 decimal tokens in the manuscript text, Tables 1–2 and captions were matched to saved values and checked against half-up rounding at the displayed precision; figure labels of Figures 1–4 were checked against the same rule (`round4_rounding_findings.csv`). Three displayed values are double-rounded from four-decimal intermediates and differ from half-up rounding of the saved value. **Proposed edits for the next rebuild (not applied):**

| Location | Displayed | Saved value | Half-up | Edit |
|---|---|---|---|---|
| Results 3.2, line 149: "stayed above 1 without its largest cluster (1.374, 1.146–1.621)" | 1.374 | 1.3734525 (`fakenewsnet_round3_exclusion_results.csv`, GossipCop mean_ratio) | 1.373 | change to 1.373 |
| Table 2, GossipCop largest-cluster-excluded row | 1.374 | same | 1.373 | change to 1.373 (Figure 1 already labels it 1.373) |
| Results 3.3, line 168: "CoAID 0.913 and 0.980" (SVM ROC-AUC) | 0.980 | 0.9794664 (`tfidf_linear_svm_holdout_results.csv`) | 0.979 | change to 0.979 |
| Results 3.3, line 176: "SVM reproduced both results (0.425 and 0.583; …)" | 0.425 | 0.4244968 (key result R017) | 0.424 | change to 0.424 |

Also outside the package: `outputs/extension_v1/reports/extension_v1_round3.md` and the round-3 terminal summary state the GossipCop exclusion ratio as 1.374; the saved value rounds to 1.373. Figure labels (Python `%.3f`) agree with half-up for every plotted value, including 1.373. Values that appear unmatched only because they are derived (0.995 = 1212/1218, 66.81% from the feasibility report, −0.003 with sign) or are section numbers, thresholds, seeds or version strings were excluded after inspection.

## Files

`tables/round4_findability.csv`, `round4_title_length.csv`, `round4_length_bins.csv`, `round4_length_spearman.csv`, `round4_length_baseline.csv`, `round4_bootstrap_conditional.csv`, `round4_bootstrap_histogram.csv`, `round4_politifact_component_titles.csv`, `round4_politifact_component_top_titles.csv`, `round4_typical_item.csv`, `round4_fakehealth_cut_offsets.csv`, `round4_fakehealth_truncation.csv`, `round4_fakehealth_top50_terms_after_cut.csv`, `round4_fold_coefficient_pairs.csv`, `round4_fold_coefficient_summary.csv`, `round4_rounding_findings.csv`; `figures/round4_bootstrap_shape_{CoAID_A,PolitiFact,GossipCop}.{png,svg}`; `protocols/round4_reviewer_checks.md`; `scripts/round4_check*.py`.
