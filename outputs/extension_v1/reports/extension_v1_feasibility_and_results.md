# Extension v1 — feasibility and results

Date: 2026-10-12. Frozen baseline untouched: nothing under `outputs/tables/`, `outputs/manuscript/`, `outputs/models/` or `data/processed/` was modified. All outputs are under `outputs/extension_v1/` (`tables/`, `reports/`, `protocols/`, `scripts/`). No frozen model was refitted; no new model was fitted. Every number below comes from saved fits, saved bootstrap draws, or transform-only scoring with frozen vectorizers and classifiers.

## Task 1 — CoAID Specification A hurdle components (reporting only)

**Source.** `outputs/tables/version_aware_hurdle_components.csv` (saved point estimates and cluster intervals) and the 1,000 successful draws for `A_primary_strict` in `outputs/tables/version_aware_bootstrap_refits.csv`, which store `post_probability_fake/real`, `positive_fake_mean/real_mean`, `positive_mean_ratio` and `mean_ratio` per draw. Model: hurdle logit + zero-truncated NB2, first-release fixed effects (May reference), both classes standardized over the full A covariate mix (N = 5,337; 922 Fake, 4,415 Real; zero share 0.498). Intervals are 2.5/97.5 percentiles of the draws (linear method); recomputing them reproduces every saved interval in the components table exactly, including the frozen 0.1322–0.5220 for the full ratio.

| Component | Fake | Real | Fake/Real ratio |
|---|---:|---:|---:|
| P(posts > 0), standardized | 0.1172 [0.1008, 0.1348] | 0.6168 [0.5922, 0.6457] | **0.1900** [0.1639, 0.2182] |
| E(posts \| posts > 0), standardized | 51.40 [36.36, 71.22] | 39.53 [23.53, 64.15] | **1.3004** [0.7285, 2.5464] |
| E(posts), full estimand | 8.83 [5.93, 12.41] | 34.92 [20.35, 57.88] | **0.2528** [0.1322, 0.5220] |

**Product check.** Occurrence ratio × conditional-count ratio = 0.1900 × 1.3004 = **0.2471** [0.1290, 0.5012], which is not the full ratio 0.2528. The difference is 0.0057 [0.0012, 0.0245]; the product exceeds the full ratio in 0.8% of draws. The frozen estimand averages the row-level product P(Y>0|X)·E(Y|Y>0,X) over the covariate mix before taking the Fake/Real ratio, so it is not the product of separately averaged components. Confirmed as expected.

**Interpretation (descriptive).** The Fake deficit in CoAID A sits entirely in occurrence: a Fake entity is about one fifth as likely to have any observed post. Conditional on at least one observed post, Fake entities have about 30% higher expected unique posts, but that interval includes 1 and is wide (0.73–2.55). This is a within-CoAID decomposition; FakeHealth was fitted as NB2 without a hurdle and no cross-context component contrast is defined.

**Available / not available.** Available: saved component point estimates and per-draw component values for all 1,000 draws, giving percentile intervals for every component and for the derived occurrence ratio and product. Not available: the fitted statsmodels objects (not pickled), per-entity predicted probabilities or conditional means for the version-aware A fit, and draw-level per-entity predictions. `outputs/tables/engagement_model_standardized_predictions.csv` belongs to the earlier primary-sample run (N = 5,073), not the version-aware A fit, and was not used. The occurrence diagnostic table (`version_aware_occurrence_results.csv`, 0.1174/0.6177) and all odds ratios were not used. Output: `tables/coaid_hurdle_components.csv` (12 rows, with source and interval provenance per row and input SHA-256s).

## Task 2 — Top Fake/Real unigrams (reporting only)

**Source.** Saved within-context natural-text LR fits `outputs/models/tfidf_logistic/final_coaid_natural.joblib` (unigrams, min_df 2, 6,428 features, training pool 2,875) and `final_fakehealth_natural.joblib` (uni+bigrams, min_df 5, 28,514 features of which 7,381 unigrams, training pool 1,317). Fake = 1, so positive coefficients lean Fake. Training document frequencies were obtained by transforming the frozen training pool with the frozen vectorizer (no fit) and verified equal to the saved `text_shared_unigram_coefficients.csv` values for every shared term; coefficients likewise match the shared table to 1e-9. Mask flags replicate `flag_features` in `src/audit_cross_context_lexical_alignment.py` on the unchanged `text_source_mask_dictionary.csv` (656 identifiers: 488 domains, 165 publisher names, 3 boilerplate).

| Dataset, leaning | In shared unigram vocab (of 25) | Sign reversed in other model | Would be removed by masking (full identifier) | Fragment of a dictionary identifier |
|---|---:|---:|---:|---|
| CoAID Fake | 17 | 11 | 1 (`facebook`) | 3 (`coronavirus`, `the`, `video`) |
| CoAID Real | 22 | 10 | 0 | 6 (`and`, `disease`, `covid`, `health`, `for`, `new`) |
| FakeHealth Fake | 19 | 14 | 0 | 4 (`news`, `and`, `the`, `cbs`) |
| FakeHealth Real | 23 | 9 | 0 | 1 (`vaccine`) |

Masking removes whole identifier occurrences only (`(?<!\w)identifier(?!\w)`), so a unigram is removed only when it equals a full dictionary identifier. Exactly one of the 100 listed terms qualifies (`facebook`, CoAID Fake rank 3). Fragment terms such as `news` or `cbs` would be removed only inside an identifier phrase (`CBS News`), not as standalone words.

Observations from `tables/top_fake_real_terms.csv`: CoAID coefficients are large (Fake up to +3.96 for `coronavirus`, Real down to −9.24 for `and`) and include function words, dates (`2020`, `april`, `march`), publisher/credit fragments (`getty`, `images`, `medscape`) and two mojibake tokens (`òthe`, `donõt`) among Fake-leaning terms, which indicates encoding artifacts in the CoAID Fake text. FakeHealth coefficients are small (|coef| ≤ 1.02) and dominated by press-release and generic health vocabulary (`release`, `news`, `mri`, `pfizer` Fake; `still`, `says`, `drug`, `costs`, `vaccine` Real). Between 9 and 14 of each top-25 list carry the opposite sign in the other model, consistent with the frozen lexical-alignment finding.

## Task 3 — FakeNewsNet as a second external target

### 3a Feasibility (from `data/raw/fakenewsnet/`, four CSVs with columns `id, news_url, title, tweet_ids`)

| | PolitiFact Fake | PolitiFact Real | GossipCop Fake | GossipCop Real |
|---|---:|---:|---:|---:|
| Rows in file | 432 | 624 | 5,323 | 16,817 |
| Ids in both label files (excluded, conflicting) | 2 | 2 | 0 | 0 |
| Title present | all | all | all | all |
| Article body present locally | none | none | none | none |
| `news_url` present | 426 | 565 | 5,067 | 16,804 |
| Rows with ≥1 tweet id | 390 | 407 | 5,135 | 15,759 |
| Zero-tweet share | 9.3% | 34.6% | 3.5% | 6.3% |
| Median tweet ids | 77.5 | 7.5 | 12 | 45 |
| Quality-screen exclusions | 1 | 26 | 8 | 18 |
| Exact-duplicate title rows | 9 | 105 | 819 | 1,498 |
| Survivors (`all_eligible`) | 429 | 596 | 5,315 | 16,799 |
| Exact source overlap removed | 0 | 0 | 0 | 0 |
| `clean_primary` | 429 | 596 | 5,315 | 16,799 |

Quality screen applied verbatim from the COVID-Rumor script: empty text, obvious crawl artifact, fewer than two alphabetic analyzer tokens after URL stripping, or less than 90% Latin letters. Exclusions were 14 artifact-only titles (13 PolitiFact Real items titled "Outlook, Office, Skype, Bing, Breaking News, and Latest Videos", 1 GossipCop "Are you a robot?"), 34 near-empty titles and 5 non-Latin-script titles. Duplicate titles are common (e.g., "Connecting People Through News" 20×, "Female First Forum" 13×) and were retained as distinct items, as in the COVID-Rumor evaluation; results are item-weighted descriptives.

Label definitions: Fake/Real follow the file names. The FakeNewsNet repository documents PolitiFact labels as fact-checker verdicts and GossipCop labels as site scores with low-scored stories treated as fake and E! Online stories as real. The local copy contains no README; these definitions were not verified here and are inherited.

Source overlap, same rules as COVID-Rumor against all 5,337 CoAID and 1,690 FakeHealth modeling-ready rows: 14 exact title matches with FakeHealth (234 + 22 pair records over 40 FakeHealth rows), all of which are the two crawl-artifact titles above and were already quality-excluded, so overlap removed no additional item. Zero exact text matches with CoAID, zero exact resource (URL) matches with either source, zero high-similarity flags. Tables: `fakenewsnet_external_cohort_summary.csv`, `fakenewsnet_external_overlap_audit.csv`, `fakenewsnet_external_overlap_evidence.csv`.

### 3b Representation decision

Body text is absent locally, so the frozen combined-text LR/SVM models are the wrong representation. The frozen **title-only** natural LR transfer fits were used: `tfidf_logistic_title/transfer_coaid_natural.joblib` (3,571 title-eligible CoAID rows, cfg_16: uni+bigram, min_df 2, C 10, no class weight, 5,230 features) and `transfer_fakehealth_natural.joblib` (1,645 FakeHealth rows, cfg_11: unigram, min_df 5, C 10, balanced, 514 features). Both passed validation: converged, classes [0,1], vectorizer and classifier match the locked `selection_lock.json` configuration, and `train_uid_sha256` equals the digest of the current title-only-eligible UIDs. No frozen title-only Linear SVM exists, so the SVM arm of the COVID-Rumor comparison is not replicated; this is an LR-only comparison.

### 3c Protocol and results

Protocol written before scoring: `protocols/fakenewsnet_external_protocol.md` (cohort rules, label-conflict gate, frozen fits, Fake = 1, LR threshold 0.5, metrics, no target-side fitting, calibration or threshold tuning). Predictions were written and hashed before labels were joined (`fakenewsnet_prediction_lock.json`). Fitting was blocked at runtime by patching `TfidfVectorizer.fit/fit_transform` and `LogisticRegression.fit`.

`clean_primary` results (identical to `all_eligible` because overlap removed nothing; `clean_without_high_similarity` identical because nothing was flagged):

| Subset | Source | N (Fake/Real) | Macro F1 | Bal. acc. | Fake F1 | ROC-AUC | AP | Actual Fake prev. | Predicted Fake prev. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PolitiFact | CoAID | 1,025 (429/596) | 0.5419 | 0.5748 | 0.5817 | 0.6084 | 0.5043 | 0.4185 | 0.6683 |
| PolitiFact | FakeHealth | 1,025 (429/596) | 0.5370 | 0.5395 | 0.4213 | 0.5034 | 0.4262 | 0.4185 | 0.3317 |
| GossipCop | CoAID | 22,114 (5,315/16,799) | 0.4624 | 0.5245 | 0.3601 | 0.5323 | 0.2609 | 0.2403 | 0.5693 |
| GossipCop | FakeHealth | 22,114 (5,315/16,799) | 0.4742 | 0.4814 | 0.2689 | 0.4770 | 0.2233 | 0.2403 | 0.3691 |

CoAID-source minus FakeHealth-source (`fakenewsnet_external_source_comparison.csv`):

| Subset | Macro F1 | Bal. acc. | Fake F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| PolitiFact | +0.0049 | +0.0353 | +0.1604 | +0.1049 | +0.0781 |
| GossipCop | −0.0118 | +0.0431 | +0.0912 | +0.0553 | +0.0376 |

Vocabulary coverage (`fakenewsnet_external_vocabulary_diagnostics.csv`): the CoAID title vocabulary covers 63% (PolitiFact) and 49% (GossipCop) of target token occurrences with about 2–3% zero-feature titles; the FakeHealth title vocabulary covers 30% and 29% with 11.8% and 7.1% zero-feature titles.

**Does the CoAID-source advantage recur?** On COVID-Rumor NEWS (combined-text LR, clean), CoAID led FakeHealth on every comparison metric (macro F1 +0.061, ROC-AUC +0.178). On FakeNewsNet with title-only LR:

- **PolitiFact: yes, weakly.** CoAID leads on all five metrics, clearly in ranking (ROC-AUC 0.608 vs 0.503, AP 0.504 vs 0.426) and Fake F1, but the macro F1 difference is 0.005 and both models are near chance.
- **GossipCop: mixed.** CoAID leads on balanced accuracy, Fake F1, ROC-AUC and AP, but trails on macro F1 (0.462 vs 0.474). FakeHealth's ROC-AUC is below 0.5. Both are at or below chance on the thresholded summary.

The directional bias differs from COVID-Rumor, where both sources overpredicted Fake: here the CoAID title model overpredicts Fake (0.67 and 0.57 predicted vs 0.42 and 0.24 actual) while the FakeHealth title model underpredicts it (0.33 and 0.37). The CoAID source ordering in ranking metrics therefore persists on an unrelated political and entertainment target, but absolute performance is at chance level, the representation is title-only, the FakeHealth title model has a 514-feature vocabulary with poor coverage, and no SVM arm exists. For reference, the frozen title-only fits themselves scored macro F1 0.858 (CoAID) and 0.513 (FakeHealth) on their own holdouts and 0.41–0.47 in cross-context title transfer, so the FakeHealth title model carried little usable signal even in its source context.

## Interpretation bounds

Descriptive, item-weighted, inherited labels with differing constructs (fact-checker verdicts, site scores, review ratings, filename categories), title-only representation, no calibration. These results do not identify a COVID-specific mechanism, do not alter any frozen result, and are not manuscript content unless the authors decide to add a second external target to the Results with these caveats.

## Files

- `tables/coaid_hurdle_components.csv`, `tables/top_fake_real_terms.csv`
- `tables/fakenewsnet_external_{cohort_summary,overlap_audit,overlap_evidence,target,labels,fit_manifest,vocabulary_diagnostics,transfer_predictions,transfer_results,source_comparison}.csv`, `tables/fakenewsnet_external_cohort_lock.json`, `tables/fakenewsnet_prediction_lock.json`
- `protocols/fakenewsnet_external_protocol.md`
- `scripts/task1_hurdle_components.py`, `scripts/task2_top_terms.py`, `scripts/task3a_fakenewsnet_feasibility.py`, `scripts/task3c_fakenewsnet_score.py` (rerunnable with `.venv/bin/python -I`)
