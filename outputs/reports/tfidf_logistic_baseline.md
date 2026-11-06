# First TF-IDF + Logistic Regression baseline

Within-dataset classification only: FakeHealth HealthStory and frozen CoAID News Specification A. No transfer, alternative classifier, threshold tuning, title/body-only experiment or publisher-held-out split was run.

## Frozen protocol

Fake=1, Real=0. Only primary_modeling_eligible rows and combined_text enter the primary Pipeline. Source-masked sensitivity uses combined_text_source_masked on identical rows/splits. Targets, ratings, source/review metadata, identifiers, engagement, provenance and quality flags are never concatenated into predictors. Original target/rating definitions are retained in the prediction output for reporting.

The 24 prespecified configurations cross ngram_range=(1,1)/(1,2), min_df=2/5, C=0.1/1/10, and class_weight=None/balanced. TF-IDF uses lowercase=True, strip_accents=None, word analysis, sublinear_tf=True, max_df=1.0, float64 sparse data, L2 normalization and default smoothed IDF. Logistic Regression uses the stable sparse-compatible binary liblinear solver, pure L2 regularization, max_iter=5000, tol=1e-4 and seed 42. In installed sklearn, l1_ratio=0.0 specifies L2 without the deprecated penalty parameter. Liblinear regularizes its synthetic intercept with default intercept_scaling=1.

Each Pipeline fits a fresh vectorizer exclusively inside its training fold. Selection uses unweighted mean macro F1 across the five existing grouped folds. Configurations within 0.001 of the maximum mean are tied; among those, SD within 0.001 of the minimum is tied. Then prefer unigram, smaller C, class_weight=None, larger min_df, and stable configuration ID, in that order. SD is the sample SD (ddof=1), not an uncertainty interval; unequal fold sizes remain unchanged. Nonconverged configurations cannot win.

Both datasets’ natural-text configurations were locked before any holdout prediction. Final models fit the complete training pool and evaluate the reserved holdout once at probability >=0.5. Source masking is never retuned. Masked CV and holdout use the frozen natural-text hyperparameters. All curves, confusion matrices and diagnostics reuse saved predictions; restart checkpoints avoid reevaluation. Selected CV performance is a tuning estimate, not nested-CV generalization performance.

## Results

### FakeHealth

Selected: ngram_range=(1,2), min_df=5, C=1.0, class_weight=balanced.

| holdout_assignment | N | Fake N | Real N |
| --- | --- | --- | --- |
| test | 330 | 92 | 238 |
| train | 1317 | 367 | 950 |

| metric | CV mean | CV SD |
| --- | --- | --- |
| accuracy | 0.7351 | 0.0244 |
| balanced_accuracy | 0.6550 | 0.0268 |
| macro_f1 | 0.6598 | 0.0283 |
| fake_precision | 0.5282 | 0.0458 |
| fake_recall | 0.4743 | 0.0353 |
| fake_f1 | 0.4997 | 0.0395 |
| real_precision | 0.8044 | 0.0153 |
| real_recall | 0.8358 | 0.0205 |
| real_f1 | 0.8198 | 0.0175 |
| roc_auc | 0.7309 | 0.0087 |
| average_precision | 0.5364 | 0.0216 |

| condition | model | N | fake_N | real_N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | tfidf_logistic | 330 | 92 | 238 | 0.7515 | 0.6677 | 0.6752 | 0.5641 | 0.4783 | 0.5176 | 0.8095 | 0.8571 | 0.8327 | 0.7701 | 0.5707 |
| source_masked | tfidf_logistic | 330 | 92 | 238 | 0.7515 | 0.6644 | 0.6726 | 0.5658 | 0.4674 | 0.5119 | 0.8071 | 0.8613 | 0.8333 | 0.7648 | 0.5643 |
| dummy | dummy_most_frequent | 330 | 92 | 238 | 0.7212 | 0.5000 | 0.4190 | 0.0000 | 0.0000 | 0.0000 | 0.7212 | 1.0000 | 0.8380 | 0.5000 | 0.2788 |
| dummy | dummy_stratified | 330 | 92 | 238 | 0.5727 | 0.4671 | 0.4670 | 0.2308 | 0.2283 | 0.2295 | 0.7029 | 0.7059 | 0.7044 | 0.4671 | 0.2678 |

Confusion matrices below use rows=true Real/Fake, columns=predicted Real/Fake: [[TN, FP], [FN, TP]].

| condition | model | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| natural | tfidf_logistic | 204 | 34 | 48 | 44 |
| source_masked | tfidf_logistic | 205 | 33 | 49 | 43 |
| dummy | dummy_most_frequent | 238 | 0 | 92 | 0 |
| dummy | dummy_stratified | 168 | 70 | 71 | 21 |

Masked minus natural (paired by fold for CV; sample SD of paired differences recorded):

| scope | metric | natural | masked | masked_minus_natural | delta_sd |
| --- | --- | --- | --- | --- | --- |
| cv_mean | accuracy | 0.7351 | 0.7259 | -0.0091 | 0.0043 |
| holdout | accuracy | 0.7515 | 0.7515 | 0.0000 | nan |
| cv_mean | balanced_accuracy | 0.6550 | 0.6420 | -0.0130 | 0.0045 |
| holdout | balanced_accuracy | 0.6677 | 0.6644 | -0.0033 | nan |
| cv_mean | macro_f1 | 0.6598 | 0.6469 | -0.0129 | 0.0040 |
| holdout | macro_f1 | 0.6752 | 0.6726 | -0.0025 | nan |
| cv_mean | fake_precision | 0.5282 | 0.5116 | -0.0167 | 0.0082 |
| holdout | fake_precision | 0.5641 | 0.5658 | 0.0017 | nan |
| cv_mean | fake_recall | 0.4743 | 0.4524 | -0.0219 | 0.0156 |
| holdout | fake_recall | 0.4783 | 0.4674 | -0.0109 | nan |
| cv_mean | fake_f1 | 0.4997 | 0.4799 | -0.0198 | 0.0079 |
| holdout | fake_f1 | 0.5176 | 0.5119 | -0.0057 | nan |
| cv_mean | real_precision | 0.8044 | 0.7970 | -0.0074 | 0.0031 |
| holdout | real_precision | 0.8095 | 0.8071 | -0.0024 | nan |
| cv_mean | real_recall | 0.8358 | 0.8316 | -0.0042 | 0.0101 |
| holdout | real_recall | 0.8571 | 0.8613 | 0.0042 | nan |
| cv_mean | real_f1 | 0.8198 | 0.8139 | -0.0059 | 0.0042 |
| holdout | real_f1 | 0.8327 | 0.8333 | 0.0007 | nan |
| cv_mean | roc_auc | 0.7309 | 0.7255 | -0.0054 | 0.0028 |
| holdout | roc_auc | 0.7701 | 0.7648 | -0.0053 | nan |
| cv_mean | average_precision | 0.5364 | 0.5252 | -0.0112 | 0.0058 |
| holdout | average_precision | 0.5707 | 0.5643 | -0.0064 | nan |

Of the 60 signed top features, 2 match full tokenized source identifiers, 8 match identifier components, and 0 match the fixed crawl-artifact candidate list. Component matches are deliberately broad and can be ordinary words; they are not proof of publisher identity.

| associated_class | rank | feature | coefficient | training_document_frequency | source_identifier_match | crawl_artifact_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| Fake | 1 | news release | 1.0703 | 70 | False | False |
| Fake | 2 | release | 1.0228 | 127 | False | False |
| Fake | 3 | news | 0.8836 | 433 | False | False |
| Fake | 4 | and | 0.8417 | 1218 | False | False |
| Fake | 5 | the | 0.8383 | 1235 | False | False |
| Fake | 6 | to | 0.7763 | 1235 | False | False |
| Fake | 7 | in news | 0.6981 | 38 | False | False |
| Fake | 8 | artificial | 0.6716 | 27 | False | False |
| Real | 1 | still | -0.7700 | 437 | False | False |
| Real | 2 | was not | -0.7395 | 168 | False | False |
| Real | 3 | says | -0.7370 | 455 | False | False |
| Real | 4 | there | -0.7016 | 744 | False | False |
| Real | 5 | drug | -0.6585 | 489 | False | False |
| Real | 6 | costs | -0.6553 | 146 | False | False |
| Real | 7 | many | -0.6520 | 580 | False | False |
| Real | 8 | vaccine | -0.6427 | 40 | False | False |

### CoAID

Selected: ngram_range=(1,1), min_df=2, C=10.0, class_weight=balanced.

| holdout_assignment | N | Fake N | Real N |
| --- | --- | --- | --- |
| test | 702 | 166 | 536 |
| train | 2875 | 659 | 2216 |

| metric | CV mean | CV SD |
| --- | --- | --- |
| accuracy | 0.9555 | 0.0101 |
| balanced_accuracy | 0.9341 | 0.0108 |
| macro_f1 | 0.9375 | 0.0105 |
| fake_precision | 0.9146 | 0.0144 |
| fake_recall | 0.8939 | 0.0170 |
| fake_f1 | 0.9041 | 0.0143 |
| real_precision | 0.9677 | 0.0081 |
| real_recall | 0.9742 | 0.0070 |
| real_f1 | 0.9710 | 0.0073 |
| roc_auc | 0.9861 | 0.0016 |
| average_precision | 0.9511 | 0.0114 |

| condition | model | N | fake_N | real_N | accuracy | balanced_accuracy | macro_f1 | fake_precision | fake_recall | fake_f1 | real_precision | real_recall | real_f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| natural | tfidf_logistic | 702 | 166 | 536 | 0.9459 | 0.9313 | 0.9260 | 0.8721 | 0.9036 | 0.8876 | 0.9698 | 0.9590 | 0.9644 | 0.9827 | 0.9280 |
| source_masked | tfidf_logistic | 702 | 166 | 536 | 0.9473 | 0.9322 | 0.9278 | 0.8772 | 0.9036 | 0.8902 | 0.9699 | 0.9608 | 0.9653 | 0.9818 | 0.9269 |
| dummy | dummy_most_frequent | 702 | 166 | 536 | 0.7635 | 0.5000 | 0.4330 | 0.0000 | 0.0000 | 0.0000 | 0.7635 | 1.0000 | 0.8659 | 0.5000 | 0.2365 |
| dummy | dummy_stratified | 702 | 166 | 536 | 0.6467 | 0.5067 | 0.5067 | 0.2469 | 0.2410 | 0.2439 | 0.7667 | 0.7724 | 0.7695 | 0.5067 | 0.2390 |

Confusion matrices below use rows=true Real/Fake, columns=predicted Real/Fake: [[TN, FP], [FN, TP]].

| condition | model | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| natural | tfidf_logistic | 514 | 22 | 16 | 150 |
| source_masked | tfidf_logistic | 515 | 21 | 16 | 150 |
| dummy | dummy_most_frequent | 536 | 0 | 166 | 0 |
| dummy | dummy_stratified | 414 | 122 | 126 | 40 |

Masked minus natural (paired by fold for CV; sample SD of paired differences recorded):

| scope | metric | natural | masked | masked_minus_natural | delta_sd |
| --- | --- | --- | --- | --- | --- |
| cv_mean | accuracy | 0.9555 | 0.9528 | -0.0028 | 0.0045 |
| holdout | accuracy | 0.9459 | 0.9473 | 0.0014 | nan |
| cv_mean | balanced_accuracy | 0.9341 | 0.9325 | -0.0016 | 0.0045 |
| holdout | balanced_accuracy | 0.9313 | 0.9322 | 0.0009 | nan |
| cv_mean | macro_f1 | 0.9375 | 0.9335 | -0.0040 | 0.0064 |
| holdout | macro_f1 | 0.9260 | 0.9278 | 0.0018 | nan |
| cv_mean | fake_precision | 0.9146 | 0.9024 | -0.0122 | 0.0240 |
| holdout | fake_precision | 0.8721 | 0.8772 | 0.0051 | nan |
| cv_mean | fake_recall | 0.8939 | 0.8938 | -0.0000 | 0.0107 |
| holdout | fake_recall | 0.9036 | 0.9036 | 0.0000 | nan |
| cv_mean | fake_f1 | 0.9041 | 0.8979 | -0.0062 | 0.0099 |
| holdout | fake_f1 | 0.8876 | 0.8902 | 0.0026 | nan |
| cv_mean | real_precision | 0.9677 | 0.9673 | -0.0004 | 0.0028 |
| holdout | real_precision | 0.9698 | 0.9699 | 0.0001 | nan |
| cv_mean | real_recall | 0.9742 | 0.9711 | -0.0032 | 0.0063 |
| holdout | real_recall | 0.9590 | 0.9608 | 0.0019 | nan |
| cv_mean | real_f1 | 0.9710 | 0.9692 | -0.0018 | 0.0029 |
| holdout | real_f1 | 0.9644 | 0.9653 | 0.0010 | nan |
| cv_mean | roc_auc | 0.9861 | 0.9867 | 0.0006 | 0.0010 |
| holdout | roc_auc | 0.9827 | 0.9818 | -0.0009 | nan |
| cv_mean | average_precision | 0.9511 | 0.9533 | 0.0022 | 0.0015 |
| holdout | average_precision | 0.9280 | 0.9269 | -0.0010 | nan |

Of the 60 signed top features, 1 match full tokenized source identifiers, 10 match identifier components, and 0 match the fixed crawl-artifact candidate list. Component matches are deliberately broad and can be ordinary words; they are not proof of publisher identity.

| associated_class | rank | feature | coefficient | training_document_frequency | source_identifier_match | crawl_artifact_candidate |
| --- | --- | --- | --- | --- | --- | --- |
| Fake | 1 | coronavirus | 3.9574 | 1516 | False | False |
| Fake | 2 | trump | 3.5883 | 107 | False | False |
| Fake | 3 | facebook | 3.5235 | 137 | True | False |
| Fake | 4 | òthe | 3.1391 | 13 | False | False |
| Fake | 5 | you | 3.0340 | 371 | False | False |
| Fake | 6 | says | 2.9898 | 110 | False | False |
| Fake | 7 | the | 2.7804 | 1918 | False | False |
| Fake | 8 | stimulus | 2.7761 | 14 | False | False |
| Real | 1 | and | -9.2397 | 2204 | False | False |
| Real | 2 | 2020 | -8.1817 | 629 | False | False |
| Real | 3 | disease | -6.9610 | 819 | False | False |
| Real | 4 | 19 | -6.5716 | 1987 | False | False |
| Real | 5 | april | -5.8565 | 305 | False | False |
| Real | 6 | covid | -5.6598 | 1996 | False | False |
| Real | 7 | health | -5.3675 | 719 | False | False |
| Real | 8 | of | -5.1850 | 2275 | False | False |

## Interpretation and limitations

The ground truths differ: FakeHealth uses author review rating <3 versus >=3, whereas CoAID uses its original Fake/Real dataset labels. Scores are not directly interchangeable measurements of factual truth. These are resource/text-grouped splits, not unseen-publisher tests. Resource/group leakage checks do not remove publisher, topic, collection or time-period confounding.

CoAID has a strong source/class association: the frozen modeling-ready audit reports 2,930 rows from class-exclusive source categories, 604 from mixed-class categories (561 Fake and 43 Real), and 43 without source metadata. FakeHealth also has publisher associations. High CoAID scores may reflect these collection/source regularities rather than general misinformation language. CoAID bodies are generally truncated near 500 characters; FakeHealth often contains much longer article bodies. Differences in available text, labeling and selection affect cross-dataset comparisons.

Explicit source masking removes only known string identifiers and URL hosts. Institutional mentions may be legitimate content; stylistic cues, URL paths, topic and the [SOURCE] placeholder itself remain informative. The existing dictionary was derived label-blind from both cohorts’ metadata, including held-out metadata; this is a declared transductive sensitivity transform, not a training-only source-discovery system. Masked-minus-natural differences are descriptive paired sensitivity results, not estimates of the entire publisher-confounding effect. No significance claim or arbitrary materiality cutoff is applied.

Top coefficients describe conditional associations under the fitted vocabulary and regularization. They neither establish causal linguistic mechanisms nor prove that misinformation spreads differently. The explanatory engagement findings remain frozen.

Dummy most-frequent and seed-42 stratified baselines were fitted only to each training pool; the majority baseline reflects its training majority, not an oracle choice based on held-out labels. Stratified probabilities are a single one-hot random realization. Average precision is the non-interpolated AP definition, not trapezoidal PR area. ROC-AUC/AP use Fake probabilities, and every split contains both classes.

Cross-context transfer can be a separately authorized next experiment after locking its source-only selection/refit protocol in both directions and reviewing current cross-dataset overlap safeguards. Neither dataset’s held-out result was used to choose the other dataset’s model. Differences in labels, sources and text length limit transfer interpretation; no transfer was performed here.

## Reproducibility and validation

All 250 CV fits and four final Logistic Regression fits converged: True. Original cohort/split fingerprints are unchanged; pipeline training UID digests, vocabulary digests, convergence and selected-configuration locks are saved. Automated tests cover training-only vocabulary fitting, predictor allowlisting, split identity and group separation, masking cohort parity, holdout timing, unique output keys, predictions and metric consistency.

Run `.venv/bin/python src/run_tfidf_logistic_baseline.py` (completed runs are verified and reused). Tests: `.venv/bin/python -m unittest discover -s src -p "test_tfidf_logistic_baseline.py"`. Progress: logs/tfidf_logistic_baseline.log. Primary tables: outputs/tables/tfidf_logistic_{cv_results,model_selection,holdout_results,source_mask_sensitivity,top_features}.csv. Predictions: outputs/predictions/tfidf_logistic_holdout_predictions.csv. Four figures: outputs/figures/tfidf_logistic/. Fitted models, per-fit checkpoints and protocol locks: outputs/models/tfidf_logistic/. Metadata: outputs/reports/tfidf_logistic_baseline.json.
