# Main-paper figure and table plan

Five main-paper items are proposed. No new figures were generated. Existing figures are candidates for reuse; confirm labels and accessibility during manuscript layout. Do not recompute estimates or add uncertainty bars absent from validated artifacts.

## 1. Figure: standardized observed-post ratios and cross-context contrast

Use `outputs/figures/cross_context_engagement/within_and_cross_context_forest.png` as the existing candidate. Emphasize the primary A comparison; label B/C and dominant-component exclusion as sensitivities if included. Source: `cross_context_engagement_contrast.csv`, `cross_context_sensitivity_results.csv`, and the existing figure's validated within-context endpoints.

Caption: model-standardized expected unique observed-post ratios, including zeros; separate dataset reference populations; saved independent overlap-cluster percentile intervals. Include the null ratio line and dominant-component caveat. Do not label the estimand complete diffusion or a causal COVID effect. Existing display selection, not a new analysis.

## 2. Figure: LR/SVM within-context versus cross-context transfer

Plan matched panels by target context, with model-specific within-target and transfer macro F1; add ROC-AUC as a second panel only if legible. Use natural combined text and identical reserved target holdouts. Sources: `tfidf_logistic_holdout_results.csv`, `tfidf_logistic_transfer_results.csv`, `tfidf_linear_svm_holdout_results.csv`, `tfidf_linear_svm_transfer_results.csv`.

Existing candidates: `outputs/figures/tfidf_linear_svm/within_macro_f1.png` and `transfer_macro_f1.png`. A combined publication layout is planned, not created. Do not compare all-eligible transfer scores to holdout baselines or add significance markers.

## 3. Figure: lexical coefficient alignment and sign reversal

Reuse candidates `outputs/figures/text_feature_alignment/shared_unigram_coefficient_scatter.png` and `sign_agreement_by_importance.png` as panels in one figure. Aggregate source: `text_coefficient_alignment_summary.csv`.

Caption: shared unigrams in natural within-context LR models; positive is Fake association; ranks/cutoffs prespecified; different conditional feature sets, IDF scales and regularization. Individual labels illustrate conditional associations and must not be presented as causal or general cues. No new coefficient inspection is required.

## 4. Figure: independent COVID-Rumor NEWS transfer

Plan LR/SVM source comparisons on the identical clean NEWS target, with macro F1 and ROC-AUC panels. Include target prevalence, predicted Fake prevalence and a compact coverage/zero-feature annotation. Sources: `covid_rumor_external_transfer_results.csv`, `covid_rumor_external_source_comparison.csv`, `covid_rumor_external_vocabulary_diagnostics.csv`, `covid_rumor_external_cohort_summary.csv`.

No existing final external-transfer figure was produced, so creation is deferred to manuscript layout. Caption: descriptive external validation, exact-overlap exclusions, natural target prevalence, fixed source thresholds, short-statement/label/genre caveats. No confidence bars or winner labels.

## 5. Table: compact robustness summary

Use selected rows of `manuscript_key_results.csv`: title-only matched-target changes; publisher-held-out outer mean and fold SD; primary transformer transfer; paired budget/seed variability; engagement specification/dominant-component sensitivity. Show purpose, evaluation scope, key result and residual limitation. Fold/seed SD and range are variability, not confidence intervals. Report threshold-versus-ranking disagreements explicitly; the seed follow-up governs the budget interpretation.

The full key-results, claim-evidence and limitations tables remain drafting/supplement resources, not additional main-paper displays. Put exhaustive class metrics and protocol detail in the supplement if needed.
