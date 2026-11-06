# Manuscript results synthesis

Authoritative synthesis of the validated final aggregate artifacts. Natural combined-text LR is the primary text baseline; Linear SVM is replication. Title, publisher and transformer experiments are sensitivities. The independent COVID target supplies descriptive external validation. Earlier reports’ prospective next-stage instructions are historical and do not govern the current status.
All numbers below are copied from final tables, with source-row/field provenance in [the numeric ledger](../tables/manuscript_numeric_provenance.csv). No models, bootstrap resampling or new statistical tests were run. Text point estimates, fold variability and seed variability are not confidence intervals.

## A. Research question

How does the difference between misinformation and real-news observed engagement/sharing differ between the COVID-focused CoAID context and the non-COVID FakeHealth HealthStory comparator? The text branch asks whether learned Fake/Real associations generalize across these contexts, with independent COVID-Rumor NEWS used to assess greater portability across COVID-focused datasets. Dataset identity is inseparable from period/context in the primary comparison.

## B. Engagement findings

The standardized Fake/Real expected observed-post ratio is 0.253 in CoAID and 0.840 in FakeHealth. Their ratio-of-ratios is 0.301 (95% independent overlap-cluster bootstrap percentile interval 0.154–0.645). Thus the relative Fake disadvantage is substantially larger in CoAID. This is an observational cross-dataset contrast. [cross_context_engagement_contrast.csv](../tables/cross_context_engagement_contrast.csv)
The estimand is the ratio of model-standardized expected unique observed-post counts, including zero-count items. CoAID uses its validated first-release-adjusted hurdle mean and FakeHealth its negative-binomial mean. Each keeps its own reference population; the estimates do not compare a shared covariate population and are not complete diffusion cascades.
The B and C specification ratios-of-ratios are 0.286 and 0.343; both saved intervals remain below one. Excluding the dominant CoAID overlap component changes the ratio-of-ratios to 0.489 [0.310, 0.710]. Direction persists, but magnitude and uncertainty are sensitive to that component. This exclusion changes the target population and remains a sensitivity. [cross_context_sensitivity_results.csv](../tables/cross_context_sensitivity_results.csv)
Saved uncertainty assumes independent samples across datasets and resamples each dataset’s observed overlap components. Incomplete dependence capture and a dominant component limit nominal coverage. No bootstrap was rerun for this synthesis.

## C. Text within-context findings

CoAID is much easier than FakeHealth under the validated within-context protocols. LR and Linear SVM reproduce this pattern. These are reserved resource/group holdouts; evaluation metadata, engagement, IDs and labels were not predictors.

| Model | Context | N | Macro F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| LR | CoAID | 702 | 0.9260 | 0.9827 |
| LR | FakeHealth | 330 | 0.6752 | 0.7701 |
| Linear SVM | CoAID | 702 | 0.9128 | 0.9795 |
| Linear SVM | FakeHealth | 330 | 0.6697 | 0.7695 |

[tfidf_logistic_holdout_results.csv](../tables/tfidf_logistic_holdout_results.csv); [tfidf_linear_svm_holdout_results.csv](../tables/tfidf_linear_svm_holdout_results.csv)
The easier dataset is not evidence of intrinsically easier COVID veracity recognition: labels, sources, topic and collection differ.

## D. Cross-context transfer findings

Transfer degrades strongly when compared with target-trained models on the identical target holdout rows. Full-source transfer fits include the former source holdout; the target vocabulary and threshold remain fixed by the source protocol. All-eligible scores are supplementary and must not be substituted for these matched-holdout comparisons.

| Model | Direction | Within target F1 | Transfer F1 | Saved F1 change | Transfer ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| LR | FakeHealth → CoAID | 0.9260 | 0.4290 | -0.4970 | 0.5705 |
| LR | CoAID → FakeHealth | 0.6752 | 0.4507 | -0.2244 | 0.6191 |
| Linear SVM | FakeHealth → CoAID | 0.9128 | 0.4245 | -0.4883 | 0.5827 |
| Linear SVM | CoAID → FakeHealth | 0.6697 | 0.4493 | -0.2205 | 0.6313 |

[tfidf_logistic_transfer_results.csv](../tables/tfidf_logistic_transfer_results.csv); [tfidf_linear_svm_transfer_results.csv](../tables/tfidf_linear_svm_transfer_results.csv); [tfidf_logistic_transfer_degradation.csv](../tables/tfidf_logistic_transfer_degradation.csv); [tfidf_linear_svm_transfer_degradation.csv](../tables/tfidf_linear_svm_transfer_degradation.csv)
The absolute drop is asymmetric because the target-trained baselines differ greatly. Both classical models overpredict Fake in FakeHealth-to-CoAID transfer and underpredict it in CoAID-to-FakeHealth transfer. These descriptive patterns do not identify the mechanism of domain shift.

**Title-only sensitivity.** Small macro-F1 gains do not provide a consistent rescue: ROC-AUC worsens in both directions on identical title-eligible target rows.

| Direction | N | Title F1 | Combined F1, same rows | Title ROC-AUC | Combined ROC-AUC, same rows |
| --- | --- | --- | --- | --- | --- |
| FH_to_CA | 701 | 0.4286 | 0.4281 | 0.4742 | 0.5707 |
| CA_to_FH | 330 | 0.4636 | 0.4507 | 0.4265 | 0.6191 |

[tfidf_logistic_title_vs_combined.csv](../tables/tfidf_logistic_title_vs_combined.csv)
Matched target rows avoid cohort mismatch in these scores, but source training cohorts and tuned configurations still differ. This is a representation-protocol sensitivity, not an isolated body-length effect.

## E. Lexical-instability evidence

Across 4,174 shared unigrams, signed LR coefficients have Pearson correlation -0.0148 and Spearman correlation -0.0034. Signs reverse for 48.39% of shared features. Among the top 100 by the prespecified maximum absolute coefficient, 48.0% reverse direction. Important shared words therefore often change association direction. [text_coefficient_alignment_summary.csv](../tables/text_coefficient_alignment_summary.csv)
This audit concerns the frozen within-context training-partition models, not the later full-source external-transfer fits. Positive coefficients indicate Fake association and negative coefficients Real association. Different vocabularies, IDF scales, correlated features and regularization condition these coefficients. Weak alignment is consistent with limited portability, but does not establish that coefficient reversals caused transfer degradation. No individual word is a general misinformation cue.

## F. Publisher/source sensitivity

FakeHealth retains modest predictive association for unseen publisher families: outer-fold mean macro F1 0.6196 (sample SD 0.0210; range 0.6030–0.6432); pooled macro F1 0.6209. Mean outer-fold ROC-AUC is 0.6995. These are descriptive fold summaries, not uncertainty intervals. [fakehealth_publisher_heldout_summary.csv](../tables/fakehealth_publisher_heldout_summary.csv)
The saved comparison gives publisher outer-mean macro F1 0.6196 versus selected resource-grouped CV 0.6598, a difference of -0.0402. This is unpaired: grouping, training/tuning populations and evaluation designs differ. [fakehealth_publisher_heldout_baseline_comparison.csv](../tables/fakehealth_publisher_heldout_baseline_comparison.csv)
Source masking leaves pooled macro F1 essentially unchanged (0.6209 masked versus 0.6209 natural). [fakehealth_publisher_heldout_results.csv](../tables/fakehealth_publisher_heldout_results.csv)
Known source strings alone do not account for all surviving FakeHealth signal. Masking is a limited, label-blind transductive sensitivity; it does not remove style/topic or all publisher information. Publisher families are observed proxies, and this experiment does not establish unseen-publisher generalization for CoAID.

## G. Transformer robustness

The primary DistilRoBERTa experiment uses the original 512-token, seed-42 protocol. It reproduces the easier within-context CoAID pattern but offers no consistent transfer improvement.

| Direction | Transformer F1 | Transformer ROC-AUC | Transformer AP | Within-target transformer F1 |
| --- | --- | --- | --- | --- |
| FakeHealth → CoAID | 0.3956 | 0.4415 | 0.2009 | 0.9387 |
| CoAID → FakeHealth | 0.4638 | 0.4371 | 0.2611 | 0.6212 |

[distilroberta_transfer_results.csv](../tables/distilroberta_transfer_results.csv); [distilroberta_holdout_results.csv](../tables/distilroberta_holdout_results.csv)
The modest CoAID-to-FakeHealth macro-F1 gain over both classical comparators coexists with lower ROC-AUC and AP. FakeHealth-to-CoAID macro F1 is also lower than both classical comparators. A smaller transfer-minus-within drop can partly reflect a weaker within-target baseline and should not be read as stronger generalization by itself.
**Token budget and seeds.** The paired FakeHealth-to-CoAID follow-up qualifies the original shorter-cap result.

| Metric | Mean 256−512 | Sample SD | Range | Positive seeds / tested |
| --- | --- | --- | --- | --- |
| macro_f1 | +0.1011 | 0.0790 | +0.0238 to +0.1816 | 3 / 3 |
| roc_auc | +0.0489 | 0.1348 | -0.0297 to +0.2045 | 1 / 3 |
| average_precision | +0.0008 | 0.0812 | -0.0611 to +0.0926 | 1 / 3 |

[distilroberta_budget_seed_summary.csv](../tables/distilroberta_budget_seed_summary.csv); [distilroberta_budget_seed_paired_deltas.csv](../tables/distilroberta_budget_seed_paired_deltas.csv)
Macro F1 improves at the shorter cap across the tested seeds, while ROC-AUC and AP improve only for the original seed and decline for the other seeds. Lower predicted Fake prevalence can improve threshold behavior without improving ranking. The evidence does not support a stable claim that reducing source representation improves transfer generally. The follow-up covers one transfer direction and is not population uncertainty estimation.
At the primary cap, 71.22% of FakeHealth texts truncate versus 0% of CoAID; at the shorter cap, FakeHealth truncation is 87.86%. CoAID target token content is unchanged across the paired caps, but source budget and optimization vary together. [transformer_truncation_audit.csv](../tables/transformer_truncation_audit.csv)

## H. Independent COVID-context external validation

The clean COVID-Rumor NEWS target contains 3,667 verified items: 3,015 Fake and 652 Real. Exact source-overlap rules remove 33 otherwise eligible items. Only one statement per NEWS item is scored; Unverified entries and the Twitter subset are excluded. The models and source vectorizers are reused unchanged, with target/prediction freezes preceding outcome-label evaluation. [covid_rumor_external_cohort_summary.csv](../tables/covid_rumor_external_cohort_summary.csv)

| Source/model | Macro F1 | Balanced accuracy | ROC-AUC | Predicted Fake % |
| --- | --- | --- | --- | --- |
| CoAID LR | 0.5580 | 0.5529 | 0.6995 | 92.75 |
| FakeHealth LR | 0.4969 | 0.5021 | 0.5219 | 89.61 |
| CoAID SVM | 0.5314 | 0.5349 | 0.6816 | 94.38 |
| FakeHealth SVM | 0.4973 | 0.5059 | 0.5248 | 91.93 |

[covid_rumor_external_transfer_results.csv](../tables/covid_rumor_external_transfer_results.csv)
Actual Fake prevalence is 82.22%. Both classifiers favor CoAID-source transfer across the reported comparison metrics, with the largest separation evident in ranking. Threshold performance remains modest, and every model overpredicts Fake.
CoAID-minus-FakeHealth macro-F1 differences are +0.0611 for LR and +0.0342 for SVM; ROC-AUC differences are +0.1776 and +0.1568, respectively. These are descriptive differences, not statistical winner declarations. [covid_rumor_external_source_comparison.csv](../tables/covid_rumor_external_source_comparison.csv)
Token-occurrence coverage is 89.47% for CoAID and 81.87% for FakeHealth, identical across LR/SVM within each source. Every model has zero zero-feature rows. Higher coverage may contribute to the advantage; nonempty vectors alone do not establish semantic compatibility. [covid_rumor_external_vocabulary_diagnostics.csv](../tables/covid_rumor_external_vocabulary_diagnostics.csv)
All-eligible and approximate-overlap-excluded sensitivities retain the qualitative source ordering. This result is consistent with greater portability of linguistic Fake/Real associations across independent COVID-focused information environments. It does not identify a causal sensitive-period effect. Short-statement genre, label construction, source composition, high class imbalance and incomplete overlap/language screening remain limitations.

## I. Integrated interpretation

Observed Fake/Real engagement contrasts differ substantially across the two main dataset contexts, while learned linguistic associations also show limited portability across those contexts. Independent COVID external validation provides evidence that some textual associations are more portable within COVID-focused datasets than from the non-COVID health comparator. Together, the findings support a context-dependent interpretation of observed misinformation engagement/sharing and linguistic signals.
The engagement and text branches provide complementary evidence; no analysis links coefficient changes causally to post counts or tests mediation. Dataset identity, period/context, source composition, label construction, genre and collection procedure remain confounded.
**Consistency assessment.** No unresolved contradiction appears among the validated headline findings. Strong within-context classification and weak cross-context transfer answer different questions. Better within-COVID transfer is compatible with weak CoAID–FakeHealth transfer. Apparent disagreements concern metric choice: title-only and transformer sensitivities can improve macro F1 while worsening ranking. Dominant-component exclusion changes engagement magnitude without reversing direction. The latest seed follow-up qualifies the original single-seed budget interpretation; it does not replace the original primary protocol.
**Readiness to draft.** No additional analysis is required for the bounded observational argument. A causal period claim, a broad source-independent mechanism or a deployment claim would require a different design. Preserve those boundaries rather than adding models without a manuscript gap.

## J. Limitations

[Full limitations table](../tables/manuscript_limitations.csv). Codes link directly to the master results table.

| Code | Limitation | Residual concern |
| --- | --- | --- |
| L01 | Dataset/context confounding | Period/context, dataset, collection procedure and source composition remain inseparable. |
| L02 | Label-definition mismatch | FakeHealth expert-review quality framework, CoAID construction/veracity labels and COVID-Rumor fact-check-associated veracity are not identical ground truth. |
| L03 | Incomplete diffusion/network coverage | Counts do not recover complete cascades, exposures, network topology or population reach. |
| L04 | CoAID dependence and dominant overlap component | Graph misses other dependence; large component challenges many-small-cluster approximations; exclusion changes population; cross-dataset independence is assumed. |
| L05 | Source/publisher confounding | Style/topic remain; publisher families are proxies; masking uses label-blind evaluation metadata; no comparable unseen-publisher validation of CoAID. |
| L06 | Different article/text lengths | Title-only changes training cohorts and tuned configurations; full-text TF-IDF and truncated encoders see different representations. |
| L07 | Transformer truncation asymmetry | Source-budget changes interact with optimization; shorter cap is not a matched-information correction. |
| L08 | COVID-Rumor short-statement genre | Cross-dataset portability cannot isolate COVID vocabulary from genre, fact-check framing or collection. |
| L09 | Target class imbalance | Fixed thresholds are not calibrated to external target prevalence; no claim of deployment readiness. |
| L10 | Historical platform missingness/deletions | Deleted, private, uncaptured and unavailable activity cannot be recovered from these artifacts. |
| L11 | Non-causal observational interpretation | Text findings do not establish mediation or explain the engagement contrast. |
| L12 | Conditional coefficient comparability | Opposite coefficients are associations; correlations do not identify causes of transfer failure or causal words. |
| L13 | Non-equivalent sensitivity evaluation designs | Publisher outer CV differs from original holdout/CV; pooled scores combine models; fold SD is not a CI. |
| L14 | Limited transformer seed coverage | Limited seeds and one direction; primary experiment otherwise single-seed; MPS not asserted bitwise reproducible; no population uncertainty claim. |
| L15 | Residual external-target overlap/language uncertainty | Offline URLs cannot resolve all redirects; conservative normalization misses some equivalent text; English script is not full language identification. |

## K. Claims that ARE supported

- **C01:** The standardized Fake/Real observed-post ratio is lower in CoAID than FakeHealth, indicating a larger relative Fake disadvantage in the CoAID dataset context.
- **C02:** Both LR and Linear SVM achieve substantially higher within-context macro F1 in CoAID than FakeHealth.
- **C03:** Both classical classifiers show substantial macro-F1 degradation when trained in the other dataset context.
- **C04:** Shared-unigram LR coefficients are approximately uncorrelated, and sign reversals remain common among highly ranked shared terms.
- **C05:** Title-only transfer shows small macro-F1 gains but poorer ROC-AUC in both directions, providing no consistent rescue.
- **C06:** FakeHealth retains modest predictive signal across unseen publisher families, although performance is lower than the resource-grouped baseline.
- **C07:** Similar pooled natural and masked publisher-held-out macro F1 suggests known explicit identifiers alone do not account for the surviving FakeHealth signal.
- **C08:** The primary transformer offers no consistent transfer benefit: a modest macro-F1 gain in one direction coexists with lower ROC-AUC and AP in both directions.
- **C09:** The shorter cap improves macro F1 across tested seeds, but ROC-AUC and AP gains occur only for the original seed; the joint improvement is not stable.
- **C10:** Both LR and SVM show greater CoAID-source than FakeHealth-source portability to independent COVID-Rumor NEWS, especially in ROC-AUC.
- **C11:** No clean target row has zero TF-IDF features; CoAID has somewhat higher token coverage, which may contribute to its transfer advantage.
- **C12:** Observed Fake/Real engagement contrasts and the portability of learned linguistic associations vary across dataset contexts; independent COVID validation is consistent with greater portability within COVID-focused environments.

## L. Claims that are NOT supported

- COVID caused the observed engagement contrast or the lexical differences.
- The design identifies a causal sensitive-period effect, a universal vocabulary or causal signaling words.
- Observed post counts represent complete cascades or all exposure and diffusion.
- Lexical coefficient reversals explain or mediate the engagement contrast.
- Publisher confounding is eliminated, or FakeHealth publisher results validate CoAID on unseen publishers.
- Title-only models never improve any score, or transformers consistently improve transfer.
- Shorter transformer inputs improve ranking across seeds, or seed/fold SDs are confidence intervals.
- The independent COVID result establishes a statistical winner or deployment-ready detection.

## Validation note

Final machine-readable tables take precedence over rounded prose. Text artifacts were checked against saved validation fingerprints, and source/output integrity was rechecked after synthesis. Engagement completion metadata and saved validation flags were checked, and the final contrast tables were reconciled with their report and algebraic identities. That engagement stage did not save final-output hashes; the new synthesis fingerprint records current integrity without claiming a historical checksum. No prohibited raw data, notebooks, model checkpoints, prediction files or training logs were opened.
