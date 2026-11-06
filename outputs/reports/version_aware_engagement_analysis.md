# Version-aware within-dataset engagement analysis

Completed overlap-aware refitting. Specification A remains the CoAID primary; B/C are sensitivity analyses. Earlier CoAID estimates are superseded for substantive interpretation and preserved unchanged for provenance.

## Samples and design

| specification | N | fake_N | real_N | posts | replies |
| --- | --- | --- | --- | --- | --- |
| A_primary_strict | 5337 | 922 | 4415 | 152223 | 120604 |
| B_resource_sensitivity | 5205 | 921 | 4284 | 152144 | 120604 |
| C_high_certainty | 5082 | 920 | 4162 | 124743 | 86165 |
| FakeHealth_HealthStory | 1690 | 472 | 1218 | 384073 | 27601 |

No entity-resolution rules, raw files, validated samples, or earlier analysis scripts/results were changed. Counts include every observation; tail exclusions are explicitly separate diagnostics. CoAID models retain the existing first-recorded-release fixed effects (May reference), plus Fake/Real; FakeHealth uses Fake/Real only. No time offset, new text controls, or causal interpretation is introduced.

The common estimand averages predicted expected post counts over the SAME empirical covariate distribution under Fake and under Real, including all zeros. For a hurdle this is the row-level product P(posts>0|X) × E(posts|posts>0,X), then averaged. It is neither the count-part coefficient nor the product of separately averaged components. Each bootstrap re-standardizes to its sampled covariate distribution; HC0/model-based intervals hold the original covariates fixed, so their width comparison is not a pure isolation of dependence.

## Primary post results

| specification | standardized_fake_mean | standardized_real_mean | standardized_ratio | primary_CI_lower | primary_CI_upper | HC0_CI_lower | HC0_CI_upper | uncertainty_widening |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 8.828 | 34.92 | 0.2528 | 0.1322 | 0.522 | 0.1724 | 0.3707 | substantially wider |
| B_resource_sensitivity | 8.67 | 36.12 | 0.2401 | 0.1328 | 0.4982 | 0.1632 | 0.3532 | substantially wider |
| C_high_certainty | 8.684 | 30.11 | 0.2885 | 0.1574 | 0.4988 | 0.1981 | 0.42 | substantially wider |
| FakeHealth_HealthStory | 199.9 | 237.9 | 0.8403 | 0.7141 | 0.9967 | 0.708 | 0.9972 | similar or narrower |

Corrected A gives R=0.2528, versus the approximate old first-release value 0.237: an absolute change of +0.0158, or +6.7%. This comparison diagnoses entity-resolution methodology only. A/B/C point estimates all remain below 1: True. These are observed-count associations, not misinformation effects or rates.

## Overlap graph and effective information

| specification | overlap_components | singleton_components | multi_entity_components | maximum_component_size | percent_entities_in_non_singletons | percent_post_memberships_in_non_singletons | largest_component_post_share | post_weight_kish_equivalent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 5056 | 4986 | 70 | 187 | 6.577 | 50.46 | 0.361 | 7.591 |
| B_resource_sensitivity | 4938 | 4882 | 56 | 187 | 6.206 | 51.89 | 0.3827 | 6.758 |
| C_high_certainty | 4905 | 4848 | 57 | 103 | 4.604 | 41.63 | 0.2592 | 14.51 |
| FakeHealth_HealthStory | 1662 | 1641 | 21 | 4 | 2.899 | 4.934 | 0.01429 | 415.2 |

Edges join entities sharing a counted post ID or a counted reply ID. Connected components include every singleton; neither graph construction nor resampling changes analytical entities. Degree counts distinct neighboring entities, separately by shared post and reply; it is not the number of shared IDs. Component tables include Fake-only, Real-only, and mixed composition.

WARNING: CoAID has thousands of nominal resampling components but one dominant component per sample. The engagement-weight Kish equivalent is a concentration diagnostic, NOT a literal number of independent clusters or a degrees-of-freedom estimate. Dominant components violate the comfortable many-small-cluster approximation: their omission/repetition can make the percentile bootstrap wide, skewed, or multimodal. Cluster labels assume no remaining between-component dependence; sharing a publisher/resource without a shared observed ID is not handled by this graph. Do not present these intervals as resolving all dependence or measurement uncertainty.

FakeHealth's existing processed provenance links to the official local engagement mapping. Counted tweet/reply sets were verified against every processed row, revealing actual overlap; retweets not included in the modeled unique_post_count were excluded from graph construction. Therefore FakeHealth also receives an evidenced overlap-component bootstrap, with existing HC0/model-based intervals retained as secondary comparisons. No FakeHealth-vs-CoAID effect contrast is computed.

## Bootstrap protocol and failures

| specification | attempts | successful_post_refits | failed_post_refits | successful_any_refits | successful_reply_refits | target_successful_post_refits | target_met | pilot_only | median_attempt_seconds | largest_component_omitted_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 1000 | 1000 | 0 | 1000 | 1000 | 1000 | True | False | 0.1225 | 0.363 |
| B_resource_sensitivity | 1000 | 1000 | 0 | 1000 | 1000 | 1000 | True | False | 0.1237 | 0.363 |
| C_high_certainty | 1000 | 1000 | 0 | 1000 | 1000 | 1000 | True | False | 0.1131 | 0.37 |
| FakeHealth_HealthStory | 1000 | 1000 | 0 | 0 | 1000 | 1000 | True | False | 0.01072 | 0.38 |

Seed 42; unstratified pairs bootstrap of K components with replacement, each selected component contributing ALL its entities once per draw occurrence. Repeated selection intentionally repeats rows; singleton components are ordinary units. Replicate seeds depend only on sample and attempt index, not worker scheduling. Each refit uses the same chosen model family/design, native likelihood and bounded convergence/Hessian checks as the old tested fitter. Warm-start failures receive a documented multistart retry on the SAME sampled data. Failed/nonconvergent attempts remain in version_aware_bootstrap_refits.csv and are excluded explicitly from percentile computation. Additional attempts toward the successful-refit target are logged, never silently substituted. All post-successful draws are retained. Occurrence and provisional reply failures are tracked separately.

The script first runs a 20-attempt pilot, reports runtime/success, then continues to the target without reducing it. Bootstrap excludes model-selection uncertainty (family held fixed), entity-resolution uncertainty (bracketed by A/B/C), and potential missing social links. 'Moderately wider' means log-CI width is 1.10–<1.50 times HC0; 'substantially wider' means >=1.50; smaller ratios are labeled similar or narrower. These are descriptive thresholds, not statistical tests.

## Model reassessment

| specification | outcome | model | AIC | convergence | stable | alpha_NB2 | observed_zero_fraction | predicted_zero_fraction | pearson_chi2_per_df | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | unique_post_count | NB2 | 2.961e+04 | True | True | 3.965 | 0.4984 | 0.4897 | 5.046 | False |
| A_primary_strict | unique_post_count | Hurdle_Logit_ZTNB2 | 2.794e+04 | True | True | 4.943 | 0.4984 | 0.4984 | 6.104 | True |
| A_primary_strict | unique_reply_count | NB2 | 1.568e+04 | True | True | 17.95 | 0.7765 | 0.7793 | 4.81 | True |
| B_resource_sensitivity | unique_post_count | NB2 | 2.888e+04 | True | True | 4.007 | 0.5005 | 0.4909 | 5.108 | False |
| B_resource_sensitivity | unique_post_count | Hurdle_Logit_ZTNB2 | 2.726e+04 | True | True | 4.862 | 0.5005 | 0.5005 | 6.566 | True |
| B_resource_sensitivity | unique_reply_count | NB2 | 1.536e+04 | True | True | 17.88 | 0.7754 | 0.7783 | 4.871 | True |
| C_high_certainty | unique_post_count | NB2 | 2.725e+04 | True | True | 4.011 | 0.5116 | 0.5007 | 4.959 | False |
| C_high_certainty | unique_post_count | Hurdle_Logit_ZTNB2 | 2.563e+04 | True | True | 4.454 | 0.5116 | 0.5116 | 5.731 | True |
| C_high_certainty | unique_reply_count | NB2 | 1.384e+04 | True | True | 18.54 | 0.7885 | 0.7916 | 4.702 | True |
| FakeHealth_HealthStory | unique_post_count | NB2 | 2.171e+04 | True | True | 1.119 | 0.004142 | 0.007109 | 2.603 | True |
| FakeHealth_HealthStory | unique_post_count | Hurdle_Logit_ZTNB2 | 2.171e+04 | True | True | 1.146 | 0.004142 | 0.004142 | 2.563 | False |
| FakeHealth_HealthStory | unique_reply_count | NB2 | 8035 | True | True | 9.258 | 0.5645 | 0.5812 | 9.405 | True |

NB2 and logistic plus genuine zero-truncated NB2 were compared on the same full post outcome/sample, counting all hurdle parameters. CoAID retains a stable hurdle only if it improves AIC over NB by >2 (or NB is unstable); this rule does not depend on the Fake/Real estimate. FakeHealth retains its requested NB primary. ZINB was unnecessary as a secondary post check because the stable hurdle/NB comparison already resolves the requested family decision. Zero agreement alone is not evidence of overall fit: a logistic intercept largely enforces aggregate zero agreement. Detailed comparison tables retain residual dispersion, gradients, Hessian checks, bound flags and optimizer warnings. Heavy-tail residual misfit remains a limitation of the working means.

## Occurrence and positive-count parts

| specification | probability_fake | probability_real | percentage_point_difference | probability_difference_cluster_lower | probability_difference_cluster_upper | odds_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 0.1174 | 0.6177 | -50.03 | -0.5329 | -0.4709 | 0.02188 |
| B_resource_sensitivity | 0.1172 | 0.6183 | -50.11 | -0.5345 | -0.4703 | 0.02216 |
| C_high_certainty | 0.1165 | 0.6089 | -49.24 | -0.5183 | -0.466 | 0.02345 |

This table is any observed post OR reply, which is the requested engagement_present outcome. The separately exported any_post rows are the hurdle's posts-only occurrence part. Both use probability differences and their own cluster intervals; odds ratios are conditional logistic associations, not marginal probability ratios. The full positive-part and combined expected-count estimates/cluster intervals are in version_aware_hurdle_components.csv. No direct occurrence comparison with near-ceiling FakeHealth is made.

## Tail and dominant-cluster sensitivity

| specification | tail_specification | removed_N | mean_ratio | CI_lower | CI_upper | fixed_original_target_ratio | stable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | all_retained | 0 | 0.2528 | 0.1724 | 0.3707 | 0.2528 | True |
| A_primary_strict | exclude_top_1pct | 54 | 0.2865 | 0.1968 | 0.4171 | 0.2864 | True |
| A_primary_strict | exclude_top_5_items | 5 | 0.2611 | 0.1786 | 0.3818 | 0.2611 | True |
| A_primary_strict | exclude_largest_overlap_cluster | 187 | 0.4113 | 0.2874 | 0.5886 | 0.4103 | True |
| B_resource_sensitivity | all_retained | 0 | 0.2401 | 0.1632 | 0.3532 | 0.2401 | True |
| B_resource_sensitivity | exclude_top_1pct | 53 | 0.2831 | 0.1943 | 0.4125 | 0.2829 | True |
| B_resource_sensitivity | exclude_top_5_items | 5 | 0.249 | 0.1699 | 0.3649 | 0.249 | True |
| B_resource_sensitivity | exclude_largest_overlap_cluster | 187 | 0.3754 | 0.261 | 0.5399 | 0.3747 | True |
| C_high_certainty | all_retained | 0 | 0.2885 | 0.1981 | 0.42 | 0.2885 | True |
| C_high_certainty | exclude_top_1pct | 51 | 0.3447 | 0.239 | 0.4971 | 0.3445 | True |
| C_high_certainty | exclude_top_5_items | 5 | 0.3006 | 0.2072 | 0.4363 | 0.3006 | True |
| C_high_certainty | exclude_largest_overlap_cluster | 103 | 0.3909 | 0.273 | 0.5597 | 0.3905 | True |

Every required A/B/C tail-exclusion point estimate remains below 1: True. This preserves the direction, not a claim of overlap-adjusted statistical significance for each exclusion fit. Removing the largest component changes magnitude more than removing five extreme items.

Top 1% removes exactly ceil(0.01N) rows, sorted by descending unique posts then ascending UID to resolve ties; top five removes exactly five. All-retained remains the primary. Exclusion fits use the selected full-sample family and standardize to retained items; the fixed-original-target column isolates the fitted-parameter change from the reference-mixture change. Neither is an unbiased recovery of the untrimmed target after outcome-dependent selection. Exclusion intervals shown here are HC0 only and must NOT be interpreted as overlap-adjusted confirmation. Directional tail robustness is a point-estimate statement. Removing the largest engagement component is an additional influence diagnostic, not a revised primary.

## Replies remain secondary

| specification | standardized_ratio | primary_CI_lower | primary_CI_upper | stable | successful_refits |
| --- | --- | --- | --- | --- | --- |
| A_primary_strict | 0.4197 | 0.134 | 1.49 | True | 1000 |
| B_resource_sensitivity | 0.4042 | 0.1417 | 1.335 | True | 1000 |
| C_high_certainty | 0.5158 | 0.179 | 1.367 | True | 1000 |
| FakeHealth_HealthStory | 0.8696 | 0.2902 | 2.191 | True | 1000 |

All CoAID provisional reply NB fits are numerically stable: True; every CoAID reply cluster interval includes 1: True. Dominant-component exclusions also change reply magnitudes appreciably. The estimates are usable as provisional descriptive associations but not stable/precise enough for a headline Fake/Real reply claim.

The prior provisional NB was refitted, with alpha/Hessian/convergence recorded and overlap-bootstrap failures tracked. Numerical estimability is not substantive stability or validation of a full reply model. The tail table includes reply NB refits; inspect whether magnitudes and directions change and whether intervals cross 1. No reply-hurdle re-selection or dispersion-profile audit is claimed here. Reply estimates therefore remain provisional and cannot displace posts as the headline outcome.

## Carry-forward decision

Carry A's all-item standardized post ratio with its overlap-cluster percentile interval as the PRIMARY CoAID estimate, retaining the dominant-cluster and release-coding caveats. B/C, tail exclusions, leave-largest-component-out, HC0/model-based intervals, and provisional replies are sensitivity/secondary results. The large occurrence gap should remain a separate within-CoAID finding, not a between-context binary contrast.

The corrected within-dataset stage provides the inputs for planning a formal common-estimand cross-context analysis, but not an unconditional green light for a definitive contrast: pre-specify inference that acknowledges the dominant social component and the sensitivity-defined populations, and carry the documented uncertainty forward. No cross-context ratio-of-ratios, pooled model, or formal between-dataset comparison was calculated.

## Reproduction

Run `.venv/bin/python src/run_version_aware_engagement_analysis.py --successes 1000 --workers 4`; `--pilot` runs 20 attempts, and `--resume` reuses logged draws only when script and input fingerprints match. Full attempt records, pilot timing, sample checks, cluster assignments/composition, descriptive effects, model comparisons, occurrence results, tail sensitivities, coefficients and plots are saved under the version-aware filenames. Previous CoAID results remain archived in their original files and are explicitly superseded here.
