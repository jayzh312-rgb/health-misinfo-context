# Cross-context observed-post engagement contrast

## Primary finding

The estimated Fake/Real observed-post ratio was smaller in CoAID than in FakeHealth (standardized ratios 0.253 and 0.840, respectively; ratio-of-ratios 0.301, 95% independent overlap-cluster bootstrap percentile interval 0.154–0.645).

The primary log contrast is θ=-1.2011, with percentile 95% interval [-1.8689, -0.4382]. The C interval contains 1: False; the θ interval contains 0: False. No causal p-value is calculated.

| coaid_specification | fakehealth_ratio | coaid_ratio | ratio_of_ratios | CI_lower_ratio | CI_upper_ratio | log_ratio_of_ratios | CI_lower_log | CI_upper_log | bootstrap_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 0.8403 | 0.2528 | 0.3009 | 0.1543 | 0.6452 | -1.201 | -1.869 | -0.4382 | 0.3163 |
| B_resource_sensitivity | 0.8403 | 0.2401 | 0.2857 | 0.1541 | 0.6051 | -1.253 | -1.87 | -0.5024 | 0.2967 |
| C_high_certainty | 0.8403 | 0.2885 | 0.3433 | 0.1817 | 0.6202 | -1.069 | -1.705 | -0.4778 | 0.3535 |

## Estimand, validation, and exact uncertainty procedure

R is the model-standardized expected unique-post count under Fake divided by that under Real, averaged over all content items including zeros. CoAID retains its first-release-adjusted full logistic × zero-truncated-NB hurdle mean; FakeHealth retains its NB mean. Each context keeps its own validated reference population; this is not a pooled or common-covariate-population causal effect.

The validated within-analysis completion flag, seed 42, script hash, and every recorded input hash were checked. Each context has exactly 1,000 successful, finite positive post-ratio draws; convergence flags, standardized-mean identities, saved CI endpoints, and draw-summary counts were reconciled. The original run did not record output-file hashes, so this stage fingerprints the saved artifacts now and checks their internal consistency; it does not claim a nonexistent historical output checksum. None of the original 4,000 bootstrap models was refitted.

Treating FakeHealth and CoAID as independent samples as prespecified, form every ordered combination (i,j) of the 1,000 CoAID and 1,000 FakeHealth draws. Compute θ_ij=log(R_CoAID,i)−log(R_FH,j), then C_ij=exp(θ_ij). All 1,000,000 combinations have equal empirical weight; identical replicate numbers are not matched observations. NumPy's linear empirical quantiles supply percentile intervals. The original point estimate is the ratio of fitted ratios, NOT the bootstrap median. The Cartesian values share their marginal draws: one million combinations do not create one million independent observations or increase original bootstrap information. Reordering either marginal leaves the result unchanged.

## Quantiles and distribution shape

| coaid_specification | percentile_025_ratio | percentile_25_ratio | percentile_50_ratio | percentile_75_ratio | percentile_975_ratio | skewness | excess_kurtosis | skewness_log | excess_kurtosis_log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_primary_strict | 0.1543 | 0.2444 | 0.3163 | 0.4394 | 0.6452 | 0.7383 | 0.01596 | -0.04112 | -0.5507 |
| B_resource_sensitivity | 0.1541 | 0.2318 | 0.2967 | 0.4033 | 0.6051 | 0.8198 | 0.2221 | 0.07543 | -0.6289 |
| C_high_certainty | 0.1817 | 0.2843 | 0.3535 | 0.4416 | 0.6202 | 0.5984 | 0.2879 | -0.2302 | -0.137 |

| coaid_specification | percentile_025_log | percentile_25_log | percentile_50_log | percentile_75_log | percentile_975_log |
| --- | --- | --- | --- | --- | --- |
| A_primary_strict | -1.869 | -1.409 | -1.151 | -0.8223 | -0.4382 |
| B_resource_sensitivity | -1.87 | -1.462 | -1.215 | -0.9082 | -0.5024 |
| C_high_certainty | -1.705 | -1.258 | -1.04 | -0.8174 | -0.4778 |

Primary C has skewness 0.738: moderate right skew under the descriptive convention |skew|<0.5 small, 0.5–1 moderate, >1 strong. θ skewness is -0.041. These are moments of the empirical product distribution, not tests treating the combinations as independent observations.

Histogram/KDE figures examine both scales. KDE uses a fixed seed-42 sample of 20,000 product values for display only; all intervals/moments use all one million. The smoothing factor is multiplier ×1000^(−1/5), varied over 0.75, 1, 1.5, 2; one million is deliberately not used as an information sample size for bandwidth choice. Prominent interior density peaks use a 5%-of-maximum-density prominence threshold on a 600-point grid. This is a descriptive bandwidth check, not a reliable formal modality test; modality also need not be invariant to a log transform.

ratio: ×0.75 → 1 peaks, ×1 → 1 peaks, ×1.5 → 1 peaks, ×2 → 1 peaks; log: ×0.75 → 1 peaks, ×1 → 1 peaks, ×1.5 → 1 peaks, ×2 → 1 peaks. A shoulder or multiple visible bumps are not by themselves proof of distinct population modes. The multiplicity table below establishes resampling-mixture sensitivity without making that claim.

## Agreement across entity/version specifications

| comparison | intersection_lower | intersection_upper | intersection_over_union | relative_point_change_percent |
| --- | --- | --- | --- | --- |
| A_primary_strict versus B_resource_sensitivity | 0.1543 | 0.6051 | 0.918 | -5.042 |
| A_primary_strict versus C_high_certainty | 0.1817 | 0.6202 | 0.8932 | 14.11 |
| B_resource_sensitivity versus C_high_certainty | 0.1817 | 0.6051 | 0.9084 | 20.16 |

All A/B/C point estimates and 95% intervals are below 1. Their intervals overlap extensively; common overlap and pairwise intersection-over-union lengths are reported as descriptive interval comparisons, not tests of differences among dependent specifications. Entity/version choices change magnitude but do not reverse this qualitative observational result. B/C remain sensitivity analyses, not independent replications.

## Dominant-component influence

The removed A component has 187 entities and 36.1% of all post memberships. The retained sample has 5150 entities and 5055 intact overlap components. Its previously validated CoAID point ratio is reproduced at 0.4113. The new cross-context sensitivity gives C=0.4895 [0.3103, 0.7101], a change of +0.1886 (+62.7%) from primary A.

No validated exclusion bootstrap existed: a NEW first-release-adjusted hurdle overlap-component bootstrap was fitted only for this sensitivity. It produced 1000 successes in 1000 attempts, with 0 failed attempts. SeedSequence([42,4,attempt]) supplies a separate reproducible stream. Each replicate samples the 5,055 retained components with replacement and includes every member of each sampled component, including repeats and singletons; its empirical reference covariate mix is re-standardized. Logged failures are not silently replaced; same-draw multistart retries retain diagnostics. This changes the target population and is an influence sensitivity, NOT the new primary.

| dominant_component_multiplicity | bootstrap_draws | median_coaid_ratio | coaid_ratio_q25 | coaid_ratio_q75 | coaid_ratio_IQR | median_cross_context_C | cross_context_C_q25 | cross_context_C_q75 | cross_context_C_IQR | conditional_C_percentile_025 | conditional_C_percentile_975 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 363 | 0.4082 | 0.3586 | 0.459 | 0.1004 | 0.4825 | 0.4171 | 0.5515 | 0.1344 | 0.3041 | 0.7095 |
| 1 | 372 | 0.2486 | 0.2188 | 0.2816 | 0.06289 | 0.2951 | 0.2563 | 0.3389 | 0.08257 | 0.1929 | 0.4289 |
| 2_or_more | 265 | 0.1824 | 0.162 | 0.2134 | 0.05139 | 0.2177 | 0.1868 | 0.257 | 0.0702 | 0.1291 | 0.3311 |

Within each multiplicity class, every original A draw is combined with all 1,000 independent FakeHealth draws. The classes describe conditioning on the ORIGINAL bootstrap multiplicity; class 0 is not identical to the newly refitted exclusion bootstrap (the original draw samples K=5,056 components, rather than the exclusion sample's K=5,055). Their medians/IQRs and conditional ranges show how the dominant component shifts the primary distribution. These are neither new effects nor interchangeable alternative 95% confidence intervals. Percentile uncertainty remains sensitive to a dominant component being omitted or sampled repeatedly, even if smoothing does not establish strong multimodality.

## Tail sensitivities

| sensitivity | coaid_ratio | ratio_of_ratios | CI_lower_ratio | CI_upper_ratio | change_from_primary | relative_change_percent | uncertainty_method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_retained | 0.2528 | 0.3009 | 0.1543 | 0.6452 | 0 | 0 | Full Cartesian product of independent overlap-cluster bootstrap ratios; 1000 x 1000 empirical pairs; percentile 95% interval |
| exclude_top_1pct | 0.2865 | 0.341 | nan | nan | 0.04015 | 13.35 | Point only; no valid saved overlap-bootstrap distribution; no mixed HC0/cluster CI |
| exclude_top_5_items | 0.2611 | 0.3108 | nan | nan | 0.009904 | 3.292 | Point only; no valid saved overlap-bootstrap distribution; no mixed HC0/cluster CI |
| exclude_dominant_component | 0.4113 | 0.4895 | 0.3103 | 0.7101 | 0.1886 | 62.68 | Full Cartesian product of independent overlap-cluster bootstrap ratios; 1000 x 1000 empirical pairs; percentile 95% interval |

All-retained uses the primary empirical-product interval. Top-1% and top-five exclusions use ONLY the previously validated CoAID A point estimates divided by the unchanged full-sample FakeHealth ratio. No validated tail bootstrap exists, so their CI cells are missing; HC0 intervals are never combined with cluster intervals. Outcome-based exclusion changes the target population and remains diagnostic.

## Interpretation, reporting wording, and next stage

The direction is sufficiently consistent across the prespecified A/B/C and influence checks to report A as the primary observational dataset-context finding, WITH its cluster-percentile interval and dominant-component caveat. It is not evidence that COVID caused the difference, nor definitive proof of an intrinsic context mechanism. Dominance can undermine regular many-small-cluster bootstrap approximations; the interval is conditional on the observed graph, selected models, and observed coverage and is not a guarantee of nominal population coverage.

**Results sentence:** “The estimated Fake/Real observed-post ratio was smaller in CoAID than in FakeHealth (standardized ratios 0.253 and 0.840, respectively; ratio-of-ratios 0.301, 95% independent overlap-cluster bootstrap percentile interval 0.154–0.645).”

**Accompanying limitation sentence:** “This observational contrast cannot isolate a COVID-period effect because dataset and period/context are confounded; ground-truth definitions, engagement collection procedures, and observation windows differ, while CoAID's dominant social-overlap component, incompletely captured dependence, incomplete diffusion-cascade coverage, and missing unobserved/deleted platform activity limit interpretation.”

Additional explicit limitations: dataset and period/context are confounded; ground-truth definitions differ; collection procedures and observation windows differ; CoAID has one dominant social-overlap component; the dependence graph only partially captures social-ID overlap and other dependence; observed post counts are not complete diffusion cascades; unobserved/deleted platform activity may be missing. Cross-context sample independence is assumed, not empirically established by separate within-context graphs.

The approximately 50-percentage-point occurrence gap stays within CoAID; no engagement_present cross-context contrast is calculated because FakeHealth is near ceiling. Exploratory reply contrasts are omitted so provisional reply estimates do not enter the primary conclusion.

TF-IDF text analysis and cross-context transfer are a reasonable NEXT, separately authorized stage. Before fitting, specify leakage-safe resource/version/overlap grouping, label harmonization, and explicit within- versus cross-context evaluation; otherwise provenance duplicates and incompatible label meanings can masquerade as transfer performance. No text modeling, classification, transformers, or COVID-Rumor robustness is started here.

## Reproduction and artifacts

Run `.venv/bin/python src/run_cross_context_engagement_contrast.py --workers 4`. Validated no-dominant bootstrap draws are reused on rerun only when fingerprints match. New outputs use cross_context filenames and a separate figure directory; all raw/processed inputs and previous results remain unchanged. The metadata records input hashes, original refits reused, new sensitivity refits, pair counts, and verification status. Test with `.venv/bin/python -m unittest discover -s src -p 'test_cross_context_engagement_contrast.py'`.
