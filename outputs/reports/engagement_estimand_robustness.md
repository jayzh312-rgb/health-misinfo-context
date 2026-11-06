# Common engagement estimand and robustness audit

The common primary estimand is the fake/real ratio of marginal expected **unique post counts across all content items, including zeros**. Within each dataset, predict every retained observation under each label, hold its other covariates fixed, average those predictions, and divide. These are descriptive model-based associations. There is no cross-dataset ratio-of-ratios or interaction in this audit.

FakeHealth uses NB2. CoAID uses the FULL logistic + zero-truncated NB2 hurdle: E[Y|X] = P(Y>0|X) × E[Y|Y>0,X]. The script verifies this against native zero-truncated NB predictions. Neither the positive conditional mean nor the exponentiated positive coefficient is substituted for the overall mean. The built-in HurdleCountModel is not used.

95% intervals use the existing entity/item-level HC0 score-sandwich covariance and delta method, on log scale for positive means/ratios and identity scale for differences. No bootstrap is used (0 resamples); seed 42 is retained. Covariates are treated as a fixed empirical target. Intervals do not include model/release selection, trimming-threshold uncertainty, shared-post dependence or entity-resolution uncertainty.

## Primary all-item estimates

| dataset | N | standardized_fake_mean | standardized_real_mean | mean_difference | mean_ratio | CI_lower | CI_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth HealthStory | 1690 | 199.9 | 237.9 | -38 | 0.8403 | 0.708 | 0.9972 |
| CoAID News | 5073 | 8.757 | 36.95 | -28.2 | 0.237 | 0.1481 | 0.3792 |

The previously selected first-release CoAID specification remains the untrimmed primary reference for this audit. Its status is a working reference, not a claim that first release fully captures the observation process. All raw and processed inputs are unchanged.

## Release provenance and representation

CoAID has 4,873 single-release and 200 multi-release entities (3.94%). Multi-release entities comprise 4 fake and 196 real items and contribute 33.44% of all posts. Their mean is 251.90, versus 20.58 for single-release entities.

Counts are unions of unique post IDs over all source records in an entity, not counts observed solely at the first or last release. Distinct-release count is different from number of source records. Release labels are collection provenance, not observation-duration offsets.

| release_coding | N | standardized_fake_mean | standardized_real_mean | mean_difference | mean_ratio | CI_lower | CI_upper | stable | unsupported_profile_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_release | 5073 | 8.757 | 36.95 | -28.2 | 0.237 | 0.1481 | 0.3792 | True | 0 |
| last_release | 5073 | 9.142 | 34.84 | -25.7 | 0.2624 | 0.1774 | 0.3881 | True | 0 |
| membership_indicators | 5073 | 47.13 | 46.99 | 0.1448 | 1.003 | 0.6798 | 1.48 | True | 0.01616 |
| single_release_only | 4873 | 8.976 | 24.89 | -15.92 | 0.3606 | 0.2499 | 0.5203 | True | 0 |
| no_release_adjustment | 5073 | 11.56 | 33.62 | -22.06 | 0.344 | 0.2245 | 0.527 | True | 0 |

Membership indicators best preserve provenance for the full entity-resolved sample: they retain every recorded release without duplicating entities. However, only four fake entities span releases, and some exact profiles have no fake support. They are therefore not a defensible definitive adjustment for a precise all-item contrast here. First release describes entry cohort and is the least ambiguous single-date working label; last release describes last recorded membership, not necessarily comparable exposure. Neither models the full unioned capture process. Single-release-only analysis gives the cleanest release assignment and best overlap, but explicitly changes the target population and cannot silently replace the full-sample estimand. No-release adjustment is an unadjusted diagnostic. There is no fully satisfactory release representation for the current full-sample headline estimate; AIC cannot settle this provenance/overlap issue.

Original May/July News records have maxima of 500 observed posts, with many exactly at 500; September/November maxima are much smaller. This is an empirical collection-pattern warning, not a verified claim about a platform/API cap. Unioning multiple records can exceed the per-record maximum.

## High-count entities and tail sensitivity

The largest CoAID entity is `coaid_news_entity_518045a5c665be041af4c5bb` (real): 18,428 posts, 12.23% of the dataset total, combining 110 source records across 4 releases. Its existing status is `clear_collapsed+possible_probable duplicate`. This unusually large connected component warrants targeted entity-resolution review; the audit does not change its classification or remove it from primary results.

| dataset | tail | veracity | N | group_tail_fraction | engagement_share_dataset |
| --- | --- | --- | --- | --- | --- |
| FakeHealth HealthStory | top_20 | fake | 5 | 0.01059 | 0.03048 |
| FakeHealth HealthStory | top_20 | real | 15 | 0.01232 | 0.1224 |
| FakeHealth HealthStory | top_1pct | fake | 4 | 0.008475 | 0.02667 |
| FakeHealth HealthStory | top_1pct | real | 13 | 0.01067 | 0.1149 |
| FakeHealth HealthStory | top_5pct | fake | 23 | 0.04873 | 0.07217 |
| FakeHealth HealthStory | top_5pct | real | 62 | 0.0509 | 0.2365 |
| FakeHealth HealthStory | top_5_items | fake | 0 | 0 | 0 |
| FakeHealth HealthStory | top_5_items | real | 5 | 0.004105 | 0.06384 |
| CoAID News | top_20 | fake | 0 | 0 | 0 |
| CoAID News | top_20 | real | 20 | 0.004796 | 0.207 |
| CoAID News | top_1pct | fake | 0 | 0 | 0 |
| CoAID News | top_1pct | real | 51 | 0.01223 | 0.3102 |
| CoAID News | top_5pct | fake | 31 | 0.03433 | 0.05927 |
| CoAID News | top_5pct | real | 223 | 0.05348 | 0.6441 |
| CoAID News | top_5_items | fake | 0 | 0 | 0 |
| CoAID News | top_5_items | real | 5 | 0.001199 | 0.1476 |

Ranked tails are pooled within each dataset, not selected separately within fake/real groups. Top 1%/5% use ceil(N×p); count ties are resolved by ascending UID. The influential-entity table contains the union of top-20 and top-5% records, with top-1%, top-5-items and top-20 flags and per-item engagement contributions. Top 5% is described but is not trimmed from the primary analysis.

| dataset | tail_specification | N | mean_ratio | CI_lower | CI_upper | relative_ratio_change | change_classification | point_estimate_crosses_1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth HealthStory | all_retained | 1690 | 0.8403 | 0.708 | 0.9972 | 0 | small | False |
| FakeHealth HealthStory | exclude_top_1pct | 1673 | 0.8816 | 0.7817 | 0.9943 | 0.04921 | small | False |
| FakeHealth HealthStory | exclude_top_5_items | 1685 | 0.9142 | 0.7799 | 1.072 | 0.08797 | small | False |
| CoAID News | all_retained | 5073 | 0.237 | 0.1481 | 0.3792 | 0 | small | False |
| CoAID News | exclude_top_1pct | 5022 | 0.3641 | 0.2541 | 0.5217 | 0.5364 | substantial | False |
| CoAID News | exclude_top_5_items | 5068 | 0.2852 | 0.1961 | 0.4148 | 0.2036 | moderate | False |
| CoAID News | exclude_duplicate_candidates | 4875 | 0.3442 | 0.2391 | 0.4954 | 0.4524 | substantial | False |

Under the primary coding, neither dataset's point estimate crosses 1 after either required tail exclusion. FakeHealth moves from 0.840 to 0.882 (top 1%) or 0.914 (five largest); the five-largest exclusion CI includes 1, so confidence in a non-null difference is not fully tail-robust. CoAID moves from 0.237 to 0.364 or 0.285. Duplicate exclusion gives 0.344, a 45.2% relative increase: direction persists but magnitude changes substantially.

CoAID's top 20 and exact-size top 1% are all real items. FakeHealth's top 20 include 15 real and five fake items; account for its larger real-group sample size when interpreting this imbalance. The group-tail fractions and group-specific engagement shares in the concentration table make that comparison explicit.

Small/moderate/substantial denotes <10%, 10–<25%, or >=25% absolute relative change in the ratio. This descriptive classification is separate from crossing 1 or a confidence interval including 1. Each main row standardizes over its retained sample. Additional fixed_original_target columns standardize over the original eligible covariate mix, separating parameter changes from target-composition changes. Neither quantity recovers the untrimmed population mean after outcome-dependent removal; tail exclusions are influence diagnostics only.

An additional top-1%-with-boundary-ties specification checks dependence on arbitrary tie breaking. Tail and duplicate checks are repeated under first, last, membership, and no-release codings in the CSV, not only under the primary coding.

CoAID has 57 items tied at the 500-post cutoff; excluding every item at or above it removes 84 rather than 51 items and gives a primary-coding ratio of 0.355. The membership model's baseline ratio is 1.003; tail exclusion can move that point estimate below 1, but its intervals include 1 throughout. Thus the statement that tail removal preserves direction applies to the primary coding, not uniformly to every release representation.

## Engagement presence stays separate

| tail_specification | N | probability_fake | probability_real | probability_difference | odds_ratio | odds_ratio_CI_lower | odds_ratio_CI_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_retained | 5073 | 0.1159 | 0.6126 | -0.4967 | 0.02248 | 0.01788 | 0.02825 |
| exclude_top_1pct | 5022 | 0.1156 | 0.6083 | -0.4928 | 0.02303 | 0.01832 | 0.02895 |
| exclude_top_5_items | 5068 | 0.1159 | 0.6122 | -0.4963 | 0.02253 | 0.01792 | 0.02832 |
| exclude_top_1pct_with_boundary_ties | 4989 | 0.1126 | 0.6056 | -0.493 | 0.02269 | 0.01802 | 0.02857 |
| exclude_duplicate_candidates | 4875 | 0.1127 | 0.5994 | -0.4866 | 0.02331 | 0.0185 | 0.02938 |

The binary outcome uses any observed posts OR replies and is distinct from the hurdle's posts-only positive indicator. Its fitted label association depends on zero/nonzero status, not positive-count magnitude. Removing a few high-count entities removes a few engaged observations, but does not erase the broad occurrence gap. The tail and duplicate presence refits directly check this.

FakeHealth has only seven unengaged items (one fake, six real), with 99.79% fake and 99.51% real engagement. Both groups are at the probability ceiling (equivalently, zero frequency is at its floor). This supplies little information about label differences, making a direct between-dataset presence comparison unstable and qualitatively mismatched. No FakeHealth logistic model or between-dataset binary contrast is fitted.

## Carry-forward decision

The common all-item post-count estimand is now defined and implemented consistently. The numerical first-release CoAID ratio is not ready to be treated as a release-invariant headline result: evaluate its spread across the release and tail tables, not just its primary CI. The membership coding's overlap/extrapolation problem is part of the finding, not a reason to suppress that result. FakeHealth post-count direction and CoAID occurrence association can be carried forward as within-dataset findings, with sensitivity-qualified magnitude and uncertainty.

Reply counts remain secondary: the prior hurdle fits reached the dispersion boundary, primary ZINB fits were weakly identified, and retained NB fake/real intervals included 1 (FakeHealth approximately 0.29–2.60; CoAID approximately 0.17–1.09). They are not refitted or used in the primary estimand.

We are not yet ready for a definitive cross-context ratio-of-ratios. Next, review the giant entity-resolution component and decide whether the intended target is all unioned entities or a clearly defined single-release population, then pre-specify release/capture handling and the same target definition for later comparison. A full-sample estimate remains available but must retain the release-sensitivity qualification. This audit stops before any cross-context contrast.
