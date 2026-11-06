# Within-dataset engagement model selection

Models use separate datasets and fake versus real coding (real reference). These are associations in observed engagement counts, not causal effects or rates adjusted for observation time.

NB2 uses a log mean and estimated alpha, Var(Y|X)=mu+alpha*mu². Poisson is a diagnostic comparator. Primary CIs use item-level HC0 sandwich standard errors; model-based SEs are also exported. Neither covariance method resolves dependence between items sharing social posts or unresolved duplicate groups. CIs do not account for model selection or release measurement error.

CoAID primary/conservative models adjust for earliest recorded release (May reference). A separate membership sensitivity uses all four release membership indicators; multi-release entities remain single rows with unioned engagement. No additional text/topic controls. September/November uncollapsed checks have limited fake samples and very few positive outcomes.

The hurdle is fitted separately: logit P(Y>0), then zero-truncated NB2 on Y>0. Full log likelihood is the sum and k includes both components plus alpha. AIC/BIC use the full N. These compare models of the SAME full count outcome and SAME sample only; do not compare AIC across datasets, outcomes or sensitivity samples.

A hurdle count exp(beta) is the ratio of underlying NB mean parameters in the positive component. Because E[Y|Y>0,X]=mu/(1-P_NB(0|X)), it is not generally the positive conditional mean ratio. The prediction table reports exact model-standardized positive means and ratios, and full mixture means, with HC0 delta-method CIs. Both veracity groups use the same covariate distribution; conservative estimates are standardized to the primary release mix.

ZINB allows zeros from a latent structural-zero class and the NB process. Both components include veracity and CoAID release predictors. That latent distinction is not established by these data; ZINB is sensitivity rather than automatically preferred by AIC. Boundary estimates and unstable Hessians/SEs are flagged, even when the optimizer declares convergence. Multistart trials and all warnings are retained.

Material zero misfit is flagged when the gap exceeds max(min(0.02, half the observed zero fraction), 3/N, two plug-in Bernoulli standard errors). This also detects severe relative underprediction of rare zeros. It is exploratory, not a calibrated hypothesis test. Overall zero agreement in the logistic hurdle is largely guaranteed by its intercept score equation; group/release agreement and count/residual shape remain necessary checks. Pearson dispersion and randomized quantile residuals (seed 42) diagnose remaining tail misfit. The stable flag means numerical/parameter stability, not proof of distributional adequacy or precision.

## Selected models

- FakeHealth HealthStory / unique_post_count: NB2. Lowest numerical AIC: NB2. Stable NB2 selected: hurdle is unavailable, unstable, or has no >2 AIC improvement. Poisson is diagnostic; ZINB remains latent-mechanism sensitivity.
- FakeHealth HealthStory / unique_reply_count: NB2. Lowest numerical AIC: Hurdle_Logit_ZTNB2. Stable NB2 selected: hurdle is unavailable, unstable, or has no >2 AIC improvement. Poisson is diagnostic; ZINB remains latent-mechanism sensitivity.
- CoAID News / unique_post_count: Hurdle_Logit_ZTNB2. Lowest numerical AIC: Hurdle_Logit_ZTNB2. Hurdle has >2 AIC improvement over NB2, stable fit, and directly separates engagement occurrence from positive volume. ZINB remains sensitivity: latent structural-zero mechanism is not established.
- CoAID News / unique_reply_count: NB2. Lowest numerical AIC: Hurdle_Logit_ZTNB2. Stable NB2 selected: hurdle is unavailable, unstable, or has no >2 AIC improvement. Poisson is diagnostic; ZINB remains latent-mechanism sensitivity.
- CoAID News / engagement_present: Logit. Lowest numerical AIC: Logit. Requested binary logistic model; stability checked.

## FakeHealth engagement presence

- fake: 471/472 (99.788%) engaged.
- real: 1212/1218 (99.507%) engaged.

Only seven FakeHealth items have no observed engagement (one fake, six real); a primary presence logistic model is omitted for essentially absent outcome variation. This does not assert that separation was proven.

## Selected veracity coefficients

- FakeHealth HealthStory / unique_post_count / count: exp(beta)=0.8403, 95% HC0 CI [0.7080, 0.9972]. Fake is associated with a 15.97% lower expected observed count than real, conditional on included covariates.
- FakeHealth HealthStory / unique_reply_count / count: exp(beta)=0.8696, 95% HC0 CI [0.2906, 2.6020]. Fake is associated with a 13.04% lower expected observed count than real, conditional on included covariates.
- CoAID News / unique_post_count / positive_logit: exp(beta)=0.0228, 95% HC0 CI [0.0181, 0.0286]. Fake vs real adjusted odds of a positive observed count: OR=0.02279; not a probability ratio.
- CoAID News / unique_post_count / positive_count: exp(beta)=1.2692, 95% HC0 CI [0.7797, 2.0660]. Among items with positive unique_post_count, underlying NB2 mean parameter is +26.92% different for fake vs real. This is NOT the exact truncated conditional mean change; see standardized_positive_mean_ratio.
- CoAID News / unique_reply_count / count: exp(beta)=0.4353, 95% HC0 CI [0.1732, 1.0940]. Fake is associated with a 56.47% lower expected observed count than real, conditional on included covariates.
- CoAID News / engagement_present / positive_logit: exp(beta)=0.0225, 95% HC0 CI [0.0179, 0.0283]. Fake vs real adjusted odds of any observed engagement: OR=0.02248; not a probability ratio.

## Fit and selection interpretation

| dataset | outcome | model | AIC | BIC | stable | observed_zero_fraction | predicted_zero_fraction | pearson_chi2_per_df |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FakeHealth HealthStory | unique_post_count | Poisson | 5.134e+05 | 5.134e+05 | True | 0.004142 | 4.37e-88 | 669.9 |
| FakeHealth HealthStory | unique_post_count | NB2 | 2.171e+04 | 2.173e+04 | True | 0.004142 | 0.007109 | 2.603 |
| FakeHealth HealthStory | unique_reply_count | NB2 | 8035 | 8051 | True | 0.5645 | 0.5812 | 9.405 |
| FakeHealth HealthStory | unique_reply_count | Hurdle_Logit_ZTNB2 | 7976 | 8003 | False | 0.5645 | 0.5645 | 7.687 |
| FakeHealth HealthStory | unique_reply_count | ZINB2 | 8037 | 8064 | False | 0.5645 | 0.582 | 9.362 |
| CoAID News | unique_post_count | Poisson | 5.411e+05 | 5.412e+05 | True | 0.5066 | 0.2921 | 1147 |
| CoAID News | unique_post_count | NB2 | 2.774e+04 | 2.778e+04 | True | 0.5066 | 0.5004 | 8.457 |
| CoAID News | unique_post_count | Hurdle_Logit_ZTNB2 | 2.614e+04 | 2.622e+04 | True | 0.5066 | 0.5066 | 10.41 |
| CoAID News | unique_reply_count | NB2 | 1.445e+04 | 1.449e+04 | True | 0.78 | 0.7844 | 7.374 |
| CoAID News | unique_reply_count | Hurdle_Logit_ZTNB2 | 1.393e+04 | 1.4e+04 | False | 0.78 | 0.78 | 3.916 |
| CoAID News | unique_reply_count | ZINB2 | 1.407e+04 | 1.414e+04 | False | 0.78 | 0.7894 | 5.099 |
| CoAID News | engagement_present | Logit | 3159 | 3192 | True | 0.506 | 0.506 | 1.512 |

FakeHealth posts: NB2 is strongly preferred to Poisson and is the clearest usable count specification. FakeHealth replies: NB2 is the estimable working model, not the lowest nominal-AIC model. The NB hurdle pushes alpha to 100,000, with a singular/ill-conditioned information matrix. Fixed-alpha profiles rise over 20, 100, 1,000, 10,000, 100,000, independently confirming the boundary tendency. Its positive-count Wald CIs must not be used. FakeHealth ZINB adds no AIC improvement and its inflation contrast is weakly identified.

CoAID posts: the stable hurdle has a large AIC/BIC advantage over NB2. The occurrence contrast is strongly negative, while the positive-volume contrast is positive and imprecise in the primary sample. A single lower overall engagement mean does not imply lower positive volume. The hurdle is both substantively useful and statistically preferable for posts.

CoAID replies: the hurdle has lower nominal AIC but the same alpha-boundary issue; primary ZINB improves nominal AIC materially over NB2 but has nonidentifiable inflation parameters and no reliable full covariance. NB2 is therefore retained as a provisional explanatory baseline. The all-membership ZINB sensitivity is numerically stable and improves AIC, but it uses a different release design and does not establish a real structural-zero class. No adequate, identified primary two-part reply model is established here.

Residual Pearson ratios remain above 1 for every selected count model, especially replies and CoAID posts. Heavy extreme counts remain poorly captured even where the central distribution and zero fractions fit. HC0 intervals reduce dependence on the assumed conditional variance but are not a remedy for mean misspecification, rare influential items or sparse release strata.

## Predicted means and probabilities

| dataset | outcome | measure | estimate | CI_lower | CI_upper |
| --- | --- | --- | --- | --- | --- |
| FakeHealth HealthStory | unique_post_count | standardized_overall_mean_ratio | 0.8403 | 0.708 | 0.9972 |
| FakeHealth HealthStory | unique_post_count | standardized_positive_mean_ratio | 0.8412 | 0.7097 | 0.9972 |
| FakeHealth HealthStory | unique_post_count | standardized_probability_real | 0.9932 | 0.9911 | 0.9953 |
| FakeHealth HealthStory | unique_post_count | standardized_probability_fake | 0.9921 | 0.9894 | 0.9947 |
| FakeHealth HealthStory | unique_reply_count | standardized_overall_mean_ratio | 0.8696 | 0.2906 | 2.602 |
| FakeHealth HealthStory | unique_reply_count | standardized_positive_mean_ratio | 0.8881 | 0.3505 | 2.25 |
| FakeHealth HealthStory | unique_reply_count | standardized_probability_real | 0.4212 | 0.3905 | 0.4519 |
| FakeHealth HealthStory | unique_reply_count | standardized_probability_fake | 0.4125 | 0.3592 | 0.4657 |
| CoAID News | unique_post_count | standardized_overall_mean_ratio | 0.237 | 0.1481 | 0.3792 |
| CoAID News | unique_post_count | standardized_positive_mean_ratio | 1.235 | 0.7997 | 1.908 |
| CoAID News | unique_post_count | standardized_probability_real | 0.6119 | 0.6033 | 0.6204 |
| CoAID News | unique_post_count | standardized_probability_fake | 0.1158 | 0.09829 | 0.1333 |
| CoAID News | unique_reply_count | standardized_overall_mean_ratio | 0.4353 | 0.1732 | 1.094 |
| CoAID News | unique_reply_count | standardized_positive_mean_ratio | 0.4895 | 0.2235 | 1.072 |
| CoAID News | unique_reply_count | standardized_probability_real | 0.2212 | 0.2093 | 0.2332 |
| CoAID News | unique_reply_count | standardized_probability_fake | 0.1912 | 0.1636 | 0.2189 |
| CoAID News | engagement_present | standardized_probability_real | 0.6126 | 0.6041 | 0.6211 |
| CoAID News | engagement_present | standardized_probability_fake | 0.1159 | 0.09846 | 0.1334 |

For the CoAID hurdle, the exact standardized positive mean ratio differs from its count-component exp(beta). The standardized overall ratio combines the occurrence and positive-volume processes and is the appropriate all-item comparison, conditional on this specification. Probability predictions from an NB model are distribution-implied and should not replace the separately fitted binary logistic outcome.

## Duplicate and release sensitivity

| outcome | specification | model_component | IRR_or_OR | CI_lower | CI_upper | selected_model_changed |
| --- | --- | --- | --- | --- | --- | --- |
| unique_post_count | conservative_exclude_198 | positive_logit | 0.02331 | 0.0185 | 0.02938 | False |
| unique_post_count | conservative_exclude_198 | positive_count | 1.899 | 1.327 | 2.719 | False |
| unique_reply_count | conservative_exclude_198 | count | 0.698 | 0.3356 | 1.452 | False |
| engagement_present | conservative_exclude_198 | positive_logit | 0.02331 | 0.0185 | 0.02938 | False |
| unique_post_count | all_release_memberships | positive_logit | 0.024 | 0.01903 | 0.03028 | False |
| unique_post_count | all_release_memberships | positive_count | 1.808 | 1.293 | 2.528 | False |
| unique_reply_count | all_release_memberships | count | 0.6801 | 0.3465 | 1.335 | False |
| engagement_present | all_release_memberships | positive_logit | 0.02377 | 0.01884 | 0.02998 | False |

Excluding the 198 candidates preserves selected model families and coefficient directions. The any-engagement odds ratio changes little. Count magnitudes are not robust: the positive-post component moves farther above 1 and its CI excludes 1, while the reply NB2 IRR moves toward 1 and remains imprecise. Thus robustness of occurrence does not imply robustness of positive volume or overall counts.

The all-release-membership sensitivity also preserves selected coefficient directions but substantially changes some standardized means. Rare multi-release covariate profiles and limited fake support cause extrapolation; the comparison table reports unsupported-profile fractions and mean calibration. Its count standardizations should not be promoted to definitive estimates. The occurrence association is considerably less sensitive to the coding choice.

| release | outcome | fake_N | IRR_or_OR | CI_lower | CI_upper | small_release_warning |
| --- | --- | --- | --- | --- | --- | --- |
| 05-01-2020 | unique_post_count | 572 | 0.2934 | 0.2014 | 0.4275 | False |
| 05-01-2020 | unique_reply_count | 572 | 0.2514 | 0.1485 | 0.4256 | False |
| 05-01-2020 | engagement_present | 572 | 0.01401 | 0.01036 | 0.01894 | False |
| 07-01-2020 | unique_post_count | 266 | 0.09361 | 0.04528 | 0.1935 | False |
| 07-01-2020 | unique_reply_count | 266 | 0.1416 | 0.06731 | 0.298 | False |
| 07-01-2020 | engagement_present | 266 | 0.03631 | 0.02586 | 0.05098 | False |
| 09-01-2020 | unique_post_count | 56 | 3.527 | 0.7343 | 16.94 | True |
| 09-01-2020 | unique_reply_count | 56 | 8.859 | 1.203 | 65.27 | True |
| 09-01-2020 | engagement_present | 56 | 0.9799 | 0.3436 | 2.794 | True |
| 11-01-2020 | unique_post_count | 31 | 0.6164 | 0.08027 | 4.734 | True |
| 11-01-2020 | unique_reply_count | 31 | 4.12 | 0.3996 | 42.47 | True |
| 11-01-2020 | engagement_present | 31 | 0.6069 | 0.08104 | 4.545 | True |

May and July both show lower marginal observed counts and occurrence for fake items, but their sizes differ. September (56 fake records) and November (31) have very few fake positive outcomes. Treat their coefficients and Wald intervals as unstable diagnostics even when optimization converges or an interval happens to exclude 1. Strata use uncollapsed original records, so they are not independent estimates of the primary entity-resolved estimand. These results demonstrate descriptive heterogeneity; no formal interaction test has been run.

## Estimates to carry forward and next step

The FakeHealth post NB2 contrast and CoAID engagement-occurrence logistic contrast are the most stable within their own estimands. The latter is supported by both duplicate exclusion and release coding, but it cannot be compared directly to a count IRR. FakeHealth replies are imprecise; CoAID positive volume and reply means depend materially on sensitivity choices. The primary CoAID hurdle overall count ratio is potentially useful but not yet robust to release coding.

Next, review these results and pre-specify a common estimand for later cross-context work: overall observed-count mean ratios or standardized engagement probabilities, with occurrence and positive volume reported separately. Before the interaction analysis, assess influential high-count entities, release-profile overlap and uncertainty under duplicate/shared-engagement dependence. The reply positive-count boundary needs an explicit distributional decision. Do not compare a FakeHealth all-item NB IRR directly with a CoAID positive-component parameter ratio. This stage has not performed the cross-dataset comparison.

## Method references

- [statsmodels hurdle and truncated models](https://www.statsmodels.org/stable/examples/notebooks/generated/count_hurdle.html)
- [Zero-truncated NB prediction: mean versus mean-main](https://www.statsmodels.org/stable/generated/statsmodels.discrete.truncated_model.TruncatedLFNegativeBinomialP.predict.html)
- [ZINB prediction](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.predict.html)

This run stops at within-dataset regression. No pooled cross-context model, text model, machine learning, or timestamp outcome was fitted.
