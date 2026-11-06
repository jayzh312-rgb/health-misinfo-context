"""Audit all-item post-count standardizations, release coding, and influence.

Run: .venv/bin/python src/audit_engagement_estimand_robustness.py
Reuses the validated NB2 / logistic + zero-truncated NB2 fitting functions.
Inputs are read-only. No cross-dataset contrast is calculated.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from statsmodels.tools.numdiff import approx_fprime

from run_primary_engagement_models import (
    RELEASES, ROOT, SEED, fit_hurdle, fit_one, load_samples, markdown_table, moments,
)

TABLES = ROOT / "outputs/tables"
REPORT = ROOT / "outputs/reports/engagement_estimand_robustness.md"
OUTCOME = "unique_post_count"
FH = "FakeHealth HealthStory"
CO = "CoAID News"
Z = norm.ppf(.975)
METHOD = "HC0 entity/item-score sandwich delta method; log-scale means/ratio, identity-scale difference; empirical covariates held fixed; no bootstrap"


def add_release_columns(frame):
    frame = frame.copy()
    frame["releases"] = frame.member_release_periods_json.map(json.loads)
    assert frame.releases.map(lambda v: bool(v) and set(v) <= set(RELEASES)).all()
    frame["first_release"] = frame.releases.map(lambda v: min(v, key=RELEASES.index))
    frame["last_release"] = frame.releases.map(lambda v: max(v, key=RELEASES.index))
    frame["n_distinct_releases"] = frame.releases.map(lambda v: len(set(v)))
    frame["multiple_releases"] = frame.n_distinct_releases.gt(1)
    frame["membership_profile"] = frame.releases.map(lambda v: " | ".join(sorted(set(v), key=RELEASES.index)))
    return frame


def design(frame, coding):
    x = pd.DataFrame({"Intercept": 1., "fake_vs_real": frame.analysis_group.eq("fake").astype(float)}, index=frame.index)
    if coding in ("first_release", "last_release", "single_release_only"):
        values = frame["last_release" if coding == "last_release" else "first_release"]
        for release in RELEASES[1:]:
            x["release_" + release] = values.eq(release).astype(float)
    elif coding == "membership_indicators":
        for release in RELEASES:
            x["member_" + release] = frame.releases.map(lambda v: float(release in v))
    elif coding != "no_release_adjustment":
        raise ValueError(coding)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError(f"Rank deficient design for {coding}")
    return x


def ranking(frame):
    """Exact-size tails, descending count then ascending UID for boundary ties."""
    return frame.sort_values([OUTCOME, "content_uid"], ascending=[False, True], kind="stable")


def tail_sets(frame):
    ranked = ranking(frame)
    k = math.ceil(len(frame) * .01)
    cutoff = int(ranked.iloc[k - 1][OUTCOME])
    return {
        "all_retained": set(),
        "exclude_top_1pct": set(ranked.head(k).content_uid),
        "exclude_top_5_items": set(ranked.head(5).content_uid),
        "exclude_top_1pct_with_boundary_ties": set(frame.loc[frame[OUTCOME].ge(cutoff), "content_uid"]),
    }


def support_fraction(x):
    controls = x.drop(columns="fake_vs_real")
    profiles = controls.astype(str).agg("|".join, axis=1)
    counts = pd.DataFrame({"profile": profiles, "group": x.fake_vs_real}).groupby("profile").group.nunique()
    return float(profiles.map(counts).lt(2).mean())


def delta_interval(fit, function, log_scale=False):
    estimate = float(function(fit.params))
    transformed = (lambda b: np.log(function(b))) if log_scale else function
    gradient = np.atleast_1d(approx_fprime(fit.params, transformed, centered=True))
    variance = float(gradient @ fit.cov @ gradient)
    if variance < -1e-8:
        raise ValueError("Negative delta-method variance")
    se = np.sqrt(max(variance, 0.))
    center = np.log(estimate) if log_scale else estimate
    interval = center + np.array([-Z, Z]) * se
    if log_scale:
        interval = np.exp(interval)
    return estimate, float(interval[0]), float(interval[1])


def standardized_counts(fit, reference_x):
    real, fake = np.asarray(reference_x).copy(), np.asarray(reference_x).copy()
    real[:, 1], fake[:, 1] = 0., 1.
    def means(b):
        # moments()[0] is E(Y|X) for the entire hurdle, including zero items.
        return moments(fit, fake, b)[0].mean(), moments(fit, real, b)[0].mean()
    functions = {
        "standardized_fake_mean": lambda b: means(b)[0],
        "standardized_real_mean": lambda b: means(b)[1],
        "mean_difference": lambda b: means(b)[0] - means(b)[1],
        "mean_ratio": lambda b: means(b)[0] / means(b)[1],
    }
    row = {}
    for key, function in functions.items():
        value, lo, hi = delta_interval(fit, function, key != "mean_difference")
        row.update({key: value, key + "_CI_lower": lo, key + "_CI_upper": hi})
    row["CI_lower"], row["CI_upper"] = row["mean_ratio_CI_lower"], row["mean_ratio_CI_upper"]
    return row


def verify_full_hurdle_mean(fit):
    p = fit.p
    positive = fit.y > 0
    native = TruncatedLFNegativeBinomialP(fit.y[positive], fit.x[positive], p=2, truncation=0)
    conditional_mean = native.predict(fit.params[p:], exog=fit.x, which="mean")
    occurrence_probability = expit(fit.x @ fit.params[:p])
    np.testing.assert_allclose(moments(fit)[0], occurrence_probability * conditional_mean, rtol=1e-10)
    np.testing.assert_allclose(moments(fit)[3], conditional_mean, rtol=1e-10)


def presence_row(frame, coding, specification, removed):
    x = design(frame, coding)
    fit = fit_one(frame.engagement_present.to_numpy(float), x, "Logit")
    real, fake = x.to_numpy().copy(), x.to_numpy().copy()
    real[:, 1], fake[:, 1] = 0., 1.
    row = {"dataset": CO, "outcome": "engagement_present", "specification": specification,
           "release_coding": coding, "tail_specification": specification.split("__", 1)[1],
           "N": len(frame), "removed_N": removed,
           "convergence": fit.converged, "stable": fit.stable, "CI_method": METHOD}
    for name, function in {
        "probability_fake": lambda b: expit(fake @ b).mean(),
        "probability_real": lambda b: expit(real @ b).mean(),
        "probability_difference": lambda b: expit(fake @ b).mean() - expit(real @ b).mean(),
        "odds_ratio": lambda b: np.exp(b[1]),
    }.items():
        value, lo, hi = delta_interval(fit, function, name == "odds_ratio")
        row.update({name: value, name + "_CI_lower": lo, name + "_CI_upper": hi})
    for group in ("fake", "real"):
        part = frame.loc[frame.analysis_group.eq(group)]
        row[f"observed_{group}_N"] = len(part)
        row[f"observed_{group}_engaged_N"] = int(part.engagement_present.sum())
        row[f"observed_{group}_probability"] = float(part.engagement_present.mean())
    return row


def release_audit(coaid):
    rows = []
    total_posts = coaid[OUTCOME].sum()
    total_n = len(coaid)
    for kind, field in [("release_count", "n_distinct_releases"), ("single_vs_multiple", "multiple_releases"),
                        ("exact_membership_profile", "membership_profile"),
                        ("first_release", "first_release"), ("last_release", "last_release")]:
        for value, part in coaid.groupby(field):
            for group in ("all", "fake", "real"):
                subset = part if group == "all" else part.loc[part.analysis_group.eq(group)]
                y = subset[OUTCOME]
                rows.append({"record_type": kind, "release_or_profile": str(value), "veracity": group,
                             "N": len(subset), "sample_share": len(subset) / total_n,
                             "total_posts": int(y.sum()), "total_engagement_share": y.sum() / total_posts,
                             "mean_posts": y.mean(), "median_posts": y.median(), "p95_posts": y.quantile(.95),
                             "max_posts": y.max(), "zero_fraction": y.eq(0).mean(),
                             "possible_duplicate_N": int(subset.sensitivity_exclude_candidate.sum()),
                             "notes": "Unique entities. Exact profiles and first/last assignments partition entities; group=all overlaps fake/real rows."})
    raw = pd.read_csv(ROOT / "data/processed/coaid_content_level.csv")
    raw = raw.loc[raw.content_category.eq("News")]
    for release, part in raw.groupby("source_release_or_subset"):
        for group in ("all", "fake", "real"):
            subset = part if group == "all" else part.loc[part.veracity_original.str.lower().eq(group)]
            y = subset[OUTCOME]
            rows.append({"record_type": "original_release_records", "release_or_profile": release,
                         "veracity": group, "N": len(subset), "total_posts": int(y.sum()),
                         "mean_posts": y.mean(), "median_posts": y.median(), "p95_posts": y.quantile(.95),
                         "max_posts": y.max(), "zero_fraction": y.eq(0).mean(), "N_at_500": int(y.eq(500).sum()),
                         "notes": "Uncollapsed source records: do not sum across releases as unique entity engagement. Observed 500-count boundary is empirical; collection-cap mechanism is not verified."})
    return rows


def influence_rows(samples):
    rows, summaries = [], []
    for dataset, frame in samples.items():
        ranked = ranking(frame).copy()
        ranked["rank"] = np.arange(1, len(frame) + 1)
        k1, k5 = math.ceil(.01 * len(frame)), math.ceil(.05 * len(frame))
        cutoff1, cutoff5 = int(ranked.iloc[k1 - 1][OUTCOME]), int(ranked.iloc[k5 - 1][OUTCOME])
        total = int(frame[OUTCOME].sum())
        group_total = frame.groupby("analysis_group")[OUTCOME].sum()
        for item in ranked.head(max(20, k5)).itertuples():
            rows.append({"dataset": dataset, "veracity": item.analysis_group,
                         "release_or_subset": item.source_release_or_subset, "content_uid": item.content_uid,
                         "count": getattr(item, OUTCOME), "rank": item.rank,
                         "in_top_20": item.rank <= 20, "in_top_1pct": item.rank <= k1,
                         "in_top_5pct": item.rank <= k5, "in_top_5_items": item.rank <= 5,
                         "engagement_share_dataset": getattr(item, OUTCOME) / total,
                         "engagement_share_veracity_group": getattr(item, OUTCOME) / group_total[item.analysis_group],
                         "duplicate_reobservation_status": getattr(item, "duplicate_status", "Not assessed by CoAID entity-resolution audit; unique content UID"),
                         "sensitivity_exclude_candidate": getattr(item, "sensitivity_exclude_candidate", False),
                         "n_distinct_releases": getattr(item, "n_distinct_releases", np.nan),
                         "release_observation_count": getattr(item, "release_observation_count", np.nan),
                         "multiple_releases": getattr(item, "multiple_releases", False),
                         "top_1pct_cutoff": cutoff1, "top_5pct_cutoff": cutoff5,
                         "ranking_rule": "Descending count, ascending UID breaks ties; ceil(N*p) items."})
        for name, k in [("top_20", 20), ("top_1pct", k1), ("top_5pct", k5), ("top_5_items", 5)]:
            selected = ranked.head(k)
            for group in ("all", "fake", "real"):
                part = selected if group == "all" else selected.loc[selected.analysis_group.eq(group)]
                baseline = frame if group == "all" else frame.loc[frame.analysis_group.eq(group)]
                summaries.append({"dataset": dataset, "tail": name, "veracity": group, "N": len(part),
                                  "group_sample_N": len(baseline), "group_tail_fraction": len(part) / len(baseline),
                                  "posts": int(part[OUTCOME].sum()), "engagement_share_dataset": part[OUTCOME].sum() / total,
                                  "engagement_share_group": part[OUTCOME].sum() / baseline[OUTCOME].sum(),
                                  "cutoff_count": int(selected[OUTCOME].min()),
                                  "total_items_at_cutoff": int(frame[OUTCOME].eq(selected[OUTCOME].min()).sum())})
    return rows, summaries


def change_flags(ratio, baseline):
    change = ratio / baseline - 1
    crossed = (ratio - 1) * (baseline - 1) < 0
    return {"baseline_ratio": baseline, "relative_ratio_change": change,
            "same_direction": bool((ratio - 1) * (baseline - 1) > 0), "point_estimate_crosses_1": bool(crossed),
            "change_classification": "substantial" if abs(change) >= .25 else "moderate" if abs(change) >= .10 else "small",
            "classification_rule": "Absolute relative ratio change: <10% small, 10–<25% moderate, >=25% substantial; descriptive thresholds."}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "data/processed" / name for name in ("fakehealth_healthstory_primary.csv", "coaid_news_primary.csv")]
    input_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    samples = load_samples()
    samples[CO] = add_release_columns(samples[CO])
    coaid = samples[CO]
    influential, tail_summary = influence_rows(samples)
    audit = release_audit(coaid)
    common, tails, presence, coefficients = [], [], [], []
    for dataset, full_frame in samples.items():
        exclusions = tail_sets(full_frame)
        if dataset == CO:
            exclusions["exclude_duplicate_candidates"] = set(full_frame.loc[full_frame.sensitivity_exclude_candidate, "content_uid"])
            assert len(exclusions["exclude_duplicate_candidates"]) == 198
        codings = ("first_release", "last_release", "membership_indicators", "single_release_only", "no_release_adjustment") if dataset == CO else ("no_release_adjustment",)
        for coding in codings:
            # Single-release subset changes the target population. Keep it an
            # explicitly restricted sensitivity, never a replacement primary.
            eligible = full_frame.loc[~full_frame.multiple_releases] if coding == "single_release_only" else full_frame
            variants = {"all_retained": set()} if coding == "single_release_only" else exclusions
            baseline = None
            for tail, removed_uids in variants.items():
                frame = eligible.loc[~eligible.content_uid.isin(removed_uids)].copy()
                removed = eligible.loc[eligible.content_uid.isin(removed_uids)]
                specification = coding + "__" + tail
                print(f"Fitting {dataset}: {specification}; N={len(frame)}", flush=True)
                x = design(frame, coding)
                fit = fit_hurdle(frame[OUTCOME].to_numpy(), x) if dataset == CO else fit_one(frame[OUTCOME].to_numpy(), x, "NB2")
                if fit.name == "Hurdle_Logit_ZTNB2":
                    verify_full_hurdle_mean(fit)
                estimates = standardized_counts(fit, x)
                mean, _, p0, _, _ = moments(fit)
                is_primary = tail == "all_retained" and (dataset == FH or coding == "first_release")
                row = {"dataset": dataset, "outcome": OUTCOME, "model": fit.name, "specification": specification,
                       "release_coding": coding, "tail_specification": tail, "is_primary": is_primary,
                       "N": len(frame), "fake_N": int(frame.analysis_group.eq("fake").sum()),
                       "real_N": int(frame.analysis_group.eq("real").sum()), "removed_N": len(full_frame) - len(frame),
                       "standardization_population": "Retained sample, all items including zeros; both labels use the SAME covariate distribution",
                       "standardization_N": len(frame), **estimates, "CI_method": METHOD, "bootstrap_resamples": 0,
                       "convergence": fit.converged, "stable": fit.stable, "hessian_ok": fit.hessian_ok,
                       "boundary": fit.boundary, "AIC": fit.aic, "log_likelihood": fit.ll,
                       "observed_mean": frame[OUTCOME].mean(), "predicted_mean": mean.mean(),
                       "mean_calibration_ratio": mean.mean() / frame[OUTCOME].mean(),
                       "observed_zero_fraction": frame[OUTCOME].eq(0).mean(), "predicted_zero_fraction": p0.mean(),
                       "unsupported_profile_fraction": support_fraction(x),
                       "warnings": fit.warnings, "optimizer_trials_json": fit.trials,
                       "notes": "Working descriptive association. CI conditional on covariates and selected model; does not include release-coding, tail-selection or entity-resolution uncertainty."}
                # Also hold the original eligible population's covariate mix
                # fixed, so parameter influence and target-mix change are visible.
                fixed = standardized_counts(fit, design(eligible, coding))
                row.update({"fixed_original_target_" + k: v for k, v in fixed.items()})
                row["fixed_original_target_N"] = len(eligible)
                common.append(row)
                if tail == "all_retained":
                    baseline = row
                assert baseline is not None
                tail_row = {**row, **change_flags(row["mean_ratio"], baseline["mean_ratio"]),
                            "CI_includes_1": row["CI_lower"] <= 1 <= row["CI_upper"],
                            "removed_fake_N": int(removed.analysis_group.eq("fake").sum()),
                            "removed_real_N": int(removed.analysis_group.eq("real").sum()),
                            "removed_post_sum": int(removed[OUTCOME].sum()),
                            "removed_engagement_share": removed[OUTCOME].sum() / eligible[OUTCOME].sum()}
                tails.append(tail_row)
                for j, (name, component) in enumerate(zip(fit.names, fit.components)):
                    coefficients.append({"dataset": dataset, "specification": specification, "component": component,
                                         "predictor": name, "coefficient": fit.params[j], "HC0_SE": fit.se[j]})
                if dataset == CO:
                    presence.append(presence_row(frame, coding, specification, len(full_frame) - len(frame)))
                    audit.append({"record_type": "model_specification", **row,
                                  "notes": "Same hurdle family across release/tail alternatives. AIC is diagnostic, not the basis for choosing provenance representation; do not compare AIC across different samples."})

    common = pd.DataFrame(common)
    tails = pd.DataFrame(tails)
    presence = pd.DataFrame(presence)
    # Include changed/unsupported standardizations honestly rather than filtering
    # sensitivity rows according to their direction or numerical fit ranking.
    assert len(common.loc[common.is_primary]) == 2
    assert common[["standardized_fake_mean", "standardized_real_mean", "mean_ratio", "CI_lower", "CI_upper"]].notna().all().all()
    assert np.allclose(common.mean_ratio, common.standardized_fake_mean / common.standardized_real_mean)
    assert np.allclose(common.mean_difference, common.standardized_fake_mean - common.standardized_real_mean)
    previous = pd.read_csv(TABLES / "engagement_model_standardized_predictions.csv")
    for row in common.loc[common.is_primary].itertuples():
        old = previous.loc[previous.dataset.eq(row.dataset) & previous.outcome.eq(OUTCOME) & previous.specification.eq("primary") & previous.measure.eq("standardized_overall_mean_ratio"), "estimate"].iloc[0]
        assert np.isclose(row.mean_ratio, old, rtol=1e-5)
    for path in inputs:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == input_hashes[path.name]
    common.to_csv(TABLES / "common_engagement_estimand.csv", index=False)
    pd.DataFrame(audit).to_csv(TABLES / "coaid_release_representation_audit.csv", index=False)
    pd.DataFrame(influential).to_csv(TABLES / "engagement_influential_entities.csv", index=False)
    tails.to_csv(TABLES / "engagement_tail_sensitivity.csv", index=False)
    pd.DataFrame(tail_summary).to_csv(TABLES / "engagement_tail_concentration_summary.csv", index=False)
    presence.to_csv(TABLES / "engagement_presence_robustness.csv", index=False)
    pd.DataFrame(coefficients).to_csv(TABLES / "engagement_estimand_audit_coefficients.csv", index=False)
    write_report(samples, common, tails, presence, pd.DataFrame(tail_summary))
    metadata = {"seed": SEED, "bootstrap_resamples": 0, "CI_method": METHOD,
                "primary_input_sha256": input_hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fitting_module_sha256": hashlib.sha256((ROOT / "src/run_primary_engagement_models.py").read_bytes()).hexdigest(),
                "count_fits": len(common), "presence_fits": len(presence), "cross_context_contrast_computed": False}
    (TABLES / "engagement_estimand_audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(common[["dataset", "specification", "N", "mean_ratio", "CI_lower", "CI_upper", "stable"]].to_string(index=False))
    print(f"Saved audit: {len(common)} count specifications, {len(presence)} separate presence fits; input hashes unchanged.")


def write_report(samples, common, tails, presence, concentration):
    coaid = samples[CO]
    multi = coaid.loc[coaid.multiple_releases]
    single = coaid.loc[~coaid.multiple_releases]
    total = coaid[OUTCOME].sum()
    main = common.loc[common.is_primary]
    releases = common.loc[common.dataset.eq(CO) & common.tail_specification.eq("all_retained")]
    main_tails = tails.loc[(tails.dataset.eq(FH) | tails.release_coding.eq("first_release")) & tails.tail_specification.ne("exclude_top_1pct_with_boundary_ties")]
    binary = presence.loc[presence.release_coding.eq("first_release")]
    largest = ranking(coaid).iloc[0]
    def ratio(dataset, tail="all_retained", coding=None):
        coding = coding or ("first_release" if dataset == CO else "no_release_adjustment")
        return common.loc[common.dataset.eq(dataset) & common.release_coding.eq(coding) & common.tail_specification.eq(tail), "mean_ratio"].iloc[0]
    duplicate_change = ratio(CO, "exclude_duplicate_candidates") / ratio(CO) - 1
    lines = ["# Common engagement estimand and robustness audit", "",
             "The common primary estimand is the fake/real ratio of marginal expected **unique post counts across all content items, including zeros**. Within each dataset, predict every retained observation under each label, hold its other covariates fixed, average those predictions, and divide. These are descriptive model-based associations. There is no cross-dataset ratio-of-ratios or interaction in this audit.", "",
             "FakeHealth uses NB2. CoAID uses the FULL logistic + zero-truncated NB2 hurdle: E[Y|X] = P(Y>0|X) × E[Y|Y>0,X]. The script verifies this against native zero-truncated NB predictions. Neither the positive conditional mean nor the exponentiated positive coefficient is substituted for the overall mean. The built-in HurdleCountModel is not used.", "",
             "95% intervals use the existing entity/item-level HC0 score-sandwich covariance and delta method, on log scale for positive means/ratios and identity scale for differences. No bootstrap is used (0 resamples); seed 42 is retained. Covariates are treated as a fixed empirical target. Intervals do not include model/release selection, trimming-threshold uncertainty, shared-post dependence or entity-resolution uncertainty.", "",
             "## Primary all-item estimates", "",
             markdown_table(main[["dataset", "N", "standardized_fake_mean", "standardized_real_mean", "mean_difference", "mean_ratio", "CI_lower", "CI_upper"]]), "",
             "The previously selected first-release CoAID specification remains the untrimmed primary reference for this audit. Its status is a working reference, not a claim that first release fully captures the observation process. All raw and processed inputs are unchanged.", "",
             "## Release provenance and representation", "",
             f"CoAID has {len(single):,} single-release and {len(multi):,} multi-release entities ({100*len(multi)/len(coaid):.2f}%). Multi-release entities comprise {int(multi.analysis_group.eq('fake').sum())} fake and {int(multi.analysis_group.eq('real').sum())} real items and contribute {100*multi[OUTCOME].sum()/total:.2f}% of all posts. Their mean is {multi[OUTCOME].mean():.2f}, versus {single[OUTCOME].mean():.2f} for single-release entities.", "",
             "Counts are unions of unique post IDs over all source records in an entity, not counts observed solely at the first or last release. Distinct-release count is different from number of source records. Release labels are collection provenance, not observation-duration offsets.", "",
             markdown_table(releases[["release_coding", "N", "standardized_fake_mean", "standardized_real_mean", "mean_difference", "mean_ratio", "CI_lower", "CI_upper", "stable", "unsupported_profile_fraction"]]), "",
             "Membership indicators best preserve provenance for the full entity-resolved sample: they retain every recorded release without duplicating entities. However, only four fake entities span releases, and some exact profiles have no fake support. They are therefore not a defensible definitive adjustment for a precise all-item contrast here. First release describes entry cohort and is the least ambiguous single-date working label; last release describes last recorded membership, not necessarily comparable exposure. Neither models the full unioned capture process. Single-release-only analysis gives the cleanest release assignment and best overlap, but explicitly changes the target population and cannot silently replace the full-sample estimand. No-release adjustment is an unadjusted diagnostic. There is no fully satisfactory release representation for the current full-sample headline estimate; AIC cannot settle this provenance/overlap issue.", "",
             "Original May/July News records have maxima of 500 observed posts, with many exactly at 500; September/November maxima are much smaller. This is an empirical collection-pattern warning, not a verified claim about a platform/API cap. Unioning multiple records can exceed the per-record maximum.", "",
             "## High-count entities and tail sensitivity", "",
             f"The largest CoAID entity is `{largest.content_uid}` (real): {int(largest[OUTCOME]):,} posts, {100*largest[OUTCOME]/total:.2f}% of the dataset total, combining {int(largest.release_observation_count)} source records across {int(largest.n_distinct_releases)} releases. Its existing status is `{largest.duplicate_status}`. This unusually large connected component warrants targeted entity-resolution review; the audit does not change its classification or remove it from primary results.", "",
             markdown_table(concentration.loc[concentration.veracity.ne("all"), ["dataset", "tail", "veracity", "N", "group_tail_fraction", "engagement_share_dataset"]]), "",
             "Ranked tails are pooled within each dataset, not selected separately within fake/real groups. Top 1%/5% use ceil(N×p); count ties are resolved by ascending UID. The influential-entity table contains the union of top-20 and top-5% records, with top-1%, top-5-items and top-20 flags and per-item engagement contributions. Top 5% is described but is not trimmed from the primary analysis.", "",
             markdown_table(main_tails[["dataset", "tail_specification", "N", "mean_ratio", "CI_lower", "CI_upper", "relative_ratio_change", "change_classification", "point_estimate_crosses_1"]]), "",
             f"Under the primary coding, neither dataset's point estimate crosses 1 after either required tail exclusion. FakeHealth moves from {ratio(FH):.3f} to {ratio(FH, 'exclude_top_1pct'):.3f} (top 1%) or {ratio(FH, 'exclude_top_5_items'):.3f} (five largest); the five-largest exclusion CI includes 1, so confidence in a non-null difference is not fully tail-robust. CoAID moves from {ratio(CO):.3f} to {ratio(CO, 'exclude_top_1pct'):.3f} or {ratio(CO, 'exclude_top_5_items'):.3f}. Duplicate exclusion gives {ratio(CO, 'exclude_duplicate_candidates'):.3f}, a {100*duplicate_change:.1f}% relative increase: direction persists but magnitude changes substantially.", "",
             "CoAID's top 20 and exact-size top 1% are all real items. FakeHealth's top 20 include 15 real and five fake items; account for its larger real-group sample size when interpreting this imbalance. The group-tail fractions and group-specific engagement shares in the concentration table make that comparison explicit.", "",
             "Small/moderate/substantial denotes <10%, 10–<25%, or >=25% absolute relative change in the ratio. This descriptive classification is separate from crossing 1 or a confidence interval including 1. Each main row standardizes over its retained sample. Additional fixed_original_target columns standardize over the original eligible covariate mix, separating parameter changes from target-composition changes. Neither quantity recovers the untrimmed population mean after outcome-dependent removal; tail exclusions are influence diagnostics only.", "",
             "An additional top-1%-with-boundary-ties specification checks dependence on arbitrary tie breaking. Tail and duplicate checks are repeated under first, last, membership, and no-release codings in the CSV, not only under the primary coding.", "",
             f"CoAID has 57 items tied at the 500-post cutoff; excluding every item at or above it removes 84 rather than 51 items and gives a primary-coding ratio of {ratio(CO, 'exclude_top_1pct_with_boundary_ties'):.3f}. The membership model's baseline ratio is {ratio(CO, coding='membership_indicators'):.3f}; tail exclusion can move that point estimate below 1, but its intervals include 1 throughout. Thus the statement that tail removal preserves direction applies to the primary coding, not uniformly to every release representation.", "",
             "## Engagement presence stays separate", "",
             markdown_table(binary[["tail_specification", "N", "probability_fake", "probability_real", "probability_difference", "odds_ratio", "odds_ratio_CI_lower", "odds_ratio_CI_upper"]]), "",
             "The binary outcome uses any observed posts OR replies and is distinct from the hurdle's posts-only positive indicator. Its fitted label association depends on zero/nonzero status, not positive-count magnitude. Removing a few high-count entities removes a few engaged observations, but does not erase the broad occurrence gap. The tail and duplicate presence refits directly check this.", "",
             "FakeHealth has only seven unengaged items (one fake, six real), with 99.79% fake and 99.51% real engagement. Both groups are at the probability ceiling (equivalently, zero frequency is at its floor). This supplies little information about label differences, making a direct between-dataset presence comparison unstable and qualitatively mismatched. No FakeHealth logistic model or between-dataset binary contrast is fitted.", "",
             "## Carry-forward decision", "",
             "The common all-item post-count estimand is now defined and implemented consistently. The numerical first-release CoAID ratio is not ready to be treated as a release-invariant headline result: evaluate its spread across the release and tail tables, not just its primary CI. The membership coding's overlap/extrapolation problem is part of the finding, not a reason to suppress that result. FakeHealth post-count direction and CoAID occurrence association can be carried forward as within-dataset findings, with sensitivity-qualified magnitude and uncertainty.", "",
             "Reply counts remain secondary: the prior hurdle fits reached the dispersion boundary, primary ZINB fits were weakly identified, and retained NB fake/real intervals included 1 (FakeHealth approximately 0.29–2.60; CoAID approximately 0.17–1.09). They are not refitted or used in the primary estimand.", "",
             "We are not yet ready for a definitive cross-context ratio-of-ratios. Next, review the giant entity-resolution component and decide whether the intended target is all unioned entities or a clearly defined single-release population, then pre-specify release/capture handling and the same target definition for later comparison. A full-sample estimate remains available but must retain the release-sensitivity qualification. This audit stops before any cross-context contrast."]
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
