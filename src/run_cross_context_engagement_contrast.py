"""Independent empirical-product cross-context contrasts, not causal effects.

Run: .venv/bin/python src/run_cross_context_engagement_contrast.py --workers 4
Reuses the 4,000 validated within-sample post refits. Only the previously
unbootstrapped leave-dominant-component-out sensitivity is newly fitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew

from run_version_aware_engagement_analysis import (
    add_release_columns, fit_hurdle, model_design, point_metrics,
    refit_checked, require, resample_indices, sha,
)
from run_primary_engagement_models import markdown_table

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures/cross_context_engagement"
REPORT = ROOT / "outputs/reports/cross_context_engagement_contrast.md"
FH = "FakeHealth_HealthStory"
COAID = ["A_primary_strict", "B_resource_sensitivity", "C_high_certainty"]
NO_DOM = "A_exclude_dominant_component"
METHOD = "Full Cartesian product of independent overlap-cluster bootstrap ratios; 1000 x 1000 empirical pairs; percentile 95% interval"
CACHE = TABLES / "cross_context_no_dominant_bootstrap_refits.csv"
CACHE_META = TABLES / "cross_context_no_dominant_bootstrap_metadata.json"


def save(name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(TABLES / (name + ".csv"), index=False)
    return frame


def validated_inputs():
    meta_path = TABLES / "version_aware_engagement_run_metadata.json"
    meta = json.loads(meta_path.read_text())
    require(meta["complete"] and not meta["pilot_only"] and meta["seed"] == 42, "Within-dataset run incomplete or wrong seed")
    fingerprints = dict(meta["input_sha256"])
    script_path = ROOT / "src/run_version_aware_engagement_analysis.py"
    require(sha(script_path) == meta["script_sha256"], "Within-dataset script fingerprint mismatch")
    for relative, expected in fingerprints.items():
        require(sha(ROOT / relative) == expected, f"Validated input fingerprint mismatch: {relative}")
    names = ["version_aware_bootstrap_refits", "version_aware_engagement_results", "version_aware_bootstrap_summary",
             "version_aware_tail_sensitivity", "version_aware_social_overlap_component_summary", "coaid_social_overlap_clusters"]
    tables = {n: pd.read_csv(TABLES / (n + ".csv"), keep_default_na=False) for n in names}
    fingerprints.update({str((TABLES / (n + ".csv")).relative_to(ROOT)): sha(TABLES / (n + ".csv")) for n in names})
    fingerprints[str(meta_path.relative_to(ROOT))] = sha(meta_path)
    fingerprints[str(script_path.relative_to(ROOT))] = sha(script_path)
    results = tables["version_aware_engagement_results"]
    results = results[results.outcome.eq("unique_post_count")].set_index("specification")
    attempts = tables["version_aware_bootstrap_refits"]
    boot_summary = tables["version_aware_bootstrap_summary"].set_index("specification")
    draws, verification = {}, []
    for spec in [FH] + COAID:
        records = attempts[attempts.specification.eq(spec)].sort_values("attempt")
        accepted = records[records.success.eq(True)]
        require(len(accepted) == 1000 and accepted.attempt.is_unique, f"Expected 1000 successful unique post draws: {spec}")
        require(records.seed.eq(42).all(), "Saved bootstrap seed mismatch")
        require(accepted.post_stable.eq(True).all() and accepted.post_converged.eq(True).all(), "Saved nonconvergent post draw marked successful")
        require(len(records) == boot_summary.loc[spec, "attempts"] and len(accepted) == boot_summary.loc[spec, "successful_post_refits"], "Summary/attempt count disagreement")
        ratios = accepted.mean_ratio.to_numpy(float)
        require(np.isfinite(ratios).all() and (ratios > 0).all(), "Invalid saved ratios")
        np.testing.assert_allclose(ratios, accepted.standardized_fake_mean.to_numpy(float) / accepted.standardized_real_mean.to_numpy(float), rtol=1e-12)
        np.testing.assert_allclose(np.quantile(ratios, [.025, .975]),
                                   results.loc[spec, ["primary_CI_lower", "primary_CI_upper"]].to_numpy(float), rtol=1e-12)
        row = results.loc[spec]
        np.testing.assert_allclose(row.standardized_ratio, row.standardized_fake_mean / row.standardized_real_mean, rtol=1e-12)
        draws[spec] = accepted
        verification.append(dict(specification=spec, successful_draws=len(accepted), failed_attempts=len(records) - len(accepted),
                                 finite_positive_ratios=True, saved_interval_reproduced=True, original_models_refitted=False))
    save("cross_context_input_validation", verification)
    return results, draws, tables, fingerprints


def independent_product(coaid, fakehealth):
    coaid, fakehealth = np.asarray(coaid, float), np.asarray(fakehealth, float)
    require(np.isfinite(coaid).all() and (coaid > 0).all(), "Invalid CoAID ratios")
    require(np.isfinite(fakehealth).all() and (fakehealth > 0).all(), "Invalid FakeHealth ratios")
    theta = (np.log(coaid)[:, None] - np.log(fakehealth)[None, :]).ravel()
    return theta, np.exp(theta)


def contrast_row(spec, coaid_point, fh_point, coaid_draws, fh_draws, role):
    theta, values = independent_product(coaid_draws, fh_draws)
    q = [.025, .25, .50, .75, .975]
    cq, tq = np.quantile(values, q), np.quantile(theta, q)
    point = float(coaid_point / fh_point)
    row = dict(coaid_specification=spec, analysis_role=role, fakehealth_ratio=float(fh_point), coaid_ratio=float(coaid_point),
               ratio_of_ratios=point, log_ratio_of_ratios=float(np.log(coaid_point) - np.log(fh_point)),
               CI_lower_ratio=cq[0], CI_upper_ratio=cq[-1], CI_lower_log=tq[0], CI_upper_log=tq[-1],
               bootstrap_median=cq[2], bootstrap_median_log=tq[2], skewness=float(skew(values, bias=True)),
               excess_kurtosis=float(kurtosis(values, fisher=True, bias=True)), skewness_log=float(skew(theta, bias=True)),
               excess_kurtosis_log=float(kurtosis(theta, fisher=True, bias=True)),
               ratio_interval_contains_1=bool(cq[0] <= 1 <= cq[-1]), log_interval_contains_0=bool(tq[0] <= 0 <= tq[-1]),
               coaid_successful_draws=len(coaid_draws), fakehealth_successful_draws=len(fh_draws),
               cartesian_combinations=len(values), uncertainty_method=METHOD,
               interpretation_note="Observational dataset-context contrast; distinct standardization populations; not a causal COVID effect; dependence graph and dominant-component limitations remain")
    for label, c, t in zip(["025", "25", "50", "75", "975"], cq, tq):
        row["percentile_" + label + "_ratio"] = c
        row["percentile_" + label + "_log"] = t
    return row, theta, values


_EXCLUSION = {}


def init_exclusion(config):
    global _EXCLUSION
    _EXCLUSION = config


def exclusion_draw(attempt):
    cfg = _EXCLUSION
    # Stream 4 differs from the four validated full-sample streams 0..3.
    rng = np.random.default_rng(np.random.SeedSequence([42, 4, attempt]))
    idx, sampled = resample_indices(cfg["clusters"], rng)
    x = pd.DataFrame(cfg["x"][idx], columns=cfg["columns"])
    row = dict(specification=NO_DOM, attempt=attempt, seed=42, seed_stream=4,
               sampled_N=len(idx), distinct_sampled_clusters=len(set(sampled)), success=False, error="")
    start = time.monotonic()
    try:
        require(np.linalg.matrix_rank(x) == x.shape[1], "Rank-deficient resampled design")
        fit, retried, trials = refit_checked(cfg["y"][idx], x, "Hurdle_Logit_ZTNB2", cfg["params"])
        row.update(converged=fit.converged, stable=fit.stable, hessian_ok=fit.hessian_ok, boundary=fit.boundary,
                   same_draw_retry=retried, optimizer_trials_json=trials, alpha=float(fit.params[-1]))
        require(fit.stable, "Nonconvergent/boundary/unidentified fit")
        metrics = point_metrics(fit, x)
        require(np.isfinite(metrics["mean_ratio"]) and metrics["mean_ratio"] > 0, "Invalid ratio")
        row.update(metrics, success=True)
    except Exception as error:
        row["error"] = repr(error)
    row["elapsed_seconds"] = time.monotonic() - start
    return row


def exclusion_bootstrap(tables, fingerprints, workers):
    assignments = tables["coaid_social_overlap_clusters"]
    assignments = assignments[assignments.specification.eq(COAID[0])]
    comp = tables["version_aware_social_overlap_component_summary"]
    comp = comp[comp.specification.eq(COAID[0])].sort_values(["post_memberships", "overlap_cluster_uid"], ascending=[False, True])
    dominant = comp.iloc[0]
    require(dominant.cluster_size == 187, "Unexpected dominant-component size")
    removed = set(assignments.loc[assignments.overlap_cluster_uid.eq(dominant.overlap_cluster_uid), "entity_uid"])
    require(len(removed) == 187, "Dominant membership disagreement")
    frame = pd.read_csv(ROOT / "data/processed/coaid_news_primary_version_aware.csv", keep_default_na=False)
    require(set(assignments.entity_uid) == set(frame.content_uid), "Incomplete original cluster mapping")
    frame["analysis_group"] = frame.coaid_within_dataset_analysis_group
    frame["overlap_cluster_uid"] = assignments.set_index("entity_uid").overlap_cluster_uid.reindex(frame.content_uid).to_numpy()
    full_N = len(frame)
    frame = add_release_columns(frame[~frame.content_uid.isin(removed)].copy()).reset_index(drop=True)
    require(len(frame) == 5150 and frame.overlap_cluster_uid.nunique() == 5055, "Unexpected exclusion sample")
    # An entire connected component is removed: every other component is intact.
    original_sizes = assignments.groupby("overlap_cluster_uid").size().drop(dominant.overlap_cluster_uid)
    pd.testing.assert_series_equal(frame.groupby("overlap_cluster_uid").size().sort_index(), original_sizes.sort_index())
    tails = tables["version_aware_tail_sensitivity"]
    prior = tails[tails.specification.eq(COAID[0]) & tails.outcome.eq("unique_post_count") &
                  tails.tail_specification.eq("exclude_largest_overlap_cluster")]
    require(len(prior) == 1 and int(prior.iloc[0].removed_N) == 187 and bool(prior.iloc[0].stable), "Missing validated exclusion point estimate")
    expected_point = float(prior.iloc[0].mean_ratio)
    cache_key = dict(input_sha256=fingerprints, cross_context_script_sha256=sha(Path(__file__)),
                     removed_overlap_cluster_uid=dominant.overlap_cluster_uid, removed_entity_uids=sorted(removed),
                     seed=42, seed_stream=4, target_successful_refits=1000)
    if CACHE.exists() and CACHE_META.exists():
        cached_meta = json.loads(CACHE_META.read_text())
        require(cached_meta["cache_key"] == cache_key, "Exclusion bootstrap cache fingerprint mismatch; do not mix runs")
        draws = pd.read_csv(CACHE, keep_default_na=False).to_dict("records")
        parameters = np.asarray(cached_meta["parameters"], float)
        point = float(cached_meta["point_ratio"])
    else:
        require(not CACHE.exists() and not CACHE_META.exists(), "Partial exclusion cache; investigate before overwriting")
        x = model_design(COAID[0], frame)
        fitted = fit_hurdle(frame.unique_post_count.to_numpy(float), x)
        require(fitted.stable, "Exclusion primary fit unstable")
        point = point_metrics(fitted, x)["mean_ratio"]
        np.testing.assert_allclose(point, expected_point, rtol=1e-8)
        parameters = fitted.params
        draws = []
        cached_meta = dict(cache_key=cache_key, parameters=parameters.tolist(), point_ratio=point,
                           full_N=full_N, retained_N=len(frame), retained_clusters=frame.overlap_cluster_uid.nunique(),
                           removed_N=len(removed), removed_post_memberships=int(dominant.post_memberships),
                           removed_post_share=float(dominant.post_membership_share), complete=False)
        CACHE_META.write_text(json.dumps(cached_meta, indent=2, sort_keys=True) + "\n")
    np.testing.assert_allclose(point, expected_point, rtol=1e-8)
    x = model_design(COAID[0], frame)
    clusters = [np.flatnonzero(frame.overlap_cluster_uid.eq(uid).to_numpy()) for uid in sorted(frame.overlap_cluster_uid.unique())]
    config = dict(clusters=clusters, x=x.to_numpy(), columns=list(x.columns), y=frame.unique_post_count.to_numpy(float), params=parameters)
    success = sum(bool(r["success"]) for r in draws)
    require(len({r["attempt"] for r in draws}) == len(draws), "Duplicate exclusion attempt")
    attempted = max([int(r["attempt"]) for r in draws], default=-1) + 1
    if success < 1000:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_exclusion, initargs=(config,)) as pool:
            while success < 1000 and attempted < 1250:
                n = min(20 if attempted == 0 else 100, 1000 - success, 1250 - attempted)
                start = time.monotonic()
                batch = list(pool.map(exclusion_draw, range(attempted, attempted + n)))
                draws.extend(batch)
                success += sum(r["success"] for r in batch)
                attempted += n
                save("cross_context_no_dominant_bootstrap_refits", draws)
                print(f"EXCLUSION BOOTSTRAP {success} successful / {attempted} attempts; batch wall {time.monotonic()-start:.1f}s", flush=True)
    require(success == 1000, "Exclusion bootstrap did not reach 1000 successes; failures retained in cache")
    draws = pd.DataFrame(draws)
    accepted = draws[draws.success.eq(True)]
    require(accepted.stable.eq(True).all() and accepted.converged.eq(True).all(), "Unstable cached draw accepted")
    require(np.isfinite(accepted.mean_ratio.to_numpy(float)).all() and accepted.mean_ratio.gt(0).all(), "Invalid cached draw")
    cached_meta.update(complete=True, attempts=len(draws), successful_refits=len(accepted), failed_refits=len(draws)-len(accepted),
                       bootstrap_csv_sha256=sha(CACHE))
    CACHE_META.write_text(json.dumps(cached_meta, indent=2, sort_keys=True) + "\n")
    return point, accepted.mean_ratio.to_numpy(float), cached_meta


def dominant_diagnostic(a_draws, fh_draws):
    rows = []
    for label, mask in (("0", a_draws.largest_component_draws.eq(0)),
                         ("1", a_draws.largest_component_draws.eq(1)),
                         ("2_or_more", a_draws.largest_component_draws.ge(2))):
        values = a_draws.loc[mask, "mean_ratio"].to_numpy(float)
        theta, cross = independent_product(values, fh_draws)
        aq = np.quantile(values, [.25, .50, .75])
        cq = np.quantile(cross, [.025, .25, .50, .75, .975])
        rows.append(dict(dominant_component_multiplicity=label, bootstrap_draws=len(values), draw_fraction=len(values)/len(a_draws),
                         median_coaid_ratio=aq[1], coaid_ratio_q25=aq[0], coaid_ratio_q75=aq[2], coaid_ratio_IQR=aq[2]-aq[0],
                         median_cross_context_C=cq[2], cross_context_C_q25=cq[1], cross_context_C_q75=cq[3], cross_context_C_IQR=cq[3]-cq[1],
                         conditional_C_percentile_025=cq[0], conditional_C_percentile_975=cq[4], cartesian_pairs=len(cross),
                         notes="Diagnostic conditional bootstrap subset, NOT an alternative primary CI or independent effect test"))
    require(sum(r["bootstrap_draws"] for r in rows) == 1000, "Multiplicity classes do not partition original draws")
    return pd.DataFrame(rows)


def density_diagnostic(theta, values):
    """Visual smoothing check only; no formal modality test or pseudo-N=1e6.

    Use a fixed 20,000-product-value subsample for fast KDE evaluation; percentile
    intervals/moments always use all 1,000,000 values. Bandwidth scales with 1000
    saved draws, not the misleading Cartesian pair count as an information N.
    """
    selection = np.random.default_rng(42).choice(len(values), min(20000, len(values)), replace=False)
    rows, curves = [], {}
    for scale, all_values in (("ratio", values), ("log", theta)):
        subset = all_values[selection]
        grid = np.linspace(all_values.min(), all_values.max(), 600)
        curves[scale] = []
        for factor in (.75, 1., 1.5, 2.):
            kde = gaussian_kde(subset, bw_method=factor * 1000**(-.2))
            density = kde(grid)
            peaks, _ = find_peaks(density, prominence=.05 * density.max())
            locations = grid[peaks]
            rows.append(dict(scale=scale, bandwidth_multiplier=factor, kde_factor=kde.factor,
                             smoothing_sample_N=len(subset), reference_draw_count=1000,
                             prominent_peak_count=len(peaks), prominent_peak_locations_json=json.dumps(locations.tolist()),
                             prominence_rule="5% of maximum KDE height; interior peaks on fixed 600-point grid",
                             inference="Descriptive bandwidth sensitivity only; not a modality hypothesis test"))
            curves[scale].append((factor, grid, density))
    return pd.DataFrame(rows), curves


def figures(results, within, distributions, curves, diagnostics):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = results.iloc[0]
    for scale, vals, point, lower, upper in (
        ("ratio", distributions[COAID[0]][1], primary.ratio_of_ratios, primary.CI_lower_ratio, primary.CI_upper_ratio),
        ("log", distributions[COAID[0]][0], primary.log_ratio_of_ratios, primary.CI_lower_log, primary.CI_upper_log)):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        ax.hist(vals, bins=70, density=True, alpha=.35, color="#4477aa", label="Independent empirical product")
        for factor, grid, density in curves[scale]:
            if factor in (1., 1.5):
                ax.plot(grid, density, label=f"KDE bandwidth ×{factor:g}", linewidth=1.5)
        ax.axvline(point, color="#bb5544", label="Point estimate")
        ax.axvline(lower, color="black", linestyle="--", label="95% percentile interval")
        ax.axvline(upper, color="black", linestyle="--")
        ax.set(xlabel="A/FakeHealth ratio-of-ratios C" if scale == "ratio" else "Log contrast θ = log(R_A) − log(R_FH)", ylabel="Density",
               title="Primary independent-bootstrap contrast; 1,000 × 1,000 combinations")
        ax.legend(fontsize=8)
        fig.savefig(FIGURES / ("primary_ratio_of_ratios_histogram.png" if scale == "ratio" else "primary_log_contrast_histogram.png"), dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, labels, estimates, lows, highs, title in (
        (axes[0], ["FakeHealth R", "CoAID A R", "CoAID B R", "CoAID C R"],
         within.loc[[FH]+COAID, "standardized_ratio"], within.loc[[FH]+COAID, "primary_CI_lower"],
         within.loc[[FH]+COAID, "primary_CI_upper"], "Within-dataset Fake/Real ratios"),
        (axes[1], ["A/FakeHealth — primary", "B/FakeHealth — sensitivity", "C/FakeHealth — sensitivity"],
         results.ratio_of_ratios, results.CI_lower_ratio, results.CI_upper_ratio, "Cross-context ratios-of-ratios")):
        for i, (estimate, lo, hi) in enumerate(zip(estimates, lows, highs)):
            ax.plot([lo, hi], [i, i], color="#4477aa"); ax.scatter([estimate], [i], color="#4477aa", s=30)
        ax.axvline(1, color="black", linestyle=":")
        ax.set_xscale("log"); ax.set(yticks=np.arange(len(labels)), yticklabels=labels, title=title, xlabel="Ratio (log axis); 95% bootstrap interval")
        ax.invert_yaxis()
    fig.savefig(FIGURES / "within_and_cross_context_forest.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, row in enumerate(diagnostics.itertuples()):
        ax.plot([row.cross_context_C_q25, row.cross_context_C_q75], [i, i], linewidth=5, color="#4477aa")
        ax.scatter([row.median_cross_context_C], [i], color="#bb5544", zorder=3)
    ax.set(yticks=np.arange(3), yticklabels=["0 draws", "1 draw", "2+ draws"], xlabel="Cross-context C: median and IQR",
           title="Dominant-component multiplicity: diagnostic, not three effects")
    fig.savefig(FIGURES / "dominant_component_multiplicity.png", dpi=170); plt.close(fig)


def report(results, sensitivities, diagnostic, modes, agreement, exclusion_meta, within):
    a = results.iloc[0]
    excluded = sensitivities[sensitivities.sensitivity.eq("exclude_dominant_component")].iloc[0]
    mode_summary = "; ".join(f"{scale}: " + ", ".join(f"×{r.bandwidth_multiplier:g} → {r.prominent_peak_count} peaks" for r in part.itertuples())
                             for scale, part in modes.groupby("scale", sort=False))
    result_sentence = (f"The estimated Fake/Real observed-post ratio was smaller in CoAID than in FakeHealth "
                       f"(standardized ratios {a.coaid_ratio:.3f} and {a.fakehealth_ratio:.3f}, respectively; "
                       f"ratio-of-ratios {a.ratio_of_ratios:.3f}, 95% independent overlap-cluster bootstrap percentile interval "
                       f"{a.CI_lower_ratio:.3f}–{a.CI_upper_ratio:.3f}).")
    limitation = ("This observational contrast cannot isolate a COVID-period effect because dataset and period/context are confounded; "
                  "ground-truth definitions, engagement collection procedures, and observation windows differ, while CoAID's dominant "
                  "social-overlap component, incompletely captured dependence, incomplete diffusion-cascade coverage, and missing "
                  "unobserved/deleted platform activity limit interpretation.")
    rows = ["# Cross-context observed-post engagement contrast", "", "## Primary finding", "", result_sentence, "",
            f"The primary log contrast is θ={a.log_ratio_of_ratios:.4f}, with percentile 95% interval "
            f"[{a.CI_lower_log:.4f}, {a.CI_upper_log:.4f}]. The C interval contains 1: {a.ratio_interval_contains_1}; "
            f"the θ interval contains 0: {a.log_interval_contains_0}. No causal p-value is calculated.", "",
            markdown_table(results[["coaid_specification", "fakehealth_ratio", "coaid_ratio", "ratio_of_ratios", "CI_lower_ratio", "CI_upper_ratio", "log_ratio_of_ratios", "CI_lower_log", "CI_upper_log", "bootstrap_median"]]), "",
            "## Estimand, validation, and exact uncertainty procedure", "",
            "R is the model-standardized expected unique-post count under Fake divided by that under Real, averaged over all content items including zeros. CoAID retains its first-release-adjusted full logistic × zero-truncated-NB hurdle mean; FakeHealth retains its NB mean. Each context keeps its own validated reference population; this is not a pooled or common-covariate-population causal effect.", "",
            "The validated within-analysis completion flag, seed 42, script hash, and every recorded input hash were checked. Each context has exactly 1,000 successful, finite positive post-ratio draws; convergence flags, standardized-mean identities, saved CI endpoints, and draw-summary counts were reconciled. The original run did not record output-file hashes, so this stage fingerprints the saved artifacts now and checks their internal consistency; it does not claim a nonexistent historical output checksum. None of the original 4,000 bootstrap models was refitted.", "",
            "Treating FakeHealth and CoAID as independent samples as prespecified, form every ordered combination (i,j) of the 1,000 CoAID and 1,000 FakeHealth draws. Compute θ_ij=log(R_CoAID,i)−log(R_FH,j), then C_ij=exp(θ_ij). All 1,000,000 combinations have equal empirical weight; identical replicate numbers are not matched observations. NumPy's linear empirical quantiles supply percentile intervals. The original point estimate is the ratio of fitted ratios, NOT the bootstrap median. The Cartesian values share their marginal draws: one million combinations do not create one million independent observations or increase original bootstrap information. Reordering either marginal leaves the result unchanged.", "",
            "## Quantiles and distribution shape", "",
            markdown_table(results[["coaid_specification", "percentile_025_ratio", "percentile_25_ratio", "percentile_50_ratio", "percentile_75_ratio", "percentile_975_ratio", "skewness", "excess_kurtosis", "skewness_log", "excess_kurtosis_log"]]), "",
            markdown_table(results[["coaid_specification", "percentile_025_log", "percentile_25_log", "percentile_50_log", "percentile_75_log", "percentile_975_log"]]), "",
            f"Primary C has skewness {a.skewness:.3f}: moderate right skew under the descriptive convention |skew|<0.5 small, 0.5–1 moderate, >1 strong. θ skewness is {a.skewness_log:.3f}. These are moments of the empirical product distribution, not tests treating the combinations as independent observations.", "",
            "Histogram/KDE figures examine both scales. KDE uses a fixed seed-42 sample of 20,000 product values for display only; all intervals/moments use all one million. The smoothing factor is multiplier ×1000^(−1/5), varied over 0.75, 1, 1.5, 2; one million is deliberately not used as an information sample size for bandwidth choice. Prominent interior density peaks use a 5%-of-maximum-density prominence threshold on a 600-point grid. This is a descriptive bandwidth check, not a reliable formal modality test; modality also need not be invariant to a log transform.", "",
            mode_summary + ". A shoulder or multiple visible bumps are not by themselves proof of distinct population modes. The multiplicity table below establishes resampling-mixture sensitivity without making that claim.", "",
            "## Agreement across entity/version specifications", "",
            markdown_table(agreement), "",
            "All A/B/C point estimates and 95% intervals are below 1. Their intervals overlap extensively; common overlap and pairwise intersection-over-union lengths are reported as descriptive interval comparisons, not tests of differences among dependent specifications. Entity/version choices change magnitude but do not reverse this qualitative observational result. B/C remain sensitivity analyses, not independent replications.", "",
            "## Dominant-component influence", "",
            f"The removed A component has {exclusion_meta['removed_N']} entities and {100*exclusion_meta['removed_post_share']:.1f}% of all post memberships. "
            f"The retained sample has {exclusion_meta['retained_N']} entities and {exclusion_meta['retained_clusters']} intact overlap components. "
            f"Its previously validated CoAID point ratio is reproduced at {excluded.coaid_ratio:.4f}. The new cross-context sensitivity gives "
            f"C={excluded.ratio_of_ratios:.4f} [{excluded.CI_lower_ratio:.4f}, {excluded.CI_upper_ratio:.4f}], "
            f"a change of {excluded.change_from_primary:+.4f} ({excluded.relative_change_percent:+.1f}%) from primary A.", "",
            f"No validated exclusion bootstrap existed: a NEW first-release-adjusted hurdle overlap-component bootstrap was fitted only for this sensitivity. "
            f"It produced {exclusion_meta['successful_refits']} successes in {exclusion_meta['attempts']} attempts, with {exclusion_meta['failed_refits']} failed attempts. "
            "SeedSequence([42,4,attempt]) supplies a separate reproducible stream. Each replicate samples the 5,055 retained components with replacement and includes every member of each sampled component, including repeats and singletons; its empirical reference covariate mix is re-standardized. Logged failures are not silently replaced; same-draw multistart retries retain diagnostics. This changes the target population and is an influence sensitivity, NOT the new primary.", "",
            markdown_table(diagnostic[["dominant_component_multiplicity", "bootstrap_draws", "median_coaid_ratio", "coaid_ratio_q25", "coaid_ratio_q75", "coaid_ratio_IQR", "median_cross_context_C", "cross_context_C_q25", "cross_context_C_q75", "cross_context_C_IQR", "conditional_C_percentile_025", "conditional_C_percentile_975"]]), "",
            "Within each multiplicity class, every original A draw is combined with all 1,000 independent FakeHealth draws. The classes describe conditioning on the ORIGINAL bootstrap multiplicity; class 0 is not identical to the newly refitted exclusion bootstrap (the original draw samples K=5,056 components, rather than the exclusion sample's K=5,055). Their medians/IQRs and conditional ranges show how the dominant component shifts the primary distribution. These are neither new effects nor interchangeable alternative 95% confidence intervals. Percentile uncertainty remains sensitive to a dominant component being omitted or sampled repeatedly, even if smoothing does not establish strong multimodality.", "",
            "## Tail sensitivities", "", markdown_table(sensitivities[["sensitivity", "coaid_ratio", "ratio_of_ratios", "CI_lower_ratio", "CI_upper_ratio", "change_from_primary", "relative_change_percent", "uncertainty_method"]]), "",
            "All-retained uses the primary empirical-product interval. Top-1% and top-five exclusions use ONLY the previously validated CoAID A point estimates divided by the unchanged full-sample FakeHealth ratio. No validated tail bootstrap exists, so their CI cells are missing; HC0 intervals are never combined with cluster intervals. Outcome-based exclusion changes the target population and remains diagnostic.", "",
            "## Interpretation, reporting wording, and next stage", "",
            "The direction is sufficiently consistent across the prespecified A/B/C and influence checks to report A as the primary observational dataset-context finding, WITH its cluster-percentile interval and dominant-component caveat. It is not evidence that COVID caused the difference, nor definitive proof of an intrinsic context mechanism. Dominance can undermine regular many-small-cluster bootstrap approximations; the interval is conditional on the observed graph, selected models, and observed coverage and is not a guarantee of nominal population coverage.", "",
            "**Results sentence:** “" + result_sentence + "”", "",
            "**Accompanying limitation sentence:** “" + limitation + "”", "",
            "Additional explicit limitations: dataset and period/context are confounded; ground-truth definitions differ; collection procedures and observation windows differ; CoAID has one dominant social-overlap component; the dependence graph only partially captures social-ID overlap and other dependence; observed post counts are not complete diffusion cascades; unobserved/deleted platform activity may be missing. Cross-context sample independence is assumed, not empirically established by separate within-context graphs.", "",
            "The approximately 50-percentage-point occurrence gap stays within CoAID; no engagement_present cross-context contrast is calculated because FakeHealth is near ceiling. Exploratory reply contrasts are omitted so provisional reply estimates do not enter the primary conclusion.", "",
            "TF-IDF text analysis and cross-context transfer are a reasonable NEXT, separately authorized stage. Before fitting, specify leakage-safe resource/version/overlap grouping, label harmonization, and explicit within- versus cross-context evaluation; otherwise provenance duplicates and incompatible label meanings can masquerade as transfer performance. No text modeling, classification, transformers, or COVID-Rumor robustness is started here.", "",
            "## Reproduction and artifacts", "",
            "Run `.venv/bin/python src/run_cross_context_engagement_contrast.py --workers 4`. Validated no-dominant bootstrap draws are reused on rerun only when fingerprints match. New outputs use cross_context filenames and a separate figure directory; all raw/processed inputs and previous results remain unchanged. The metadata records input hashes, original refits reused, new sensitivity refits, pair counts, and verification status. Test with `.venv/bin/python -m unittest discover -s src -p 'test_cross_context_engagement_contrast.py'`."]
    REPORT.write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(args.workers > 0, "Positive worker count required")
    within, draws, tables, fingerprints = validated_inputs()
    print("Validated all input/script fingerprints and 4 x 1000 saved successful post-ratio draws; no original bootstrap refits.", flush=True)
    fh_point = float(within.loc[FH, "standardized_ratio"])
    fh_draws = draws[FH].mean_ratio.to_numpy(float)
    rows, distributions = [], {}
    for spec in COAID:
        row, theta, values = contrast_row(spec, float(within.loc[spec, "standardized_ratio"]), fh_point,
                                           draws[spec].mean_ratio.to_numpy(float), fh_draws,
                                           "primary observational" if spec == COAID[0] else "entity/version sensitivity")
        rows.append(row); distributions[spec] = (theta, values)
    results = save("cross_context_engagement_contrast", rows)
    no_dom_point, no_dom_draws, exclusion_meta = exclusion_bootstrap(tables, fingerprints, args.workers)
    excluded, _, _ = contrast_row(NO_DOM, no_dom_point, fh_point, no_dom_draws, fh_draws, "dominant-component influence sensitivity")
    sensitivities = []
    tails = tables["version_aware_tail_sensitivity"]
    for tail in ("all_retained", "exclude_top_1pct", "exclude_top_5_items"):
        prior = tails[tails.specification.eq(COAID[0]) & tails.outcome.eq("unique_post_count") & tails.tail_specification.eq(tail)]
        require(len(prior) == 1 and bool(prior.iloc[0].stable), "Missing/unstable validated tail point")
        ratio = float(prior.iloc[0].mean_ratio) / fh_point
        sensitivities.append(dict(sensitivity=tail, coaid_specification=COAID[0], fakehealth_ratio=fh_point,
                                  coaid_ratio=float(prior.iloc[0].mean_ratio), ratio_of_ratios=ratio, log_ratio_of_ratios=np.log(ratio),
                                  CI_lower_ratio=rows[0]["CI_lower_ratio"] if tail == "all_retained" else np.nan,
                                  CI_upper_ratio=rows[0]["CI_upper_ratio"] if tail == "all_retained" else np.nan,
                                  change_from_primary=ratio-rows[0]["ratio_of_ratios"], relative_change_percent=100*(ratio/rows[0]["ratio_of_ratios"]-1),
                                  uncertainty_method=METHOD if tail == "all_retained" else "Point only; no valid saved overlap-bootstrap distribution; no mixed HC0/cluster CI"))
    sensitivities.append(dict(excluded, sensitivity="exclude_dominant_component",
                              change_from_primary=excluded["ratio_of_ratios"]-rows[0]["ratio_of_ratios"],
                              relative_change_percent=100*(excluded["ratio_of_ratios"]/rows[0]["ratio_of_ratios"]-1)))
    sensitivities = save("cross_context_sensitivity_results", sensitivities)
    diagnostic = save("cross_context_dominant_component_diagnostic", dominant_diagnostic(draws[COAID[0]], fh_draws))
    modes, curves = density_diagnostic(*distributions[COAID[0]])
    save("cross_context_modality_diagnostic", modes)
    agreement_rows = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        a, b = rows[left], rows[right]
        low, high = max(a["CI_lower_ratio"], b["CI_lower_ratio"]), min(a["CI_upper_ratio"], b["CI_upper_ratio"])
        union_width = max(a["CI_upper_ratio"], b["CI_upper_ratio"]) - min(a["CI_lower_ratio"], b["CI_lower_ratio"])
        agreement_rows.append(dict(comparison=a["coaid_specification"]+" versus "+b["coaid_specification"],
                                   intersection_lower=low, intersection_upper=high,
                                   intersection_over_union=max(0., high-low)/union_width,
                                   relative_point_change_percent=100*(b["ratio_of_ratios"]/a["ratio_of_ratios"]-1)))
    agreement = save("cross_context_specification_agreement", agreement_rows)
    figures(results, within, distributions, curves, diagnostic)
    report(results, sensitivities, diagnostic, modes, agreement, exclusion_meta, within)
    for relative, expected in fingerprints.items():
        require(sha(ROOT / relative) == expected, f"Input changed during contrast: {relative}")
    metadata = dict(input_sha256=fingerprints, cross_context_script_sha256=sha(Path(__file__)),
                    sample_independence="assumed by prespecified design", original_successful_draws_reused=4000,
                    original_bootstrap_models_refitted=0, new_exclusion_bootstrap_successes=1000,
                    cartesian_pairs_per_specification=1000000, percentiles_method="numpy.quantile linear",
                    common_ABC_CI_intersection=[float(results.CI_lower_ratio.max()), float(results.CI_upper_ratio.min())],
                    all_points_below_1=bool(results.ratio_of_ratios.lt(1).all()),
                    all_ratio_intervals_below_1=bool(results.CI_upper_ratio.lt(1).all()),
                    input_fingerprints_verified=True, no_input_changes=True, no_causal_p_value=True,
                    no_binary_cross_context_contrast=True, no_text_analysis=True, complete=True)
    (TABLES / "cross_context_engagement_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    print(results[["coaid_specification", "ratio_of_ratios", "CI_lower_ratio", "CI_upper_ratio", "log_ratio_of_ratios", "CI_lower_log", "CI_upper_log"]].to_string(index=False))
    print(sensitivities[["sensitivity", "coaid_ratio", "ratio_of_ratios", "CI_lower_ratio", "CI_upper_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
