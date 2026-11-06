"""Corrected within-dataset EDA/modeling and social-component bootstrap.

Run: .venv/bin/python src/run_version_aware_engagement_analysis.py
Optional pilot: --pilot; continuation: --resume. Seed 42; default target 1,000
successful primary post refits per sample, with every attempted draw recorded.
No entity-rule changes, raw writes, or between-dataset contrasts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

# Avoid nested BLAS oversubscription in the bootstrap process pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.special import expit
from statsmodels.discrete.discrete_model import Logit, NegativeBinomialP
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP

from audit_engagement_estimand_robustness import (
    add_release_columns, delta_interval, design, standardized_counts,
    support_fraction, tail_sets, verify_full_hurdle_mean,
)
from build_primary_analysis_samples import DSU
from run_primary_engagement_models import (
    Fit, comparison_row, fit_hurdle, fit_one, markdown_table, moments, optimize,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures/version_aware_engagement"
REPORT = ROOT / "outputs/reports/version_aware_engagement_analysis.md"
SAMPLES = {
    "A_primary_strict": ("coaid_news_primary_version_aware.csv", 5337, 922, 4415),
    "B_resource_sensitivity": ("coaid_news_resource_level_sensitivity.csv", 5205, 921, 4284),
    "C_high_certainty": ("coaid_news_high_certainty_sensitivity.csv", 5082, 920, 4162),
    "FakeHealth_HealthStory": ("fakehealth_healthstory_primary.csv", 1690, 472, 1218),
}
BOOT_PATH = TABLES / "version_aware_bootstrap_refits.csv"
RUN_META = TABLES / "version_aware_engagement_run_metadata.json"
BOOT_METRICS = ["standardized_fake_mean", "standardized_real_mean", "mean_difference", "mean_ratio",
                "post_probability_fake", "post_probability_real", "post_probability_difference", "post_odds_ratio",
                "positive_fake_mean", "positive_real_mean", "positive_mean_ratio",
                "any_probability_fake", "any_probability_real", "any_probability_difference", "any_odds_ratio",
                "reply_mean_ratio"]


def require(value, message):
    if not value:
        raise ValueError(message)


def js(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(TABLES / (name + ".csv"), index=False)
    return frame


def load_samples():
    frames, hashes, confirmation = {}, {}, []
    for spec, (filename, n, fake_n, real_n) in SAMPLES.items():
        path = ROOT / "data/processed" / filename
        frame = pd.read_csv(path, keep_default_na=False)
        label = "fakehealth_within_dataset_analysis_group" if spec.startswith("FakeHealth") else "coaid_within_dataset_analysis_group"
        frame["analysis_group"] = frame[label]
        require((len(frame), sum(frame.analysis_group.eq("fake")), sum(frame.analysis_group.eq("real"))) == (n, fake_n, real_n), "Sample composition drift")
        require(frame.content_uid.is_unique, "Nonunique sample entity UID")
        for col in ("unique_post_count", "unique_reply_count"):
            require(frame[col].ge(0).all() and frame[col].mod(1).eq(0).all(), "Invalid count")
        require((frame.engagement_present == frame[["unique_post_count", "unique_reply_count"]].gt(0).any(axis=1)).all(), "Presence/count mismatch")
        if not spec.startswith("FakeHealth"):
            frame = add_release_columns(frame)
        frames[spec] = frame
        hashes[str(path.relative_to(ROOT))] = sha(path)
        confirmation.append(dict(specification=spec, N=n, fake_N=fake_n, real_N=real_n,
                                 posts=int(frame.unique_post_count.sum()), replies=int(frame.unique_reply_count.sum())))
    overlap_path = TABLES / "coaid_cross_entity_social_overlap.csv"
    overlaps = pd.read_csv(overlap_path, dtype={"social_id": str}, keep_default_na=False)
    hashes[str(overlap_path.relative_to(ROOT))] = sha(overlap_path)
    # Verify the samples/overlap against each other when constructing graphs.
    # FakeHealth uses only counted tweets/replies, NEVER unmodeled retweets.
    fh = frames["FakeHealth_HealthStory"]
    provenance = fh.provenance.map(json.loads)
    paths = {p["raw_engagement_file"] for p in provenance}
    require(len(paths) == 1, "Unexpected FakeHealth mapping layout")
    mapping_path = ROOT / next(iter(paths))
    mapping = json.loads(mapping_path.read_text())
    hashes[str(mapping_path.relative_to(ROOT))] = sha(mapping_path)
    fh_overlap = []
    for kind, field, count_col in (("post", "tweets", "unique_post_count"), ("reply", "replies", "unique_reply_count")):
        owners = defaultdict(set)
        for row in fh.itertuples():
            key = json.loads(row.provenance)["engagement_mapping_key"]
            ids = set(map(str, mapping[key][field]))
            require(len(ids) == getattr(row, count_col), "FakeHealth raw mapping count mismatch")
            for identifier in ids:
                owners[identifier].add(row.content_uid)
        for identifier, members in sorted(owners.items()):
            if len(members) > 1:
                fh_overlap.append(dict(specification="FakeHealth_HealthStory", social_type=kind,
                                       social_id=identifier, final_entity_uids_json=js(sorted(members))))
    save("fakehealth_version_aware_social_overlap", fh_overlap)
    overlaps = pd.concat([overlaps, pd.DataFrame(fh_overlap)], ignore_index=True)
    # Preserve and fingerprint earlier substantive results without modifying them.
    for path in (ROOT / "data/processed/coaid_news_primary.csv", TABLES / "common_engagement_estimand.csv",
                 TABLES / "primary_engagement_model_results.csv", TABLES / "primary_engagement_descriptives.csv",
                 ROOT / "src/run_primary_engagement_models.py", ROOT / "src/run_primary_engagement_eda.py",
                 ROOT / "src/audit_engagement_estimand_robustness.py"):
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = sha(path)
    return frames, overlaps, hashes, confirmation


def overlap_graph(spec, frame, overlaps):
    nodes = sorted(frame.content_uid)
    dsu = DSU(nodes)
    neighbors = {kind: defaultdict(set) for kind in ("post", "reply")}
    typed_ids = {kind: defaultdict(int) for kind in ("post", "reply")}
    for r in overlaps[overlaps.specification.eq(spec)].itertuples():
        group = sorted(json.loads(r.final_entity_uids_json))
        require(len(group) == len(set(group)) and len(group) >= 2, "Invalid overlap ownership")
        require(set(group) <= set(nodes), "Overlap references missing entity")
        for uid in group[1:]:
            dsu.union(group[0], uid)
        for uid in group:
            typed_ids[r.social_type][uid] += 1
            neighbors[r.social_type][uid].update(set(group) - {uid})
    groups = sorted([sorted(g) for g in dsu.groups()], key=lambda g: (-len(g), g))
    index = frame.set_index("content_uid")
    rows, components = [], []
    for group in groups:
        cid = "social_overlap_" + hashlib.sha256(js(group).encode()).hexdigest()[:24]
        part = index.loc[group]
        fake_n = int(part.analysis_group.eq("fake").sum())
        posts = int(part.unique_post_count.sum())
        components.append(dict(specification=spec, overlap_cluster_uid=cid, cluster_size=len(group),
                               fake_N=fake_n, real_N=len(group) - fake_n,
                               composition="mixed" if 0 < fake_n < len(group) else "fake_only" if fake_n else "real_only",
                               post_memberships=posts, post_membership_share=posts / frame.unique_post_count.sum(),
                               entity_share=len(group) / len(frame)))
        for uid in group:
            rows.append(dict(specification=spec, entity_uid=uid, overlap_cluster_uid=cid, cluster_size=len(group),
                             shared_post_degree=len(neighbors["post"][uid]), shared_reply_degree=len(neighbors["reply"][uid]),
                             shared_post_id_count=typed_ids["post"][uid], shared_reply_id_count=typed_ids["reply"][uid],
                             veracity=index.loc[uid, "analysis_group"]))
    assignments = pd.DataFrame(rows)
    comp = pd.DataFrame(components)
    if not spec.startswith("FakeHealth"):
        for kind in ("post", "reply"):
            observed = assignments.set_index("entity_uid")[f"shared_{kind}_id_count"].reindex(frame.content_uid).to_numpy()
            require(np.array_equal(observed, frame[f"cross_entity_{kind}_overlap_count"]), "Overlap audit/sample count mismatch")
    multi = comp.cluster_size.gt(1)
    weights = comp.post_memberships.to_numpy(float)
    sizes = comp.cluster_size.to_numpy(float)
    summary = dict(specification=spec, overlap_components=len(comp), singleton_components=int((~multi).sum()),
                   multi_entity_components=int(multi.sum()), median_component_size=float(comp.cluster_size.median()),
                   maximum_component_size=int(comp.cluster_size.max()), entities_in_non_singletons=int(comp.loc[multi, "cluster_size"].sum()),
                   percent_entities_in_non_singletons=100 * comp.loc[multi, "cluster_size"].sum() / len(frame),
                   percent_post_memberships_in_non_singletons=100 * comp.loc[multi, "post_memberships"].sum() / weights.sum(),
                   fake_only_components=int(comp.composition.eq("fake_only").sum()), real_only_components=int(comp.composition.eq("real_only").sum()),
                   mixed_components=int(comp.composition.eq("mixed").sum()),
                   largest_component_post_share=float(comp.post_membership_share.max()),
                   top3_components_post_share=float(comp.post_membership_share.nlargest(3).sum()),
                   entity_weight_kish_equivalent=sizes.sum()**2 / np.square(sizes).sum(),
                   post_weight_kish_equivalent=weights.sum()**2 / np.square(weights).sum(),
                   concentration_warning=bool(comp.post_membership_share.max() >= .20 or len(comp) < 50))
    # Kish equivalents are concentration diagnostics, NOT independent cluster counts.
    frame = frame.copy()
    frame["overlap_cluster_uid"] = assignments.set_index("entity_uid").overlap_cluster_uid.reindex(frame.content_uid).to_numpy()
    return frame, assignments, comp, summary


def descriptive(spec, frame):
    rows, effects = [], []
    for outcome in ("unique_post_count", "unique_reply_count", "engagement_present"):
        for label in ("fake", "real"):
            y = frame.loc[frame.analysis_group.eq(label), outcome].to_numpy(float)
            ranked = np.sort(y)[::-1]
            total = y.sum()
            row = dict(specification=spec, outcome=outcome, veracity=label, N=len(y), mean=y.mean(), median=np.median(y),
                       SD=y.std(ddof=1), IQR=np.quantile(y, .75) - np.quantile(y, .25),
                       percentile_90=np.quantile(y, .90), percentile_95=np.quantile(y, .95), maximum=y.max(),
                       variance=y.var(ddof=1), variance_to_mean=y.var(ddof=1) / y.mean() if y.mean() else np.nan,
                       zero_count=int((y == 0).sum()), zero_percentage=100 * np.mean(y == 0),
                       top_1pct_share=ranked[:math.ceil(.01 * len(y))].sum() / total if total else np.nan,
                       top_5pct_share=ranked[:math.ceil(.05 * len(y))].sum() / total if total else np.nan)
            rows.append(row)
        fake = frame.loc[frame.analysis_group.eq("fake"), outcome].to_numpy(float)
        real = frame.loc[frame.analysis_group.eq("real"), outcome].to_numpy(float)
        effects.append(dict(specification=spec, outcome=outcome, fake_N=len(fake), real_N=len(real),
                            fake_mean=fake.mean(), real_mean=real.mean(), mean_difference=fake.mean() - real.mean(),
                            mean_ratio=fake.mean() / real.mean(), median_difference=np.median(fake) - np.median(real),
                            percentage_point_difference=100 * (fake.mean() - real.mean()) if outcome == "engagement_present" else np.nan,
                            notes="Descriptive only; all observations retained; no row-independent inferential CI"))
    return rows, effects


def model_design(spec, frame):
    return design(frame, "no_release_adjustment" if spec.startswith("FakeHealth") else "first_release")


def point_metrics(fit, x):
    real, fake = np.asarray(x).copy(), np.asarray(x).copy()
    real[:, 1], fake[:, 1] = 0., 1.
    mf, mr = moments(fit, fake), moments(fit, real)
    return dict(standardized_fake_mean=float(mf[0].mean()), standardized_real_mean=float(mr[0].mean()),
                mean_difference=float(mf[0].mean() - mr[0].mean()), mean_ratio=float(mf[0].mean() / mr[0].mean()),
                post_probability_fake=float((1 - mf[2]).mean()), post_probability_real=float((1 - mr[2]).mean()),
                post_probability_difference=float((1 - mf[2]).mean() - (1 - mr[2]).mean()),
                post_odds_ratio=float(np.exp(fit.params[1])) if fit.name == "Hurdle_Logit_ZTNB2" else np.nan,
                positive_fake_mean=float(mf[3].mean()), positive_real_mean=float(mr[3].mean()),
                positive_mean_ratio=float(mf[3].mean() / mr[3].mean()))


def binary_metrics(fit, x):
    real, fake = np.asarray(x).copy(), np.asarray(x).copy()
    real[:, 1], fake[:, 1] = 0, 1
    pf, pr = expit(fake @ fit.params).mean(), expit(real @ fit.params).mean()
    return dict(any_probability_fake=float(pf), any_probability_real=float(pr),
                any_probability_difference=float(pf - pr), any_odds_ratio=float(np.exp(fit.params[1])))


def binary_intervals(fit, x):
    real, fake = np.asarray(x).copy(), np.asarray(x).copy()
    real[:, 1], fake[:, 1] = 0, 1
    funcs = {"probability_fake": lambda b: expit(fake @ b).mean(),
             "probability_real": lambda b: expit(real @ b).mean(),
             "probability_difference": lambda b: expit(fake @ b).mean() - expit(real @ b).mean(),
             "odds_ratio": lambda b: np.exp(b[1])}
    result = {}
    for name, fun in funcs.items():
        for method, obj in (("HC0", fit), ("model_based", replace(fit, cov=fit.bread))):
            est, lo, hi = delta_interval(obj, fun, name == "odds_ratio")
            result[name] = est
            result[f"{name}_{method}_lower"], result[f"{name}_{method}_upper"] = lo, hi
    return result


def warm_fit(y, x, name, initial):
    """Same tested likelihood, bounds, convergence/Hessian checks; warm start.

    A failed warm optimization is retried on the SAME bootstrap sample using
    the old multistart fitter; the attempt records that retry and its trials.
    """
    y, xx = np.asarray(y, float), np.asarray(x, float)
    p = xx.shape[1]
    columns = list(x.columns)
    bounds = [(-30., 30.)] * p
    if name == "Logit":
        return optimize(Logit(y, xx), [initial], bounds, columns, ["positive_logit"] * p, name, xx, y)
    if name in ("NB2", "ZTNB2"):
        native = NegativeBinomialP(y, xx, p=2) if name == "NB2" else TruncatedLFNegativeBinomialP(y, xx, p=2, truncation=0)
        return optimize(native, [initial], bounds + [(1e-5, 1e5)], columns + ["alpha"],
                        ["count"] * p + ["dispersion"], name, xx, y)
    if name == "Hurdle_Logit_ZTNB2":
        positive = y > 0
        binary = warm_fit(positive.astype(float), x, "Logit", initial[:p])
        count = warm_fit(y[positive], x.loc[positive], "ZTNB2", initial[p:])
        scores = np.zeros((len(y), binary.k + count.k))
        scores[:, :binary.k] = binary.scores
        scores[positive, binary.k:] = count.scores
        bread = block_diag(binary.bread, count.bread)
        return Fit(name, np.r_[binary.params, count.params], bread, bread @ (scores.T @ scores) @ bread,
                   scores, binary.ll + count.ll, binary.converged and count.converged,
                   binary.warnings + " | " + count.warnings,
                   js({"logit": json.loads(binary.trials), "positive": json.loads(count.trials)}),
                   binary.names + count.names, binary.components + count.components,
                   binary.boundary or count.boundary, max(binary.gradient, count.gradient),
                   binary.hessian_ok and count.hessian_ok, p, xx, y)
    raise ValueError(name)


def refit_checked(y, x, model, initial):
    first_error = ""
    try:
        fit = warm_fit(y, x, model, initial)
        if fit.stable:
            return fit, False, fit.trials
        first_error = fit.trials
    except Exception as error:
        first_error = repr(error)
    fit = fit_hurdle(y, x) if model == "Hurdle_Logit_ZTNB2" else fit_one(y, x, model)
    return fit, True, js({"warm_attempt": first_error, "same_draw_multistart": json.loads(fit.trials)})


_BOOT = {}


def initialize_workers(config):
    global _BOOT
    _BOOT = config


def resample_indices(clusters, rng):
    sampled = rng.integers(0, len(clusters), len(clusters))
    return np.concatenate([clusters[i] for i in sampled]), sampled


def bootstrap_draw(task):
    spec, attempt = task
    cfg = _BOOT[spec]
    rng = np.random.default_rng(np.random.SeedSequence([42, cfg["ordinal"], attempt]))
    idx, sampled = resample_indices(cfg["clusters"], rng)
    x = pd.DataFrame(cfg["x"][idx], columns=cfg["columns"])
    y = cfg["posts"][idx]
    row = dict(specification=spec, attempt=attempt, seed=42, sampled_N=len(idx),
               distinct_sampled_clusters=len(set(sampled)), largest_component_draws=int((sampled == cfg["largest_cluster_index"]).sum()),
               success=False, any_success=False, reply_success=False, post_error="", any_error="", reply_error="")
    started = time.monotonic()
    try:
        require(np.linalg.matrix_rank(x) == x.shape[1], "Bootstrap design rank loss")
        fit, retry, trials = refit_checked(y, x, cfg["model"], cfg["post_start"])
        row.update(post_converged=fit.converged, post_stable=fit.stable, post_hessian_ok=fit.hessian_ok,
                   post_boundary=fit.boundary, post_alpha=float(fit.params[-1]), post_retry=retry,
                   post_optimizer_trials_json=trials)
        require(fit.stable, "Nonconvergent/unidentified/boundary post fit")
        metrics = point_metrics(fit, x)  # Re-standardize to the sampled all-item covariate mix.
        require(np.isfinite([metrics[k] for k in ("standardized_fake_mean", "standardized_real_mean", "mean_ratio")]).all(), "Nonfinite standardized means")
        row.update(metrics, success=True)
    except Exception as error:
        row["post_error"] = repr(error)
    if cfg["any_start"] is not None:
        try:
            binary, retry, trials = refit_checked(cfg["presence"][idx], x, "Logit", cfg["any_start"])
            require(binary.stable, "Unstable any-engagement fit")
            row.update(binary_metrics(binary, x), any_success=True, any_retry=retry, any_optimizer_trials_json=trials)
        except Exception as error:
            row["any_error"] = repr(error)
    try:
        reply, retry, trials = refit_checked(cfg["replies"][idx], x, "NB2", cfg["reply_start"])
        row.update(reply_converged=reply.converged, reply_stable=reply.stable, reply_alpha=float(reply.params[-1]),
                   reply_retry=retry, reply_optimizer_trials_json=trials)
        require(reply.stable, "Unstable provisional reply NB")
        row.update(reply_mean_ratio=point_metrics(reply, x)["mean_ratio"], reply_success=True)
    except Exception as error:
        row["reply_error"] = repr(error)
    row["elapsed_seconds"] = time.monotonic() - started
    return row


def bootstrap(frames, fitted, hashes, target, workers, resume):
    config = {}
    for ordinal, (spec, frame) in enumerate(frames.items()):
        ids = sorted(frame.overlap_cluster_uid.unique())
        clusters = [np.flatnonzero(frame.overlap_cluster_uid.eq(cid).to_numpy()) for cid in ids]
        totals = [frame.iloc[idx].unique_post_count.sum() for idx in clusters]
        x = model_design(spec, frame)
        config[spec] = dict(ordinal=ordinal, clusters=clusters, largest_cluster_index=int(np.argmax(totals)),
                            x=x.to_numpy(), columns=list(x.columns), posts=frame.unique_post_count.to_numpy(),
                            replies=frame.unique_reply_count.to_numpy(), presence=frame.engagement_present.to_numpy(float),
                            model=fitted[spec]["post"].name, post_start=fitted[spec]["post"].params,
                            reply_start=fitted[spec]["reply"].params,
                            any_start=fitted[spec]["any"].params if fitted[spec]["any"] is not None else None)
    draws = []
    if resume and BOOT_PATH.exists():
        meta = json.loads(RUN_META.read_text())
        require(meta["input_sha256"] == hashes and meta["script_sha256"] == sha(Path(__file__)), "Cannot resume a changed input/script run")
        draws = pd.read_csv(BOOT_PATH, keep_default_na=False).to_dict("records")
        require(len({(r["specification"], r["attempt"]) for r in draws}) == len(draws), "Repeated bootstrap draw")
    RUN_META.write_text(json.dumps(dict(input_sha256=hashes, script_sha256=sha(Path(__file__)), seed=42,
                                        target_successful_post_refits=target, workers=workers, complete=False), indent=2) + "\n")
    pilot_rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_workers, initargs=(config,)) as pool:
        for spec in frames:
            existing = [r for r in draws if r["specification"] == spec]
            attempted = max([int(r["attempt"]) for r in existing], default=-1) + 1
            successes = sum(bool(r["success"]) for r in existing)
            start = time.monotonic()
            # Always report a 20-attempt pilot before committing to the full target.
            if attempted < 20:
                batch = list(pool.map(bootstrap_draw, [(spec, i) for i in range(attempted, 20)]))
                draws.extend(batch)
                attempted = 20
                successes += sum(r["success"] for r in batch)
                save("version_aware_bootstrap_refits", draws)
                elapsed = time.monotonic() - start
                print(f"PILOT {spec}: {successes}/20 successful post refits; wall={elapsed:.1f}s; projected {target} successes ~{elapsed * target / max(successes, 1):.1f}s", flush=True)
                pilot_rows.append(dict(specification=spec, pilot_attempts=20, successful_post_refits=successes,
                                       wall_seconds=elapsed, projected_target_seconds=elapsed * target / max(successes, 1), target=target))
            limit = max(target + 200, math.ceil(target * 1.25))
            while successes < target and attempted < limit:
                n = min(50, target - successes, limit - attempted)
                batch = list(pool.map(bootstrap_draw, [(spec, i) for i in range(attempted, attempted + n)]))
                draws.extend(batch)
                attempted += n
                successes += sum(r["success"] for r in batch)
                save("version_aware_bootstrap_refits", draws)
                print(f"BOOTSTRAP {spec}: successes={successes}, attempts={attempted}, failed={attempted - successes}", flush=True)
            print(f"COMPLETE {spec}: {successes} successful post refits / {attempted} attempts", flush=True)
    if pilot_rows:
        save("version_aware_bootstrap_pilot", pilot_rows)
    return pd.DataFrame(draws)


def percentile(draws, column, success="success"):
    values = pd.to_numeric(draws.loc[draws[success].eq(True), column], errors="coerce").dropna()
    return tuple(np.quantile(values, [.025, .975])) if len(values) >= 2 else (np.nan, np.nan)


def fitting_stage(frames, components):
    fitted, comparisons, tails, coefficient_rows = {}, [], [], []
    for spec, frame in frames.items():
        x = model_design(spec, frame)
        post = frame.unique_post_count.to_numpy(float)
        nb, hurdle = fit_one(post, x, "NB2"), fit_hurdle(post, x)
        # Predeclared family selection, independent of effect direction.
        selected = nb if spec.startswith("FakeHealth") else hurdle if hurdle.stable and (not nb.stable or hurdle.aic < nb.aic - 2) else nb
        require(selected.stable, "No stable primary post model")
        if selected.name == "Hurdle_Logit_ZTNB2":
            verify_full_hurdle_mean(selected)
        reply = fit_one(frame.unique_reply_count.to_numpy(float), x, "NB2")
        binary = None if spec.startswith("FakeHealth") else fit_one(frame.engagement_present.to_numpy(float), x, "Logit")
        fitted[spec] = dict(post=selected, reply=reply, any=binary)
        for outcome, candidates in (("unique_post_count", [nb, hurdle]), ("unique_reply_count", [reply])):
            for fit in candidates:
                row = comparison_row("FakeHealth HealthStory" if spec.startswith("FakeHealth") else "CoAID News", outcome, fit, spec)
                row["selected"] = fit is selected if outcome == "unique_post_count" else True
                row["status"] = "primary posts" if outcome == "unique_post_count" else "provisional secondary NB; not a validated reply family selection"
                comparisons.append(row)
                for j, (name, component) in enumerate(zip(fit.names, fit.components)):
                    coefficient_rows.append(dict(specification=spec, outcome=outcome, model=fit.name,
                                                 component=component, predictor=name, coefficient=fit.params[j],
                                                 HC0_SE=fit.se[j], model_based_SE=np.sqrt(max(fit.bread[j, j], 0))))
        variants = tail_sets(frame)
        variants.pop("exclude_top_1pct_with_boundary_ties")
        if not spec.startswith("FakeHealth"):
            biggest = components[components.specification.eq(spec)].sort_values(["post_memberships", "overlap_cluster_uid"], ascending=[False, True]).iloc[0].overlap_cluster_uid
            variants["exclude_largest_overlap_cluster"] = set(frame.loc[frame.overlap_cluster_uid.eq(biggest), "content_uid"])
        for tail, excluded in variants.items():
            part = frame[~frame.content_uid.isin(excluded)].copy()
            xx = model_design(spec, part)
            for outcome in ("unique_post_count", "unique_reply_count"):
                family = selected.name if outcome == "unique_post_count" else "NB2"
                fit = fitted[spec]["post" if outcome == "unique_post_count" else "reply"] if not excluded else (
                    fit_hurdle(part[outcome].to_numpy(), xx) if family == "Hurdle_Logit_ZTNB2" else fit_one(part[outcome].to_numpy(), xx, "NB2"))
                estimates = standardized_counts(fit, xx)
                fixed = point_metrics(fit, x)
                tails.append(dict(specification=spec, outcome=outcome, tail_specification=tail,
                                  N=len(part), removed_N=len(excluded), removed_fake_N=int(frame.loc[frame.content_uid.isin(excluded), "analysis_group"].eq("fake").sum()),
                                  removed_post_memberships=int(frame.loc[frame.content_uid.isin(excluded), "unique_post_count"].sum()),
                                  model=fit.name, stable=fit.stable, converged=fit.converged, alpha=float(fit.params[-1]),
                                  **estimates, fixed_original_target_ratio=fixed["mean_ratio"],
                                  notes="HC0 secondary intervals only for exclusions; retained-sample standardization; influence diagnostic, not the original all-item target"))
            if binary is not None:
                bfit = binary if not excluded else fit_one(part.engagement_present.to_numpy(float), xx, "Logit")
                tails.append(dict(specification=spec, outcome="engagement_present", tail_specification=tail,
                                  N=len(part), removed_N=len(excluded), stable=bfit.stable, model="Logit",
                                  **binary_metrics(bfit, xx), notes="Any post OR reply; HC0/direction diagnostic only"))
        print(f"FITTED {spec}: selected {selected.name}, R={point_metrics(selected, x)['mean_ratio']:.6f}, stable={selected.stable}", flush=True)
    save("version_aware_model_coefficients", coefficient_rows)
    return fitted, pd.DataFrame(comparisons), pd.DataFrame(tails)


def assemble_results(frames, fitted, draws, effects, target):
    results, occurrence, summaries, decomposition = [], [], [], []
    for spec, frame in frames.items():
        x = model_design(spec, frame)
        sample = draws[draws.specification.eq(spec)]
        successful = sample.success.eq(True)
        summaries.append(dict(specification=spec, attempts=len(sample), successful_post_refits=int(successful.sum()),
                              failed_post_refits=int((~successful).sum()), successful_any_refits=int(sample.any_success.eq(True).sum()),
                              successful_reply_refits=int(sample.reply_success.eq(True).sum()),
                              target_successful_post_refits=target, target_met=int(successful.sum()) >= target,
                              pilot_only=target < 1000, median_attempt_seconds=float(sample.elapsed_seconds.median()),
                              largest_component_omitted_fraction=float(sample.largest_component_draws.eq(0).mean())))
        for outcome, key, draw_col, flag in (("unique_post_count", "post", "mean_ratio", "success"),
                                            ("unique_reply_count", "reply", "reply_mean_ratio", "reply_success")):
            fit = fitted[spec][key]
            hc0 = standardized_counts(fit, x)
            model_based = standardized_counts(replace(fit, cov=fit.bread), x)
            lo, hi = percentile(sample, draw_col, flag)
            raw = effects[(effects.specification.eq(spec)) & effects.outcome.eq(outcome)].iloc[0]
            width = np.log(hi / lo) / np.log(hc0["CI_upper"] / hc0["CI_lower"])
            results.append(dict(specification=spec, outcome=outcome, fake_N=int(raw.fake_N), real_N=int(raw.real_N),
                                fake_mean=raw.fake_mean, real_mean=raw.real_mean, raw_mean_ratio=raw.mean_ratio,
                                standardized_fake_mean=hc0["standardized_fake_mean"], standardized_real_mean=hc0["standardized_real_mean"],
                                standardized_difference=hc0["mean_difference"], standardized_ratio=hc0["mean_ratio"],
                                primary_CI_lower=lo, primary_CI_upper=hi,
                                uncertainty_method="Social-overlap connected-component pairs bootstrap; percentile 95%; empirical reference mix re-sampled; seed 42",
                                successful_refits=int(sample[flag].eq(True).sum()), bootstrap_attempts=len(sample),
                                HC0_CI_lower=hc0["CI_lower"], HC0_CI_upper=hc0["CI_upper"],
                                model_based_CI_lower=model_based["CI_lower"], model_based_CI_upper=model_based["CI_upper"],
                                bootstrap_to_HC0_log_interval_width=width,
                                uncertainty_widening="substantially wider" if width >= 1.5 else "moderately wider" if width >= 1.1 else "similar or narrower",
                                zero_proportion=float(frame[outcome].eq(0).mean()), model=fit.name, stable=fit.stable,
                                release_adjustment="none" if spec.startswith("FakeHealth") else "first release fixed effects; May reference",
                                unsupported_profile_fraction=support_fraction(x),
                                status="primary" if spec == "A_primary_strict" and key == "post" else "separate FakeHealth primary" if spec.startswith("FakeHealth") and key == "post" else "sensitivity" if key == "post" else "secondary provisional reply model",
                                notes="Pilot interval only; not final" if target < 1000 else "Cluster concentration and first-release target limitations remain; no cross-context contrast"))
        fit = fitted[spec]["post"]
        if not spec.startswith("FakeHealth"):
            metrics = point_metrics(fit, x)
            for metric, value in metrics.items():
                lo, hi = percentile(sample, metric)
                decomposition.append(dict(specification=spec, measure=metric, estimate=value,
                                          cluster_CI_lower=lo, cluster_CI_upper=hi,
                                          notes="Positive-volume standardization uses the common all-item covariate mix; overall mean averages probability times positive mean, NOT product of averaged components"))
            binary = fitted[spec]["any"]
            # Report posts-only hurdle occurrence and any-post-or-reply separately.
            post_binary = replace(fit, name="Logit", params=fit.params[:fit.p],
                                  cov=fit.cov[:fit.p, :fit.p], bread=fit.bread[:fit.p, :fit.p]) if fit.name == "Hurdle_Logit_ZTNB2" else fit_one(frame.unique_post_count.gt(0).to_numpy(float), x, "Logit")
            for label, bfit, prefix, success_flag in (("any_post", post_binary, "post", "success"),
                                                       ("any_post_or_reply", binary, "any", "any_success")):
                estimates = binary_intervals(bfit, x)
                row = dict(specification=spec, outcome=label, fake_N=int(frame.analysis_group.eq("fake").sum()),
                           real_N=int(frame.analysis_group.eq("real").sum()), **estimates,
                           uncertainty_method="Overlap-cluster percentile bootstrap; seed 42", stable=bfit.stable)
                for metric in ("probability_fake", "probability_real", "probability_difference", "odds_ratio"):
                    lo, hi = percentile(sample, prefix + "_" + metric, success_flag)
                    row[metric + "_cluster_lower"], row[metric + "_cluster_upper"] = lo, hi
                row["percentage_point_difference"] = 100 * row["probability_difference"]
                occurrence.append(row)
    return pd.DataFrame(results), pd.DataFrame(occurrence), pd.DataFrame(summaries), pd.DataFrame(decomposition)


def plots(frames, results, draws, components):
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    coaid = [s for s in frames if not s.startswith("FakeHealth")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, spec in zip(axes, coaid):
        for label in ("fake", "real"):
            y = np.sort(frames[spec].loc[frames[spec].analysis_group.eq(label), "unique_post_count"].to_numpy())
            ax.step(np.log1p(y), np.arange(1, len(y) + 1) / len(y), where="post", label=label)
        ax.set(title=spec.split("_")[0], xlabel="log(1 + unique posts)", ylabel="Empirical cumulative proportion")
        ax.legend()
    fig.savefig(FIGURES / "post_count_ecdf.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, spec in zip(axes, coaid):
        sample = draws[draws.specification.eq(spec) & draws.success.eq(True)]
        for omitted, color in ((True, "#bb5544"), (False, "#4477aa")):
            part = sample[sample.largest_component_draws.eq(0) == omitted]
            ax.hist(pd.to_numeric(part.mean_ratio), bins=25, alpha=.55, color=color,
                    label="Largest cluster omitted" if omitted else "Largest cluster sampled")
        ax.axvline(1, color="black", linestyle=":")
        ax.set(title=spec.split("_")[0], xlabel="Standardized Fake/Real ratio", ylabel="Successful bootstrap draws")
        ax.legend(fontsize=7)
    fig.savefig(FIGURES / "cluster_bootstrap_distributions.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    for i, spec in enumerate(coaid):
        r = results[(results.specification.eq(spec)) & results.outcome.eq("unique_post_count")].iloc[0]
        for shift, lo, hi, label, color in ((.09, r.primary_CI_lower, r.primary_CI_upper, "Overlap-cluster 95% CI", "#4477aa"),
                                           (-.09, r.HC0_CI_lower, r.HC0_CI_upper, "HC0 95% CI (secondary)", "#999999")):
            ax.plot([lo, hi], [i + shift] * 2, color=color, label=label if i == 0 else None)
            ax.scatter([r.standardized_ratio], [i + shift], color=color, s=20)
    ax.axvline(1, color="black", linestyle=":"); ax.set_xscale("log")
    ax.set(yticks=range(3), yticklabels=[s.split("_")[0] for s in coaid], xlabel="All-item standardized Fake/Real post-count ratio")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "coaid_standardized_post_ratios.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, spec in zip(axes, coaid):
        shares = components[components.specification.eq(spec)].post_membership_share.sort_values(ascending=False).head(10)
        ax.bar(np.arange(1, len(shares) + 1), 100 * shares)
        ax.set(title=spec.split("_")[0], xlabel="Component engagement rank", ylabel="% of all post memberships")
    fig.savefig(FIGURES / "overlap_component_concentration.png", dpi=170); plt.close(fig)


def write_report(confirmation, results, occurrence, clusters, components, comparisons, tails, boot_summary, draws, target):
    post = results[results.outcome.eq("unique_post_count")]
    coa = post[~post.specification.str.startswith("FakeHealth")]
    a = coa[coa.specification.eq("A_primary_strict")].iloc[0]
    any_occ = occurrence[occurrence.outcome.eq("any_post_or_reply")]
    post_tail = tails[tails.outcome.eq("unique_post_count") & ~tails.specification.str.startswith("FakeHealth")]
    replies = results[results.outcome.eq("unique_reply_count")]
    required_tails = post_tail[post_tail.tail_specification.isin(["exclude_top_1pct", "exclude_top_5_items"])]
    coaid_replies = replies[~replies.specification.str.startswith("FakeHealth")]
    old = .237
    text = ["# Version-aware within-dataset engagement analysis", "",
            "PILOT ONLY: intervals below are provisional; complete the 1,000-success run before carrying estimates forward." if target < 1000 else "Completed overlap-aware refitting. Specification A remains the CoAID primary; B/C are sensitivity analyses. Earlier CoAID estimates are superseded for substantive interpretation and preserved unchanged for provenance.", "",
            "## Samples and design", "", markdown_table(pd.DataFrame(confirmation)), "",
            "No entity-resolution rules, raw files, validated samples, or earlier analysis scripts/results were changed. Counts include every observation; tail exclusions are explicitly separate diagnostics. CoAID models retain the existing first-recorded-release fixed effects (May reference), plus Fake/Real; FakeHealth uses Fake/Real only. No time offset, new text controls, or causal interpretation is introduced.", "",
            "The common estimand averages predicted expected post counts over the SAME empirical covariate distribution under Fake and under Real, including all zeros. For a hurdle this is the row-level product P(posts>0|X) × E(posts|posts>0,X), then averaged. It is neither the count-part coefficient nor the product of separately averaged components. Each bootstrap re-standardizes to its sampled covariate distribution; HC0/model-based intervals hold the original covariates fixed, so their width comparison is not a pure isolation of dependence.", "",
            "## Primary post results", "", markdown_table(post[["specification", "standardized_fake_mean", "standardized_real_mean", "standardized_ratio", "primary_CI_lower", "primary_CI_upper", "HC0_CI_lower", "HC0_CI_upper", "uncertainty_widening"]]), "",
            f"Corrected A gives R={a.standardized_ratio:.4f}, versus the approximate old first-release value 0.237: an absolute change of {a.standardized_ratio-old:+.4f}, or {100*(a.standardized_ratio/old-1):+.1f}%. This comparison diagnoses entity-resolution methodology only. A/B/C point estimates all remain below 1: {bool(coa.standardized_ratio.lt(1).all())}. These are observed-count associations, not misinformation effects or rates.", "",
            "## Overlap graph and effective information", "",
            markdown_table(clusters[["specification", "overlap_components", "singleton_components", "multi_entity_components", "maximum_component_size", "percent_entities_in_non_singletons", "percent_post_memberships_in_non_singletons", "largest_component_post_share", "post_weight_kish_equivalent"]]), "",
            "Edges join entities sharing a counted post ID or a counted reply ID. Connected components include every singleton; neither graph construction nor resampling changes analytical entities. Degree counts distinct neighboring entities, separately by shared post and reply; it is not the number of shared IDs. Component tables include Fake-only, Real-only, and mixed composition.", "",
            "WARNING: CoAID has thousands of nominal resampling components but one dominant component per sample. The engagement-weight Kish equivalent is a concentration diagnostic, NOT a literal number of independent clusters or a degrees-of-freedom estimate. Dominant components violate the comfortable many-small-cluster approximation: their omission/repetition can make the percentile bootstrap wide, skewed, or multimodal. Cluster labels assume no remaining between-component dependence; sharing a publisher/resource without a shared observed ID is not handled by this graph. Do not present these intervals as resolving all dependence or measurement uncertainty.", "",
            "FakeHealth's existing processed provenance links to the official local engagement mapping. Counted tweet/reply sets were verified against every processed row, revealing actual overlap; retweets not included in the modeled unique_post_count were excluded from graph construction. Therefore FakeHealth also receives an evidenced overlap-component bootstrap, with existing HC0/model-based intervals retained as secondary comparisons. No FakeHealth-vs-CoAID effect contrast is computed.", "",
            "## Bootstrap protocol and failures", "", markdown_table(boot_summary), "",
            "Seed 42; unstratified pairs bootstrap of K components with replacement, each selected component contributing ALL its entities once per draw occurrence. Repeated selection intentionally repeats rows; singleton components are ordinary units. Replicate seeds depend only on sample and attempt index, not worker scheduling. Each refit uses the same chosen model family/design, native likelihood and bounded convergence/Hessian checks as the old tested fitter. Warm-start failures receive a documented multistart retry on the SAME sampled data. Failed/nonconvergent attempts remain in version_aware_bootstrap_refits.csv and are excluded explicitly from percentile computation. Additional attempts toward the successful-refit target are logged, never silently substituted. All post-successful draws are retained. Occurrence and provisional reply failures are tracked separately.", "",
            "The script first runs a 20-attempt pilot, reports runtime/success, then continues to the target without reducing it. Bootstrap excludes model-selection uncertainty (family held fixed), entity-resolution uncertainty (bracketed by A/B/C), and potential missing social links. 'Moderately wider' means log-CI width is 1.10–<1.50 times HC0; 'substantially wider' means >=1.50; smaller ratios are labeled similar or narrower. These are descriptive thresholds, not statistical tests.", "",
            "## Model reassessment", "", markdown_table(comparisons[["specification", "outcome", "model", "AIC", "convergence", "stable", "alpha_NB2", "observed_zero_fraction", "predicted_zero_fraction", "pearson_chi2_per_df", "selected"]]), "",
            "NB2 and logistic plus genuine zero-truncated NB2 were compared on the same full post outcome/sample, counting all hurdle parameters. CoAID retains a stable hurdle only if it improves AIC over NB by >2 (or NB is unstable); this rule does not depend on the Fake/Real estimate. FakeHealth retains its requested NB primary. ZINB was unnecessary as a secondary post check because the stable hurdle/NB comparison already resolves the requested family decision. Zero agreement alone is not evidence of overall fit: a logistic intercept largely enforces aggregate zero agreement. Detailed comparison tables retain residual dispersion, gradients, Hessian checks, bound flags and optimizer warnings. Heavy-tail residual misfit remains a limitation of the working means.", "",
            "## Occurrence and positive-count parts", "", markdown_table(any_occ[["specification", "probability_fake", "probability_real", "percentage_point_difference", "probability_difference_cluster_lower", "probability_difference_cluster_upper", "odds_ratio"]]), "",
            "This table is any observed post OR reply, which is the requested engagement_present outcome. The separately exported any_post rows are the hurdle's posts-only occurrence part. Both use probability differences and their own cluster intervals; odds ratios are conditional logistic associations, not marginal probability ratios. The full positive-part and combined expected-count estimates/cluster intervals are in version_aware_hurdle_components.csv. No direct occurrence comparison with near-ceiling FakeHealth is made.", "",
            "## Tail and dominant-cluster sensitivity", "", markdown_table(post_tail[["specification", "tail_specification", "removed_N", "mean_ratio", "CI_lower", "CI_upper", "fixed_original_target_ratio", "stable"]]), "",
            f"Every required A/B/C tail-exclusion point estimate remains below 1: {bool(required_tails.mean_ratio.lt(1).all())}. This preserves the direction, not a claim of overlap-adjusted statistical significance for each exclusion fit. Removing the largest component changes magnitude more than removing five extreme items.", "",
            "Top 1% removes exactly ceil(0.01N) rows, sorted by descending unique posts then ascending UID to resolve ties; top five removes exactly five. All-retained remains the primary. Exclusion fits use the selected full-sample family and standardize to retained items; the fixed-original-target column isolates the fitted-parameter change from the reference-mixture change. Neither is an unbiased recovery of the untrimmed target after outcome-dependent selection. Exclusion intervals shown here are HC0 only and must NOT be interpreted as overlap-adjusted confirmation. Directional tail robustness is a point-estimate statement. Removing the largest engagement component is an additional influence diagnostic, not a revised primary.", "",
            "## Replies remain secondary", "", markdown_table(replies[["specification", "standardized_ratio", "primary_CI_lower", "primary_CI_upper", "stable", "successful_refits"]]), "",
            f"All CoAID provisional reply NB fits are numerically stable: {bool(coaid_replies.stable.all())}; every CoAID reply cluster interval includes 1: {bool(((coaid_replies.primary_CI_lower <= 1) & (coaid_replies.primary_CI_upper >= 1)).all())}. Dominant-component exclusions also change reply magnitudes appreciably. The estimates are usable as provisional descriptive associations but not stable/precise enough for a headline Fake/Real reply claim.", "",
            "The prior provisional NB was refitted, with alpha/Hessian/convergence recorded and overlap-bootstrap failures tracked. Numerical estimability is not substantive stability or validation of a full reply model. The tail table includes reply NB refits; inspect whether magnitudes and directions change and whether intervals cross 1. No reply-hurdle re-selection or dispersion-profile audit is claimed here. Reply estimates therefore remain provisional and cannot displace posts as the headline outcome.", "",
            "## Carry-forward decision", "",
            "Carry A's all-item standardized post ratio with its overlap-cluster percentile interval as the PRIMARY CoAID estimate, retaining the dominant-cluster and release-coding caveats. B/C, tail exclusions, leave-largest-component-out, HC0/model-based intervals, and provisional replies are sensitivity/secondary results. The large occurrence gap should remain a separate within-CoAID finding, not a between-context binary contrast.", "",
            "The corrected within-dataset stage provides the inputs for planning a formal common-estimand cross-context analysis, but not an unconditional green light for a definitive contrast: pre-specify inference that acknowledges the dominant social component and the sensitivity-defined populations, and carry the documented uncertainty forward. No cross-context ratio-of-ratios, pooled model, or formal between-dataset comparison was calculated.", "",
            "## Reproduction", "",
            "Run `.venv/bin/python src/run_version_aware_engagement_analysis.py --successes 1000 --workers 4`; `--pilot` runs 20 attempts, and `--resume` reuses logged draws only when script and input fingerprints match. Full attempt records, pilot timing, sample checks, cluster assignments/composition, descriptive effects, model comparisons, occurrence results, tail sensitivities, coefficients and plots are saved under the version-aware filenames. Previous CoAID results remain archived in their original files and are explicitly superseded here."]
    REPORT.write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    target = 20 if args.pilot else args.successes
    require(target >= 20 and args.workers >= 1, "Use at least 20 target successes and one worker")
    frames, overlaps, hashes, confirmation = load_samples()
    print(pd.DataFrame(confirmation).to_string(index=False), flush=True)
    assignments, components, clusters, descriptions, effects = [], [], [], [], []
    for spec in frames:
        frames[spec], assn, comp, summary = overlap_graph(spec, frames[spec], overlaps)
        assignments.append(assn); components.append(comp); clusters.append(summary)
        desc, eff = descriptive(spec, frames[spec]); descriptions.extend(desc); effects.extend(eff)
        print(f"GRAPH {spec}: K={summary['overlap_components']}; largest post share={summary['largest_component_post_share']:.1%}; concentration warning={summary['concentration_warning']}", flush=True)
    all_assignments = pd.concat(assignments, ignore_index=True)
    components = pd.concat(components, ignore_index=True)
    clusters = pd.DataFrame(clusters)
    save("coaid_social_overlap_clusters", all_assignments[~all_assignments.specification.str.startswith("FakeHealth")])
    save("fakehealth_social_overlap_clusters", all_assignments[all_assignments.specification.str.startswith("FakeHealth")])
    save("version_aware_social_overlap_component_summary", components)
    save("version_aware_social_overlap_graph_summary", clusters)
    save("version_aware_sample_confirmation", confirmation)
    save("version_aware_engagement_descriptives", descriptions)
    effects = save("version_aware_raw_effect_sizes", effects)
    fitted, comparisons, tails = fitting_stage(frames, components)
    save("version_aware_model_comparison", comparisons)
    draws = bootstrap(frames, fitted, hashes, target, args.workers, args.resume)
    results, occurrence, boot_summary, decomposition = assemble_results(frames, fitted, draws, effects, target)
    save("version_aware_engagement_results", results)
    save("version_aware_occurrence_results", occurrence)
    save("version_aware_bootstrap_summary", boot_summary)
    save("version_aware_hurdle_components", decomposition)
    save("version_aware_tail_sensitivity", tails)
    # Track dominance directly rather than relying only on a pooled percentile CI.
    influence = []
    for (spec, multiplicity), part in draws[draws.success.eq(True)].groupby(["specification", "largest_component_draws"]):
        vals = pd.to_numeric(part.mean_ratio)
        influence.append(dict(specification=spec, largest_component_draws=multiplicity, successful_refits=len(part),
                              median_ratio=float(vals.median()), percentile_025=float(vals.quantile(.025)), percentile_975=float(vals.quantile(.975))))
    save("version_aware_bootstrap_dominant_cluster_influence", influence)
    plots(frames, results, draws, components)
    write_report(confirmation, results, occurrence, clusters, components, comparisons, tails, boot_summary, draws, target)
    for relative, fingerprint in hashes.items():
        require(sha(ROOT / relative) == fingerprint, f"Input mutated: {relative}")
    meta = json.loads(RUN_META.read_text())
    meta.update(complete=bool(boot_summary.target_met.all()), pilot_only=target < 1000,
                no_raw_or_sample_changes=True, no_cross_context_contrast=True,
                input_hashes_verified=True, timestamp_variables_used_as_primary=False)
    RUN_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(results[["specification", "outcome", "standardized_ratio", "primary_CI_lower", "primary_CI_upper", "uncertainty_widening"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
