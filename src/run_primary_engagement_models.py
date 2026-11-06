"""Separate explanatory engagement regressions; run with .venv/bin/python.

NB2: Var(Y|X) = mu + alpha*mu**2. Hurdle = Bernoulli-logit for
Y>0 plus a genuinely zero-truncated NB2, not an ordinary NB on positives.
No exposure offset: these are observed-count associations, not rates per time.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import nbinom, norm, poisson
import statsmodels
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, NegativeBinomialP, Poisson
from statsmodels.discrete.truncated_model import TruncatedLFNegativeBinomialP
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.tools.numdiff import approx_fprime

SEED = 42
TABLES = ROOT / "outputs/tables"
FIGURES = ROOT / "outputs/figures/engagement_models"
REPORT = ROOT / "outputs/reports/primary_engagement_models.md"
COUNTS = ("unique_post_count", "unique_reply_count")
RELEASES = ("05-01-2020", "07-01-2020", "09-01-2020", "11-01-2020")
Z = norm.ppf(.975)


def design(frame, release_coding="earliest"):
    x = pd.DataFrame({"Intercept": 1., "fake_vs_real": frame.analysis_group.eq("fake").astype(float)}, index=frame.index)
    if "member_release_periods_json" in frame and release_coding != "none":
        memberships = frame.member_release_periods_json.map(json.loads)
        if release_coding == "earliest":
            earliest = memberships.map(lambda values: min(values, key=RELEASES.index))
            for release in RELEASES[1:]:
                x["release_" + release] = earliest.eq(release).astype(float)
        elif release_coding == "membership":
            # Four membership indicators are full rank with an intercept because
            # entities can belong to more than one release. No entity is repeated.
            for release in RELEASES:
                x["member_" + release] = memberships.map(lambda values: float(release in values))
        else:
            raise ValueError(release_coding)
    assert np.linalg.matrix_rank(x) == x.shape[1], "Rank-deficient design"
    return x


@dataclass
class Fit:
    name: str
    params: np.ndarray
    bread: np.ndarray
    cov: np.ndarray
    scores: np.ndarray
    ll: float
    converged: bool
    warnings: str
    trials: str
    names: list[str]
    components: list[str]
    boundary: bool
    gradient: float
    hessian_ok: bool
    p: int
    x: np.ndarray
    y: np.ndarray

    @property
    def k(self):
        return len(self.params)

    @property
    def aic(self):
        return -2 * self.ll + 2 * self.k

    @property
    def bic(self):
        # Same full outcome and N for count candidates, including joint hurdle.
        return -2 * self.ll + self.k * np.log(len(self.y))

    @property
    def se(self):
        return np.sqrt(np.maximum(np.diag(self.cov), 0))

    @property
    def stable(self):
        return bool(self.converged and self.hessian_ok and not self.boundary
                    and np.isfinite(self.se).all() and self.se.max() < 10
                    and np.abs(self.params[:-1] if self.name != "Logit" else self.params).max() < 20)


def optimize(model, starts, bounds, names, components, name, x, y):
    """Native statsmodels likelihood/derivatives, bounded multistart L-BFGS.

    Alpha is strictly positive. We retain warnings from ALL starts. Covariance
    is HC0 from the observed-information bread and item-level score vectors;
    model-based covariance is saved too. No regularization is applied.
    """
    trials, usable = [], []
    for start in starts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = model.fit(start_params=np.asarray(start), method="lbfgs",
                                   bounds=bounds, maxiter=4000, disp=False,
                                   pgtol=1e-9, factr=100)
                pars = np.asarray(result.params)
                ll = float(model.loglike(pars))
                score = np.asarray(model.score_obs(pars))
                gradient = float(np.max(np.abs(score.mean(axis=0))))
                information = -np.asarray(model.hessian(pars))
                eig = np.linalg.eigvalsh((information + information.T) / 2)
                good_hessian = bool(np.isfinite(eig).all() and eig.min() > 1e-9)
                bread = np.linalg.inv(information) if good_hessian else np.full_like(information, np.nan)
                cov = bread @ (score.T @ score) @ bread
                convergence = bool(result.mle_retvals.get("converged", False) and gradient < 1e-4)
                boundary = any((lo is not None and v <= lo + 1e-4) or
                               (hi is not None and v >= hi - 1e-4)
                               for v, (lo, hi) in zip(pars, bounds))
                trial = {"ll": ll, "optimizer_converged": bool(result.mle_retvals.get("converged", False)),
                         "converged": convergence, "gradient_per_item": gradient,
                         "hessian_ok": good_hessian, "boundary": bool(boundary)}
                if np.isfinite(ll):
                    usable.append((ll, pars, bread, cov, score, convergence, boundary, gradient, good_hessian))
            except Exception as error:
                trial = {"error": str(error)}
            trial["warnings"] = sorted(set(str(w.message) for w in caught))
            trials.append(trial)
    if not usable:
        raise RuntimeError(f"All {name} optimizations failed: {trials}")
    # Numerically tied likelihoods prefer converged/identified solutions.
    best_ll = max(t[0] for t in usable)
    tied = [t for t in usable if t[0] >= best_ll - 1e-6]
    ll, pars, bread, cov, score, converged, boundary, gradient, good_hessian = max(tied, key=lambda t: (t[5], t[8], not t[6], t[0]))
    return Fit(name, pars, bread, cov, score, ll, converged,
               " | ".join(sorted(set(w for t in trials for w in t["warnings"]))),
               json.dumps(trials), names, components, boundary, gradient, good_hessian,
               x.shape[1], np.asarray(x), np.asarray(y))


def fit_one(y, x, name, nb=None):
    y, xx = np.asarray(y, float), np.asarray(x, float)
    p = xx.shape[1]
    columns = list(x.columns)
    bounds = [(-30., 30.)] * p
    if name == "Logit":
        return optimize(Logit(y, xx), [np.zeros(p)], bounds, columns,
                        ["positive_logit"] * p, name, xx, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = np.asarray(sm.GLM(y, xx, family=sm.families.Poisson()).fit().params)
    if name == "Poisson":
        return optimize(Poisson(y, xx), [beta], bounds, columns, ["count"] * p, name, xx, y)
    if name in ("NB2", "ZTNB2"):
        model = NegativeBinomialP(y, xx, p=2) if name == "NB2" else TruncatedLFNegativeBinomialP(y, xx, p=2, truncation=0)
        starts = [np.r_[beta, alpha] for alpha in (1., 5., 20.)]
        return optimize(model, starts, bounds + [(1e-5, 1e5)], columns + ["alpha"],
                        ["count" if name == "NB2" else "positive_count"] * p + ["dispersion"], name, xx, y)
    if name == "ZINB2":
        assert nb is not None
        model = ZeroInflatedNegativeBinomialP(y, xx, exog_infl=xx, inflation="logit", p=2)
        starts = [np.r_[np.r_[intercept, np.zeros(p - 1)], nb.params] for intercept in (-4., -1., 1.)]
        return optimize(model, starts, bounds + bounds + [(1e-5, 1e5)],
                        columns + columns + ["alpha"], ["inflation_logit"] * p + ["count"] * p + ["dispersion"], name, xx, y)
    raise ValueError(name)


def fit_hurdle(y, x):
    positive = np.asarray(y) > 0
    binary = fit_one(positive.astype(float), x, "Logit")
    count = fit_one(np.asarray(y)[positive], x.loc[positive], "ZTNB2")
    scores = np.zeros((len(y), binary.k + count.k))
    scores[:, :binary.k] = binary.scores
    scores[positive, binary.k:] = count.scores
    bread = block_diag(binary.bread, count.bread)
    return Fit("Hurdle_Logit_ZTNB2", np.r_[binary.params, count.params], bread,
               bread @ (scores.T @ scores) @ bread, scores, binary.ll + count.ll,
               binary.converged and count.converged, binary.warnings + " | " + count.warnings,
               json.dumps({"logit": json.loads(binary.trials), "positive": json.loads(count.trials)}),
               binary.names + count.names, binary.components + count.components,
               binary.boundary or count.boundary, max(binary.gradient, count.gradient),
               binary.hessian_ok and count.hessian_ok, x.shape[1], np.asarray(x), np.asarray(y))


def moments(fit, x=None, params=None):
    """Return overall mean, variance, P(0), conditional-positive mean, NB mean."""
    x = fit.x if x is None else np.asarray(x)
    b = fit.params if params is None else params
    p = fit.p
    if fit.name == "Logit":
        prob = expit(x @ b)
        return prob, prob * (1 - prob), 1 - prob, np.ones(len(x)), prob
    offset = p if fit.name in ("Hurdle_Logit_ZTNB2", "ZINB2") else 0
    mu = np.exp(x @ b[offset:offset + p])
    if fit.name == "Poisson":
        p0 = np.exp(-mu)
        return mu, mu, p0, mu / (-np.expm1(-mu)), mu
    alpha = b[-1]
    base_var = mu + alpha * mu**2
    log_p0 = -np.log1p(alpha * mu) / alpha
    base_p0, nonzero = np.exp(log_p0), -np.expm1(log_p0)
    positive_mean = mu / nonzero
    positive_var = (base_var + mu**2) / nonzero - positive_mean**2
    if fit.name == "Hurdle_Logit_ZTNB2":
        prob = expit(x @ b[:p])
        mean = prob * positive_mean
        var = prob * positive_var + prob * (1 - prob) * positive_mean**2
        return mean, var, 1 - prob, positive_mean, mu
    if fit.name == "ZINB2":
        pi = expit(x @ b[:p])
        return (1 - pi) * mu, (1 - pi) * base_var + pi * (1 - pi) * mu**2, pi + (1 - pi) * base_p0, positive_mean, mu
    return mu, base_var, base_p0, positive_mean, mu


def cdf(fit, value):
    mean, var, pzero, positive_mean, mu = moments(fit)
    if fit.name == "Logit":
        return np.where(value < 0, 0., np.where(value < 1, pzero, 1.))
    if fit.name == "Poisson":
        return poisson.cdf(value, mu)
    alpha = fit.params[-1]
    base = nbinom.cdf(value, 1 / alpha, 1 / (1 + alpha * mu))
    base_zero = np.exp(-np.log1p(alpha * mu) / alpha)
    if fit.name == "Hurdle_Logit_ZTNB2":
        result = pzero + (1 - pzero) * (base - base_zero) / (1 - base_zero)
    elif fit.name == "ZINB2":
        pi = expit(fit.x @ fit.params[:fit.p])
        result = pi + (1 - pi) * base
    else:
        result = base
    return np.where(np.asarray(value) < 0, 0., np.clip(result, 0, 1))


def comparison_row(dataset, outcome, fit, specification):
    mean, var, p0, _, _ = moments(fit)
    y = fit.y
    residual = (y - mean) / np.sqrt(np.maximum(var, 1e-12))
    observed_zero = float(np.mean(y == 0))
    zero_gap = observed_zero - p0.mean()
    threshold = max(min(.02, .5 * observed_zero), 3 / len(y),
                    2 * np.sqrt(np.sum(p0 * (1 - p0))) / len(y))
    finite_se = fit.se[np.isfinite(fit.se)]
    max_se = finite_se.max() if len(finite_se) else np.nan
    rng = np.random.default_rng(SEED)
    lower, upper = cdf(fit, y - 1), cdf(fit, y)
    rqr = norm.ppf(np.clip(lower + rng.random(len(y)) * (upper - lower), 1e-10, 1 - 1e-10))
    notes = ["HC0 item-level sandwich SE; model-based SE also saved; real is reference.",
             "BIC = -2*LL + k*log(full N); all free parameters counted."]
    if dataset == "CoAID News":
        notes.append("Primary/conservative release FE use earliest recorded member release, May reference; membership check uses all four memberships.")
    if fit.name == "Hurdle_Logit_ZTNB2":
        notes.append("Joint Bernoulli-logit(Y>0) + zero-truncated NB2 likelihood; exp(count beta) is underlying NB mean-parameter ratio, not exact positive conditional mean ratio.")
    if fit.name == "ZINB2":
        notes.append("Both inflation-logit and NB2 count components use the same predictors; latent structural and sampling zeros.")
    if not fit.stable:
        notes.append("UNSTABLE: inspect bounds, Hessian, gradients, SE and trial warnings; Wald CI may be unreliable.")
    if fit.boundary or not fit.hessian_ok:
        notes.append("AIC/BIC are nominal numerical values only: boundary/nonidentification violates regular information-criterion assumptions.")
    pattern_frame = pd.DataFrame(fit.x)
    controls = [i for i in range(fit.p) if i != 1]
    supports = pattern_frame.groupby(controls)[1].nunique()
    bad_patterns = set(supports[supports < 2].index)
    keys = [tuple(row) if len(controls) > 1 else row[0] for row in fit.x[:, controls]]
    unsupported_fraction = np.mean([key in bad_patterns for key in keys])
    return {"dataset": dataset, "outcome": outcome, "model": fit.name, "specification": specification,
            "N": len(y), "N_positive": int((y > 0).sum()), "k_parameters": fit.k,
            "log_likelihood": fit.ll, "AIC": fit.aic, "BIC": fit.bic,
            "observed_zero_fraction": observed_zero, "predicted_zero_fraction": p0.mean(),
            "zero_gap_observed_minus_predicted": zero_gap, "materially_underfits_zeros": bool(zero_gap > threshold),
            "materially_overfits_zeros": bool(-zero_gap > threshold), "zero_flag_threshold": threshold,
            "convergence": fit.converged, "stable": fit.stable,
            "hessian_positive_definite": fit.hessian_ok, "boundary": fit.boundary,
            "max_abs_gradient_per_item": fit.gradient, "max_HC0_standard_error": max_se,
            "extreme_standard_errors": bool(not np.isfinite(fit.se).all() or max_se > 10),
            "regular_information_criteria": bool(fit.hessian_ok and not fit.boundary),
            "fraction_covariate_profiles_without_both_veracity_groups": unsupported_fraction,
            "alpha_NB2": fit.params[-1] if fit.name not in ("Poisson", "Logit") else np.nan,
            "pearson_chi2_per_df": np.sum(residual**2) / (len(y) - fit.k),
            "max_abs_pearson_residual": np.max(np.abs(residual)),
            "RQR_mean": rqr.mean(), "RQR_sd": rqr.std(ddof=1), "RQR_abs_gt_2_fraction": np.mean(np.abs(rqr) > 2),
            "observed_mean": y.mean(), "predicted_mean": mean.mean(),
            "mean_calibration_ratio": mean.mean() / y.mean(),
            "large_mean_miscalibration": bool(mean.mean() / y.mean() < .5 or mean.mean() / y.mean() > 2),
            "warnings_occurred": bool(fit.warnings.strip(" |")), "warnings": fit.warnings,
            "optimizer_trials_json": fit.trials, "notes": " ".join(notes)}


def coefficients(dataset, outcome, fit, specification, selected=False):
    rows = []
    for j, (name, component) in enumerate(zip(fit.names, fit.components)):
        beta, se = fit.params[j], fit.se[j]
        dispersion = component == "dispersion"
        ratio = np.exp(beta) if not dispersion else np.nan
        lo, hi = np.exp(np.clip([beta - Z * se, beta + Z * se], -700, 700)) if not dispersion else (np.nan, np.nan)
        interpretation = "Nuisance/control parameter; see model specification."
        if name == "fake_vs_real":
            if component == "positive_logit":
                interpretation = f"Fake vs real adjusted odds of {'any observed engagement' if outcome == 'engagement_present' else 'a positive observed count'}: OR={ratio:.4g}; not a probability ratio."
            elif component == "positive_count":
                interpretation = f"Among items with positive {outcome}, underlying NB2 mean parameter is {100 * (ratio - 1):+.2f}% different for fake vs real. This is NOT the exact truncated conditional mean change; see standardized_positive_mean_ratio."
            elif component == "inflation_logit":
                interpretation = f"Fake vs real odds of membership in the latent structural-zero component: OR={ratio:.4g}; this latent interpretation is unverified."
            else:
                interpretation = f"Fake is associated with a {abs(100 * (ratio - 1)):.2f}% {'lower' if ratio < 1 else 'higher'} expected observed count than real, conditional on included covariates{' and the latent NB component' if fit.name == 'ZINB2' else ''}."
        if dispersion:
            interpretation = "NB2 dispersion alpha on its original scale: Var(base count|X)=mu+alpha*mu^2."
        rows.append({"dataset": dataset, "outcome": outcome, "specification": specification,
                     "selected_model": fit.name, "model": fit.name, "selected": selected,
                     "model_component": component, "predictor": name, "coefficient": beta,
                     "standard_error": se, "standard_error_model_based": np.sqrt(max(fit.bread[j, j], 0)),
                     "IRR_or_OR": ratio, "CI_lower": lo, "CI_upper": hi,
                     "coefficient_CI_lower": beta - Z * se, "coefficient_CI_upper": beta + Z * se,
                     "CI_method": "95% Wald, HC0 item-level sandwich", "stable": fit.stable,
                     "interpretation": interpretation})
    return rows


def contrasts(dataset, outcome, fit, specification, reference_x=None):
    x = fit.x if reference_x is None else np.asarray(reference_x)
    real, fake = x.copy(), x.copy()
    real[:, 1], fake[:, 1] = 0, 1
    # Positive-mean standardization uses the SAME covariate distribution for
    # both groups, not the separate observed positive groups' covariates.
    def values(params):
        r, f = moments(fit, real, params), moments(fit, fake, params)
        return {"standardized_mean_real": r[0].mean(), "standardized_mean_fake": f[0].mean(),
                "standardized_overall_mean_ratio": f[0].mean() / r[0].mean(),
                "standardized_positive_mean_real": r[3].mean(), "standardized_positive_mean_fake": f[3].mean(),
                "standardized_positive_mean_ratio": f[3].mean() / r[3].mean(),
                "standardized_probability_real": (1 - r[2]).mean(),
                "standardized_probability_fake": (1 - f[2]).mean(),
                "standardized_probability_difference": (r[2] - f[2]).mean()}
    rows = []
    for key, estimate in values(fit.params).items():
        if fit.name == "Logit" and ("mean" in key):
            continue
        positive = "mean" in key
        transform = np.log if positive else lambda v: v
        gradient = np.atleast_1d(approx_fprime(fit.params, lambda b: transform(values(b)[key]), centered=True))
        se = np.sqrt(max(float(gradient @ fit.cov @ gradient), 0))
        lo, hi = transform(estimate) + np.array([-Z, Z]) * se
        if positive:
            lo, hi = np.exp(np.clip([lo, hi], -700, 700))
        rows.append({"dataset": dataset, "outcome": outcome, "specification": specification,
                     "selected_model": fit.name, "measure": key, "estimate": estimate,
                     "CI_lower": lo, "CI_upper": hi, "stable": fit.stable,
                     "standardization_N": len(x), "CI_method": "HC0 delta method, log scale for means/ratios; identity for probabilities",
                     "notes": "Both groups standardized to the same full sample release distribution; not a causal counterfactual. Probability Wald CI may extend beyond [0,1]."})
    return rows


def candidates(dataset, outcome, frame, coding="earliest"):
    x = design(frame, coding)
    y = frame[outcome].to_numpy(float)
    if outcome == "engagement_present":
        return [fit_one(y, x, "Logit")]
    fits = []
    if outcome == "unique_post_count":
        fits.append(fit_one(y, x, "Poisson"))
    nb = fit_one(y, x, "NB2")
    fits.append(nb)
    if dataset == "CoAID News" or outcome == "unique_reply_count":
        fits.append(fit_hurdle(y, x))
    if outcome == "unique_reply_count":
        fits.append(fit_one(y, x, "ZINB2", nb))
    return fits


def select(fits):
    lookup = {f.name: f for f in fits}
    if "Logit" in lookup:
        return lookup["Logit"], "Requested binary logistic model; stability checked."
    nb = lookup["NB2"]
    hurdle = lookup.get("Hurdle_Logit_ZTNB2")
    if hurdle is not None and hurdle.stable and (not nb.stable or hurdle.aic + 2 < nb.aic):
        return hurdle, "Hurdle has >2 AIC improvement over NB2, stable fit, and directly separates engagement occurrence from positive volume. ZINB remains sensitivity: latent structural-zero mechanism is not established."
    if nb.stable:
        return nb, "Stable NB2 selected: hurdle is unavailable, unstable, or has no >2 AIC improvement. Poisson is diagnostic; ZINB remains latent-mechanism sensitivity."
    raise RuntimeError("Neither NB2 nor hurdle is sufficiently stable for selection")


def plot_fit(dataset, outcome, fits, selected):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cutpoints = np.array([0, 1, 4, 9, 24, 49, 99, 499, np.inf])
    labels = ["0", "1", "2–4", "5–9", "10–24", "25–49", "50–99", "100–499", "500+"]
    observed = np.diff(np.r_[0, [(selected.y <= c).mean() for c in cutpoints]])
    ax = axes[0]
    ax.bar(np.arange(len(labels)), observed, alpha=.3, color="black", label="Observed")
    for fit in fits:
        predicted = np.diff(np.r_[0, [cdf(fit, c).mean() for c in cutpoints]])
        ax.plot(predicted, marker=".", label=fit.name + (" (unstable)" if not fit.stable else ""))
    ax.set_xticks(np.arange(len(labels)), labels, rotation=60)
    ax.set_ylabel("Fraction of items")
    ax.legend(fontsize=7)
    ax.set_title("Observed and fitted distribution")
    means, variance, zeros, _, _ = moments(selected)
    ax = axes[1]
    locations = np.arange(2)
    groups = selected.x[:, 1]
    ax.bar(locations - .16, [np.mean(selected.y[groups == g] == 0) for g in (0, 1)], .32, label="Observed")
    ax.bar(locations + .16, [zeros[groups == g].mean() for g in (0, 1)], .32, label="Fitted")
    ax.set_xticks(locations, ["real", "fake"])
    ax.set_ylim(0, 1)
    ax.set_title("Zero fractions by observed group")
    ax.legend(fontsize=8)
    ax = axes[2]
    rng = np.random.default_rng(SEED)
    low, high = cdf(selected, selected.y - 1), cdf(selected, selected.y)
    rqr = np.sort(norm.ppf(np.clip(low + rng.random(len(low)) * (high - low), 1e-10, 1 - 1e-10)))
    theoretical = norm.ppf((np.arange(len(low)) + .5) / len(low))
    ax.scatter(theoretical, rqr, s=4, alpha=.5)
    limits = [min(theoretical.min(), rqr.min()), max(theoretical.max(), rqr.max())]
    ax.plot(limits, limits, color="black", lw=1)
    ax.set(xlabel="Standard normal quantile", ylabel="Randomized quantile residual", title="Selected-model residual Q–Q")
    fig.suptitle(f"{dataset}: {outcome}\nSelected: {selected.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset.lower().replace(' ', '_')}_{outcome}_diagnostics.png", dpi=180)
    plt.close(fig)


def load_samples():
    samples = {}
    for dataset, filename, group in [
        ("FakeHealth HealthStory", "fakehealth_healthstory_primary.csv", "fakehealth_within_dataset_analysis_group"),
        ("CoAID News", "coaid_news_primary.csv", "coaid_within_dataset_analysis_group")]:
        frame = pd.read_csv(ROOT / "data/processed" / filename)
        frame["analysis_group"] = frame[group]
        assert set(frame.analysis_group) == {"fake", "real"}
        assert frame.content_uid.is_unique
        for outcome in COUNTS:
            y = frame[outcome]
            assert y.notna().all() and (y >= 0).all() and (y == np.floor(y)).all()
        assert frame.engagement_present.isin([True, False, 0, 1]).all()
        assert np.array_equal(frame.engagement_present.astype(bool), frame[list(COUNTS)].gt(0).any(axis=1))
        samples[dataset] = frame
    return samples


def profile_reply_dispersion(samples):
    """Check that boundary fits reflect a rising profile, not just a bad start.

    Fixed-alpha coefficients are optimized separately; SciPy's NB log-PMF
    independently validates statsmodels' zero-truncated log likelihood.
    """
    rows = []
    for dataset, frame in samples.items():
        positive = frame.unique_reply_count.gt(0)
        x = design(frame).loc[positive].to_numpy()
        y = frame.loc[positive, "unique_reply_count"].to_numpy()
        model = TruncatedLFNegativeBinomialP(y, x, p=2, truncation=0)
        start = np.linalg.lstsq(x, np.log(y), rcond=None)[0]
        for alpha in (20., 100., 1000., 10000., 100000.):
            result = minimize(lambda b: -model.loglike(np.r_[b, alpha]), start,
                              jac=lambda b: -model.score(np.r_[b, alpha])[:-1],
                              method="BFGS", options={"gtol": 1e-6, "maxiter": 1000})
            start = result.x
            mu = np.exp(x @ start)
            logp0 = -np.log1p(alpha * mu) / alpha
            scipy_ll = np.sum(nbinom.logpmf(y, 1 / alpha, 1 / (1 + alpha * mu)) - np.log(-np.expm1(logp0)))
            assert np.isclose(-result.fun, scipy_ll, atol=1e-5, rtol=0)
            rows.append({"dataset": dataset, "outcome": "unique_reply_count", "fixed_alpha": alpha,
                         "positive_log_likelihood": -result.fun, "independent_scipy_log_likelihood": scipy_ll,
                         "convergence": bool(result.success),
                         "max_abs_gradient_per_item": np.max(np.abs(result.jac)) / len(y),
                         "fake_log_mean_parameter_ratio": start[1],
                         "notes": "Likelihood increases toward large-alpha boundary; ordinary joint Wald inference for NB2 positive component is not regular."})
    return rows


def markdown_table(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v) for v in values) + " |")
    return "\n".join(rows)


def main():
    for directory in (TABLES, FIGURES, REPORT.parent):
        directory.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    comparisons, all_coefs, selected_coefs, predictions, choices, sensitivity = [], [], [], [], [], []
    primary_fits = {}
    primary_choices = {}
    for dataset, frame in samples.items():
        outcomes = COUNTS + (("engagement_present",) if dataset == "CoAID News" else ())
        for outcome in outcomes:
            print(f"Fitting {dataset} / {outcome} / primary", flush=True)
            fits = candidates(dataset, outcome, frame)
            selected, reason = select(fits)
            primary_fits[dataset, outcome] = fits
            primary_choices[dataset, outcome] = selected
            choices.append({"dataset": dataset, "outcome": outcome, "selected_model": selected.name,
                            "lowest_AIC_model": min(fits, key=lambda f: f.aic).name, "reason": reason})
            for fit in fits:
                row = comparison_row(dataset, outcome, fit, "primary")
                row["selected"] = fit is selected
                comparisons.append(row)
                all_coefs.extend(coefficients(dataset, outcome, fit, "primary", fit is selected))
            selected_coefs.extend(coefficients(dataset, outcome, selected, "primary", True))
            predictions.extend(contrasts(dataset, outcome, selected, "primary"))
            if outcome in COUNTS:
                plot_fit(dataset, outcome, fits, selected)
            print(f"Selected {selected.name}; AIC={selected.aic:.2f}; stable={selected.stable}", flush=True)

    coaid = samples["CoAID News"]
    excluded = coaid.sensitivity_exclude_candidate.astype(bool)
    assert excluded.sum() == 198, f"Expected 198 exclusion candidates, found {excluded.sum()}"
    conservative = coaid.loc[~excluded].copy()
    for specification, frame, coding in [("conservative_exclude_198", conservative, "earliest"),
                                        ("all_release_memberships", coaid, "membership")]:
        for outcome in (*COUNTS, "engagement_present"):
            print(f"Fitting CoAID / {outcome} / {specification}", flush=True)
            fits = candidates("CoAID News", outcome, frame, coding)
            selected, reason = select(fits)
            base = primary_choices["CoAID News", outcome]
            for fit in fits:
                row = comparison_row("CoAID News", outcome, fit, specification)
                row["selected"] = fit is selected
                comparisons.append(row)
                coef_rows = coefficients("CoAID News", outcome, fit, specification, fit is selected)
                all_coefs.extend(coef_rows)
                for coef in coef_rows:
                    if coef["predictor"] != "fake_vs_real":
                        continue
                    same_model = next((f for f in primary_fits["CoAID News", outcome] if f.name == fit.name), None)
                    base_coef = next((c for c in coefficients("CoAID News", outcome, same_model, "primary")
                                      if c["predictor"] == "fake_vs_real" and c["model_component"] == coef["model_component"]), {})
                    sensitivity.append({**coef, "N": len(frame), "excluded_N": len(coaid) - len(frame),
                                        "log_likelihood": fit.ll, "AIC": fit.aic, "BIC": fit.bic,
                                        "convergence": fit.converged, "selected_model_changed": selected.name != base.name,
                                        "primary_selected_model": base.name, "sensitivity_selected_model": selected.name,
                                        "primary_same_model_IRR_or_OR": base_coef.get("IRR_or_OR"),
                                        "direction_changed": bool(np.sign(coef["coefficient"]) != np.sign(base_coef.get("coefficient", np.nan))),
                                        "relative_ratio_change": coef["IRR_or_OR"] / base_coef.get("IRR_or_OR", np.nan) - 1,
                                        "primary_CI_lower": base_coef.get("CI_lower"), "primary_CI_upper": base_coef.get("CI_upper"),
                                        "selection_reason": reason})
            # For exclusion sensitivity, use primary sample release mix for
            # standardized predictions to separate composition from parameters.
            reference_x = design(coaid, coding).to_numpy()
            predictions.extend(contrasts("CoAID News", outcome, selected, specification, reference_x))

    # Release strata use original, uncollapsed News records. Estimates across
    # strata are not independent; no pooling or meta-analysis is attempted.
    observations = pd.read_csv(ROOT / "data/processed/coaid_content_level.csv")
    observations = observations.loc[observations.content_category.eq("News")].copy()
    observations["analysis_group"] = observations.veracity_original.str.lower()
    release_rows = []
    for release, frame in observations.groupby("source_release_or_subset"):
        for group, part in frame.groupby("analysis_group"):
            release_rows.append({"release": release, "group": group, "N": len(part),
                                 "post_positive": int((part.unique_post_count > 0).sum()),
                                 "reply_positive": int((part.unique_reply_count > 0).sum()),
                                 "any_engagement": int(part.engagement_present.sum())})
        for outcome in (*COUNTS, "engagement_present"):
            print(f"Release diagnostic {release} / {outcome}", flush=True)
            fit = fit_one(frame[outcome].to_numpy(float), design(frame, "none"),
                          "Logit" if outcome == "engagement_present" else "NB2")
            specification = "release_uncollapsed_" + release
            comparisons.append(comparison_row("CoAID News", outcome, fit, specification))
            coefs = coefficients("CoAID News", outcome, fit, specification)
            all_coefs.extend(coefs)
            for coef in coefs:
                if coef["predictor"] == "fake_vs_real":
                    sensitivity.append({**coef, "N": len(frame), "fake_N": int(frame.analysis_group.eq("fake").sum()),
                                        "release": release, "convergence": fit.converged,
                                        "AIC": fit.aic, "BIC": fit.bic, "log_likelihood": fit.ll,
                                        "small_release_warning": release in RELEASES[2:],
                                        "selection_reason": "Diagnostic only; uncollapsed records; sparse September/November fake positives make these estimates unstable regardless of optimizer success."})

    comparison = pd.DataFrame(comparisons)
    comparison["selected"] = comparison["selected"].fillna(False).astype(bool)
    comparison.to_csv(TABLES / "primary_engagement_model_comparison.csv", index=False)
    pd.DataFrame(selected_coefs).to_csv(TABLES / "primary_engagement_model_results.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(TABLES / "coaid_model_sensitivity_results.csv", index=False)
    pd.DataFrame(all_coefs).to_csv(TABLES / "engagement_model_all_coefficients.csv", index=False)
    pd.DataFrame(predictions).to_csv(TABLES / "engagement_model_standardized_predictions.csv", index=False)
    pd.DataFrame(release_rows).to_csv(TABLES / "coaid_model_release_support.csv", index=False)
    pd.DataFrame(choices).to_csv(TABLES / "engagement_model_selection_decisions.csv", index=False)
    pd.DataFrame(profile_reply_dispersion(samples)).to_csv(TABLES / "engagement_reply_dispersion_profiles.csv", index=False)
    metadata = {"seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT / "data/processed").glob("*_primary.csv")}}
    (TABLES / "engagement_model_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = ["# Within-dataset engagement model selection", "",
              "Models use separate datasets and fake versus real coding (real reference). These are associations in observed engagement counts, not causal effects or rates adjusted for observation time.", "",
              "NB2 uses a log mean and estimated alpha, Var(Y|X)=mu+alpha*mu². Poisson is a diagnostic comparator. Primary CIs use item-level HC0 sandwich standard errors; model-based SEs are also exported. Neither covariance method resolves dependence between items sharing social posts or unresolved duplicate groups. CIs do not account for model selection or release measurement error.", "",
              "CoAID primary/conservative models adjust for earliest recorded release (May reference). A separate membership sensitivity uses all four release membership indicators; multi-release entities remain single rows with unioned engagement. No additional text/topic controls. September/November uncollapsed checks have limited fake samples and very few positive outcomes.", "",
              "The hurdle is fitted separately: logit P(Y>0), then zero-truncated NB2 on Y>0. Full log likelihood is the sum and k includes both components plus alpha. AIC/BIC use the full N. These compare models of the SAME full count outcome and SAME sample only; do not compare AIC across datasets, outcomes or sensitivity samples.", "",
              "A hurdle count exp(beta) is the ratio of underlying NB mean parameters in the positive component. Because E[Y|Y>0,X]=mu/(1-P_NB(0|X)), it is not generally the positive conditional mean ratio. The prediction table reports exact model-standardized positive means and ratios, and full mixture means, with HC0 delta-method CIs. Both veracity groups use the same covariate distribution; conservative estimates are standardized to the primary release mix.", "",
              "ZINB allows zeros from a latent structural-zero class and the NB process. Both components include veracity and CoAID release predictors. That latent distinction is not established by these data; ZINB is sensitivity rather than automatically preferred by AIC. Boundary estimates and unstable Hessians/SEs are flagged, even when the optimizer declares convergence. Multistart trials and all warnings are retained.", "",
              "Material zero misfit is flagged when the gap exceeds max(min(0.02, half the observed zero fraction), 3/N, two plug-in Bernoulli standard errors). This also detects severe relative underprediction of rare zeros. It is exploratory, not a calibrated hypothesis test. Overall zero agreement in the logistic hurdle is largely guaranteed by its intercept score equation; group/release agreement and count/residual shape remain necessary checks. Pearson dispersion and randomized quantile residuals (seed 42) diagnose remaining tail misfit. The stable flag means numerical/parameter stability, not proof of distributional adequacy or precision.", "",
              "## Selected models", ""]
    for choice in choices:
        report.append(f"- {choice['dataset']} / {choice['outcome']}: {choice['selected_model']}. Lowest numerical AIC: {choice['lowest_AIC_model']}. {choice['reason']}")
    report += ["", "## FakeHealth engagement presence", ""]
    for group, frame in samples["FakeHealth HealthStory"].groupby("analysis_group"):
        report.append(f"- {group}: {int(frame.engagement_present.sum())}/{len(frame)} ({100 * frame.engagement_present.mean():.3f}%) engaged.")
    report += ["", "Only seven FakeHealth items have no observed engagement (one fake, six real); a primary presence logistic model is omitted for essentially absent outcome variation. This does not assert that separation was proven.", "",
               "## Selected veracity coefficients", ""]
    for row in selected_coefs:
        if row["predictor"] == "fake_vs_real":
            report.append(f"- {row['dataset']} / {row['outcome']} / {row['model_component']}: exp(beta)={row['IRR_or_OR']:.4f}, 95% HC0 CI [{row['CI_lower']:.4f}, {row['CI_upper']:.4f}]. {row['interpretation']}")
    report += ["", "## Fit and selection interpretation", "",
               markdown_table(comparison.loc[comparison.specification.eq("primary"),
                   ["dataset", "outcome", "model", "AIC", "BIC", "stable", "observed_zero_fraction", "predicted_zero_fraction", "pearson_chi2_per_df"]]), "",
               "FakeHealth posts: NB2 is strongly preferred to Poisson and is the clearest usable count specification. FakeHealth replies: NB2 is the estimable working model, not the lowest nominal-AIC model. The NB hurdle pushes alpha to 100,000, with a singular/ill-conditioned information matrix. Fixed-alpha profiles rise over 20, 100, 1,000, 10,000, 100,000, independently confirming the boundary tendency. Its positive-count Wald CIs must not be used. FakeHealth ZINB adds no AIC improvement and its inflation contrast is weakly identified.", "",
               "CoAID posts: the stable hurdle has a large AIC/BIC advantage over NB2. The occurrence contrast is strongly negative, while the positive-volume contrast is positive and imprecise in the primary sample. A single lower overall engagement mean does not imply lower positive volume. The hurdle is both substantively useful and statistically preferable for posts.", "",
               "CoAID replies: the hurdle has lower nominal AIC but the same alpha-boundary issue; primary ZINB improves nominal AIC materially over NB2 but has nonidentifiable inflation parameters and no reliable full covariance. NB2 is therefore retained as a provisional explanatory baseline. The all-membership ZINB sensitivity is numerically stable and improves AIC, but it uses a different release design and does not establish a real structural-zero class. No adequate, identified primary two-part reply model is established here.", "",
               "Residual Pearson ratios remain above 1 for every selected count model, especially replies and CoAID posts. Heavy extreme counts remain poorly captured even where the central distribution and zero fractions fit. HC0 intervals reduce dependence on the assumed conditional variance but are not a remedy for mean misspecification, rare influential items or sparse release strata.", "",
               "## Predicted means and probabilities", "",
               markdown_table(pd.DataFrame(predictions).loc[lambda d: d.specification.eq("primary") & d.measure.isin([
                   "standardized_overall_mean_ratio", "standardized_positive_mean_ratio", "standardized_probability_real", "standardized_probability_fake"]),
                   ["dataset", "outcome", "measure", "estimate", "CI_lower", "CI_upper"]]), "",
               "For the CoAID hurdle, the exact standardized positive mean ratio differs from its count-component exp(beta). The standardized overall ratio combines the occurrence and positive-volume processes and is the appropriate all-item comparison, conditional on this specification. Probability predictions from an NB model are distribution-implied and should not replace the separately fitted binary logistic outcome.", "",
               "## Duplicate and release sensitivity", "",
               markdown_table(pd.DataFrame(sensitivity).loc[lambda d: d.selected.eq(True),
                   ["outcome", "specification", "model_component", "IRR_or_OR", "CI_lower", "CI_upper", "selected_model_changed"]]), "",
               "Excluding the 198 candidates preserves selected model families and coefficient directions. The any-engagement odds ratio changes little. Count magnitudes are not robust: the positive-post component moves farther above 1 and its CI excludes 1, while the reply NB2 IRR moves toward 1 and remains imprecise. Thus robustness of occurrence does not imply robustness of positive volume or overall counts.", "",
               "The all-release-membership sensitivity also preserves selected coefficient directions but substantially changes some standardized means. Rare multi-release covariate profiles and limited fake support cause extrapolation; the comparison table reports unsupported-profile fractions and mean calibration. Its count standardizations should not be promoted to definitive estimates. The occurrence association is considerably less sensitive to the coding choice.", "",
               markdown_table(pd.DataFrame(sensitivity).loc[lambda d: d.specification.str.startswith("release_uncollapsed"),
                   ["release", "outcome", "fake_N", "IRR_or_OR", "CI_lower", "CI_upper", "small_release_warning"]]), "",
               "May and July both show lower marginal observed counts and occurrence for fake items, but their sizes differ. September (56 fake records) and November (31) have very few fake positive outcomes. Treat their coefficients and Wald intervals as unstable diagnostics even when optimization converges or an interval happens to exclude 1. Strata use uncollapsed original records, so they are not independent estimates of the primary entity-resolved estimand. These results demonstrate descriptive heterogeneity; no formal interaction test has been run.", "",
               "## Estimates to carry forward and next step", "",
               "The FakeHealth post NB2 contrast and CoAID engagement-occurrence logistic contrast are the most stable within their own estimands. The latter is supported by both duplicate exclusion and release coding, but it cannot be compared directly to a count IRR. FakeHealth replies are imprecise; CoAID positive volume and reply means depend materially on sensitivity choices. The primary CoAID hurdle overall count ratio is potentially useful but not yet robust to release coding.", "",
               "Next, review these results and pre-specify a common estimand for later cross-context work: overall observed-count mean ratios or standardized engagement probabilities, with occurrence and positive volume reported separately. Before the interaction analysis, assess influential high-count entities, release-profile overlap and uncertainty under duplicate/shared-engagement dependence. The reply positive-count boundary needs an explicit distributional decision. Do not compare a FakeHealth all-item NB IRR directly with a CoAID positive-component parameter ratio. This stage has not performed the cross-dataset comparison."]
    report += ["", "## Method references", "",
               "- [statsmodels hurdle and truncated models](https://www.statsmodels.org/stable/examples/notebooks/generated/count_hurdle.html)",
               "- [Zero-truncated NB prediction: mean versus mean-main](https://www.statsmodels.org/stable/generated/statsmodels.discrete.truncated_model.TruncatedLFNegativeBinomialP.predict.html)",
               "- [ZINB prediction](https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.predict.html)", "",
               "This run stops at within-dataset regression. No pooled cross-context model, text model, machine learning, or timestamp outcome was fitted."]
    REPORT.write_text("\n".join(report) + "\n")
    print(comparison.loc[comparison.specification.eq("primary"),
                         ["dataset", "outcome", "model", "AIC", "convergence", "stable", "observed_zero_fraction", "predicted_zero_fraction"]].to_string(index=False))
    print(f"Saved {len(comparison)} comparisons, {len(selected_coefs)} selected coefficients, {len(sensitivity)} sensitivity effects.")


if __name__ == "__main__":
    main()
