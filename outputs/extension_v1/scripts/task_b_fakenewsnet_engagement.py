"""Round 2 Task B: FakeNewsNet engagement under the frozen estimand. Reuses frozen fitting/standardization helpers; writes only under outputs/extension_v1/ and logs/."""
import sys, re, json, csv, time, math, hashlib, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import run_primary_engagement_models as pem
import run_version_aware_engagement_analysis as vae
from run_cross_context_engagement_contrast import independent_product
T = ROOT/'outputs/extension_v1/tables'; LOG = ROOT/'logs/extension_v1_fakenewsnet_engagement.log'
logging.basicConfig(filename=LOG, level=logging.INFO, format='%(asctime)s %(message)s'); log = logging.info
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
csv.field_size_limit(10**9)
CONFLICT = {'politifact14940', 'politifact14920'}
def load(subset):
    parts = []
    for lab in ['fake', 'real']:
        d = pd.read_csv(ROOT/f'data/raw/fakenewsnet/{subset}_{lab}.csv', dtype=str, keep_default_na=False, engine='python')
        d['label'] = lab; parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d = d[~d.id.isin(CONFLICT)].reset_index(drop=True); assert d.id.is_unique
    d['tweet_set'] = d.tweet_ids.map(lambda s: frozenset(t.strip() for t in s.split('\t') if t.strip()))
    d['unique_post_count'] = d.tweet_set.map(len).astype(float)
    d['analysis_group'] = d.label; d['content_uid'] = f'fakenewsnet:{subset}:' + d.id
    d['title_norm'] = d.title.map(lambda s: re.sub(r'\s+', ' ', s).strip())
    return d
def components(d):
    parent = list(range(len(d)))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    first = {}
    for i, s in enumerate(d.tweet_set):
        for t in s:
            if t in first: union(first[t], i)
            else: first[t] = i
    comp = np.array([find(i) for i in range(len(d))])
    ids = {c: k for k, c in enumerate(sorted(set(comp)))}
    return np.array([ids[c] for c in comp])
def design(d): return pd.DataFrame({'Intercept': 1., 'fake_vs_real': d.analysis_group.eq('fake').astype(float)}, index=d.index)
_CFG = {}
def init(cfg):
    global _CFG; _CFG = cfg
def draw(task):
    subset, attempt = task; c = _CFG[subset]
    rng = np.random.default_rng(np.random.SeedSequence([42, c['ordinal'], attempt]))
    idx, sampled = vae.resample_indices(c['clusters'], rng)
    x = pd.DataFrame(c['x'][idx], columns=c['columns']); y = c['y'][idx]
    row = dict(subset=subset, attempt=attempt, seed=42, sampled_N=len(idx), distinct_sampled_clusters=len(set(sampled)), largest_component_draws=int((sampled == c['largest']).sum()), success=False, error='')
    t0 = time.monotonic()
    try:
        if np.linalg.matrix_rank(x) != x.shape[1]: raise ValueError('rank loss')
        fit, retry, trials = vae.refit_checked(y, x, c['model'], c['start'])
        row.update(converged=fit.converged, stable=fit.stable, hessian_ok=fit.hessian_ok, boundary=fit.boundary, alpha=float(fit.params[-1]), retry=retry)
        if not fit.stable: raise ValueError('unstable refit')
        m = vae.point_metrics(fit, x)
        if not np.isfinite([m['standardized_fake_mean'], m['standardized_real_mean'], m['mean_ratio']]).all(): raise ValueError('nonfinite')
        row.update(m, success=True)
    except Exception as e: row['error'] = repr(e)
    row['elapsed_seconds'] = time.monotonic() - t0
    return row
def main():
    meta = dict(seed=42, target=1000, attempt_cap=1250, inputs={}, started=time.strftime('%Y-%m-%d %H:%M:%S'))
    graph_rows, comp_rows, result_rows, contrast_rows, dup_rows, all_draws, cfg = [], [], [], [], [], [], {}
    saved = pd.read_csv(ROOT/'outputs/tables/version_aware_bootstrap_refits.csv', low_memory=False)
    refs = {}
    for name, spec in [('CoAID_A_primary_strict', 'A_primary_strict'), ('FakeHealth_HealthStory', 'FakeHealth_HealthStory')]:
        acc = saved[saved.specification.eq(spec) & saved.success.astype(str).eq('True')]
        assert len(acc) == 1000; refs[name] = acc.mean_ratio.to_numpy(float)
    ref_points = {'CoAID_A_primary_strict': 0.2528032930591203, 'FakeHealth_HealthStory': 0.8402633149648507}
    for ordinal, subset in enumerate(['politifact', 'gossipcop']):
        name = 'PolitiFact' if subset == 'politifact' else 'GossipCop'
        d = load(subset); meta['inputs'][subset] = {f'{subset}_{l}.csv': sha(ROOT/f'data/raw/fakenewsnet/{subset}_{l}.csv') for l in ['fake','real']}
        # dependence graph (label-blind)
        comp = components(d); d['component'] = comp
        sizes = pd.Series(comp).value_counts(); posts_by_comp = d.groupby('component').unique_post_count.sum()
        largest = int(posts_by_comp.idxmax()); total_posts = d.unique_post_count.sum()
        fake = d.analysis_group.eq('fake')
        graph_rows.append(dict(subset=name, N=len(d), fake_N=int(fake.sum()), real_N=int((~fake).sum()), zero_N=int(d.unique_post_count.eq(0).sum()), total_unique_posts=int(total_posts),
            components=int(len(sizes)), singleton_components=int((sizes == 1).sum()), multi_row_components=int((sizes > 1).sum()), rows_in_multi_row_components=int(sizes[sizes > 1].sum()),
            largest_component_rows=int(sizes.max()), largest_component_post_membership_share=float(posts_by_comp.max() / total_posts), largest_component_row_share=float(sizes.max() / len(d)),
            largest_component_fake_N=int(fake[d.component.eq(largest)].sum()), largest_component_real_N=int((~fake)[d.component.eq(largest)].sum()),
            tweet_ids_shared_across_rows=int(sum(1 for t, n in pd.Series([t for s in d.tweet_set for t in s]).value_counts().items() if n > 1))))
        log(f'{name}: N={len(d)} components={len(sizes)} largest={sizes.max()} largest_post_share={posts_by_comp.max()/total_posts:.4f}')
        # model selection (frozen rule)
        x = design(d); y = d.unique_post_count.to_numpy(float)
        fits = [pem.fit_one(y, x, 'Poisson'), pem.fit_one(y, x, 'NB2'), pem.fit_hurdle(y, x)]
        chosen, reason = pem.select(fits)
        for f in fits: comp_rows.append(dict(subset=name, model=f.name, k=f.k, log_likelihood=f.ll, AIC=f.aic, BIC=f.bic, converged=f.converged, stable=f.stable, hessian_ok=f.hessian_ok, boundary=f.boundary, selected=f.name == chosen.name, selection_reason=reason if f.name == chosen.name else ''))
        log(f'{name}: selected {chosen.name}; AIC NB2={fits[1].aic:.2f} hurdle={fits[2].aic:.2f}; {reason[:80]}')
        pm = vae.point_metrics(chosen, x)
        ids = sorted(set(comp)); clusters = [np.flatnonzero(comp == c) for c in ids]
        cfg[subset] = dict(ordinal=ordinal, clusters=clusters, largest=ids.index(largest), x=x.to_numpy(), columns=list(x.columns), y=y, model=chosen.name, start=chosen.params)
        # bootstrap
        draws = []; attempted = 0; successes = 0; t0 = time.monotonic()
        with ProcessPoolExecutor(max_workers=4, initializer=init, initargs=(cfg,)) as pool:
            while successes < 1000 and attempted < 1250:
                n = min(50, 1000 - successes, 1250 - attempted)
                batch = list(pool.map(draw, [(subset, a) for a in range(attempted, attempted + n)]))
                draws.extend(batch); attempted += n; successes += sum(r['success'] for r in batch)
                log(f'{name}: bootstrap successes={successes} attempts={attempted} elapsed={time.monotonic()-t0:.0f}s')
        D = pd.DataFrame(draws); D['subset'] = name; all_draws.append(D)
        acc = D[D.success]
        def q(col): v = acc[col].to_numpy(float); return float(np.quantile(v, .025)), float(np.quantile(v, .975))
        measures = ['standardized_fake_mean', 'standardized_real_mean', 'mean_difference', 'mean_ratio']
        if chosen.name == 'Hurdle_Logit_ZTNB2': measures += ['post_probability_fake', 'post_probability_real', 'post_probability_difference', 'positive_fake_mean', 'positive_real_mean', 'positive_mean_ratio']
        for mname in measures:
            lo, hi = q(mname) if len(acc) >= 2 else (np.nan, np.nan)
            result_rows.append(dict(subset=name, model=chosen.name, measure=mname, estimate=pm[mname], CI_lower=lo, CI_upper=hi, successful_draws=len(acc), attempts=attempted, N=len(d), fake_N=int(fake.sum()), real_N=int((~fake).sum()),
                uncertainty_method='component pairs bootstrap, percentile 95%, linear quantiles, seed 42', population='all rows incl. zeros; label-conflict ids excluded'))
        log(f'{name}: ratio={pm["mean_ratio"]:.4f} CI=({q("mean_ratio")[0]:.4f},{q("mean_ratio")[1]:.4f}) successes={len(acc)}/{attempted}')
        # contrasts vs saved references
        for rname, rdraws in refs.items():
            theta, values = independent_product(acc.mean_ratio.to_numpy(float), rdraws)
            qs = [.025, .25, .5, .75, .975]; cq, tq = np.quantile(values, qs), np.quantile(theta, qs)
            point = pm['mean_ratio'] / ref_points[rname]
            contrast_rows.append(dict(subset=name, reference=rname, subset_ratio=pm['mean_ratio'], reference_ratio=ref_points[rname], ratio_of_ratios=point, log_ratio_of_ratios=float(np.log(point)),
                CI_lower_ratio=cq[0], CI_upper_ratio=cq[-1], CI_lower_log=tq[0], CI_upper_log=tq[-1], bootstrap_median=cq[2], percentile_25_ratio=cq[1], percentile_75_ratio=cq[3],
                ratio_interval_contains_1=bool(cq[0] <= 1 <= cq[-1]), subset_successful_draws=len(acc), reference_successful_draws=len(rdraws), cartesian_combinations=len(values),
                direction='subset ratio divided by reference ratio; C<1 = larger relative Fake deficit in the FakeNewsNet subset',
                uncertainty_method='Full Cartesian product of independent component-bootstrap ratios; 1000 x 1000 empirical pairs; percentile 95% interval',
                interpretation_note='Observational dataset-context contrast; different collection (title search), years, label constructs; no replies/retweets; not a causal effect'))
        # duplicate-title sensitivity (fixed family, point estimate only)
        dup = d.title_norm.duplicated(keep=False); dd = d[~dup].reset_index(drop=True)
        xd = design(dd); yd = dd.unique_post_count.to_numpy(float)
        fd = pem.fit_hurdle(yd, xd) if chosen.name == 'Hurdle_Logit_ZTNB2' else pem.fit_one(yd, xd, chosen.name)
        pmd = vae.point_metrics(fd, xd)
        dup_rows.append(dict(subset=name, model=chosen.name, primary_N=len(d), dropped_rows=int(dup.sum()), dropped_fake=int((dup & fake).sum()), dropped_real=int((dup & ~fake).sum()), sensitivity_N=len(dd),
            primary_ratio=pm['mean_ratio'], sensitivity_ratio=pmd['mean_ratio'], ratio_change=pmd['mean_ratio'] - pm['mean_ratio'], log_ratio_change=float(np.log(pmd['mean_ratio']) - np.log(pm['mean_ratio'])), sensitivity_fit_stable=fd.stable))
        log(f'{name}: duplicate-title sensitivity dropped {int(dup.sum())} rows; ratio {pm["mean_ratio"]:.4f} -> {pmd["mean_ratio"]:.4f}')
    pd.DataFrame(graph_rows).to_csv(T/'fakenewsnet_overlap_graph_summary.csv', index=False)
    pd.DataFrame(comp_rows).to_csv(T/'fakenewsnet_engagement_model_comparison.csv', index=False)
    pd.concat(all_draws, ignore_index=True).to_csv(T/'fakenewsnet_engagement_bootstrap_refits.csv', index=False)
    pd.DataFrame(result_rows).to_csv(T/'fakenewsnet_engagement_results.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(T/'fakenewsnet_engagement_contrast.csv', index=False)
    pd.DataFrame(dup_rows).to_csv(T/'fakenewsnet_duplicate_title_sensitivity.csv', index=False)
    meta.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), saved_draws_sha256=sha(ROOT/'outputs/tables/version_aware_bootstrap_refits.csv'), script_sha256=sha(__file__), complete=True)
    json.dump(meta, open(T/'fakenewsnet_engagement_run_metadata.json', 'w'), indent=1)
    log('COMPLETE')
if __name__ == '__main__': main()
