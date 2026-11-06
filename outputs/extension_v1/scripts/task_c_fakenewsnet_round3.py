"""Round 3 Task A: dominant-component exclusion, multi-membership diagnostic, domain composition. Reuses round-2 loaders and frozen helpers."""
import sys, json, time, hashlib, logging
from pathlib import Path
from collections import Counter
from urllib.parse import urlsplit
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'outputs/extension_v1/scripts'))
import run_primary_engagement_models as pem, run_version_aware_engagement_analysis as vae
from run_cross_context_engagement_contrast import independent_product
import task_b_fakenewsnet_engagement as r2
T = ROOT/'outputs/extension_v1/tables'; LOG = ROOT/'logs/extension_v1_fakenewsnet_round3.log'
logging.basicConfig(filename=LOG, level=logging.INFO, format='%(asctime)s %(message)s'); log = logging.info
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    saved = pd.read_csv(ROOT/'outputs/tables/version_aware_bootstrap_refits.csv', low_memory=False)
    refs = {n: saved[saved.specification.eq(s) & saved.success.astype(str).eq('True')].mean_ratio.to_numpy(float) for n, s in [('CoAID_A_primary_strict','A_primary_strict'),('FakeHealth_HealthStory','FakeHealth_HealthStory')]}
    ref_points = {'CoAID_A_primary_strict': 0.2528032930591203, 'FakeHealth_HealthStory': 0.8402633149648507}
    r2res = pd.read_csv(T/'fakenewsnet_engagement_results.csv'); r2cmp = pd.read_csv(T/'fakenewsnet_engagement_model_comparison.csv')
    excl_rows, con_rows, draws_all, mm_rows, dom_rows, cfg = [], [], [], [], [], {}
    for ordinal, subset in [(2, 'politifact'), (3, 'gossipcop')]:
        name = 'PolitiFact' if subset == 'politifact' else 'GossipCop'
        d = r2.load(subset); comp = r2.components(d); d['component'] = comp
        posts_by_comp = d.groupby('component').unique_post_count.sum(); largest = int(posts_by_comp.idxmax())
        family = r2cmp[r2cmp.subset.eq(name) & r2cmp.selected.astype(str).eq('True')].model.iloc[0]; assert family == 'Hurdle_Logit_ZTNB2'
        primary_ratio = float(r2res[r2res.subset.eq(name) & r2res.measure.eq('mean_ratio')].estimate.iloc[0])
        # ---- A2 multi-membership (label-blind)
        cnt = Counter(t for s in d.tweet_set for t in s); vals = np.array(list(cnt.values()))
        bins = {'1': (vals == 1).sum(), '2': (vals == 2).sum(), '3-4': ((vals >= 3) & (vals <= 4)).sum(), '5-9': ((vals >= 5) & (vals <= 9)).sum(), '10+': (vals >= 10).sum()}
        memberships = vals.sum()
        mm = dict(subset=name, scope='all rows', distinct_tweet_ids=len(vals), total_memberships=int(memberships), ids_attached_to_multiple_rows=int((vals > 1).sum()), share_ids_multiple_rows=float((vals > 1).mean()),
                  share_memberships_from_multi_row_ids=float(vals[vals > 1].sum() / memberships), max_rows_per_id=int(vals.max()), **{f'ids_rows_{k}': int(v) for k, v in bins.items()}, **{f'share_ids_rows_{k}': float(v / len(vals)) for k, v in bins.items()})
        mm_rows.append(mm)
        big = d[d.component.eq(largest)]; cntb = Counter(t for s in big.tweet_set for t in s); vb = np.array(list(cntb.values()))
        mm_rows.append(dict(subset=name, scope=f'largest component ({len(big)} rows)', distinct_tweet_ids=len(vb), total_memberships=int(vb.sum()), ids_attached_to_multiple_rows=int((vb > 1).sum()), share_ids_multiple_rows=float((vb > 1).mean()),
                            share_memberships_from_multi_row_ids=float(vb[vb > 1].sum() / vb.sum()), max_rows_per_id=int(vb.max()), ids_rows_10plus=int((vb >= 10).sum()), share_ids_rows_10plus=float((vb >= 10).mean()), component_rows=len(big),
                            component_fake_N=int(big.analysis_group.eq('fake').sum()), component_real_N=int(big.analysis_group.eq('real').sum())))
        log(f'{name}: distinct ids={len(vals)} multi-row share={(vals>1).mean():.4f}; largest comp ids={len(vb)} 10+ share={(vb>=10).mean():.4f}')
        # ---- A3 domain composition (extraction label-blind; tabulated by class)
        def domain(u):
            u = (u or '').strip()
            if not u: return '(missing)'
            try: h = urlsplit(u if '://' in u else 'https://' + u).netloc.lower()
            except Exception: return '(unparsable)'
            h = h.split(':')[0]; return h[4:] if h.startswith('www.') else (h or '(unparsable)')
        d['domain'] = d.news_url.map(domain)
        for cls in ['fake', 'real']:
            g = d[d.analysis_group.eq(cls)]; vc = g.domain.value_counts(); top = vc.head(10)
            for rank, (dom, n) in enumerate(top.items(), 1):
                dom_rows.append(dict(subset=name, label=cls.title(), rank=rank, domain=dom, rows=int(n), class_N=len(g), row_share_within_class=float(n / len(g)), top10_share_within_class=float(top.sum() / len(g)), distinct_domains_in_class=int(vc.size)))
        # ---- A1 exclusion
        keep = d[~d.component.eq(largest)].reset_index(drop=True); x = r2.design(keep); y = keep.unique_post_count.to_numpy(float)
        fit = pem.fit_hurdle(y, x); assert fit.stable, 'exclusion hurdle unstable'
        pm = vae.point_metrics(fit, x)
        ids = sorted(keep.component.unique()); clusters = [np.flatnonzero(keep.component.eq(c).to_numpy()) for c in ids]
        totals = [keep.iloc[i].unique_post_count.sum() for i in clusters]
        cfg[subset] = dict(ordinal=ordinal, clusters=clusters, largest=int(np.argmax(totals)), x=x.to_numpy(), columns=list(x.columns), y=y, model='Hurdle_Logit_ZTNB2', start=fit.params)
        draws, attempted, successes, t0 = [], 0, 0, time.monotonic()
        with ProcessPoolExecutor(max_workers=4, initializer=r2.init, initargs=(cfg,)) as pool:
            while successes < 1000 and attempted < 1250:
                n = min(50, 1000 - successes, 1250 - attempted)
                batch = list(pool.map(r2.draw, [(subset, a) for a in range(attempted, attempted + n)]))
                draws.extend(batch); attempted += n; successes += sum(r['success'] for r in batch)
                log(f'{name} exclusion: successes={successes} attempts={attempted} elapsed={time.monotonic()-t0:.0f}s')
        D = pd.DataFrame(draws); D['subset'] = name; D['analysis'] = 'exclude_dominant_component'; draws_all.append(D); acc = D[D.success]
        q = lambda c: (float(np.quantile(acc[c], .025)), float(np.quantile(acc[c], .975)))
        fake = keep.analysis_group.eq('fake')
        for m in ['standardized_fake_mean','standardized_real_mean','mean_ratio','post_probability_fake','post_probability_real','positive_fake_mean','positive_real_mean','positive_mean_ratio']:
            lo, hi = q(m)
            excl_rows.append(dict(subset=name, analysis='exclude_dominant_component', model='Hurdle_Logit_ZTNB2', measure=m, estimate=pm[m], CI_lower=lo, CI_upper=hi, primary_estimate=float(r2res[r2res.subset.eq(name) & r2res.measure.eq(m)].estimate.iloc[0]),
                                  N=len(keep), fake_N=int(fake.sum()), real_N=int((~fake).sum()), excluded_rows=int(d.component.eq(largest).sum()), excluded_post_membership_share=float(posts_by_comp.max() / d.unique_post_count.sum()),
                                  successful_draws=len(acc), attempts=attempted, note='reference population changed by exclusion; not a replacement primary estimate'))
        log(f'{name} exclusion: ratio={pm["mean_ratio"]:.4f} CI=({q("mean_ratio")[0]:.4f},{q("mean_ratio")[1]:.4f}) vs primary {primary_ratio:.4f}')
        for rname, rdraws in refs.items():
            theta, values = independent_product(acc.mean_ratio.to_numpy(float), rdraws); qs = np.quantile(values, [.025, .5, .975])
            con_rows.append(dict(subset=name, analysis='exclude_dominant_component', reference=rname, subset_ratio=pm['mean_ratio'], reference_ratio=ref_points[rname], ratio_of_ratios=pm['mean_ratio'] / ref_points[rname],
                                 CI_lower_ratio=qs[0], CI_upper_ratio=qs[2], bootstrap_median=qs[1], ratio_interval_contains_1=bool(qs[0] <= 1 <= qs[2]), cartesian_combinations=len(values),
                                 primary_ratio_of_ratios=primary_ratio / ref_points[rname], direction='subset ratio / reference ratio; C<1 = larger relative Fake deficit in the FakeNewsNet subset',
                                 note='subset population excludes its dominant component; reference draws are the saved full-population draws, as in the frozen CoAID exclusion contrast'))
    pd.DataFrame(excl_rows).to_csv(T/'fakenewsnet_round3_exclusion_results.csv', index=False); pd.DataFrame(con_rows).to_csv(T/'fakenewsnet_round3_exclusion_contrast.csv', index=False)
    pd.concat(draws_all, ignore_index=True).to_csv(T/'fakenewsnet_round3_exclusion_bootstrap_refits.csv', index=False)
    pd.DataFrame(mm_rows).to_csv(T/'fakenewsnet_round3_multimembership.csv', index=False); pd.DataFrame(dom_rows).to_csv(T/'fakenewsnet_round3_domain_composition.csv', index=False)
    json.dump(dict(seed=42, ordinals={'politifact': 2, 'gossipcop': 3}, inputs={k: sha(T/k) for k in ['fakenewsnet_engagement_results.csv','fakenewsnet_engagement_model_comparison.csv']} | {'version_aware_bootstrap_refits.csv': sha(ROOT/'outputs/tables/version_aware_bootstrap_refits.csv')} | {f'{s}_{l}.csv': sha(ROOT/f'data/raw/fakenewsnet/{s}_{l}.csv') for s in ['politifact','gossipcop'] for l in ['fake','real']},
              finished=time.strftime('%Y-%m-%d %H:%M:%S'), complete=True), open(T/'fakenewsnet_round3_run_metadata.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    E = pd.DataFrame(excl_rows); print(E[E.measure.isin(['mean_ratio','post_probability_fake','post_probability_real','positive_mean_ratio'])][['subset','measure','estimate','CI_lower','CI_upper','primary_estimate','N','excluded_rows','excluded_post_membership_share','successful_draws','attempts']].round(4).to_string(index=False))
    print(); print(pd.DataFrame(con_rows)[['subset','reference','ratio_of_ratios','CI_lower_ratio','CI_upper_ratio','bootstrap_median','ratio_interval_contains_1','primary_ratio_of_ratios']].round(4).to_string(index=False))
    print(); print(pd.DataFrame(mm_rows).T.to_string())
    print(); D = pd.DataFrame(dom_rows); print(D[['subset','label','rank','domain','rows','class_N','row_share_within_class','top10_share_within_class']].round(4).to_string(index=False))
    log('COMPLETE')
if __name__ == '__main__': main()
