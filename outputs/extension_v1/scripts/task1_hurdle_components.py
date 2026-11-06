"""Task 1: report saved CoAID A hurdle components; no refitting. Reads frozen tables only."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
COMP = ROOT/'outputs/tables/version_aware_hurdle_components.csv'
DRAWS = ROOT/'outputs/tables/version_aware_bootstrap_refits.csv'
RESULTS = ROOT/'outputs/tables/version_aware_engagement_results.csv'
OUT = ROOT/'outputs/extension_v1/tables/coaid_hurdle_components.csv'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
comp = pd.read_csv(COMP); comp = comp[comp.specification.eq('A_primary_strict')].set_index('measure')
res = pd.read_csv(RESULTS); res = res[res.specification.eq('A_primary_strict') & res.outcome.eq('unique_post_count')].iloc[0]
draws = pd.read_csv(DRAWS, low_memory=False)
a = draws[draws.specification.eq('A_primary_strict') & draws.success.astype(str).eq('True')].copy()
assert len(a) == 1000, len(a)
a['post_probability_ratio'] = a.post_probability_fake / a.post_probability_real
a['component_product'] = a.post_probability_ratio * a.positive_mean_ratio
def pct(x): lo, hi = np.percentile(np.asarray(x, float), [2.5, 97.5], method='linear'); return float(lo), float(hi)
# saved percentile interval for mean_ratio must reproduce from the same draws
lo, hi = pct(a.mean_ratio); assert abs(lo-comp.loc['mean_ratio','cluster_CI_lower'])<1e-9 and abs(hi-comp.loc['mean_ratio','cluster_CI_upper'])<1e-9
rows = []
def add(measure, est, source, lo=None, hi=None, interval_source='', note=''):
    rows.append(dict(specification='A_primary_strict', measure=measure, estimate=float(est), CI_lower=lo, CI_upper=hi,
        point_source=source, interval_source=interval_source, note=note))
S = 'outputs/tables/version_aware_hurdle_components.csv (saved point estimate)'
D = 'percentile 2.5/97.5 of 1000 successful saved draws in version_aware_bootstrap_refits.csv (linear method; reproduces saved mean_ratio interval exactly)'
for m, label in [('post_probability_fake','P_posts_gt_0_fake'),('post_probability_real','P_posts_gt_0_real'),
                 ('positive_fake_mean','E_posts_given_positive_fake'),('positive_real_mean','E_posts_given_positive_real'),
                 ('positive_mean_ratio','conditional_count_ratio_fake_over_real'),
                 ('standardized_fake_mean','E_posts_fake_full'),('standardized_real_mean','E_posts_real_full'),('mean_ratio','full_ratio_fake_over_real')]:
    lo, hi = pct(a[m]); add(label, comp.loc[m,'estimate'], S, lo, hi, D,
        'standardized over the A population covariate mix (first-release fixed effects); hurdle logit + zero-truncated NB2')
    assert abs(lo-comp.loc[m,'cluster_CI_lower'])<1e-9 and abs(hi-comp.loc[m,'cluster_CI_upper'])<1e-9, m
pr = comp.loc['post_probability_fake','estimate']/comp.loc['post_probability_real','estimate']
lo, hi = pct(a.post_probability_ratio)
add('occurrence_probability_ratio_fake_over_real', pr, 'ratio of the two saved standardized post probabilities', lo, hi, D,
    'ratio of marginal standardized probabilities; NOT the conditional logistic odds ratio')
prod = pr*comp.loc['positive_mean_ratio','estimate']; lo, hi = pct(a.component_product)
add('product_of_component_ratios', prod, 'occurrence_probability_ratio x conditional_count_ratio (point estimates)', lo, hi, D,
    'product of averaged components; differs from the frozen estimand, which averages row-level P(Y>0|X)*E(Y|Y>0,X) before taking the Fake/Real ratio')
add('full_ratio_minus_product', comp.loc['mean_ratio','estimate']-prod, 'difference of the two rows above', *pct(a.mean_ratio-a.component_product), D,
    f'share of draws with product > full ratio = {float((a.component_product>a.mean_ratio).mean()):.3f}')
add('draw_level_median_product', float(a.component_product.median()), 'median over draws', note='descriptive only')
out = pd.DataFrame(rows)
out['inputs_sha256'] = json.dumps({str(p.relative_to(ROOT)): sha(p) for p in [COMP, DRAWS, RESULTS]})
out.to_csv(OUT, index=False)
print(out[['measure','estimate','CI_lower','CI_upper']].to_string(index=False))
print('model:', res.model, '| N fake/real', res.fake_N, res.real_N, '| zero proportion', round(res.zero_proportion,4), '| draws', res.successful_refits)
