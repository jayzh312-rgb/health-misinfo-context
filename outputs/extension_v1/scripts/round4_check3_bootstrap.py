"""Round 4 Check 3: bootstrap shape conditional on dominant-component inclusions (saved draws only), per-context figures, and a label-blind PolitiFact title audit."""
import sys, os, re, tempfile, itertools
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'round4-mpl'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'outputs/extension_v1/scripts'))
T = ROOT/'outputs/extension_v1/tables'; F = ROOT/'outputs/extension_v1/figures'
va = pd.read_csv(ROOT/'outputs/tables/version_aware_bootstrap_refits.csv', low_memory=False); fn = pd.read_csv(T/'fakenewsnet_engagement_bootstrap_refits.csv', low_memory=False)
ctx = {'CoAID_A': va[va.specification.eq('A_primary_strict') & va.success.astype(str).eq('True')], 'PolitiFact': fn[fn.subset.eq('PolitiFact') & fn.success.astype(str).eq('True')], 'GossipCop': fn[fn.subset.eq('GossipCop') & fn.success.astype(str).eq('True')]}
rows, hist = [], []
for name, d in ctx.items():
    assert len(d) == 1000; k = d.largest_component_draws.astype(int).to_numpy(); r = d.mean_ratio.to_numpy(float)
    cls = np.where(k == 0, '0', np.where(k == 1, '1', '2+'))
    for c in ['all', '0', '1', '2+']:
        m = np.ones(len(r), bool) if c == 'all' else cls == c
        if m.sum() == 0: rows.append(dict(context=name, inclusions=c, draws=0)); continue
        q = np.quantile(r[m], [.025, .25, .5, .75, .975])
        rows.append(dict(context=name, inclusions=c, draws=int(m.sum()), share_of_draws=float(m.mean()), ratio_p025=q[0], ratio_p25=q[1], ratio_median=q[2], ratio_p75=q[3], ratio_p975=q[4], ratio_mean=float(r[m].mean())))
    kde = gaussian_kde(np.log(r)); xs = np.linspace(np.log(r).min(), np.log(r).max(), 400); dens = kde(xs); peaks = int(((dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])).sum())
    counts, edges = np.histogram(np.log(r), bins=20)
    for i in range(20): hist.append(dict(context=name, bin=i+1, log_ratio_low=edges[i], log_ratio_high=edges[i+1], ratio_low=np.exp(edges[i]), ratio_high=np.exp(edges[i+1]), draws=int(counts[i]), draws_with_0_inclusions=int(((cls == '0') & (np.log(r) >= edges[i]) & (np.log(r) < edges[i+1] + (1e-12 if i == 19 else 0))).sum())))
    rows.append(dict(context=name, inclusions='kde_local_maxima_log_ratio', draws=peaks, share_of_draws=float((k >= 1).mean()), ratio_mean=float(r.mean())))
    fig, ax = plt.subplots(figsize=(6.4, 3.6)); bins_ = np.linspace(np.log(r).min(), np.log(r).max(), 31)
    for c, col in [('0', '#295C7A'), ('1', '#A65D32'), ('2+', '#707980')]:
        if (cls == c).any(): ax.hist(np.log(r[cls == c]), bins=bins_, alpha=.75, color=col, label=f'{c} inclusions (n={int((cls==c).sum())})')
    ax.axvline(0, color='k', lw=.8, ls='--'); ax.set_xlabel('log Fake/Real ratio (draw)'); ax.set_ylabel('draws'); ax.set_title(f'{name}: bootstrap ratio by dominant-component inclusions; KDE local maxima = {peaks}', fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F/f'round4_bootstrap_shape_{name}.png', dpi=200); fig.savefig(F/f'round4_bootstrap_shape_{name}.svg'); plt.close(fig)
R = pd.DataFrame(rows); R.to_csv(T/'round4_bootstrap_conditional.csv', index=False); pd.DataFrame(hist).to_csv(T/'round4_bootstrap_histogram.csv', index=False)
pd.set_option('display.width', 250); print(R.round(4).to_string(index=False))
# ---- label-blind PolitiFact title audit
import task_b_fakenewsnet_engagement as r2
from prepare_text_analysis_cohorts import near_pairs
d = r2.load('politifact'); comp = r2.components(d); d['component'] = comp
largest = int(d.groupby('component').unique_post_count.sum().idxmax()); d['in_dominant'] = d.component.eq(largest)
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
d['t'] = d.title.map(norm)
def bigrams(t): w = re.findall(r'\w+', t); return {tuple(w[i:i+2]) for i in range(len(w)-1)}
def audit(g, name):
    titles = g.t.tolist(); n = len(titles); exact_dup = g.t.duplicated(keep=False).mean()
    def near_share(minimum):
        pairs, _ = near_pairs(set(titles), 2, .95, minimum); involved = {a for a, b, _ in pairs} | {b for a, b, _ in pairs}; return float(g.t.isin(involved).mean()), len(pairs)
    n8, p8 = near_share(8); n4, p4 = near_share(4)
    sets = [bigrams(t) for t in titles]; js = []
    for a, b in itertools.combinations(range(n), 2):
        u = sets[a] | sets[b]; js.append(len(sets[a] & sets[b]) / len(u) if u else 0.)
    return dict(group=name, rows=n, fake_N=int(g.analysis_group.eq('fake').sum()), real_N=int(g.analysis_group.eq('real').sum()), exact_duplicate_title_share=float(exact_dup), distinct_titles=int(g.t.nunique()),
                near_duplicate_share_frozen_rule_min8=n8, near_pairs_min8=p8, near_duplicate_share_relaxed_min4=n4, near_pairs_min4=p4, mean_pairwise_title_bigram_jaccard=float(np.mean(js)), pairs=len(js), median_title_tokens=float(g.t.map(lambda s: len(re.findall(r'\w+', s))).median()))
A = pd.DataFrame([audit(d[d.in_dominant], 'dominant_component_84'), audit(d[~d.in_dominant], 'rest_of_PolitiFact')]); A.to_csv(T/'round4_politifact_component_titles.csv', index=False)
top = Counter(d[d.in_dominant].t).most_common(10); pd.DataFrame([dict(rank=i+1, normalized_title=t, count=c) for i, (t, c) in enumerate(top)]).to_csv(T/'round4_politifact_component_top_titles.csv', index=False)
print(); print(A.round(4).to_string(index=False)); print(); print(pd.DataFrame(top, columns=['title','count']).to_string(index=False))
