"""Round 4 Check 4: mean versus typical item, zero-inclusive, four contexts by class. Descriptive only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'outputs/extension_v1/scripts'))
import task_b_fakenewsnet_engagement as r2
T = ROOT/'outputs/extension_v1/tables'
ctx = {}
co = pd.read_csv(ROOT/'data/processed/coaid_news_primary_version_aware.csv', usecols=['coaid_within_dataset_analysis_group','unique_post_count'], keep_default_na=False); ctx['CoAID_A'] = (co.coaid_within_dataset_analysis_group.str.lower(), co.unique_post_count.astype(float))
fh = pd.read_csv(ROOT/'data/processed/fakehealth_healthstory_primary.csv', usecols=['fakehealth_within_dataset_analysis_group','unique_post_count'], keep_default_na=False); ctx['FakeHealth'] = (fh.fakehealth_within_dataset_analysis_group.str.lower(), fh.unique_post_count.astype(float))
for s, n in [('politifact','PolitiFact'), ('gossipcop','GossipCop')]:
    d = r2.load(s); ctx[n] = (d.analysis_group, d.unique_post_count.astype(float))
rows = []
for name, (lab, y) in ctx.items():
    f, r = y[lab.eq('fake')].to_numpy(), y[lab.eq('real')].to_numpy()
    u = mannwhitneyu(f, r, alternative='two-sided').statistic; p_gt = u / (len(f) * len(r))   # ties count one half
    for cls, v in [('Fake', f), ('Real', r)]:
        rows.append(dict(context=name, label=cls, N=len(v), share_zero=float((v == 0).mean()), median=float(np.median(v)), p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)), mean=float(v.mean()), prob_random_fake_exceeds_random_real_ties_half=float(p_gt)))
R = pd.DataFrame(rows); R.to_csv(T/'round4_typical_item.csv', index=False); pd.set_option('display.width', 220); print(R.round(4).to_string(index=False))
