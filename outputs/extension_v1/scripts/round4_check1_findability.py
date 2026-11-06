"""Round 4 Check 1: CoAID A findability. Label-blind status/release/site construction, then class tabulation. Counts and shares only."""
import sys, re, json
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import audit_publisher_heldout_feasibility as pf
from audit_engagement_estimand_robustness import add_release_columns
from sklearn.feature_extraction.text import TfidfVectorizer
T = ROOT/'outputs/extension_v1/tables'
eng = pd.read_csv(ROOT/'data/processed/coaid_news_primary_version_aware.csv', keep_default_na=False, low_memory=False)
assert len(eng) == 5337
eng = add_release_columns(eng)
meta = pd.read_csv(ROOT/'data/processed/coaid_text_modeling_ready.csv', usecols=['dataset','content_uid','publisher_source','article_host','article_url'], keep_default_na=False)
fam = pf.build_family_map(meta)
fam = fam[0] if isinstance(fam, tuple) else fam
if not isinstance(fam, pd.DataFrame): fam = pd.DataFrame(fam)
status = fam.set_index('content_uid').identity_status.reindex(eng.content_uid).fillna('missing_source_metadata').to_numpy()
eng['resolution'] = pd.Series(status).map({'identified': 'resolved', 'identified_account_proxy': 'resolved', 'unresolved_platform_publisher': 'unresolved', 'missing_source_metadata': 'missing'}).fillna('unresolved').to_numpy()
eng['label'] = eng.coaid_within_dataset_analysis_group.str.title()
eng['any_post'] = eng.unique_post_count.astype(float).gt(0)
def host(u):
    u = (u or '').strip()
    if not u: return ''
    try: h = urlsplit(u if '://' in u else 'https://' + u).netloc.lower().split(':')[0]
    except Exception: return ''
    return h[4:] if h.startswith('www.') else h
eng['fact_check_site'] = eng.fact_check_url.map(host)
an = TfidfVectorizer().build_analyzer(); eng['title_tokens'] = eng.title.map(lambda s: len(an(s)))
rows = []
def tab(g, **keys):
    rows.append(dict(**keys, N=len(g), fake_N=int(g.label.eq('Fake').sum()), real_N=int(g.label.eq('Real').sum()), share_any_post=float(g.any_post.mean()), any_post_N=int(g.any_post.sum())))
for (lab, res), g in eng.groupby(['label', 'resolution']): tab(g, breakdown='class_x_resolution', label=lab, group=res)
for lab, g in eng.groupby('label'): tab(g, breakdown='class', label=lab, group='all')
for (lab, rel), g in eng.groupby(['label', 'first_release']): tab(g, breakdown='class_x_first_release', label=lab, group=rel)
fake = eng[eng.label.eq('Fake')]
site_counts = fake.fact_check_site.replace('', '(not recorded)').value_counts()
for site, n in site_counts.items():
    g = fake[fake.fact_check_site.replace('', '(not recorded)').eq(site)]
    if n >= 10 or site == '(not recorded)': tab(g, breakdown='fake_by_fact_check_site', label='Fake', group=site)
small = fake[fake.fact_check_site.replace('', '(not recorded)').map(site_counts).lt(10) & fake.fact_check_site.ne('')]
if len(small): tab(small, breakdown='fake_by_fact_check_site', label='Fake', group=f'other sites (<10 each; {small.fact_check_site.nunique()} sites)')
pd.DataFrame(rows).to_csv(T/'round4_findability.csv', index=False)
lens = []
for (lab, res), g in eng.groupby(['label', 'resolution']):
    q = g.title_tokens.quantile([.25, .5, .75, .9])
    lens.append(dict(label=lab, resolution=res, N=len(g), mean_title_tokens=float(g.title_tokens.mean()), median=float(q[.5]), p25=float(q[.25]), p75=float(q[.75]), p90=float(q[.9]), empty_title_N=int(g.title_tokens.eq(0).sum()),
                     share_any_post_titles_le5=float(g[g.title_tokens.le(5)].any_post.mean()) if g.title_tokens.le(5).any() else np.nan, share_any_post_titles_gt5=float(g[g.title_tokens.gt(5)].any_post.mean()) if g.title_tokens.gt(5).any() else np.nan))
pd.DataFrame(lens).to_csv(T/'round4_title_length.csv', index=False)
pd.set_option('display.width', 220)
R = pd.DataFrame(rows); print(R[R.breakdown.isin(['class_x_resolution','class'])].round(4).to_string(index=False)); print(); print(R[R.breakdown.eq('class_x_first_release')].round(4).to_string(index=False)); print(); print(R[R.breakdown.eq('fake_by_fact_check_site')].round(4).to_string(index=False)); print(); print(pd.DataFrame(lens).round(2).to_string(index=False))
