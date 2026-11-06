"""Task 3a: FakeNewsNet feasibility, quality screen and source-overlap audit, mirroring run_covid_rumor_external_transfer rules.
Reads raw FakeNewsNet CSVs and the frozen *_text_modeling_ready.csv source tables. Writes only under outputs/extension_v1/."""
import sys, re, json, hashlib, unicodedata, csv
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from prepare_text_analysis_cohorts import normalize, quality, near_pairs, url_key, specific_url
from sklearn.feature_extraction.text import TfidfVectorizer
RAW = ROOT/'data/raw/fakenewsnet'; OUTT = ROOT/'outputs/extension_v1/tables'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
csv.field_size_limit(10**9)
# ---- read target; label kept in a separate frame, joined only for feasibility class counts and final evaluation
items, labels = [], []
for subset in ['politifact', 'gossipcop']:
    for lab in ['fake', 'real']:
        p = RAW/f'{subset}_{lab}.csv'
        d = pd.read_csv(p, keep_default_na=False, dtype=str, engine='python')
        assert list(d.columns) == ['id','news_url','title','tweet_ids'], d.columns
        d['target_uid'] = f'fakenewsnet:{subset}:' + d.id
        d['subset'] = 'PolitiFact' if subset == 'politifact' else 'GossipCop'
        d['tweet_id_N'] = d.tweet_ids.map(lambda s: len([t for t in s.split('\t') if t.strip()]))
        d['predictor_text'] = d.title.map(normalize)
        items.append(d[['target_uid','subset','news_url','predictor_text','tweet_id_N']])
        labels.append(pd.DataFrame(dict(target_uid=d.target_uid, label=lab, binary_target=int(lab == 'fake'))))
target = pd.concat(items, ignore_index=True); labels = pd.concat(labels, ignore_index=True)
# Eligibility gate (reveals eligibility only): ids listed in BOTH the fake and the real file of a subset carry conflicting labels.
conflict = set(labels[labels.duplicated('target_uid', keep=False)].target_uid)
target['label_conflict_excluded'] = target.target_uid.isin(conflict)
target = target.drop_duplicates('target_uid').reset_index(drop=True)
labels = labels[~labels.target_uid.isin(conflict)].reset_index(drop=True)
assert target.target_uid.is_unique and labels.target_uid.is_unique
# ---- quality screen (identical rules to covid script)
tokenizer = TfidfVectorizer().build_analyzer()
def english_script(s):
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and sum('LATIN' in unicodedata.name(c, '') for c in letters)/len(letters) >= .9
def reason(s):
    if not s: return 'missing_or_empty_text'
    if quality(s, 'body') == 'clear_crawl_artifact': return 'obvious_artifact_only'
    without_urls = re.sub(r'https?://\S+|www\.\S+', '', s)
    words = [w for w in tokenizer(without_urls) if any(c.isalpha() for c in w)]
    if len(words) < 2: return 'near_empty_or_metadata_only'
    if not english_script(s): return 'fails_english_script_check'
    return ''
target['analyzer_token_N'] = target.predictor_text.map(lambda s: len(tokenizer(s)))
target['quality_exclusion_reason'] = target.predictor_text.map(reason)
target['exact_duplicate_text'] = target.predictor_text.duplicated(keep=False)
# ---- source overlap audit (all modeling-ready source rows, eligible or not)
cols = ['content_uid','primary_modeling_eligible','title_only_modeling_eligible','title_text','body_text','combined_text','article_url','canonical_resource_key','source_review_url']
sources = {ds: pd.read_csv(ROOT/f'data/processed/{p}_text_modeling_ready.csv', usecols=cols, keep_default_na=False).sort_values('content_uid')
           for ds, p in [('FakeHealth','fakehealth'),('CoAID','coaid')]}
def resource(v):
    try: k = url_key(v); return k if specific_url(k) else ''
    except Exception: return ''
target_text = defaultdict(list)
for r in target.itertuples():
    if r.predictor_text: target_text[r.predictor_text].append(r.target_uid)
website_index = defaultdict(set)
for r in target.itertuples():
    k = resource(r.news_url)
    if k: website_index[k].add(r.target_uid)
    for v in re.findall(r'https?://[^\s<>]+', r.predictor_text):
        k = resource(v.rstrip('.,;!'))
        if k: website_index[k].add(r.target_uid)
evidence = []
def add(ds, suid, tuid, kind, field, score):
    evidence.append(dict(source_dataset=ds, source_uid=suid, target_uid=tuid, overlap_type=kind, source_field=field, similarity=score, automatic_exclusion=kind.startswith('exact_')))
for ds, source in sources.items():
    for field in ['title_text','body_text','combined_text']:
        lookup = defaultdict(list)
        for suid, text in source[['content_uid', field]].itertuples(index=False, name=None):
            text = normalize(text)
            if text: lookup[text].append(suid)
        for text in set(lookup) & set(target_text):
            for suid in lookup[text]:
                for tuid in target_text[text]: add(ds, suid, tuid, 'exact_text', field, 1.)
        width, threshold, minimum = (2, .95, 8) if field == 'title_text' else (5, .90, 30)
        matches, _ = near_pairs(set(lookup) | set(target_text), width, threshold, minimum)
        for left, right, score in matches:
            for t, s in [(left, right), (right, left)]:
                if t in target_text and s in lookup:
                    for suid in lookup[s]:
                        for tuid in target_text[t]: add(ds, suid, tuid, 'high_similarity', field, score)
    for field in ['article_url','canonical_resource_key','source_review_url']:
        for suid, value in source[['content_uid', field]].itertuples(index=False, name=None):
            k = resource(value)
            for tuid in sorted(website_index.get(k, [])): add(ds, suid, tuid, 'exact_resource', field, 1.)
ecols = ['source_dataset','source_uid','target_uid','overlap_type','source_field','similarity','automatic_exclusion']
evidence = pd.DataFrame(evidence, columns=ecols).drop_duplicates().sort_values(ecols[:5])
evidence.to_csv(OUTT/'fakenewsnet_external_overlap_evidence.csv', index=False)
reasons = defaultdict(set)
for r in evidence[evidence.automatic_exclusion].itertuples(): reasons[r.target_uid].add(r.source_dataset + ':' + r.overlap_type)
target['overlap_exclusion_reason'] = target.target_uid.map(lambda x: ';'.join(sorted(reasons[x])))
target['high_similarity_flag'] = target.target_uid.isin(evidence.loc[evidence.overlap_type.eq('high_similarity'), 'target_uid'])
target['all_eligible'] = target.quality_exclusion_reason.eq('') & ~target.label_conflict_excluded
target['clean_primary'] = target.all_eligible & target.overlap_exclusion_reason.eq('')
target.to_csv(OUTT/'fakenewsnet_external_target.csv', index=False)   # no labels in this file
labels.to_csv(OUTT/'fakenewsnet_external_labels.csv', index=False)
# ---- feasibility summary per subset x class (labels joined here only for counting)
conflict_rows = target[target.label_conflict_excluded][['target_uid','subset']]
j = target.merge(labels, on='target_uid')
rows = []
for (subset, lab), g in j.groupby(['subset','label']):
    rows.append(dict(subset=subset, label=lab, rows=len(g), title_present=int(g.predictor_text.ne('').sum()), body_text_present_locally=0,
        news_url_present=int(g.news_url.ne('').sum()), tweet_ids_present=int(g.tweet_id_N.gt(0).sum()), zero_tweet_rows=int(g.tweet_id_N.eq(0).sum()),
        zero_tweet_share=float(g.tweet_id_N.eq(0).mean()), median_tweet_ids=float(g.tweet_id_N.median()),
        quality_excluded=int(~g.all_eligible).sum() if False else int((~g.all_eligible).sum()),
        **{f'quality_{k}': int(v) for k, v in g.quality_exclusion_reason.value_counts().items() if k},
        exact_duplicate_title_rows=int(g.exact_duplicate_text.sum()), median_analyzer_tokens=float(g.analyzer_token_N.median()),
        all_eligible=int(g.all_eligible.sum()), exact_source_overlap_excluded=int((g.all_eligible & ~g.clean_primary).sum()),
        clean_primary=int(g.clean_primary.sum()), high_similarity_flagged_clean=int((g.clean_primary & g.high_similarity_flag).sum())))
summary = pd.DataFrame(rows).fillna(0)
for subset in ['PolitiFact','GossipCop']: summary.loc[summary.subset.eq(subset),'label_conflict_excluded_ids_in_subset'] = int(conflict_rows.subset.eq(subset).sum())
summary.to_csv(OUTT/'fakenewsnet_external_cohort_summary.csv', index=False)
ov = []
for ds in sources:
    for kind in ['exact_text','exact_resource','high_similarity']:
        e = evidence[evidence.source_dataset.eq(ds) & evidence.overlap_type.eq(kind)]
        for subset in ['PolitiFact','GossipCop']:
            tu = set(e.target_uid) & set(target.loc[target.subset.eq(subset),'target_uid'])
            ov.append(dict(source_dataset=ds, overlap_type=kind, subset=subset, pair_evidence_N=int(len(e[e.target_uid.isin(tu)])), target_items_N=len(tu),
                           fields=';'.join(sorted(e[e.target_uid.isin(tu)].source_field.unique()))))
pd.DataFrame(ov).to_csv(OUTT/'fakenewsnet_external_overlap_audit.csv', index=False)
json.dump(dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in sorted(RAW.glob('*.csv'))} | {f'data/processed/{p}_text_modeling_ready.csv': sha(ROOT/f'data/processed/{p}_text_modeling_ready.csv') for p in ['coaid','fakehealth']},
               target_sha256=sha(OUTT/'fakenewsnet_external_target.csv'), labels_sha256=sha(OUTT/'fakenewsnet_external_labels.csv'),
               all_eligible_N=int(target.all_eligible.sum()), clean_primary_N=int(target.clean_primary.sum())), open(OUTT/'fakenewsnet_external_cohort_lock.json','w'), indent=1)
pd.set_option('display.width', 250); print(summary.T.to_string()); print(); print(pd.DataFrame(ov).to_string(index=False))
print('\nexample dup titles:', j[j.exact_duplicate_text].predictor_text.value_counts().head(3).to_dict())
print('resource matches (first 10):'); print(evidence[evidence.overlap_type.eq('exact_resource')].head(10).to_string(index=False))
