"""Round 4 Check 5b: locked FakeHealth LR refit on text before / after the frozen 512-token cut; top-50 term occurrence audit."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import run_tfidf_logistic_baseline as base
T = ROOT/'outputs/extension_v1/tables'
cfg = json.load(open(ROOT/'outputs/models/tfidf_logistic/selection_lock.json'))['selections']['FakeHealth']
d = base.load_cohort('FakeHealth'); cuts = pd.read_csv(T/'round4_fakehealth_cut_offsets.csv').set_index('content_uid')
full = pd.read_csv(ROOT/'data/processed/fakehealth_text_modeling_ready.csv', usecols=['content_uid','title_text','body_text','combined_text'], keep_default_na=False).set_index('content_uid').loc[d.content_uid]
off = cuts.cut_char_offset.reindex(d.content_uid).to_numpy()
before = [ (t + '\n\n' + b[:o]) if t and b[:o] else (t or b[:o]) for t, b, o in zip(full.title_text, full.body_text, off)]
after = [b[o:] for b, o in zip(full.body_text, off)]
assert all((t + '\n\n' + b) if t and b else (t or b) == c for t, b, c in zip(full.title_text, full.body_text, full.combined_text)), 'combine rule mismatch'
train, hold = d[d.holdout_assignment.eq('train')], d[d.holdout_assignment.eq('test')]
frozen = pd.read_csv(ROOT/'outputs/tables/tfidf_logistic_holdout_results.csv'); fr = frozen[frozen.dataset.eq('FakeHealth') & frozen.condition.eq('natural') & frozen.model.eq('tfidf_logistic')].iloc[0]
rows = [dict(variant='frozen_saved', macro_f1=fr.macro_f1, roc_auc=fr.roc_auc, average_precision=fr.average_precision, vocabulary_size=int(fr.vocabulary_size), empty_training_texts=0)]
for name, texts in [('natural_refit_check', full.combined_text.tolist()), ('before_512_cut', before), ('after_512_cut', after)]:
    tr = train.copy(); ho = hold.copy(); idx = {u: i for i, u in enumerate(d.content_uid)}
    tr['combined_text'] = [texts[idx[u]] for u in tr.content_uid]; ho['combined_text'] = [texts[idx[u]] for u in ho.content_uid]
    model, info = base.fit_pipeline(tr, 'natural', cfg); proba = model.predict_proba(base.texts(ho, 'natural'))[:, 1]; m = base.metrics(ho.binary_analysis_target.to_numpy(), proba)
    rows.append(dict(variant=name, macro_f1=m['macro_f1'], roc_auc=m['roc_auc'], average_precision=m['average_precision'], vocabulary_size=info['vocabulary_size'], empty_training_texts=int(tr.combined_text.eq('').sum()), converged=info['converged']))
R = pd.DataFrame(rows); R['config'] = json.dumps(cfg, sort_keys=True); R.to_csv(T/'round4_fakehealth_truncation.csv', index=False)
# term audit: top-50 |coef| terms of the frozen FakeHealth LR; occurrences before vs after the cut in the training pool
b = joblib.load(ROOT/'outputs/models/tfidf_logistic/final_fakehealth_natural.joblib'); m = b['model']; names = m['tfidf'].get_feature_names_out(); coef = m['logistic'].coef_[0]
top = np.argsort(-np.abs(coef))[:50]; analyzer = m['tfidf'].build_analyzer()
from collections import Counter
cb, ca = Counter(), Counter(); tset = set(names[top])
for u in train.content_uid:
    i = idx[u]
    for g in analyzer(before[i]):
        if g in tset: cb[g] += 1
    for g in analyzer(after[i]):
        if g in tset: ca[g] += 1
terms = pd.DataFrame([dict(rank=k+1, term=names[j], coefficient=float(coef[j]), leaning='Fake' if coef[j] > 0 else 'Real', occurrences_before_cut=cb[names[j]], occurrences_after_cut=ca[names[j]], share_after_cut=(ca[names[j]] / (cb[names[j]] + ca[names[j]])) if (cb[names[j]] + ca[names[j]]) else np.nan) for k, j in enumerate(top)])
terms.to_csv(T/'round4_fakehealth_top50_terms_after_cut.csv', index=False)
pd.set_option('display.width', 220); print(R.round(4).to_string(index=False)); print('\ntop-50 terms: mean share after cut', round(terms.share_after_cut.mean(), 4), '| occurrence-weighted', round(terms.occurrences_after_cut.sum() / (terms.occurrences_before_cut.sum() + terms.occurrences_after_cut.sum()), 4)); print(terms.head(12).round(3).to_string(index=False))
