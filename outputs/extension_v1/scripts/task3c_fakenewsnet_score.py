"""Task 3c: score FakeNewsNet title cohort with the frozen title-only LR transfer fits. No fitting, no tuning, labels joined after predictions are hashed."""
import sys, json, hashlib
from pathlib import Path
from unittest.mock import patch
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import run_tfidf_logistic_baseline as base
T = ROOT/'outputs/extension_v1/tables'; MD = ROOT/'outputs/models/tfidf_logistic_title'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = json.load(open(T/'fakenewsnet_external_cohort_lock.json'))
assert sha(T/'fakenewsnet_external_target.csv') == lock['target_sha256'], 'cohort changed since feasibility'
target = pd.read_csv(T/'fakenewsnet_external_target.csv', keep_default_na=False)
for c in ['all_eligible','clean_primary','high_similarity_flag','label_conflict_excluded']: target[c] = target[c].astype(str).eq('True')
target = target[target.all_eligible].reset_index(drop=True)   # quality-eligible, label-conflict-free rows only
assert 'label' not in target and 'binary_target' not in target
# ---- validate frozen title fits
sel = json.load(open(MD/'selection_lock.json'))
cfgs = sel.get('selections', sel)
models = {}
for ds, prefix, bundle_name in [('CoAID','coaid','transfer_coaid_natural'),('FakeHealth','fakehealth','transfer_fakehealth_natural')]:
    path = MD/f'{bundle_name}.joblib'; b = joblib.load(path); fit = b['fit']; m = b['model']; v, c = m['tfidf'], m['logistic']
    cfg = eval(fit['source_config']) if isinstance(fit['source_config'], str) else fit['source_config']
    locked = cfgs[ds] if isinstance(cfgs, dict) and ds in cfgs else cfgs.get(ds.lower()) if isinstance(cfgs, dict) else None
    src = pd.read_csv(ROOT/f'data/processed/{prefix}_text_modeling_ready.csv', usecols=['content_uid','title_only_modeling_eligible','title_text'], keep_default_na=False)
    elig = src[src.title_only_modeling_eligible.astype(str).eq('True')]
    base.require(fit['converged'] and fit['predictor_field'] == 'title_text' and fit['evaluation'] == 'transfer' and fit['condition'] == 'natural', 'unexpected fit')
    base.require(fit['train_uid_sha256'] == base.digest(elig.content_uid), 'title-eligible membership differs from frozen fit')
    base.require(int(fit['train_N']) == len(elig) and list(c.classes_) == [0,1], 'train N / classes mismatch')
    base.require(v.ngram_range == (1, cfg['ngram_max']) and v.min_df == cfg['min_df'] and c.C == cfg['C'] and (c.class_weight or 'none') == cfg['class_weight'], 'vectorizer/classifier differ from source_config')
    base.require(locked is None or {k: locked[k] for k in cfg if k in locked} == {k: cfg[k] for k in cfg if k in locked}, f'selection lock differs: {locked} vs {cfg}')
    models[ds] = dict(path=path, bundle=b, cfg=cfg, source_titles=elig.title_text, lock_checked=locked is not None)
# ---- predict (guards against any fit)
preds, fits, diags = [], [], []
with patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('Forbidden target fit')), patch.object(TfidfVectorizer,'fit_transform',side_effect=AssertionError('Forbidden')), \
     patch.object(LogisticRegression,'fit',side_effect=AssertionError('Forbidden LR fit')):
    for ds, M in models.items():
        m = M['bundle']['model']; v = m['tfidf']; vocab, idf = dict(v.vocabulary_), v.idf_.copy()
        X = v.transform(target.predictor_text.to_numpy()); s = m['logistic'].predict_proba(X)[:,1]
        base.require(np.isfinite(s).all() and vocab == v.vocabulary_ and np.array_equal(idf, v.idf_), 'vectorizer mutated')
        p = target[['target_uid','subset','clean_primary','high_similarity_flag']].copy(); p['source_dataset'] = ds; p['classifier'] = 'LR'
        p['score'] = s; p['predicted_label'] = (s >= .5).astype(int); p['threshold'] = .5; preds.append(p)
        fits.append(dict(classifier='LR', source_dataset=ds, action='reused', source_model_path=str(M['path'].relative_to(ROOT)), source_model_sha256=sha(M['path']),
            source_N=len(M['source_titles']), source_uid_sha256=M['bundle']['fit']['train_uid_sha256'], locked_configuration=json.dumps(M['cfg'], sort_keys=True),
            selection_lock_checked=M['lock_checked'], target_N=len(target), threshold=.5, predictor_field='predictor_text(normalized title)', source_predictor_field='title_text',
            target_text_fitted=False, target_labels_available=False, source_only_vocabulary=True))
        pre, tok = v.build_preprocessor(), v.build_tokenizer(); V = set(v.vocabulary_)
        for subset in ['PolitiFact','GossipCop']:
            for scope, mask in [('clean_primary', (target.subset.eq(subset) & target.clean_primary).to_numpy()), ('all_eligible', target.subset.eq(subset).to_numpy())]:
                toks = [tok(pre(t)) for t in target.predictor_text[mask]]; known = [sum(w in V for w in ts) for ts in toks]; n = [len(ts) for ts in toks]
                nnz = np.asarray(X[mask].getnnz(axis=1)).ravel()
                diags.append(dict(classifier='LR', source_dataset=ds, subset=subset, scope=scope, vocabulary_size=len(V), target_N=int(mask.sum()),
                    target_documents_with_known_token_percent=100*float(np.mean(np.array(known) > 0)), target_token_occurrence_coverage_percent=100*sum(known)/max(1,sum(n)),
                    target_median_known_token_fraction=float(np.median([k/x if x else 0 for k, x in zip(known, n)])), zero_feature_rows=int((nnz == 0).sum()), zero_feature_percent=100*float(np.mean(nnz == 0))))
P = pd.concat(preds, ignore_index=True)
base.require(not P.duplicated(['target_uid','source_dataset']).any() and P.groupby('source_dataset').size().eq(len(target)).all(), 'prediction coverage')
P.to_csv(T/'fakenewsnet_external_transfer_predictions.csv', index=False)
json.dump(dict(predictions_sha256=sha(T/'fakenewsnet_external_transfer_predictions.csv'), target_sha256=lock['target_sha256'], target_labels_joined=False), open(T/'fakenewsnet_prediction_lock.json','w'), indent=1)
pd.DataFrame(fits).to_csv(T/'fakenewsnet_external_fit_manifest.csv', index=False); pd.DataFrame(diags).to_csv(T/'fakenewsnet_external_vocabulary_diagnostics.csv', index=False)
# ---- labels joined only now
labels = pd.read_csv(T/'fakenewsnet_external_labels.csv'); assert sha(T/'fakenewsnet_external_labels.csv') == lock['labels_sha256']
J = P.merge(labels[['target_uid','binary_target']], on='target_uid', how='left'); base.require(J.binary_target.notna().all(), 'missing labels')
rows = []
for (ds, subset), g in J.groupby(['source_dataset','subset']):
    for scope, q in [('clean_primary', g[g.clean_primary]), ('all_eligible', g), ('clean_without_high_similarity', g[g.clean_primary & ~g.high_similarity_flag])]:
        y = q.binary_target.to_numpy(int)
        rows.append(dict(classifier='LR', source_dataset=ds, subset=subset, scope=scope, representation='title_only', **base.metrics(y, q.score, q.predicted_label), threshold=.5,
            actual_fake_prevalence=float(y.mean()), predicted_fake_prevalence=float(q.predicted_label.mean()), evaluation_uid_sha256=base.digest(q.target_uid)))
R = pd.DataFrame(rows)
for (subset, scope), g in R.groupby(['subset','scope']): base.require(g.evaluation_uid_sha256.nunique() == 1, 'models evaluated on different rows')
R.to_csv(T/'fakenewsnet_external_transfer_results.csv', index=False)
# source-difference table parallel to covid comparison
comp = []
for (subset, scope), g in R.groupby(['subset','scope']):
    a, b = g[g.source_dataset.eq('CoAID')].iloc[0], g[g.source_dataset.eq('FakeHealth')].iloc[0]
    for met in ['macro_f1','balanced_accuracy','fake_f1','roc_auc','average_precision']:
        comp.append(dict(subset=subset, scope=scope, metric=met, coaid_source=a[met], fakehealth_source=b[met], coaid_minus_fakehealth=a[met]-b[met]))
C = pd.DataFrame(comp); C.to_csv(T/'fakenewsnet_external_source_comparison.csv', index=False)
pd.set_option('display.width', 250)
print(R[R.scope.eq('clean_primary')][['source_dataset','subset','N','fake_N','real_N','macro_f1','balanced_accuracy','fake_f1','roc_auc','average_precision','actual_fake_prevalence','predicted_fake_prevalence']].round(4).to_string(index=False))
print(); print(C[C.scope.eq('clean_primary')].round(4).to_string(index=False))
print(); print(pd.DataFrame(diags)[lambda d: d.scope.eq('clean_primary')][['source_dataset','subset','vocabulary_size','target_documents_with_known_token_percent','target_token_occurrence_coverage_percent','zero_feature_percent']].round(2).to_string(index=False))
print('\nselection lock checked:', {ds: M['lock_checked'] for ds, M in models.items()}, '| lock keys:', list(sel.keys())[:6])
