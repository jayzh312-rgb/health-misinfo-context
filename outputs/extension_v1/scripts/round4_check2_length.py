"""Round 4 Check 2: length as a shared cue. Transform-only scoring with frozen pipelines; saved predictions for external targets; a NEW diagnostic length-only LR baseline."""
import sys, json, re, csv
from pathlib import Path
from unittest.mock import patch
import numpy as np, pandas as pd, joblib
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src')); T = ROOT/'outputs/extension_v1/tables'; MD = ROOT/'outputs/models'
an = TfidfVectorizer().build_analyzer(); ntok = lambda s: len(an(s))
def cohort(prefix):
    d = pd.read_csv(ROOT/f'data/processed/{prefix}_text_modeling_ready.csv', usecols=['content_uid','combined_text','title_text','binary_analysis_target','primary_modeling_eligible','title_only_modeling_eligible','holdout_assignment','cv_fold'], keep_default_na=False)
    return d
co, fh = cohort('coaid'), cohort('fakehealth')
elig = lambda d: d[d.primary_modeling_eligible.astype(str).eq('True')]
targets = {}
targets['within_CoAID_holdout'] = (elig(co)[elig(co).holdout_assignment.eq('test')], 'combined_text', 'CoAID')
targets['within_FakeHealth_holdout'] = (elig(fh)[elig(fh).holdout_assignment.eq('test')], 'combined_text', 'FakeHealth')
targets['transfer_CoAID_to_FakeHealth_all_eligible'] = (elig(fh), 'combined_text', 'CoAID')
targets['transfer_FakeHealth_to_CoAID_all_eligible'] = (elig(co), 'combined_text', 'FakeHealth')
models = {('within_CoAID_holdout','LR'): MD/'tfidf_logistic/final_coaid_natural.joblib', ('within_CoAID_holdout','SVM'): MD/'tfidf_linear_svm/within_coaid.joblib',
          ('within_FakeHealth_holdout','LR'): MD/'tfidf_logistic/final_fakehealth_natural.joblib', ('within_FakeHealth_holdout','SVM'): MD/'tfidf_linear_svm/within_fakehealth.joblib',
          ('transfer_CoAID_to_FakeHealth_all_eligible','LR'): MD/'tfidf_logistic_transfer/frozen_CA_to_FH_natural.joblib', ('transfer_CoAID_to_FakeHealth_all_eligible','SVM'): MD/'tfidf_linear_svm/transfer_coaid.joblib',
          ('transfer_FakeHealth_to_CoAID_all_eligible','LR'): MD/'tfidf_logistic_transfer/frozen_FH_to_CA_natural.joblib', ('transfer_FakeHealth_to_CoAID_all_eligible','SVM'): MD/'tfidf_linear_svm/transfer_fakehealth.joblib'}
scored = []   # dict(target, model, source, y, score, pred, tokens)
with patch.object(TfidfVectorizer,'fit',side_effect=AssertionError), patch.object(TfidfVectorizer,'fit_transform',side_effect=AssertionError), patch.object(LogisticRegression,'fit',side_effect=AssertionError), patch.object(LinearSVC,'fit',side_effect=AssertionError):
    for (tname, clf), path in models.items():
        frame, field, source = targets[tname]; b = joblib.load(path); m = b['model']
        X = m['tfidf'].transform(frame[field].to_numpy(str))
        if clf == 'LR': s = m['logistic'].predict_proba(X)[:, 1]; pred = (s >= .5).astype(int)
        else: s = m['svm'].decision_function(X); pred = (s >= 0).astype(int)
        scored.append(dict(target=tname, classifier=clf, source=source, y=frame.binary_analysis_target.to_numpy(int), score=s, pred=pred, tokens=frame[field].map(ntok).to_numpy()))
# COVID-Rumor clean: saved predictions + labels from NEWS table (as the frozen script's join)
csv.field_size_limit(10**9)
news = pd.read_csv(ROOT/'data/raw/COVID-19-rumor-dataset-master/Data/news/news.csv', header=None, names=['rumor_id','label','content','sentiment'], usecols=['rumor_id','label'], keep_default_na=False)
news['target_uid'] = 'covid_rumor:NEWS:' + news.rumor_id.astype(str); lab = news.set_index('target_uid').label.map({'F': 1, 'T': 0})
ct = pd.read_csv(ROOT/'data/processed/covid_rumor_news_external_target.csv', keep_default_na=False).set_index('target_uid')
cp = pd.read_csv(ROOT/'outputs/predictions/covid_rumor_external_transfer_predictions.csv'); cp = cp[cp.clean_primary.astype(str).eq('True')]
for (src, clf), g in cp.groupby(['source_dataset','classifier']):
    y = lab.reindex(g.target_uid).to_numpy(); assert not np.isnan(y).any()
    scored.append(dict(target='external_COVIDRumor_clean', classifier=clf, source=src, y=y.astype(int), score=g.score.to_numpy(), pred=g.predicted_label.to_numpy(int), tokens=ct.predictor_text.reindex(g.target_uid).map(ntok).to_numpy()))
ft = pd.read_csv(T/'fakenewsnet_external_target.csv', keep_default_na=False).set_index('target_uid'); fl = pd.read_csv(T/'fakenewsnet_external_labels.csv').set_index('target_uid').binary_target
fp = pd.read_csv(T/'fakenewsnet_external_transfer_predictions.csv'); fp = fp[fp.clean_primary.astype(str).eq('True')]
for (subset, src), g in fp.groupby(['subset','source_dataset']):
    scored.append(dict(target=f'external_FakeNewsNet_{subset}_title_clean', classifier='LR', source=src, y=fl.reindex(g.target_uid).to_numpy(int), score=g.score.to_numpy(), pred=g.predicted_label.to_numpy(int), tokens=ft.predictor_text.reindex(g.target_uid).map(ntok).to_numpy()))
bins, spear = [], []
for r in scored:
    tok = r['tokens']; q = pd.qcut(pd.Series(tok).rank(method='first'), 5, labels=[1,2,3,4,5]).to_numpy()
    for b in range(1, 6):
        m = q == b; y, s, p = r['y'][m], r['score'][m], r['pred'][m]
        bins.append(dict(target=r['target'], classifier=r['classifier'], source=r['source'], quintile=b, N=int(m.sum()), tokens_min=int(tok[m].min()), tokens_max=int(tok[m].max()), predicted_fake_rate=float(p.mean()), actual_fake_rate=float(y.mean()),
                         macro_f1=float(f1_score(y, p, average='macro', labels=[0,1], zero_division=0)) if len(set(y)) > 1 else np.nan, roc_auc=float(roc_auc_score(y, s)) if len(set(y)) > 1 else np.nan))
    rho, pv = spearmanr(np.log1p(tok), r['score'])
    spear.append(dict(target=r['target'], classifier=r['classifier'], source=r['source'], N=len(tok), spearman_log_tokens_vs_score=float(rho), frozen_roc_auc=float(roc_auc_score(r['y'], r['score']))))
pd.DataFrame(bins).to_csv(T/'round4_length_bins.csv', index=False); S = pd.DataFrame(spear); S.to_csv(T/'round4_length_spearman.csv', index=False)
# NEW diagnostic model: length-only logistic regression on log(1+tokens), fitted on each source training pool
base_rows = []
def fit_len(frame, field):
    x = np.log1p(frame[field].map(ntok).to_numpy()).reshape(-1, 1); y = frame.binary_analysis_target.to_numpy(int)
    return LogisticRegression(C=1., class_weight='balanced', random_state=42, max_iter=1000).fit(x, y)
pools = {('CoAID','combined_text'): elig(co)[elig(co).holdout_assignment.eq('train')], ('FakeHealth','combined_text'): elig(fh)[elig(fh).holdout_assignment.eq('train')],
         ('CoAID','title_text'): co[co.title_only_modeling_eligible.astype(str).eq('True') & co.holdout_assignment.eq('train')], ('FakeHealth','title_text'): fh[fh.title_only_modeling_eligible.astype(str).eq('True') & fh.holdout_assignment.eq('train')]}
baselines = {k: fit_len(v, k[1]) for k, v in pools.items()}
for r in scored:
    rep = 'title_text' if 'title' in r['target'] else 'combined_text'
    if r['classifier'] != 'LR': continue
    bl = baselines[(r['source'], rep)]; s = bl.predict_proba(np.log1p(r['tokens']).reshape(-1, 1))[:, 1]
    base_rows.append(dict(target=r['target'], source_pool=r['source'], representation=rep, model='length_only_logistic_NEW_DIAGNOSTIC', train_N=len(pools[(r['source'], rep)]), coefficient_log_tokens=float(bl.coef_[0][0]), roc_auc_length_only=float(roc_auc_score(r['y'], s)), roc_auc_frozen_LR=float(roc_auc_score(r['y'], r['score']))))
B = pd.DataFrame(base_rows); B.to_csv(T/'round4_length_baseline.csv', index=False)
pd.set_option('display.width', 250); print(S.round(4).to_string(index=False)); print(); print(B.round(4).to_string(index=False)); print(); print(pd.DataFrame(bins)[lambda d: d.classifier.eq('LR')].round(3).to_string(index=False))
