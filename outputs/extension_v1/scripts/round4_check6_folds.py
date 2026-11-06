"""Round 4 Check 6: refit the locked LR configuration on each saved CV training fold; pairwise Spearman of coefficients over the shared unigram vocabulary."""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import run_tfidf_logistic_baseline as base
T = ROOT/'outputs/extension_v1/tables'
shared = pd.read_csv(ROOT/'outputs/tables/text_shared_unigram_coefficients.csv', usecols=['feature','fakehealth_coefficient','coaid_coefficient'], keep_default_na=False); vocab = shared.feature.tolist()
cross = spearmanr(shared.fakehealth_coefficient, shared.coaid_coefficient)[0]
locks = json.load(open(ROOT/'outputs/models/tfidf_logistic/selection_lock.json'))['selections']
rows, summ = [], []
for ds in ['FakeHealth', 'CoAID']:
    d = base.load_cohort(ds); cfg = locks[ds]; coefs = {}
    for f in range(5):
        tr = d[(d.cv_fold >= 0) & d.cv_fold.ne(f)]; model, info = base.fit_pipeline(tr, 'natural', cfg)
        names = model['tfidf'].get_feature_names_out(); c = pd.Series(model['logistic'].coef_[0], index=names); coefs[f] = c.reindex(vocab)
        rows.append(dict(dataset=ds, fold=f, train_N=len(tr), vocabulary_size=info['vocabulary_size'], shared_terms_present=int(c.reindex(vocab).notna().sum()), converged=info['converged']))
    vals = []
    for a, b in itertools.combinations(range(5), 2):
        m = coefs[a].notna() & coefs[b].notna(); rho = spearmanr(coefs[a][m], coefs[b][m])[0]
        vals.append(rho); rows.append(dict(dataset=ds, fold=f'{a}-{b}', pair=True, shared_terms_present=int(m.sum()), spearman=float(rho)))
    summ.append(dict(dataset=ds, pairs=10, spearman_mean=float(np.mean(vals)), spearman_min=float(np.min(vals)), spearman_max=float(np.max(vals)), cross_dataset_spearman_frozen=float(cross), config=json.dumps(cfg, sort_keys=True)))
pd.DataFrame(rows).to_csv(T/'round4_fold_coefficient_pairs.csv', index=False); S = pd.DataFrame(summ); S.to_csv(T/'round4_fold_coefficient_summary.csv', index=False)
pd.set_option('display.width', 220); print(S.round(4).to_string(index=False))
