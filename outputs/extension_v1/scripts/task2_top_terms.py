"""Task 2: top Fake/Real unigrams from the saved within-context natural LR fits. Reporting only; no fitting."""
import hashlib, json, re
from pathlib import Path
from unittest.mock import patch
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[3]
MODELS = {'CoAID': ROOT/'outputs/models/tfidf_logistic/final_coaid_natural.joblib',
          'FakeHealth': ROOT/'outputs/models/tfidf_logistic/final_fakehealth_natural.joblib'}
COHORT = {'CoAID': ROOT/'data/processed/coaid_text_modeling_ready.csv', 'FakeHealth': ROOT/'data/processed/fakehealth_text_modeling_ready.csv'}
SHARED = ROOT/'outputs/tables/text_shared_unigram_coefficients.csv'
DICT = ROOT/'outputs/tables/text_source_mask_dictionary.csv'
OUT = ROOT/'outputs/extension_v1/tables/top_fake_real_terms.csv'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
shared = pd.read_csv(SHARED, keep_default_na=False).set_index('feature')
dictionary = pd.read_csv(DICT, keep_default_na=False)
bundles = {ds: joblib.load(p) for ds, p in MODELS.items()}
def coef_table(ds):
    m = bundles[ds]['model']; v, c = m['tfidf'], m['logistic']
    assert list(c.classes_) == [0, 1]  # Fake = 1, so positive coefficients lean Fake
    names = np.array(v.get_feature_names_out()); coef = c.coef_[0]
    return pd.DataFrame(dict(feature=names, coefficient=coef)).set_index('feature')
coefs = {ds: coef_table(ds) for ds in MODELS}
# mask dictionary flags, replicating audit_cross_context_lexical_alignment.flag_features (tokenizer of the model)
tok = bundles['CoAID']['model']['tfidf'].build_tokenizer()
exact, fragments = {}, {}
for identifier in sorted(set(dictionary.identifier)):
    tokens = tok(identifier.lower()); normalized = ' '.join(tokens)
    if len(normalized) >= 3: exact.setdefault(normalized, []).append(identifier)
    for t in set(tokens):
        if len(t) >= 3: fragments.setdefault(t, []).append(identifier)
# training-pool document frequency via frozen vectorizer transform (no fit), verified against saved shared-table df
with patch.object(TfidfVectorizer, 'fit', side_effect=AssertionError), patch.object(TfidfVectorizer, 'fit_transform', side_effect=AssertionError), \
     patch.object(LogisticRegression, 'fit', side_effect=AssertionError):
    dfreq = {}
    for ds in MODELS:
        d = pd.read_csv(COHORT[ds], usecols=['content_uid','combined_text','primary_modeling_eligible','holdout_assignment'], keep_default_na=False)
        pool = d[d.primary_modeling_eligible.astype(str).eq('True') & d.holdout_assignment.eq('train')]
        assert len(pool) == bundles[ds]['result']['train_N'], (len(pool), bundles[ds]['result']['train_N'])
        X = bundles[ds]['model']['tfidf'].transform(pool.combined_text.to_numpy())
        dfreq[ds] = pd.Series(np.asarray((X > 0).sum(axis=0)).ravel(), index=coefs[ds].index)
rows = []
for ds, other in [('CoAID','FakeHealth'), ('FakeHealth','CoAID')]:
    t = coefs[ds]; uni = t[~t.index.str.contains(' ')]
    col_self, col_other = ('coaid','fakehealth') if ds == 'CoAID' else ('fakehealth','coaid')
    for lean, sel in [('Fake', uni.coefficient.sort_values(ascending=False).head(25)), ('Real', uni.coefficient.sort_values().head(25))]:
        for rank, (feat, co) in enumerate(sel.items(), 1):
            in_shared = feat in shared.index
            oc = float(coefs[other].coefficient[feat]) if feat in coefs[other].index else np.nan
            if in_shared:
                assert abs(shared.loc[feat, f'{col_self}_coefficient'] - co) < 1e-9 and abs(shared.loc[feat, f'{col_other}_coefficient'] - oc) < 1e-9
                assert int(shared.loc[feat, f'{col_self}_training_document_frequency']) == int(dfreq[ds][feat])
            rows.append(dict(dataset=ds, leaning=lean, rank=rank, term=feat, coefficient=float(co),
                training_document_frequency=int(dfreq[ds][feat]), training_N=int(bundles[ds]['result']['train_N']),
                in_shared_unigram_vocabulary=in_shared, in_other_model_vocabulary=feat in coefs[other].index,
                other_model=other, other_model_coefficient=oc,
                other_model_sign=('' if np.isnan(oc) else ('Fake' if oc > 0 else 'Real' if oc < 0 else 'zero')),
                sign_reversal=(not np.isnan(oc)) and (oc * co < 0),
                masked_by_dictionary_as_full_identifier=feat in exact,
                fragment_of_dictionary_identifier=feat in fragments,
                matching_identifiers=json.dumps(exact.get(feat, [])), fragment_identifiers_N=len(fragments.get(feat, []))))
out = pd.DataFrame(rows)
out['model_sha256'] = out.dataset.map({ds: sha(p) for ds, p in MODELS.items()})
out['shared_table_sha256'] = sha(SHARED); out['mask_dictionary_sha256'] = sha(DICT)
out.to_csv(OUT, index=False)
for ds in MODELS:
    s = out[out.dataset.eq(ds)]
    print(f"== {ds}: vocab {len(coefs[ds])} (unigrams {int((~coefs[ds].index.str.contains(' ')).sum())}); train_N {bundles[ds]['result']['train_N']}")
    for lean in ['Fake','Real']:
        q = s[s.leaning.eq(lean)]
        print(f"  {lean}-leaning top25: shared={int(q.in_shared_unigram_vocabulary.sum())} reversed={int(q.sign_reversal.sum())} full-identifier-masked={int(q.masked_by_dictionary_as_full_identifier.sum())} fragment={int(q.fragment_of_dictionary_identifier.sum())}")
        print('   ', ', '.join(f"{r.term}({r.coefficient:+.2f}{'*' if r.masked_by_dictionary_as_full_identifier else ''}{'/rev' if r.sign_reversal else ''})" for r in q.itertuples()))
