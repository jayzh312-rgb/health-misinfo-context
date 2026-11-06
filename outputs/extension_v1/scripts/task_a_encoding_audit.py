"""Round 2 Task A: mojibake prevalence and locked-protocol text-repair sensitivity for CoAID LR. Writes only under outputs/extension_v1/."""
import sys, re, json, hashlib, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT/'src'))
import run_tfidf_logistic_baseline as base
T = ROOT/'outputs/extension_v1/tables'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
PAT = {
 'cp1252_utf8': re.compile(r'â€|Ã.|Â'),
 'macroman_quote': re.compile(r'(?<=\w)[õÕ](?=\w)|\b[òÒ](?=\w)|(?<=\w)[óÓ]\b'),
 'c1_control': re.compile(r'[\u0080-\u009f]'),
 'replacement_char': re.compile('�'),
}
PAT['mojibake_any'] = re.compile('|'.join(p.pattern for p in PAT.values()))
MONTHS = ['january','february','march','april','may','june','july','august','september','october','november','december']
CUE = {'year_token': re.compile(r'\b(?:1[89]|20)\d{2}\b'),
       'month_name': re.compile(r'\b(?:' + '|'.join(MONTHS) + r')\b', re.I),
       'month_name_excl_may': re.compile(r'\b(?:' + '|'.join(m for m in MONTHS if m != 'may') + r')\b', re.I),
       'credit_fragment': re.compile(r'\b(?:getty|images|reuters|ap photo|medscape)\b', re.I)}
MAP = [('â€™',"'"),('â€˜',"'"),('â€œ','"'),('â€\x9d','"'),('â€“','-'),('â€”','-'),('â€¦','...'),('â€¢','•'),('â€','"'),
       ('Ã©','é'),('Ã¨','è'),('Ã¡','á'),('Ã ','à'),('Ã³','ó'),('Ã­','í'),('Ãº','ú'),('Ã±','ñ'),('Ã¼','ü'),('Ã¶','ö'),('Ã¤','ä'),('Ã§','ç'),('Ã‰','É'),('Â°','°'),('Â£','£'),('Â',''),]
def repair(s):
    for a, b in MAP: s = s.replace(a, b)
    s = re.sub(r'(?<=\w)[õÕ](?=\w)', "'", s); s = re.sub(r'\b[òÒ](?=\w)', '"', s); s = re.sub(r'(?<=\w)[óÓ]\b', '"', s)
    s = re.sub(r'[\u0080-\u009f�]', '', s)
    return unicodedata.normalize('NFKC', s)
def strict(s):
    s = repair(s)
    for k in ('year_token', 'month_name', 'credit_fragment'): s = CUE[k].sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip()
# ---- cohorts (frozen loaders; read-only)
co = base.load_cohort('CoAID'); fh = base.load_cohort('FakeHealth')
fam = pd.read_csv(ROOT/'outputs/tables/publisher_family_map.csv', keep_default_na=False)
fam = fam[fam.dataset.eq('CoAID')].set_index('content_uid')
co['family'] = co.content_uid.map(fam.publisher_family).fillna('')
co['family_status'] = co.content_uid.map(fam.identity_status).fillna('missing')
co.loc[~co.family_status.eq('identified') | co.family.eq(''), 'family'] = 'unresolved/missing'
from collections import Counter
for nm, fr in [('CoAID', co), ('FakeHealth', fh)]:
    c = Counter()
    for t in fr.combined_text:
        c.update(set(ch for ch in t if ord(ch) > 127))
    print(nm, 'top non-ASCII chars by doc count:', [(ch, 'U+%04X' % ord(ch), n) for ch, n in c.most_common(14)])
# ---- A1 prevalence (patterns are label-blind; labels used only to tabulate)
def flags(frame):
    f = pd.DataFrame(index=frame.index)
    for k, p in {**PAT, **CUE}.items(): f[k] = frame.combined_text.map(lambda s: bool(p.search(s)))
    return f
rows = []
def tab(frame, f, dataset, group_name, group_value):
    for cls, g in [('Fake', frame.binary_analysis_target.eq(1)), ('Real', frame.binary_analysis_target.eq(0)), ('All', pd.Series(True, index=frame.index))]:
        sub = f[g]
        if len(sub) == 0: continue
        rows.append(dict(dataset=dataset, group_type=group_name, group=group_value, label=cls, N=len(sub), **{k + '_share': float(sub[k].mean()) for k in sub.columns}))
fco, ffh = flags(co), flags(fh)
tab(co, fco, 'CoAID', 'cohort', 'all'); tab(fh, ffh, 'FakeHealth', 'cohort', 'all')
counts = co.family.value_counts(); big = set(counts[counts >= 20].index)
co['family_group'] = co.family.where(co.family.isin(big), 'other (<20 rows)')
for g, idx in co.groupby('family_group').groups.items(): tab(co.loc[idx], fco.loc[idx], 'CoAID', 'publisher_family', g)
prev = pd.DataFrame(rows); prev.to_csv(T/'coaid_encoding_prevalence.csv', index=False)
pd.set_option('display.width', 250)
show = prev[prev.group_type.eq('cohort')][['dataset','label','N','mojibake_any_share','cp1252_utf8_share','macroman_quote_share','c1_control_share','replacement_char_share','year_token_share','month_name_share','month_name_excl_may_share','credit_fragment_share']]
print(show.round(4).to_string(index=False))
famtab = prev[prev.group_type.eq('publisher_family') & prev.label.ne('All')].sort_values(['group','label'])[['group','label','N','mojibake_any_share','year_token_share','credit_fragment_share']]
print(); print(famtab.round(3).to_string(index=False))
# ---- A2 locked-protocol sensitivity
cfg = json.load(open(ROOT/'outputs/models/tfidf_logistic/selection_lock.json'))['selections']['CoAID']
frozen = pd.read_csv(ROOT/'outputs/tables/tfidf_logistic_holdout_results.csv'); frozen = frozen[frozen.dataset.eq('CoAID') & frozen.condition.eq('natural') & frozen.model.eq('tfidf_logistic')].iloc[0]
train, hold = co[co.holdout_assignment.eq('train')], co[co.holdout_assignment.eq('test')]
assert len(train) == 2875 and len(hold) == 702 and not set(train.leakage_group) & set(hold.leakage_group)
out = [dict(variant='frozen_saved', text_transform='none (saved result)', N_train=int(frozen.train_N), N_holdout=int(frozen.N), macro_f1=frozen.macro_f1, roc_auc=frozen.roc_auc, average_precision=frozen.average_precision,
            balanced_accuracy=frozen.balanced_accuracy, fake_f1=frozen.fake_f1, vocabulary_size=int(frozen.vocabulary_size), converged=bool(frozen.converged), train_docs_changed=0, holdout_docs_changed=0)]
y_tr, y_ho = train.binary_analysis_target.to_numpy(), hold.binary_analysis_target.to_numpy()
for variant, fn in [('natural_refit_check', lambda s: s), ('repaired', repair), ('repaired_strict', strict)]:
    tr = train.copy(); ho = hold.copy()
    tr['combined_text'] = tr.combined_text.map(fn); ho['combined_text'] = ho.combined_text.map(fn)
    model, info = base.fit_pipeline(tr, 'natural', cfg)
    proba = model.predict_proba(base.texts(ho, 'natural'))[:, 1]
    m = base.metrics(y_ho, proba)
    out.append(dict(variant=variant, text_transform={'natural_refit_check':'none','repaired':'mojibake map + NFKC','repaired_strict':'mojibake map + NFKC + remove years/months/credits'}[variant],
        N_train=len(tr), N_holdout=len(ho), macro_f1=m['macro_f1'], roc_auc=m['roc_auc'], average_precision=m['average_precision'], balanced_accuracy=m['balanced_accuracy'], fake_f1=m['fake_f1'],
        vocabulary_size=info['vocabulary_size'], converged=info['converged'], train_docs_changed=int((tr.combined_text != train.combined_text).sum()), holdout_docs_changed=int((ho.combined_text != hold.combined_text).sum())))
res = pd.DataFrame(out)
res['config'] = json.dumps(cfg, sort_keys=True); res['ftfy_used'] = False; res['repair_mapping'] = 'hand-written (scripts/task_a_encoding_audit.py MAP + 3 regex rules)'
res['inputs_sha256'] = json.dumps({'coaid_text_modeling_ready.csv': sha(ROOT/'data/processed/coaid_text_modeling_ready.csv'), 'selection_lock.json': sha(ROOT/'outputs/models/tfidf_logistic/selection_lock.json')})
res.to_csv(T/'coaid_encoding_sensitivity.csv', index=False)
print(); print(res[['variant','N_train','N_holdout','macro_f1','roc_auc','average_precision','balanced_accuracy','fake_f1','vocabulary_size','converged','train_docs_changed','holdout_docs_changed']].round(4).to_string(index=False))
# rule-level change counts (how many docs each repair rule touches), for the limitation note
rule = {'cp1252_map': lambda s: any(a in s for a, _ in MAP), 'macroman_o_tilde': lambda s: bool(re.search(r'(?<=\w)[õÕ](?=\w)', s)), 'macroman_o_grave_open': lambda s: bool(re.search(r'\b[òÒ](?=\w)', s)), 'macroman_o_acute_close': lambda s: bool(re.search(r'(?<=\w)[óÓ]\b', s)), 'c1_or_fffd': lambda s: bool(re.search(r'[\u0080-\u009f�]', s)), 'nfkc_changes': lambda s: unicodedata.normalize('NFKC', s) != s}
print(); print('CoAID docs touched per repair rule:', {k: int(co.combined_text.map(f).sum()) for k, f in rule.items()})
print('FakeHealth docs touched per repair rule:', {k: int(fh.combined_text.map(f).sum()) for k, f in rule.items()})
