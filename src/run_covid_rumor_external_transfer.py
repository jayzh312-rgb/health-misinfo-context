"""Independent NEWS external transfer; frozen natural LR/SVM models, no fitting.

Run: .venv/bin/python src/run_covid_rumor_external_transfer.py
Existing outputs are never overwritten. Completed runs verify fingerprints.
"""
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch
import hashlib
import json
import logging
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import run_tfidf_logistic_baseline as base
import run_tfidf_logistic_cross_context_transfer as lr
import run_tfidf_linear_svm_replication as svm
from prepare_text_analysis_cohorts import normalize, quality, near_pairs, url_key, specific_url

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT/'data/raw/COVID-19-rumor-dataset-master/Data/news'
NAME = 'covid_rumor_external'
STATE = ROOT/'outputs/models'/NAME
REPORT = ROOT/'outputs/reports/covid_rumor_external_transfer.md'
META = REPORT.with_suffix('.json')
COHORT = ROOT/'data/processed/covid_rumor_news_external_target.csv'
PRED = ROOT/'outputs/predictions/covid_rumor_external_transfer_predictions.csv'
COMPARE = ['macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']
SEED = 42
require, sha = base.require, base.sha
EVENTS = []
WRITTEN = []


def write_csv(frame, path):
    path = Path(path)
    require(not path.exists(), 'Refusing overwrite: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    WRITTEN.append(path)


def write_json(value, path):
    require(not path.exists(), 'Refusing overwrite: '+str(path))
    base.write_json(path, value)
    WRITTEN.append(path)


def table(frame, suffix):
    write_csv(frame, ROOT/f'outputs/tables/{NAME}_{suffix}.csv')


def read_features():
    # Never request labels, sentiment, stance, or dates at this boundary.
    d = pd.read_csv(NEWS/'news.csv', header=None, names=['rumor_id','label','content','sentiment'], usecols=['rumor_id', 'content'], keep_default_na=False)
    require(d.rumor_id.notna().all() and d.rumor_id.is_unique, 'NEWS IDs missing/duplicated: stop')
    d['target_uid'] = 'covid_rumor:NEWS:'+d.rumor_id.astype(str)
    d['subset'] = 'NEWS'
    d['predictor_text'] = d.content.map(normalize)
    return d.drop(columns='content').sort_values('target_uid').reset_index(drop=True)


def quality_view(d):
    require(not {'label', 'binary_target'} & set(d), 'Quality input contains labels')
    d = d.copy()
    tokenizer = TfidfVectorizer().build_analyzer()
    d['analyzer_token_N'] = d.predictor_text.map(lambda s: len(tokenizer(s)))
    d['normalized_character_N'] = d.predictor_text.str.len()
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
    d['quality_exclusion_reason'] = d.predictor_text.map(reason)
    d['exact_duplicate_text'] = d.predictor_text.duplicated(keep=False)
    return d


def resource(value):
    try:
        key = url_key(value)
        return key if specific_url(key) else ''
    except ValueError:
        return ''


def source_views():
    cols = ['content_uid', 'primary_modeling_eligible', 'title_text', 'body_text',
            'combined_text', 'article_url', 'canonical_resource_key', 'source_review_url']
    return {ds: pd.read_csv(ROOT/f'data/processed/{prefix}_text_modeling_ready.csv',
                           usecols=cols, keep_default_na=False).sort_values('content_uid')
            for ds, prefix in base.PREFIXES.items()}


def overlap_audit(target, sources):
    require(not {'label', 'binary_target'} & set(target), 'Overlap input contains labels')
    evidence = []
    target_text = defaultdict(list)
    for r in target.itertuples():
        if r.predictor_text: target_text[r.predictor_text].append(r.target_uid)
    website_index = defaultdict(set)
    files_present = 0
    for r in target.itertuples():
        path = NEWS/f'rumor{int(r.rumor_id):04d}.csv'
        if not path.exists(): continue
        files_present += 1
        records = pd.read_csv(path, usecols=lambda c: c == 'website', keep_default_na=False)
        if 'website' in records:
            for value in records.website:
                key = resource(str(value))
                if key: website_index[key].add(r.target_uid)
    # URLs embedded in item text provide additional known-resource evidence.
    for r in target.itertuples():
        for value in re.findall(r'https?://[^\s<>]+', r.predictor_text):
            key = resource(value.rstrip('.,;!'))
            if key: website_index[key].add(r.target_uid)
    def add(ds, suid, tuid, kind, field, score):
        evidence.append(dict(source_dataset=ds, source_uid=suid, target_uid=tuid,
            overlap_type=kind, source_field=field, similarity=score,
            automatic_exclusion=kind.startswith('exact_')))
    for ds, source in sources.items():
        require(not {'label', 'binary_target', 'binary_analysis_target'} & set(source), 'Source audit contains labels')
        # Audit all modeling-ready source records, including ineligible records.
        for field in ['title_text', 'body_text', 'combined_text']:
            lookup = defaultdict(list)
            for r in source[['content_uid', field]].itertuples(index=False, name=None):
                text = normalize(r[1])
                if text: lookup[text].append(r[0])
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
        for field in ['article_url', 'canonical_resource_key', 'source_review_url']:
            for suid, value in source[['content_uid', field]].itertuples(index=False, name=None):
                key = resource(value)
                for tuid in sorted(website_index.get(key, [])):
                    add(ds, suid, tuid, 'exact_resource', field, 1.)
    cols = ['source_dataset','source_uid','target_uid','overlap_type','source_field','similarity','automatic_exclusion']
    evidence = pd.DataFrame(evidence, columns=cols).drop_duplicates().sort_values(cols[:5])
    table(evidence, 'overlap_evidence')
    result = target.copy()
    reasons = defaultdict(set)
    for r in evidence[evidence.automatic_exclusion].itertuples():
        reasons[r.target_uid].add(r.source_dataset+':'+r.overlap_type)
    result['overlap_exclusion_reason'] = result.target_uid.map(lambda x: ';'.join(sorted(reasons[x])))
    result['high_similarity_flag'] = result.target_uid.isin(evidence.loc[evidence.overlap_type.eq('high_similarity'), 'target_uid'])
    return result, evidence, dict(items_with_website_file=files_present,
        items_with_specific_resource=len(set().union(*website_index.values())) if website_index else 0)


def binary_gate():
    # This gate reveals eligibility only. F/T outcomes are not returned or used.
    d = pd.read_csv(NEWS/'news.csv', header=None, names=['rumor_id','label','content','sentiment'], usecols=['rumor_id', 'label'], keep_default_na=False)
    require(set(d.label) <= {'F','T','U','U(Twitter)'}, 'Unexpected label coding')
    return set(d.loc[d.label.isin(['F','T']), 'rumor_id'])


def validate_sources(sources):
    metas = {'LR': json.loads(lr.META.read_text()), 'SVM': json.loads(svm.REPORT.with_suffix('.json').read_text())}
    paths = {}
    configs = {'LR': json.loads((base.STATE/'selection_lock.json').read_text())['selections'],
               'SVM': json.loads((svm.STATE/'selection_lock.json').read_text())['configs']}
    require(configs['LR'] == lr.CONFIGS, 'Locked LR configuration changed')
    for classifier, meta in metas.items():
        require(meta['validation'] == 'passed', 'Unvalidated source models')
        require(meta['signature']['versions'] == base.get_signature()['versions'], 'Environment differs')
        for ds, prefix in base.PREFIXES.items():
            path = lr.STATE/f'frozen_{"CA_to_FH" if ds == "CoAID" else "FH_to_CA"}_natural.joblib' if classifier == 'LR' else svm.STATE/f'transfer_{prefix}.joblib'
            require(path.exists(), 'Frozen source artifact missing; locked refit would be required')
            require(meta['outputs'].get(str(path.relative_to(ROOT))) == sha(path), 'Model fingerprint mismatch')
            source_path = ROOT/f'data/processed/{prefix}_text_modeling_ready.csv'
            require(meta['signature']['inputs'].get(str(source_path.relative_to(ROOT))) == sha(source_path), 'Source cohort fingerprint mismatch')
            bundle = joblib.load(path)
            config = bundle['fit']['source_config'] if classifier == 'LR' else bundle['selected_config']
            require(config == configs[classifier][ds], 'Model configuration mismatch')
            eligible = sources[ds][sources[ds].primary_modeling_eligible]
            require(bundle['fit']['train_uid_sha256'] == base.digest(eligible.content_uid), 'Full-source membership differs')
            model = bundle['model']; v = model['tfidf']; c = model['logistic' if classifier == 'LR' else 'svm']
            require(v.ngram_range == (1,config['ngram_max']) and v.min_df == config['min_df'], 'Vectorizer changed')
            require(c.C == config['C'] and c.class_weight == config['class_weight'], 'Classifier changed')
            require(np.array_equal(c.classes_, [0,1]) and bundle['fit']['converged'], 'Invalid source fit')
            paths[(classifier,ds)] = (path,bundle,config)
    return paths


def predict_all(target, sources, models):
    require(not {'label','binary_target'} & set(target), 'Target outcomes reached prediction')
    predictions, fits, diagnostics = [], [], []
    # Runtime guards make fitting impossible, including an accidental vectorizer fit.
    with patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('Forbidden target fit')), \
         patch.object(TfidfVectorizer,'fit_transform',side_effect=AssertionError('Forbidden vocabulary fit')), \
         patch.object(LogisticRegression,'fit',side_effect=AssertionError('Forbidden LR fit')), \
         patch.object(LinearSVC,'fit',side_effect=AssertionError('Forbidden SVM fit')):
        for (classifier,ds),(path,bundle,config) in models.items():
            model = bundle['model']; v = model['tfidf']
            vocabulary, idf = dict(v.vocabulary_), v.idf_.copy()
            matrix = v.transform(target.predictor_text.to_numpy())
            scores = model['logistic'].predict_proba(matrix)[:,1] if classifier == 'LR' else model['svm'].decision_function(matrix)
            threshold = .5 if classifier == 'LR' else 0.
            require(np.isfinite(scores).all(), 'Nonfinite predictions')
            require(vocabulary == v.vocabulary_ and np.array_equal(idf,v.idf_), 'Target changed vectorizer')
            p = target[['target_uid','clean_primary']].copy()
            p['source_dataset'],p['classifier'] = ds,classifier
            p['score'],p['predicted_label'],p['threshold'] = scores,(scores>=threshold).astype(int),threshold
            predictions.append(p)
            fits.append(dict(classifier=classifier,source_dataset=ds,action='reused',source_model_path=str(path.relative_to(ROOT)),
                source_model_sha256=sha(path),source_N=int(sources[ds].primary_modeling_eligible.sum()),
                source_uid_sha256=bundle['fit']['train_uid_sha256'],locked_configuration=json.dumps(config,sort_keys=True),
                target_N=len(target),threshold=threshold,predictor_field='predictor_text',source_predictor_field='combined_text',
                target_text_fitted=False,target_labels_available=False,source_only_vocabulary=True))
            for scope, mask in [('clean_primary',target.clean_primary.to_numpy()),('all_eligible',np.ones(len(target),dtype=bool))]:
                diag = lr.vocabulary_diagnostics(model,sources[ds].loc[sources[ds].primary_modeling_eligible,'combined_text'],target.loc[mask,'predictor_text'],matrix[mask])
                diagnostics.append(dict(classifier=classifier,source_dataset=ds,scope=scope,**diag))
    p = pd.concat(predictions,ignore_index=True)
    require(not p.duplicated(['target_uid','classifier','source_dataset']).any(), 'Duplicate prediction keys')
    require(p.groupby(['classifier','source_dataset']).size().eq(len(target)).all(), 'Incomplete predictions')
    write_csv(p,PRED)  # Intentionally never append labels or raw text to this file.
    write_json(dict(predictions_sha256=sha(PRED),target_sha256=sha(COHORT),
        target_labels_joined=False,all_four_models_frozen=True),STATE/'prediction_lock.json')
    EVENTS.append('all_predictions_frozen')
    return p,pd.DataFrame(fits),pd.DataFrame(diagnostics)


def join_labels(predictions):
    require(EVENTS[-1] == 'all_predictions_frozen', 'Label join attempted before prediction freeze')
    lock = json.loads((STATE/'prediction_lock.json').read_text())
    require(sha(PRED)==lock['predictions_sha256'] and sha(COHORT)==lock['target_sha256'], 'Frozen outputs changed')
    labels = pd.read_csv(NEWS/'news.csv',header=None,names=['rumor_id','label','content','sentiment'],usecols=['rumor_id','label'],keep_default_na=False)
    labels['target_uid'] = 'covid_rumor:NEWS:'+labels.rumor_id.astype(str)
    labels['binary_target'] = labels.label.map({'F':1,'T':0})
    joined = predictions.merge(labels[['target_uid','label','binary_target']],on='target_uid',validate='many_to_one')
    require(joined.binary_target.notna().all() and joined.label.isin(['F','T']).all(),'Unverified included')
    EVENTS.append('target_labels_joined_for_evaluation')
    return joined,labels


def evaluate(joined):
    rows=[]
    for (classifier,ds),p in joined.groupby(['classifier','source_dataset']):
        for scope,mask in [('clean_primary',p.clean_primary),('all_eligible',pd.Series(True,index=p.index)),
                           ('clean_without_high_similarity',p.clean_primary & ~p.target_uid.isin(HIGH_UIDS))]:
            q=p[mask]; y=q.binary_target.astype(int)
            rows.append(dict(classifier=classifier,source_dataset=ds,scope=scope,
                **base.metrics(y,q.score,q.predicted_label),threshold=float(q.threshold.iloc[0]),
                actual_fake_prevalence=float(y.mean()),predicted_fake_prevalence=float(q.predicted_label.mean()),
                evaluation_uid_sha256=base.digest(q.target_uid)))
    results=pd.DataFrame(rows)
    comparison=[]
    for (classifier,scope),g in results.groupby(['classifier','scope']):
        s=g.set_index('source_dataset')
        require(s.evaluation_uid_sha256.nunique()==1,'Models evaluated on different target rows')
        for metric in COMPARE:
            comparison.append(dict(classifier=classifier,scope=scope,metric=metric,
                coaid_source=s.loc['CoAID',metric],fakehealth_source=s.loc['FakeHealth',metric],
                coaid_minus_fakehealth=s.loc['CoAID',metric]-s.loc['FakeHealth',metric]))
    return results,pd.DataFrame(comparison)


def protected_files():
    dirs=[base.STATE,lr.STATE,svm.STATE]
    paths=[p for d in dirs for p in d.glob('*') if p.is_file()]
    paths += [p for p in (ROOT/'outputs/tables').glob('tfidf_*.csv') if p.name.startswith(('tfidf_logistic_','tfidf_linear_svm_')) and 'title' not in p.name]
    paths += [ROOT/f'data/processed/{v}_text_modeling_ready.csv' for v in base.PREFIXES.values()]
    paths += [lr.META,svm.REPORT.with_suffix('.json'),Path(base.__file__),Path(lr.__file__),Path(svm.__file__)]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def input_hashes():
    paths=[NEWS/'news.csv',Path(__file__),ROOT/'src/prepare_text_analysis_cohorts.py',ROOT/'src/audit_coaid_entity_resolution.py']
    paths += sorted(NEWS.glob('rumor*.csv'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def report(results,comparison,diagnostics,summary,quality_stats,overlap,resources):
    primary=results[results.scope.eq('clean_primary')]
    differences=comparison[comparison.scope.eq('clean_primary')]
    support=all((differences[(differences.classifier==c)&differences.metric.isin(['macro_f1','balanced_accuracy','fake_f1','roc_auc','average_precision'])].coaid_minus_fakehealth>0).all() for c in ['LR','SVM'])
    interpretation=('Both classifiers show a consistent CoAID-source advantage across the five comparison metrics. This is consistent with shared COVID-context linguistic associations across datasets.' if support else
        'Both classifiers do not show a consistent CoAID-source advantage across all comparison metrics. These results do not support a simple COVID-context lexical-alignment explanation; inspect the metric differences below for mixed threshold and ranking behavior.')
    text=['# Independent COVID-Rumor NEWS external transfer','',
        '## Design and schema','',
        'The headerless Data/news/news.csv is read with positional columns rumor_id, label, content, sentiment. One row per rumor_id; content is the item-level rumor/news statement. Labels are F=False/Fake=1, T=True/Real=0; U and U(Twitter) are unverified and excluded. The U(Twitter) marker occurs within NEWS and does not authorize use of the Twitter subset. No Twitter files were read.',
        'The main table has no item publication date. Associated rumor####.csv files contain website URLs, repost dates and sometimes stance. Only website URLs were read for overlap screening; no repost is an independent item and no repost text is concatenated. IDs, URLs, source identities, sentiment, stance, dates and engagement are never supplied as separate predictors.', '',
        '## Cohort and quality','',base.md(summary),'',base.md(quality_stats),'',
        'Rules were fixed before predictions: whitespace/obvious-null normalization preserving case and punctuation; remove missing text, existing obvious artifact-only patterns, fewer than two alphabetic analyzer tokens after URLs are stripped for quality inspection, or <90% Latin letters. The repository designates these as English; the script check cannot certify language or exclude every mixed-language sentence. This language limitation is retained. No keyword dictionary or outcome-driven exclusions were introduced. Normalized exact duplicates within NEWS retain distinct item IDs; this is an item-weighted descriptive evaluation, not independent observations for inferential testing.', '',
        '## Source overlap','',base.md(overlap),'',
        'Exact comparison uses the previous case-preserving whitespace/null normalization and checks each statement against source title, body and combined text, including source records not eligible for training. Known-resource comparisons use the prior offline URL normalizer (tracking parameters removed, path identity retained); homepage-only matches do not count. Source article URLs, canonical resource keys and review URLs are compared with associated target website URLs and URLs in statements. Shared review/repost resources are conservatively removed without claiming identical underlying claims. Redirects and unobserved resources cannot be resolved offline.',
        f'URL availability: {resources}. Missing website files do not exclude an item.',
        'High-similarity screening reuses case-preserving title bigram Jaccard >=0.95 with >=8 words, and body/combined five-gram Jaccard >=0.90 with >=30 words. These matches are flagged, not automatically removed. Exact source overlaps are removed from the primary target; all-eligible and clean-without-high-similarity results are sensitivity analyses. Pair evidence and every item exclusion reason are saved separately.', '',
        '## Frozen models and evaluation','',
        'All four validated natural combined-text full-source models were reused, with matching artifact/cohort/configuration fingerprints. No source or target model was fitted. Runtime guards prohibit LR, SVM and vectorizer fitting. Only the existing source vocabulary/IDF transforms target statements. LR uses probability >=0.5; SVM uses decision score >=0. Target text and UIDs were fingerprinted before scoring; all four prediction sets and their fingerprint were written before outcome labels were joined. The binary eligibility gate exposes only membership before scoring. Predictions remain label-free and contain no raw text.', '',
        base.md(primary.drop(columns=['evaluation_uid_sha256'])),'',
        'Confusion matrix columns: tn=Real predicted Real, fp=Real predicted Fake, fn=Fake predicted Real, tp=Fake predicted Fake. Fake is the positive class. Average precision depends on the high Fake prevalence; it should be read with prevalence and ROC-AUC. No target calibration, rebalancing, adaptation or tuning occurred.', '',
        '## CoAID minus FakeHealth source comparisons','',base.md(differences),'',
        'Differences are descriptive, with no statistical winner designation or prespecified materiality cutoff. Effect magnitudes and agreement across metrics determine how strongly the pattern can be interpreted.', '',
        '## Representation','',base.md(diagnostics[diagnostics.scope.eq('clean_primary')][['classifier','source_dataset','vocabulary_size','target_token_occurrence_coverage_percent','zero_feature_rows','zero_feature_percent','median_target_nonzero_features']]),'',
        'Coverage counts occurrences of default analyzer unigrams found in the source vocabulary; bigrams are not double-counted. Nonzero rows can still have poor semantic or genre alignment.', '',
        '## Sensitivities','',base.md(results[~results.scope.eq('clean_primary')].drop(columns='evaluation_uid_sha256')),'',
        '## Interpretation and limitations','',interpretation,
        'H1 would be supported descriptively by a material, consistent CoAID advantage in both LR and SVM. H2 applies when that advantage is absent or inconsistent: the evidence does not support the simple COVID-context explanation. This experiment does not establish COVID-caused lexical change, a causal pandemic/sensitive-period effect, universal COVID misinformation vocabulary, or causal signaling words.',
        'FakeHealth labels derive from its expert-review framework (review-rating threshold), CoAID labels derive from its dataset construction/veracity scheme, and COVID-Rumor uses True/False/Unverified annotations associated with fact-checking sources. These are not perfectly harmonized ground-truth mechanisms. Short target statements versus longer source combined texts, dataset construction, genre, labeling and source composition remain alternative explanations. Duplicate statements, imperfect language checks and incomplete overlap metadata further limit interpretation.', '',
        '## Existing transfer comparators','']
    for classifier,path in [('LR',ROOT/'outputs/tables/tfidf_logistic_transfer_results.csv'),('SVM',ROOT/'outputs/tables/tfidf_linear_svm_transfer_results.csv')]:
        old=pd.read_csv(path);old=old[old.condition.eq('natural')&old.scope.eq('all_eligible')]
        text += [classifier,base.md(old[['source_dataset','target_dataset','N','macro_f1','roc_auc']]),'']
    text += ['These saved comparators use different target populations and label/prevalence distributions; cross-table score changes do not isolate context.','',
        '## Validation and files','',
        'Runtime checks passed: NEWS-only identity, unique IDs/output keys, binary labels, label-blind quality/overlap functions, frozen cohort and predictions before evaluation-label join, unchanged source vectorizers/configurations, zero fitting, fixed thresholds, identical clean UIDs and one prediction per model/item, and unchanged prior LR/SVM artifacts. Automated test results are recorded separately in covid_rumor_external_transfer_tests.json.',
        'Tables: outputs/tables/covid_rumor_external_*.csv. Target: data/processed/covid_rumor_news_external_target.csv. Predictions: outputs/predictions/covid_rumor_external_transfer_predictions.csv. Freeze locks: outputs/models/covid_rumor_external/. Validation: outputs/reports/covid_rumor_external_transfer.json. No figures were required.','']
    require(not REPORT.exists(),'Refusing report overwrite')
    REPORT.write_text('\n'.join(text)); WRITTEN.append(REPORT)


HIGH_UIDS=set()

def main():
    global HIGH_UIDS
    if META.exists():
        meta=json.loads(META.read_text())
        require(meta['inputs']==input_hashes(),'Inputs changed since completion')
        require(meta['protected_artifacts']==protected_files(),'Previous source artifacts changed')
        for p,h in meta['outputs'].items(): require(sha(ROOT/p)==h,'Output changed: '+p)
        print('Completed experiment fingerprints verified; no outputs modified.')
        return
    require(not COHORT.exists() and not PRED.exists() and not STATE.exists(),'Partial outputs already exist; refusing overwrite')
    (ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT/'logs/covid_rumor_external_transfer.log',level=logging.INFO)
    before,inputs=protected_files(),input_hashes()
    sources=source_views(); models=validate_sources(sources)
    logging.info('All four full-source artifacts validated; no refits required')
    raw=quality_view(read_features())
    audited,evidence,resources=overlap_audit(raw,sources)
    EVENTS.append('label_blind_quality_and_overlap_complete')
    gate=binary_gate()
    audited['binary_eligible']=audited.rumor_id.isin(gate)
    audited['clean_primary']=audited.binary_eligible & audited.quality_exclusion_reason.eq('') & audited.overlap_exclusion_reason.eq('')
    audited['exclusion_reason']=audited.apply(lambda r: ';'.join(x for x in [r.quality_exclusion_reason,r.overlap_exclusion_reason,'unverified' if not r.binary_eligible else ''] if x),axis=1)
    table(audited.drop(columns='predictor_text'),'item_audit')
    target=audited[audited.binary_eligible & audited.quality_exclusion_reason.eq('')].copy()
    require(target.target_uid.is_unique and target.subset.eq('NEWS').all(),'Target identity failure')
    write_csv(target,COHORT)
    write_json(dict(target_sha256=sha(COHORT),all_eligible_uid_sha256=base.digest(target.target_uid),
        clean_uid_sha256=base.digest(target.loc[target.clean_primary,'target_uid']),predictor='predictor_text',
        seed=SEED,outcome_labels_available=False),STATE/'target_lock.json')
    EVENTS.append('target_cohort_frozen')
    logging.info('Target frozen; eligible N=%d clean N=%d',len(target),target.clean_primary.sum())
    p,fits,diagnostics=predict_all(target,sources,models)
    joined,labels=join_labels(p)
    HIGH_UIDS=set(target.loc[target.high_similarity_flag,'target_uid'])
    results,comparison=evaluate(joined)
    lab=labels.set_index('target_uid')
    summary=[]
    def stat(name,value): summary.append(dict(measure=name,value=value))
    for name,value in [('original_NEWS_N',len(raw)),('True_N',labels.label.eq('T').sum()),('False_N',labels.label.eq('F').sum()),
        ('Unverified_N',labels.label.str.startswith('U').sum()),('unusable_or_missing_text_N',raw.quality_exclusion_reason.ne('').sum()),
        ('duplicate_ID_excess_N',raw.rumor_id.duplicated().sum()),('otherwise_eligible_binary_N',len(target)),
        ('exact_source_overlap_excluded_N',(~target.clean_primary).sum()),('final_clean_binary_N',target.clean_primary.sum()),
        ('clean_high_similarity_flagged_N',(target.clean_primary&target.high_similarity_flag).sum())]: stat(name,int(value))
    for scope,frame in [('all_NEWS',raw),('all_eligible',target),('clean_primary',target[target.clean_primary])]:
        stat(scope+'_exact_duplicate_text_excess_N',int(frame.predictor_text.duplicated().sum()))
        stat(scope+'_exact_duplicate_text_involved_N',int(frame.predictor_text.duplicated(keep=False).sum()))
    for name,value in lab.loc[target.loc[target.clean_primary,'target_uid'],'label'].value_counts().items(): stat('clean_'+('Fake' if name=='F' else 'Real')+'_N',int(value))
    stat('clean_Fake_prevalence',float(lab.loc[target.loc[target.clean_primary,'target_uid'],'binary_target'].mean()))
    stats=[]
    for scope,frame in [('all_NEWS',raw),('all_eligible',target),('clean_primary',target[target.clean_primary])]:
        for field in ['normalized_character_N','analyzer_token_N']:
            stats.append(dict(scope=scope,measure=field,N=len(frame),minimum=frame[field].min(),p05=frame[field].quantile(.05),median=frame[field].median(),p95=frame[field].quantile(.95),maximum=frame[field].max()))
    overlap=[]
    for ds in base.PREFIXES:
        for kind in ['exact_text','exact_resource','high_similarity']:
            e=evidence[evidence.source_dataset.eq(ds)&evidence.overlap_type.eq(kind)]
            overlap.append(dict(source_dataset=ds,overlap_type=kind,pair_evidence_N=len(e),
                all_NEWS_target_N=e.target_uid.nunique(),eligible_binary_target_N=e[e.target_uid.isin(target.target_uid)].target_uid.nunique(),
                clean_primary_target_N=e[e.target_uid.isin(target.loc[target.clean_primary,'target_uid'])].target_uid.nunique()))
    summary,stats,overlap=pd.DataFrame(summary),pd.DataFrame(stats),pd.DataFrame(overlap)
    for suffix,frame in [('cohort_summary',summary),('quality_summary',stats),('overlap_audit',overlap),('transfer_results',results),
                         ('source_comparison',comparison),('vocabulary_diagnostics',diagnostics),('fit_manifest',fits)]: table(frame,suffix)
    require(before==protected_files(),'Previous LR/SVM artifacts changed')
    require(inputs==input_hashes(),'Input changed during experiment')
    require(results[results.scope.eq('clean_primary')].evaluation_uid_sha256.nunique()==1,'Clean targets differ')
    report(results,comparison,diagnostics,summary,stats,overlap,resources)
    write_json(dict(validation='passed',inputs=inputs,protected_artifacts=before,events=EVENTS,source_refits=0,
        source_artifacts_unchanged=True,outputs={str(p.relative_to(ROOT)):sha(p) for p in WRITTEN}),META)
    print('External NEWS transfer complete; runtime validation passed. '+str(REPORT.relative_to(ROOT)))


if __name__=='__main__': main()
