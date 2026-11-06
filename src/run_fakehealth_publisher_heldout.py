"""FakeHealth-only nested unseen-publisher TF-IDF/LR evaluation.

Run: .venv/bin/python src/run_fakehealth_publisher_heldout.py
Reuse audited publisher/resource logic, not its feasibility runner. Never reads
CoAID data, refits previous models, or alters original cohort/split artifacts.
"""
from pathlib import Path
import hashlib
import json
import logging
import os
import platform
import tempfile
import time
import warnings

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'fakehealth_publisher_matplotlib'))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_tfidf_logistic_baseline as base
import audit_publisher_heldout_feasibility as audit

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'outputs/models/fakehealth_publisher_heldout'
TABLES = ROOT / 'outputs/tables'
REPORT = ROOT / 'outputs/reports/fakehealth_publisher_heldout.md'
OUTER_PATH = ROOT / 'outputs/splits/fakehealth_publisher_heldout_outer_folds.csv'
PRED_PATH = ROOT / 'outputs/predictions/fakehealth_publisher_heldout_predictions.csv'
FIELDS = {'natural': 'combined_text', 'source_masked': 'combined_text_source_masked'}
METRICS = ['macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']
COLUMNS = ['dataset', 'content_uid', 'primary_modeling_eligible', 'combined_text', 'combined_text_source_masked',
           'binary_analysis_target', 'binary_analysis_label', 'leakage_group']
SEED = 42
PROTOCOL = dict(seed=SEED, dataset='FakeHealth', outer_folds=3, outer_shuffle=True,
    inner_folds_preference=[3, 2], inner_shuffle=True, minimum_evaluation_class_count=20,
    grid=base.GRID, numerical_pipeline=base.PROTOCOL, tie_tolerance=base.TOLERANCE,
    predictor_fields=FIELDS, threshold=.5, masked_retuning=False,
    grouping='unchanged audited publisher/resource connected components',
    outer_test_labels='used for fixed stratification/support validation only, never hyperparameter selection')
require, sha, digest, write_json = base.require, base.sha, base.digest, base.write_json


def signature():
    audit_meta_path = audit.REPORT.with_suffix('.json')
    am = json.loads(audit_meta_path.read_text())
    bm = json.loads(base.META_PATH.read_text())
    require(am['validation'] == bm['validation'] == 'passed', 'Previous validation failed')
    require(sha(audit.__file__) == am['script_sha256'], 'Audited component code changed')
    require(sha(base.__file__) == bm['signature']['script_sha256'], 'Baseline utility code changed')
    paths = [ROOT / 'data/processed/fakehealth_text_modeling_ready.csv']
    paths += [TABLES / f'{name}.csv' for name in ['publisher_family_map', 'publisher_label_support',
        'publisher_resource_group_conflicts', 'publisher_heldout_feasibility',
        'tfidf_logistic_model_selection', 'tfidf_logistic_holdout_results']]
    expected = {**bm['signature']['inputs'], **bm['outputs'], **am['inputs'], **am['outputs']}
    for path in paths:
        require(expected.get(str(path.relative_to(ROOT))) == sha(path), 'Validated input fingerprint failed: '+path.name)
    paths += [ROOT / f'outputs/splits/fakehealth_text_{suffix}.csv' for suffix in ['holdout', 'grouped_cv']]
    for path in paths[-2:]:
        require(am['unchanged_splits'][str(path.relative_to(ROOT))] == sha(path), 'Original resource split changed')
    paths += [audit_meta_path, base.META_PATH, Path(audit.__file__), Path(base.__file__)]
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, pandas=pd.__version__)
    require(versions == bm['signature']['versions'], 'Baseline environment differs')
    return dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, versions=versions,
                script_sha256=sha(__file__), protocol=PROTOCOL)


def load_cohort():
    d = pd.read_csv(ROOT / 'data/processed/fakehealth_text_modeling_ready.csv', usecols=COLUMNS, keep_default_na=False)
    require(d.primary_modeling_eligible.dtype == bool, 'Eligibility must be boolean')
    d = d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
    require(len(d) == 1647 and d.dataset.eq('FakeHealth').all() and d.content_uid.is_unique, 'Frozen FakeHealth cohort differs')
    require(d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all(), 'Fake=1 coding mismatch')
    for field in FIELDS.values():
        require(d[field].str.strip().ne('').all(), f'Empty predictor: {field}')
    mapping = pd.read_csv(TABLES / 'publisher_family_map.csv',
        usecols=['dataset', 'content_uid', 'publisher_family', 'identity_status', 'identity_resolved'], keep_default_na=False)
    mapping = mapping[mapping.dataset.eq('FakeHealth')]
    require(mapping.content_uid.is_unique and set(mapping.content_uid) == set(d.content_uid), 'Audited map cohort mismatch')
    d = d.merge(mapping, on=['dataset', 'content_uid'], validate='one_to_one').sort_values('content_uid').reset_index(drop=True)
    # Pure audited component construction is reused; no normalization, candidate
    # design search, feasibility runner, or CoAID loader is invoked.
    d, conflicts = audit.resource_components(d)
    require(d.strict_publisher_eligible.all() and d.publisher_family.ne('').all(), 'Unidentified FakeHealth publisher')
    expected_conflicts = pd.read_csv(TABLES / 'publisher_resource_group_conflicts.csv')
    require(len(conflicts) == len(expected_conflicts[expected_conflicts.dataset.eq('FakeHealth')]) == 0, 'Audited resource conflicts differ')
    support = pd.read_csv(TABLES / 'publisher_label_support.csv', keep_default_na=False)
    support = support[support.dataset.eq('FakeHealth') & support.row_type.eq('publisher_family')].set_index('publisher_family')
    counts = d.groupby('publisher_family').binary_analysis_target.agg(['size', 'sum'])
    require(set(counts.index) == set(support.index), 'Audited family set changed')
    require(np.array_equal(counts['size'], support.loc[counts.index, 'N']) and
            np.array_equal(counts['sum'], support.loc[counts.index, 'fake_N']), 'Audited family counts changed')
    return d


def check_partition(train, test, minimum=1):
    for key in ['content_uid', 'publisher_family', 'constraint_component', 'leakage_group']:
        require(set(train[key]).isdisjoint(test[key]), f'{key} overlaps train/test')
    for name, part in [('train', train), ('test', test)]:
        require(part.content_uid.is_unique and set(part.binary_analysis_target) == {0, 1}, f'{name} lacks both classes or unique UIDs')
    require(min(int(test.binary_analysis_target.sum()), int(test.binary_analysis_target.eq(0).sum())) >= minimum,
            f'Evaluation class support below {minimum}')


def outer_assignments(d):
    result = d[['content_uid', 'publisher_family', 'constraint_component', 'leakage_group', 'binary_analysis_target']].copy()
    result['outer_fold'] = -1
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(splitter.split(np.zeros((len(d), 1)), d.binary_analysis_target, d.constraint_component)):
        check_partition(d.iloc[tr], d.iloc[te], minimum=20)
        require(result.iloc[te].outer_fold.eq(-1).all(), 'Repeated outer assignment')
        result.loc[result.index[te], 'outer_fold'] = fold
    require(result.outer_fold.isin([0, 1, 2]).all(), 'Missing outer assignment')
    for key in ['publisher_family', 'constraint_component', 'leakage_group']:
        require(result.groupby(key).outer_fold.nunique().eq(1).all(), 'Group crosses outer folds')
    previous = pd.read_csv(TABLES / 'publisher_heldout_feasibility.csv')
    previous = previous[previous.dataset.eq('FakeHealth') & previous.design.eq('A') &
        previous.algorithm.eq('StratifiedGroupKFold') & previous.folds.eq(3) & previous.partition.eq('validation')].set_index('fold')
    require(len(previous) == 3, 'Missing audited three-fold specification')
    for fold, part in result.groupby('outer_fold'):
        require(len(part) == previous.loc[fold, 'N'] and part.binary_analysis_target.sum() == previous.loc[fold, 'fake_N'],
                'Final outer counts differ from audited recommendation')
    return result


def choose_inner_splits(outer_train):
    """Accepts only outer-training rows; all tests are count/barrier diagnostics."""
    failures = []
    for count in [3, 2]:
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=SEED)
        candidates = list(splitter.split(np.zeros((len(outer_train), 1)), outer_train.binary_analysis_target, outer_train.constraint_component))
        try:
            for tr, va in candidates:
                check_partition(outer_train.iloc[tr], outer_train.iloc[va], minimum=20)
            return candidates, '; '.join(failures) or 'Three folds retain >=20 of each validation class with intact barriers.'
        except ValueError as exc:
            failures.append(f'{count}-fold unavailable: {exc}')
    raise ValueError('No defensible inner split: '+'; '.join(failures))


def texts(frame, condition):
    return frame[FIELDS[condition]].to_numpy(dtype=str, copy=True)


def fit_training(train, condition, config):
    require(train.dataset.eq('FakeHealth').all() and train.primary_modeling_eligible.all(), 'Fit outside eligible FakeHealth')
    model = base.make_pipeline(config)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(texts(train, condition), train.binary_analysis_target.to_numpy())
    iterations = int(model['logistic'].n_iter_.max())
    converged = iterations < 5000 and not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    require(np.array_equal(model['logistic'].classes_, [0, 1]), 'Class orientation failed')
    details = dict(converged=converged, n_iter=iterations, fit_seconds=time.perf_counter()-started,
        warnings='; '.join(str(w.message) for w in caught), train_N=len(train), predictor_field=FIELDS[condition],
        train_uid_sha256=digest(train.content_uid), train_publisher_sha256=digest(train.publisher_family.unique()),
        train_component_sha256=digest(train.constraint_component.unique()),
        vocabulary_size=len(model['tfidf'].vocabulary_), vocabulary_sha256=digest(model['tfidf'].vocabulary_))
    if not converged:
        logging.warning('Nonconverged fit: %s', details)
    return model, details


def predict_unlabeled(model, target, condition):
    require(not {'binary_analysis_target', 'binary_analysis_label', 'true_binary_label'} & set(target.columns), 'Labels in prediction interface')
    vec = model['tfidf']
    vocabulary, idf = dict(vec.vocabulary_), vec.idf_.copy()
    matrix = vec.transform(texts(target, condition))
    probabilities = model['logistic'].predict_proba(matrix)[:, 1]
    require(vocabulary == vec.vocabulary_ and np.array_equal(idf, vec.idf_), 'Target modified fitted TF-IDF')
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid probabilities')
    return probabilities


def inner_cv(outer_train, outer_fold, splits):
    """Only this outer-training frame can affect this fold's selected model."""
    rows = []
    for config in base.GRID:
        for fold, (tr, va) in enumerate(splits):
            train, valid = outer_train.iloc[tr], outer_train.iloc[va]
            check_partition(train, valid, minimum=20)
            path = STATE / f'inner_outer{outer_fold}_{config["config_id"]}_fold{fold}.json'
            if path.exists():
                row = json.loads(path.read_text())
                require(row['train_uid_sha256'] == digest(train.content_uid) and row['validation_uid_sha256'] == digest(valid.content_uid), 'Stale inner checkpoint')
            else:
                model, fit = fit_training(train, 'natural', config)
                probability = predict_unlabeled(model, valid.drop(columns=['binary_analysis_target', 'binary_analysis_label']), 'natural')
                row = dict(dataset='FakeHealth', condition='natural', outer_fold=outer_fold, inner_fold=fold,
                    inner_folds=len(splits), **config, **fit, **base.metrics(valid.binary_analysis_target, probability),
                    validation_uid_sha256=digest(valid.content_uid), outer_test_used=False)
                write_json(path, row)
            rows.append(row)
        logging.info('Outer %d: %s inner CV complete', outer_fold, config['config_id'])
    return rows


def summarize_inner(cv):
    rows = []
    for (outer_fold, config_id), part in cv.groupby(['outer_fold', 'config_id'], sort=True):
        require(len(part) == int(part.inner_folds.iloc[0]) and part.inner_fold.nunique() == len(part), 'Incomplete inner CV')
        config = next(c for c in base.GRID if c['config_id'] == config_id)
        row = dict(dataset='FakeHealth', condition='natural', outer_fold=int(outer_fold), **config,
                   inner_folds=len(part), all_converged=bool(part.converged.all()))
        for metric in base.METRICS:
            row[metric+'_mean'] = float(part[metric].mean())
            row[metric+'_sd'] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def lock_configs(selection):
    configs = {str(f): base.select_config(selection[selection.outer_fold.eq(f)]) for f in range(3)}
    payload = dict(configs=configs, inner_summary_sha256=hashlib.sha256(selection.to_json(orient='records').encode()).hexdigest(),
                   outer_split_sha256=sha(OUTER_PATH), protocol=PROTOCOL,
                   scope='All configurations selected using inner CV only; frozen before any outer prediction')
    path = STATE / 'selection_lock.json'
    if path.exists():
        require(json.loads(path.read_text()) == payload, 'Selection lock changed')
    else:
        write_json(path, payload)
    return configs


def outer_predictions(d, assignments, configs):
    lock_path = STATE / 'selection_lock.json'
    require(lock_path.exists(), 'Missing configuration lock')
    lock = json.loads(lock_path.read_text())
    require(lock['configs'] == configs and set(configs) == {'0', '1', '2'}, 'Incomplete or different configuration lock')
    bundles = []
    for fold in range(3):
        train, test = d[assignments.outer_fold.ne(fold)], d[assignments.outer_fold.eq(fold)]
        check_partition(train, test, minimum=20)
        for condition in FIELDS:
            path = STATE / f'outer{fold}_{condition}.joblib'
            if path.exists():
                bundle = joblib.load(path)
                require(bundle['selection_lock_sha256'] == sha(lock_path), 'Outer checkpoint selection mismatch')
                require(bundle['fit']['train_uid_sha256'] == digest(train.content_uid) and bundle['test_uid_sha256'] == digest(test.content_uid), 'Outer checkpoint identity mismatch')
            else:
                model, fit = fit_training(train, condition, configs[str(fold)])
                require(fit['converged'], 'Outer fit did not converge; stopping without grid changes')
                probability = predict_unlabeled(model, test.drop(columns=['binary_analysis_target', 'binary_analysis_label']), condition)
                predictions = test[['content_uid', 'publisher_family', 'constraint_component', 'leakage_group']].copy()
                predictions['outer_fold'], predictions['condition'] = fold, condition
                predictions['fake_probability'], predictions['predicted_label'] = probability, (probability >= .5).astype(int)
                bundle = dict(model=model, fit=fit, predictions=predictions, test_uid_sha256=digest(test.content_uid),
                              selection_lock_sha256=sha(lock_path), threshold=.5, selected_config=configs[str(fold)])
                # Freeze unlabeled predictions before reading test labels for metrics.
                base.dump(path, bundle)
                logging.info('Outer %d %s predictions frozen', fold, condition)
            bundles.append(bundle)
    return bundles


def evaluate(d, bundles):
    outputs, fit_rows = [], []
    label_index = d.set_index('content_uid').binary_analysis_target
    for bundle in bundles:
        p = bundle['predictions'].copy()
        require('true_binary_label' not in p, 'Prediction checkpoint was already labeled')
        p['true_binary_label'] = label_index.loc[p.content_uid].to_numpy()
        outputs.append(p)
        fit_rows.append(dict(outer_fold=int(p.outer_fold.iloc[0]), condition=p.condition.iloc[0], **bundle['fit']))
    predictions = pd.concat(outputs, ignore_index=True).sort_values(['condition', 'outer_fold', 'content_uid']).reset_index(drop=True)
    results = []
    for condition, part in predictions.groupby('condition'):
        scopes = [('outer_fold', int(f), p) for f, p in part.groupby('outer_fold')] + [('pooled', -1, part)]
        for scope, fold, p in scopes:
            results.append(dict(condition=condition, scope=scope, outer_fold=fold, threshold=.5,
                **base.metrics(p.true_binary_label, p.fake_probability),
                actual_fake_prevalence=float(p.true_binary_label.mean()), predicted_fake_prevalence=float(p.predicted_label.mean()),
                mean_fake_probability=float(p.fake_probability.mean()), evaluation_uid_sha256=digest(p.content_uid)))
    return predictions, pd.DataFrame(results), pd.DataFrame(fit_rows)


def summarize_results(results):
    rows = []
    for condition, part in results[results.scope.eq('outer_fold')].groupby('condition'):
        pooled = results[results.condition.eq(condition) & results.scope.eq('pooled')].iloc[0]
        for metric in METRICS:
            rows.append(dict(condition=condition, metric=metric, outer_mean=float(part[metric].mean()),
                outer_sample_sd=float(part[metric].std(ddof=1)), outer_min=float(part[metric].min()),
                outer_max=float(part[metric].max()), pooled=float(pooled[metric])))
    return pd.DataFrame(rows)


def mask_sensitivity(results, summary):
    rows = []
    for (scope, fold), part in results.groupby(['scope', 'outer_fold']):
        part = part.set_index('condition')
        require(part.loc['natural', 'evaluation_uid_sha256'] == part.loc['source_masked', 'evaluation_uid_sha256'], 'Masked cohort differs')
        for metric in METRICS:
            a, b = float(part.loc['natural', metric]), float(part.loc['source_masked', metric])
            rows.append(dict(scope=scope, outer_fold=int(fold), metric=metric, natural=a, source_masked=b, masked_minus_natural=b-a))
    for metric in METRICS:
        part = summary[summary.metric.eq(metric)].set_index('condition')
        a, b = float(part.loc['natural', 'outer_mean']), float(part.loc['source_masked', 'outer_mean'])
        rows.append(dict(scope='outer_mean', outer_fold=-1, metric=metric, natural=a, source_masked=b, masked_minus_natural=b-a))
    return pd.DataFrame(rows)


def publisher_diagnostics(predictions):
    rows = []
    for (condition, fold, family), p in predictions.groupby(['condition', 'outer_fold', 'publisher_family']):
        fake = int(p.true_binary_label.sum()); real = len(p)-fake
        rows.append(dict(condition=condition, outer_fold=int(fold), publisher_family=family, N=len(p), fake_N=fake, real_N=real,
            accuracy=float(accuracy_score(p.true_binary_label, p.predicted_label)),
            macro_f1=float(f1_score(p.true_binary_label, p.predicted_label, labels=[0,1], average='macro', zero_division=0)) if fake and real else np.nan,
            fake_recall=float(recall_score(p.true_binary_label, p.predicted_label, pos_label=1, zero_division=0)) if fake else np.nan,
            small_publisher=bool(len(p)<10), one_class_publisher=bool(not (fake and real))))
    return pd.DataFrame(rows)


def baseline_comparison(summary):
    selection = pd.read_csv(TABLES / 'tfidf_logistic_model_selection.csv')
    selected = selection[selection.dataset.eq('FakeHealth') & selection.condition.eq('natural') & selection.selected]
    holdout = pd.read_csv(TABLES / 'tfidf_logistic_holdout_results.csv')
    holdout = holdout[holdout.dataset.eq('FakeHealth') & holdout.condition.eq('natural') & holdout.model.eq('tfidf_logistic')]
    require(len(selected) == len(holdout) == 1, 'Missing baseline comparator')
    rows = []
    for metric in METRICS:
        row = summary[summary.condition.eq('natural') & summary.metric.eq(metric)].iloc[0]
        for scope, value in [('resource_grouped_cv', selected.iloc[0][metric+'_mean']), ('resource_grouped_holdout', holdout.iloc[0][metric])]:
            for aggregation in ['outer_mean', 'pooled']:
                rows.append(dict(metric=metric, baseline_scope=scope, publisher_aggregation=aggregation,
                    resource_grouped=float(value), publisher_heldout=float(row[aggregation]),
                    publisher_minus_resource=float(row[aggregation]-value), paired_comparison=False))
    return pd.DataFrame(rows)


def validate_results(d, assignments, cv, selection, predictions, results, fit_manifest, diagnostics):
    require(len(cv) == int(selection.groupby('outer_fold').inner_folds.first().sum())*len(base.GRID), 'Inner fit count mismatch')
    require(not cv.duplicated(['outer_fold','config_id','inner_fold']).any() and not cv.outer_test_used.any(), 'Invalid inner CV keys/selection scope')
    require(selection.groupby('outer_fold').selected.sum().eq(1).all(), 'Wrong selection count')
    require(len(fit_manifest) == 6 and fit_manifest.converged.all(), 'Outer fit failure')
    require(len(predictions) == 2*len(d) and not predictions.duplicated(['condition','content_uid']).any(), 'Missing/duplicate predictions')
    require(np.array_equal(predictions.predicted_label, (predictions.fake_probability >= .5).astype(int)), 'Threshold changed')
    require(results.threshold.eq(.5).all() and not results.duplicated(['condition','scope','outer_fold']).any(), 'Results keys/threshold invalid')
    indexed_assignments = assignments.set_index('content_uid')
    for condition, p in predictions.groupby('condition'):
        require(set(p.content_uid) == set(d.content_uid), 'Condition misses cohort rows')
        require(np.array_equal(indexed_assignments.loc[p.content_uid].outer_fold, p.outer_fold), 'Wrong outer prediction assignment')
        require(p.groupby('publisher_family').outer_fold.nunique().eq(1).all(), 'Publisher crossed folds')
    natural = predictions[predictions.condition.eq('natural')].set_index('content_uid')
    masked = predictions[predictions.condition.eq('source_masked')].set_index('content_uid').loc[natural.index]
    require(np.array_equal(natural.outer_fold, masked.outer_fold) and np.array_equal(natural.true_binary_label, masked.true_binary_label), 'Mask row mismatch')
    require(not diagnostics.duplicated(['condition','outer_fold','publisher_family']).any(), 'Publisher diagnostic duplicate')
    require(diagnostics.groupby('condition').N.sum().eq(len(d)).all(), 'Publisher diagnostic accounting failed')


def figures(results, summary, comparison, predictions):
    directory = ROOT / 'outputs/figures/fakehealth_publisher_heldout'
    directory.mkdir(parents=True, exist_ok=True)
    natural = summary[summary.condition.eq('natural')].set_index('metric')
    base_values = comparison[comparison.metric.eq('macro_f1') & comparison.publisher_aggregation.eq('outer_mean')].set_index('baseline_scope')
    values = [base_values.loc['resource_grouped_cv','resource_grouped'], base_values.loc['resource_grouped_holdout','resource_grouped'],
              natural.loc['macro_f1','outer_mean'], natural.loc['macro_f1','pooled']]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(4), values, color=['#7a99ab','#7a99ab','#277c8e','#277c8e'])
    ax.errorbar(2, values[2], yerr=natural.loc['macro_f1','outer_sample_sd'], color='black', capsize=5, fmt='none')
    ax.set(xticks=range(4), xticklabels=['Resource CV','Resource holdout','Publisher mean','Publisher pooled'], ylim=(0,1), ylabel='Macro F1', title='Descriptive comparison; different evaluation designs')
    fig.tight_layout(); fig.savefig(directory/'resource_vs_publisher_macro_f1.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    part = results[results.condition.eq('natural') & results.scope.eq('outer_fold')].sort_values('outer_fold')
    for metric in METRICS:
        ax.plot(part.outer_fold+1, part[metric], marker='o', label=metric.replace('_',' '))
    ax.set(xticks=[1,2,3], xlabel='Outer test fold (displayed 1–3)', ylabel='Metric', ylim=(0,1), title='Unseen-publisher performance by fold')
    ax.legend(fontsize=8, loc='best'); fig.tight_layout(); fig.savefig(directory/'outer_fold_metrics.png',dpi=180); plt.close(fig)
    p = predictions[predictions.condition.eq('natural')]
    cm = confusion_matrix(p.true_binary_label,p.predicted_label,labels=[0,1])
    fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='Actual',title='Pooled natural outer predictions')
    fig.tight_layout();fig.savefig(directory/'pooled_confusion_matrix.png',dpi=180);plt.close(fig)


def write_report(d, split_summary, selection, results, summary, sensitivity, comparison, diagnostics, cv, fits):
    md = base.md
    outer = results[results.condition.eq('natural') & results.scope.eq('outer_fold')].sort_values('outer_fold')
    pooled = results[results.condition.eq('natural') & results.scope.eq('pooled')].iloc[0]
    stats = summary[summary.condition.eq('natural')].set_index('metric')
    macro_comparison = comparison[comparison.metric.eq('macro_f1')]
    selected = selection[selection.selected]
    majority_f1 = (1-pooled.actual_fake_prevalence)/(2-pooled.actual_fake_prevalence)
    # Fixed diagnostic reference, calculated analytically; no dummy classifier fit.
    leave_one_means = [outer[outer.outer_fold.ne(f)].macro_f1.mean() for f in range(3)]
    signal_all = bool(outer.balanced_accuracy.gt(.5).all() and outer.roc_auc.gt(.5).all())
    findings = (f'All three folds retain above-0.5 balanced accuracy and ROC-AUC: {signal_all}. '
        f'Outer macro F1 ranges {outer.macro_f1.min():.4f}–{outer.macro_f1.max():.4f}; '
        f'mean macro F1 after omitting any single fold ranges {min(leave_one_means):.4f}–{max(leave_one_means):.4f}. '
        f'All outer folds contain {int(outer.N.min())}–{int(outer.N.max())} rows, so no fold dominates pooled sample size. '
        f'The pooled analytic always-Real macro-F1 reference is {majority_f1:.4f}, compared with model macro F1 {pooled.macro_f1:.4f}. '
        'These descriptive checks distinguish surviving predictive association from a result sustained by one favorable fold; they are not significance tests.')
    lines = ['# FakeHealth unseen-publisher TF-IDF + Logistic Regression', '',
        f'Natural outer macro F1: {", ".join(f"{x:.4f}" for x in outer.macro_f1)}; '
        f'mean {stats.loc["macro_f1","outer_mean"]:.4f} ± sample SD {stats.loc["macro_f1","outer_sample_sd"]:.4f}; '
        f'pooled {pooled.macro_f1:.4f}.', '', findings, '',
        '## Cohort, grouping and split protocol', '',
        f'The unchanged primary modeling-ready FakeHealth cohort contains {len(d)} rows, '
        f'{int(d.binary_analysis_target.sum())} Fake and {int(d.binary_analysis_target.eq(0).sum())} Real, '
        f'{d.publisher_family.nunique()} audited publisher/source families and {d.constraint_component.nunique()} publisher/resource components. '
        'Only combined_text is used for natural prediction; Fake=1, Real=0. The saved publisher-family map and pure audited component function are reused without renormalizing sources or rerunning the feasibility audit.', '',
        'Rows sort by content_uid. Outer StratifiedGroupKFold uses n_splits=3, shuffle=True, random_state=42. '
        'Publisher families, components and original leakage groups never cross train/test, and every outer test has >=20 Fake and >=20 Real. '
        'These authorized publisher folds are new files; the original resource splits remain unchanged. Families are observed website/name proxies, not a fully established corporate-ownership ontology.', '',
        md(split_summary), '',
        '## Nested selection and leakage controls', '',
        'Each outer-training partition gets its own publisher/component-aware inner StratifiedGroupKFold, shuffle=True, random_state=42. '
        'Three inner folds are preferred; two are permitted only if three fail class/barrier support, with the reason saved. '
        'The unchanged 24-point grid crosses unigrams/unigrams+bigrams, min_df=2/5, C=0.1/1/10, and class_weight=None/balanced. '
        'TF-IDF uses sublinear_tf=True, max_df=1.0, lowercase=True; LR uses L2/liblinear, max_iter=5000 and seed=42.', '',
        'The baseline tie policy is reused exactly: candidates within 0.001 of maximum mean inner macro F1, '
        'then within 0.001 of the lowest candidate sample SD, then unigram, smaller C, unweighted, larger min_df, config ID. '
        'No outer outcome affects selection. Labels are used to stratify and validate fixed split support, as required; '
        'after partitions are fixed only outer-training labels enter fitting/selection. '
        'All three configurations lock before any outer prediction. Natural and masked models refit only outer-training rows, '
        'and predictions are frozen without labels before metrics are joined. Threshold is always 0.5.', '',
        md(selected[['outer_fold','inner_folds','config_id','ngram_max','min_df','C','class_weight','macro_f1_mean','macro_f1_sd']]), '',
        '## Outer and pooled performance', '',
        md(results[['condition','scope','outer_fold','N','accuracy','balanced_accuracy','macro_f1','fake_precision','fake_recall','fake_f1','real_f1','roc_auc','average_precision']]), '',
        md(summary), '',
        'Means are unweighted across three outer folds; SD uses ddof=1. Min/max describe fold heterogeneity, not uncertainty intervals. '
        'Pooled metrics use each eligible row exactly once per condition. Pooled AUC/AP rank scores from different fitted models; '
        'fold-specific AUC/AP and their means should also be read because score scales can differ.', '',
        '## Prediction bias and confusion counts', '',
        md(results[['condition','scope','outer_fold','actual_fake_prevalence','predicted_fake_prevalence','mean_fake_probability','tn','fp','fn','tp']]), '',
        'TN/FP/FN/TP use Fake as the positive class. No recalibration, threshold optimization or output-based feature changes were made.', '',
        '## Comparison with resource-grouped baseline', '',
        md(macro_comparison), '',
        md(comparison[comparison.metric.ne('macro_f1')]), '',
        'All comparator metrics are reused from validated natural-text baseline tables. This is an unpaired descriptive comparison: '
        'publisher outer CV covers the entire eligible cohort, whereas the original holdout covers its frozen subset, '
        'and grouping, training samples, tuning partitions and selected hyperparameters differ. '
        'The original selected-configuration CV estimate is not a nested outer-CV estimate. '
        'A degradation is consistent with publisher-dependent signal but does not identify a causal publisher effect.', '',
        '## Source-masked sensitivity', '',
        'Every masked model uses its natural counterpart’s locked hyperparameters on identical train/test rows. No masked retuning occurs. '
        'The prepared mask dictionary is unchanged and was constructed label-blind from metadata across the prepared datasets, '
        'including evaluation metadata; this is the previously declared transductive explicit-identifier sensitivity. '
        'It can remove legitimate institutional mentions and retain publisher style, topic, URL paths and SOURCE placeholders. '
        'Masked−natural changes do not estimate the entire publisher-confounding contribution.', '',
        md(sensitivity), '',
        '## Publisher-level diagnostics and limitations', '',
        f'The diagnostic CSV contains {len(diagnostics)} publisher/condition rows. Natural diagnostics flag '
        f'{int(diagnostics[diagnostics.condition.eq("natural")].small_publisher.sum())} publishers with N<10 and '
        f'{int(diagnostics[diagnostics.condition.eq("natural")].one_class_publisher.sum())} one-class publishers. '
        'Publisher macro F1 is reported only with both classes; Fake recall is undefined with no Fake items. '
        'Do not treat small publisher estimates as stable rankings.', '',
        findings, '',
        'Surviving signal is an association under the audited unseen-source-family definition. It does not establish universal misinformation cues, '
        'unseen corporate-owner generalization, or a COVID/sensitive-period causal effect. Three outer folds provide limited evidence about '
        'variation across publishers; content selection and FakeHealth review-derived labels remain part of the dataset context.', '',
        'CoAID source/class structure is too entangled for this experiment to be interpreted as a clean unseen-publisher linguistic-generalization test. '
        'No CoAID data were opened or CoAID models fitted.', '',
        '## Validation and reproducibility', '',
        f'{len(cv)} inner fits and {len(fits)} outer fits completed. All inner fits converged: {bool(cv.converged.all())}; '
        f'all outer fits converged: {bool(fits.converged.all())}. Any inner convergence failures are logged and excluded by the unchanged selection policy. '
        'Final-fit failure would stop execution. Checkpoints avoid repeating completed fits or outer prediction calls.', '',
        'Validation covers audited map/count identity, deterministic outer counts, publisher/resource/component separation, '
        'inner membership restricted to outer training, training-only TF-IDF, allowlisted predictors, selection locks, '
        'threshold 0.5, exactly one outer prediction per eligible row per condition, natural/masked parity, unique output keys '
        'and unchanged protected input fingerprints. Report JSON contains provenance and checksums.', '',
        'Outputs: `outputs/splits/fakehealth_publisher_heldout_outer_folds.csv`; '
        '`outputs/tables/fakehealth_publisher_heldout_{inner_cv,model_selection,results,source_mask_sensitivity,summary,baseline_comparison,split_summary,fit_manifest}.csv`; '
        '`outputs/tables/fakehealth_publisher_level_diagnostics.csv`; '
        '`outputs/predictions/fakehealth_publisher_heldout_predictions.csv`; '
        '`outputs/figures/fakehealth_publisher_heldout/`. Models and locks: `outputs/models/fakehealth_publisher_heldout/`. '
        'Progress log: `logs/fakehealth_publisher_heldout.log`. Script: `src/run_fakehealth_publisher_heldout.py`.', '']
    REPORT.write_text('\n'.join(lines))


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    (ROOT/'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT/'logs/fakehealth_publisher_heldout.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', force=True)
    sig = signature()
    meta_path = REPORT.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        require(meta['signature'] == sig, 'Completed experiment signature changed; do not silently rerun')
        require(meta['validation'] == 'passed' and all(sha(ROOT/p) == h for p,h in meta['outputs'].items()), 'Saved output changed')
        print('Validated FakeHealth publisher-held-out results reused; no fits or predictions repeated.')
        return
    sig_path = STATE/'run_signature.json'
    if sig_path.exists():
        require(json.loads(sig_path.read_text()) == sig, 'Checkpoint signature mismatch')
    else:
        write_json(sig_path,sig)
    d=load_cohort()
    assignments=outer_assignments(d)
    if OUTER_PATH.exists():
        old=pd.read_csv(OUTER_PATH,keep_default_na=False)
        require(old.equals(assignments),'Existing publisher outer folds differ')
    else:
        assignments.to_csv(OUTER_PATH,index=False)
    inner_splits, split_rows = {}, []
    # Complete outer/inner support and leakage preflight before any fit.
    for outer_fold in range(3):
        train=d[assignments.outer_fold.ne(outer_fold)].reset_index(drop=True)
        test=d[assignments.outer_fold.eq(outer_fold)]
        splits,reason=choose_inner_splits(train)
        inner_splits[outer_fold]=splits
        for scope,fold,part in [('outer_train',-1,train),('outer_test',-1,test)]+[
            (scope,f,train.iloc[indices]) for f,(tr,va) in enumerate(splits) for scope,indices in [('inner_train',tr),('inner_validation',va)]]:
            split_rows.append(dict(outer_fold=outer_fold,scope=scope,inner_fold=fold,inner_folds=len(splits),
                N=len(part),fake_N=int(part.binary_analysis_target.sum()),real_N=int(part.binary_analysis_target.eq(0).sum()),
                publisher_families=part.publisher_family.nunique(),constraint_components=part.constraint_component.nunique(),
                uid_sha256=digest(part.content_uid),inner_choice_reason=reason))
    split_summary=pd.DataFrame(split_rows)
    split_summary.to_csv(TABLES/'fakehealth_publisher_heldout_split_summary.csv',index=False)
    logging.info('Preflight passed: %d rows, %d families; outer and all inner barriers/class supports valid',len(d),d.publisher_family.nunique())
    cv=pd.DataFrame([row for f in range(3) for row in inner_cv(
        d[assignments.outer_fold.ne(f)].reset_index(drop=True),f,inner_splits[f])])
    selection=summarize_inner(cv)
    configs=lock_configs(selection)
    selection['selected']=[r.config_id == configs[str(r.outer_fold)]['config_id'] for r in selection.itertuples()]
    bundles=outer_predictions(d,assignments,configs)
    predictions,results,fits=evaluate(d,bundles)
    summary=summarize_results(results)
    sensitivity=mask_sensitivity(results,summary)
    diagnostics=publisher_diagnostics(predictions)
    comparison=baseline_comparison(summary)
    validate_results(d,assignments,cv,selection,predictions,results,fits,diagnostics)
    tables=dict(inner_cv=cv,model_selection=selection,results=results,source_mask_sensitivity=sensitivity,
                summary=summary,baseline_comparison=comparison,split_summary=split_summary,fit_manifest=fits)
    paths=[OUTER_PATH]
    for name,frame in tables.items():
        path=TABLES/f'fakehealth_publisher_heldout_{name}.csv';frame.to_csv(path,index=False);paths.append(path)
    diag_path=TABLES/'fakehealth_publisher_level_diagnostics.csv';diagnostics.to_csv(diag_path,index=False);paths.append(diag_path)
    predictions.to_csv(PRED_PATH,index=False);paths.append(PRED_PATH)
    figures(results,summary,comparison,predictions)
    write_report(d,split_summary,selection,results,summary,sensitivity,comparison,diagnostics,cv,fits)
    require(signature()==sig,'Previous inputs changed during experiment')
    paths += [REPORT,*STATE.glob('*.json'),*STATE.glob('*.joblib'),*(ROOT/'outputs/figures/fakehealth_publisher_heldout').glob('*.png')]
    write_json(meta_path,dict(signature=sig,validation='passed',inner_fit_count=len(cv),outer_fit_count=len(fits),
        all_fits_converged=bool(cv.converged.all() and fits.converged.all()),threshold=.5,
        predictions_per_row_per_condition=1,source_metadata_used_as_predictors=False,
        previous_inputs_unchanged=True,outer_predictions_after_selection_lock=True,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}))
    logging.info('Complete: validation passed; %d inner fits, %d outer fits',len(cv),len(fits))
    print('FakeHealth publisher-held-out experiment complete; validation passed. Report: outputs/reports/fakehealth_publisher_heldout.md')


if __name__=='__main__':
    main()
