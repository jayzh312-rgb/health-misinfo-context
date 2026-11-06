"""Frozen-split title-only sensitivity. No combined-text fitting or modification.

Run: .venv/bin/python src/run_tfidf_logistic_title_sensitivity.py
Both natural configurations lock before holdout/transfer. Transfer targets expose
no labels until every prediction bundle is frozen. Checkpoints avoid repeat fits.
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

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'title_sensitivity_matplotlib'))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import run_tfidf_logistic_baseline as base
import run_tfidf_logistic_cross_context_transfer as transfer

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'outputs/models/tfidf_logistic_title'
TABLES = ROOT / 'outputs/tables'
REPORT = ROOT / 'outputs/reports/tfidf_logistic_title_sensitivity.md'
FIELDS = {'natural': 'title_text', 'source_masked': 'title_text_source_masked'}
DIRECTIONS = dict(transfer.DIRECTIONS)
COMPARE = ['macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']
FEATURE_COLUMNS = ['dataset', 'content_uid', 'title_only_modeling_eligible', 'primary_modeling_eligible',
                   'title_text', 'title_text_source_masked', 'title_text_source_cue_flag',
                   'leakage_group', 'cv_fold', 'holdout_assignment',
                   'cross_dataset_overlap_flag', 'cross_dataset_overlap_group_flag']
PROTOCOL = dict(seed=42, fields=FIELDS, eligibility='title_only_modeling_eligible',
                grid=base.GRID, selection_tolerance=base.TOLERANCE, numerical_pipeline=base.PROTOCOL,
                threshold=.5, masked_retuning=False, target_labels='join after all predictions frozen',
                transfer_training='all title-eligible source rows', selection='natural source grouped CV only')
require, sha, digest, write_json = base.require, base.sha, base.digest, base.write_json


def input_signature():
    baseline_meta = json.loads(base.META_PATH.read_text())
    transfer_meta = json.loads(transfer.META.read_text())
    audit_path = ROOT / 'outputs/reports/text_quality_and_source_cue_audit_metadata.json'
    audit = json.loads(audit_path.read_text())
    require(all(m['validation'] == 'passed' for m in [baseline_meta, transfer_meta, audit]), 'Prior validation failed')
    require('title_text_source_masked' in audit['feature_allowlist'], 'Masked title field lacks prior audit')
    expected = {**baseline_meta['signature']['inputs'], **baseline_meta['outputs'], **transfer_meta['outputs']}
    paths = [ROOT / f'data/processed/{p}_text_modeling_ready.csv' for p in base.PREFIXES.values()]
    paths += [ROOT / f'outputs/splits/{p}_text_{s}.csv' for p in base.PREFIXES.values() for s in ['grouped_cv', 'holdout']]
    paths += [TABLES / f'tfidf_logistic_{n}.csv' for n in ['holdout_results', 'transfer_results', 'transfer_degradation']]
    paths += [ROOT / f'outputs/predictions/{n}.csv' for n in ['tfidf_logistic_holdout_predictions', 'tfidf_logistic_cross_context_predictions']]
    for path in paths:
        rel = str(path.relative_to(ROOT))
        require(expected.get(rel) == sha(path), f'Validated input fingerprint failed: {rel}')
    dictionary = TABLES / 'text_source_mask_dictionary.csv'
    require(sha(dictionary) == audit['dictionary_sha256'], 'Prior masking dictionary fingerprint failed')
    paths += [dictionary, audit_path, base.META_PATH, transfer.META, Path(base.__file__), Path(transfer.__file__)]
    require(sha(base.__file__) == baseline_meta['signature']['script_sha256'], 'Baseline utility code changed')
    require(sha(transfer.__file__) == transfer_meta['signature']['script_sha256'], 'Transfer utility code changed')
    versions = dict(python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, pandas=pd.__version__)
    require(versions == baseline_meta['signature']['versions'], 'Environment differs from baseline')
    return dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}, versions=versions,
                script_sha256=sha(__file__), protocol=PROTOCOL)


def load_features(dataset):
    path = ROOT / f'data/processed/{base.PREFIXES[dataset]}_text_modeling_ready.csv'
    d = pd.read_csv(path, usecols=FEATURE_COLUMNS, keep_default_na=False)
    require(d.title_only_modeling_eligible.dtype == bool, 'Eligibility not boolean')
    d = d[d.title_only_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
    require(d.content_uid.is_unique and d.dataset.eq(dataset).all(), 'Cohort identity failed')
    require(d.primary_modeling_eligible.all(), 'Title rows outside audited primary cohort')
    require(not d.cross_dataset_overlap_flag.any() and not d.cross_dataset_overlap_group_flag.any(), 'Cross-dataset overlap')
    require(d.cv_fold.isin([-1, 0, 1, 2, 3, 4]).all(), 'Invalid frozen fold')
    require((d.cv_fold.eq(-1) == d.holdout_assignment.eq('test')).all(), 'Holdout mismatch')
    require(d.groupby('leakage_group').cv_fold.nunique().eq(1).all(), 'Leakage group crosses folds')
    tokenizer = base.make_pipeline(base.GRID[0])['tfidf'].build_analyzer()
    for field in FIELDS.values():
        require(d[field].map(lambda x: bool(x.strip()) and bool(tokenizer(x))).all(), f'Unusable {field}')
    for suffix, old_field, field in [('grouped_cv', 'fold', 'cv_fold'), ('holdout', 'assignment', 'holdout_assignment')]:
        old = pd.read_csv(ROOT / f'outputs/splits/{base.PREFIXES[dataset]}_text_{suffix}.csv',
                          usecols=['content_uid', 'leakage_group', old_field]).set_index('content_uid')
        require(old.index.is_unique and set(d.content_uid) <= set(old.index), 'Frozen split IDs differ')
        aligned = old.loc[d.content_uid]
        require(aligned[old_field].tolist() == d[field].tolist(), 'Split reassignment')
        require(aligned.leakage_group.tolist() == d.leakage_group.tolist(), 'Group reassignment')
    require(not set(transfer.TARGET_COLUMNS) & set(d.columns), 'Labels in feature view')
    return d


def labels(dataset, uids):
    # Shared read-only loader validates Fake=1; no body/title columns are read.
    return transfer.read_labels(dataset, uids)[['binary_analysis_target', 'binary_analysis_label']]


def source_view(dataset, features):
    require(features.dataset.eq(dataset).all(), 'Source dataset mixing')
    d = features.copy()
    d['binary_analysis_target'] = labels(dataset, features.content_uid).binary_analysis_target.to_numpy()
    return d


def texts(frame, condition):
    """The ONLY predictor boundary. Never concatenates metadata or body text."""
    return frame[FIELDS[condition]].to_numpy(dtype=str, copy=True)


def fit_source(source, condition, config):
    require(source.dataset.nunique() == 1, 'Mixed-source fit')
    require(source.title_only_modeling_eligible.all(), 'Ineligible training rows')
    require(set(source.binary_analysis_target) == {0, 1}, 'Missing training class')
    model = base.make_pipeline(config)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(texts(source, condition), source.binary_analysis_target.to_numpy())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    converged = converged and int(model['logistic'].n_iter_.max()) < model['logistic'].max_iter
    require(np.array_equal(model['logistic'].classes_, [0, 1]), 'Fake probability orientation error')
    fit = dict(converged=converged, n_iter=int(model['logistic'].n_iter_.max()), fit_seconds=time.perf_counter()-started,
        warnings='; '.join(str(w.message) for w in caught), predictor_field=FIELDS[condition],
        source_dataset=source.dataset.iloc[0], train_N=len(source), train_uid_sha256=digest(source.content_uid),
        train_group_sha256=digest(source.leakage_group.unique()), vocabulary_size=len(model['tfidf'].vocabulary_),
        vocabulary_sha256=digest(model['tfidf'].vocabulary_))
    return model, fit


def predict_features(model, features, condition):
    require(not set(transfer.TARGET_COLUMNS) & set(features.columns), 'Target labels entered prediction interface')
    v = model['tfidf']
    vocabulary, idf = dict(v.vocabulary_), v.idf_.copy()
    matrix = v.transform(texts(features, condition))
    probability = model['logistic'].predict_proba(matrix)[:, 1]
    require(vocabulary == v.vocabulary_ and np.array_equal(idf, v.idf_), 'Target changed fitted TF-IDF')
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Invalid probability')
    p = features[['content_uid', 'leakage_group', 'cv_fold', 'holdout_assignment']].copy()
    p['fake_probability'], p['predicted_label'] = probability, (probability >= .5).astype(int)
    return p, matrix


def run_cv(dataset, features):
    train = source_view(dataset, features[features.cv_fold.ge(0)])
    rows = []
    for config in base.GRID:
        for fold in range(5):
            tr, va = train[train.cv_fold.ne(fold)], train[train.cv_fold.eq(fold)]
            require(set(va.binary_analysis_target) == {0, 1}, 'Missing validation class')
            require(not set(tr.leakage_group) & set(va.leakage_group), 'CV leakage')
            path = STATE / f'cv_{base.PREFIXES[dataset]}_{config["config_id"]}_{fold}.json'
            if path.exists():
                row = json.loads(path.read_text())
                require(row['train_uid_sha256'] == digest(tr.content_uid) and row['validation_uid_sha256'] == digest(va.content_uid), 'Stale CV cache')
            else:
                model, fit = fit_source(tr, 'natural', config)
                prediction, _ = predict_features(model, va.drop(columns='binary_analysis_target'), 'natural')
                row = dict(dataset=dataset, condition='natural', **config, fold=fold, **fit,
                    validation_N=len(va), validation_uid_sha256=digest(va.content_uid), holdout_used=False,
                    **base.metrics(va.binary_analysis_target, prediction.fake_probability))
                write_json(path, row)
            rows.append(row)
        logging.info('%s %s title CV complete', dataset, config['config_id'])
    return rows


def lock_selections(selection):
    selections = {ds: base.select_config(selection[selection.dataset.eq(ds)]) for ds in base.PREFIXES}
    payload = dict(selections=selections, protocol=PROTOCOL,
        cv_summary_sha256=hashlib.sha256(selection.to_json(orient='records').encode()).hexdigest(),
        scope='natural source training-pool CV only; both configurations locked before holdout/transfer')
    path = STATE / 'selection_lock.json'
    if path.exists():
        require(json.loads(path.read_text()) == payload, 'Selection lock changed')
    else:
        write_json(path, payload)
    return selections


def verify_lock(source_dataset, config):
    path = STATE / 'selection_lock.json'
    require(path.exists(), 'Both title configurations must be locked first')
    lock = json.loads(path.read_text())
    require(set(lock['selections']) == set(base.PREFIXES), 'Incomplete configuration lock')
    require(lock['selections'][source_dataset] == config, 'Unfrozen source configuration')
    return sha(path)


def freeze_predictions(views, configs):
    bundles = {}
    for evaluation in ['within', 'transfer']:
        for direction, (source_ds, target_ds) in DIRECTIONS.items():
            if evaluation == 'within':
                target_ds = source_ds
            source_features = views[source_ds]
            target_features = views[target_ds]
            if evaluation == 'within':
                source_features = source_features[source_features.cv_fold.ge(0)]
                target_features = target_features[target_features.cv_fold.eq(-1)]
            require(not set(source_features.content_uid) & set(target_features.content_uid), 'Fit/evaluation UID overlap')
            require(not set(source_features.leakage_group) & set(target_features.leakage_group), 'Fit/evaluation group overlap')
            for condition in FIELDS:
                config = configs[source_ds]
                lock_hash = verify_lock(source_ds, config)
                key = f'{evaluation}_{base.PREFIXES[source_ds]}_{condition}'
                path = STATE / f'{key}.joblib'
                if path.exists():
                    bundle = joblib.load(path)
                    require(bundle['fit']['selection_lock_sha256'] == lock_hash, 'Stale prediction cache')
                    require(bundle['fit']['train_uid_sha256'] == digest(source_features.content_uid), 'Stale source cache')
                    require(bundle['fit']['target_uid_sha256'] == digest(target_features.content_uid), 'Stale target cache')
                else:
                    source = source_view(source_ds, source_features)
                    model, fit = fit_source(source, condition, config)
                    require(fit['converged'], 'Final fit did not converge; stop without changing grid')
                    predictions, matrix = predict_features(model, target_features, condition)
                    predictions['evaluation'], predictions['condition'] = evaluation, condition
                    predictions['source_dataset'], predictions['target_dataset'] = source_ds, target_ds
                    predictions['direction'] = direction if evaluation == 'transfer' else 'within_' + base.PREFIXES[source_ds]
                    fit.update(evaluation=evaluation, condition=condition, source_config=config, target_dataset=target_ds,
                        target_uid_sha256=digest(target_features.content_uid), target_N=len(target_features),
                        selection_lock_sha256=lock_hash, threshold=.5, target_text_fitted=False,
                        target_labels_available=False)
                    diagnostic = None
                    if evaluation == 'transfer':
                        diagnostic = dict(direction=direction, condition=condition, source_dataset=source_ds, target_dataset=target_ds,
                            **transfer.vocabulary_diagnostics(model, texts(source, condition), texts(target_features, condition), matrix))
                    bundle = dict(model=model, predictions=predictions, fit=fit, diagnostic=diagnostic)
                    base.dump(path, bundle)
                    logging.info('%s prediction bundle frozen', key)
                bundles[key] = bundle
    hashes = {str((STATE / f'{key}.joblib').relative_to(ROOT)): sha(STATE / f'{key}.joblib') for key in bundles}
    lock = dict(selection_lock_sha256=sha(STATE / 'selection_lock.json'), prediction_bundles=hashes,
                target_evaluation_labels_joined=False, threshold=.5)
    path = STATE / 'prediction_lock.json'
    if path.exists():
        require(json.loads(path.read_text()) == lock, 'Prediction lock changed')
    else:
        write_json(path, lock)
    return bundles


def join_evaluation_labels(bundles):
    lock_path = STATE / 'prediction_lock.json'
    require(lock_path.exists(), 'Cannot evaluate before predictions freeze')
    lock = json.loads(lock_path.read_text())
    require(len(lock['prediction_bundles']) == 8, 'Incomplete prediction freeze')
    require(all(sha(ROOT / p) == h for p, h in lock['prediction_bundles'].items()), 'Prediction fingerprint failed')
    rows = []
    for bundle in bundles.values():
        p = bundle['predictions'].copy()
        require('true_binary_label' not in p and 'binary_analysis_target' not in p, 'Frozen predictions already labeled')
        target = p.target_dataset.iloc[0]
        target_labels = labels(target, p.content_uid)
        p['true_binary_label'] = target_labels.binary_analysis_target.to_numpy()
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def evaluate(predictions):
    within = []
    for (dataset, condition), p in predictions[predictions.evaluation.eq('within')].groupby(['target_dataset', 'condition']):
        require(set(p.true_binary_label) == {0, 1}, 'Holdout lacks class support')
        within.append(dict(dataset=dataset, condition=condition, threshold=.5, **base.metrics(p.true_binary_label, p.fake_probability),
                           evaluation_uid_sha256=digest(p.content_uid)))
    external = transfer.evaluation_results(predictions[predictions.evaluation.eq('transfer')])
    return pd.DataFrame(within), external


def cohort_summary(views):
    # Preflight aggregate class support only; no labels enter feature views,
    # selection, or transfer prediction interfaces through this summary.
    rows = []
    for ds, d in views.items():
        d = source_view(ds, d)
        partitions = [('all', d), ('training_pool', d[d.cv_fold.ge(0)]), ('holdout', d[d.cv_fold.eq(-1)])]
        partitions += [(f'cv_{f}_validation', d[d.cv_fold.eq(f)]) for f in range(5)]
        partitions += [(f'cv_{f}_training', d[d.cv_fold.ge(0) & d.cv_fold.ne(f)]) for f in range(5)]
        for name, part in partitions:
            require(set(part.binary_analysis_target) == {0, 1}, f'{ds} {name} lacks a class')
            rows.append(dict(dataset=ds, partition=name, N=len(part), fake_N=int(part.binary_analysis_target.sum()),
                real_N=int(part.binary_analysis_target.eq(0).sum()), leakage_groups=part.leakage_group.nunique(),
                title_source_cue_N=int(part.title_text_source_cue_flag.sum()),
                changed_by_mask_N=int(part.title_text.ne(part.title_text_source_masked).sum()),
                uid_sha256=digest(part.content_uid)))
    return pd.DataFrame(rows)


def comparisons(within, external, predictions):
    # Read comparator outcomes only after all title predictions have frozen.
    old_within = pd.read_csv(TABLES / 'tfidf_logistic_holdout_results.csv')
    old_transfer = pd.read_csv(TABLES / 'tfidf_logistic_transfer_results.csv')
    old_drop = pd.read_csv(TABLES / 'tfidf_logistic_transfer_degradation.csv')
    saved_within = pd.read_csv(ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv')
    saved_transfer = pd.read_csv(ROOT / 'outputs/predictions/tfidf_logistic_cross_context_predictions.csv')
    changes, degradation = [], []
    for r in external.itertuples():
        if r.scope != 'same_target_holdout':
            continue
        w = within[within.dataset.eq(r.target_dataset) & within.condition.eq(r.condition)].iloc[0]
        require(w.evaluation_uid_sha256 == r.evaluation_uid_sha256, 'Title degradation rows differ')
        for metric in COMPARE:
            degradation.append(dict(direction=r.direction, target_dataset=r.target_dataset, condition=r.condition,
                metric=metric, target_holdout_N=r.N, within_target=float(w[metric]), cross_context=float(getattr(r, metric)),
                transfer_drop=float(getattr(r, metric)-w[metric]), target_uid_sha256=r.evaluation_uid_sha256))
    for (evaluation, direction, condition), p in predictions.groupby(['evaluation', 'direction', 'condition']):
        scopes = [('same_target_holdout', p)] if evaluation == 'within' else [('all_eligible', p), ('same_target_holdout', p[p.cv_fold.eq(-1)])]
        target = p.target_dataset.iloc[0]
        for scope, part in scopes:
            if evaluation == 'within':
                old_p = saved_within[saved_within.dataset.eq(target) & saved_within.condition.eq(condition)]
                old_row = old_within[old_within.dataset.eq(target) & old_within.condition.eq(condition) & old_within.model.eq('tfidf_logistic')].iloc[0]
            else:
                old_p = saved_transfer[saved_transfer.direction.eq(direction) & saved_transfer.condition.eq(condition)]
                if scope == 'same_target_holdout':
                    old_p = old_p[old_p.cv_fold.eq(-1)]
                old_row = old_transfer[old_transfer.direction.eq(direction) & old_transfer.condition.eq(condition) & old_transfer.scope.eq(scope)].iloc[0]
            require(old_p.content_uid.is_unique and set(part.content_uid) <= set(old_p.content_uid), 'Comparator missing title-eligible rows')
            aligned = old_p.set_index('content_uid').loc[part.content_uid]
            require(np.array_equal(aligned.true_binary_label, part.true_binary_label), 'Comparator label mismatch')
            require(np.array_equal(aligned.leakage_group, part.leakage_group), 'Comparator group mismatch')
            new_metrics = base.metrics(part.true_binary_label, part.fake_probability)
            matched_old = base.metrics(aligned.true_binary_label, aligned.fake_probability)
            for metric in COMPARE:
                changes.append(dict(evaluation=evaluation, direction=direction, target_dataset=target, condition=condition,
                    scope=scope, metric=metric, title_N=len(part), combined_original_N=int(old_row.N), matched_target_N=len(part),
                    title=float(new_metrics[metric]), combined_same_rows=float(matched_old[metric]),
                    title_minus_combined=float(new_metrics[metric]-matched_old[metric]),
                    combined_original=float(old_row[metric]), unpaired_title_minus_combined_original=float(new_metrics[metric]-old_row[metric]),
                    exact_same_target_rows=True, original_cohorts_identical=(digest(old_p.content_uid) == digest(part.content_uid)),
                    source_training_cohorts_identical=False, evaluation_uid_sha256=digest(part.content_uid)))
    # Validate comparator transfer-drop arithmetic without rerunning anything.
    require(np.allclose(old_drop.cross_context-old_drop.within_target, old_drop.transfer_drop), 'Saved comparator drop inconsistency')
    return pd.DataFrame(changes), pd.DataFrame(degradation)


def validate_outputs(cv, selection, within, external, predictions, changes, degradation, views, fit_manifest):
    require(len(cv) == 240 and not cv.duplicated(['dataset', 'config_id', 'fold']).any(), 'CV key/count error')
    require(not cv.holdout_used.any(), 'Holdout used for selection')
    require(selection.groupby('dataset').selected.sum().eq(1).all(), 'Selection count error')
    require(len(fit_manifest) == 8 and fit_manifest.converged.all(), 'Final fit failure')
    for table, keys in [(within, ['dataset', 'condition']), (external, ['direction', 'condition', 'scope']),
                        (predictions, ['evaluation', 'direction', 'condition', 'content_uid']),
                        (changes, ['evaluation', 'direction', 'condition', 'scope', 'metric']),
                        (degradation, ['direction', 'condition', 'metric'])]:
        require(not table.duplicated(keys).any(), 'Duplicate output keys')
    require(np.array_equal(predictions.predicted_label, (predictions.fake_probability >= .5).astype(int)), 'Threshold changed')
    for (evaluation, direction, condition), p in predictions.groupby(['evaluation', 'direction', 'condition']):
        expected = views[p.target_dataset.iloc[0]]
        if evaluation == 'within':
            expected = expected[expected.cv_fold.eq(-1)]
        require(set(p.content_uid) == set(expected.content_uid), 'Incomplete eligible-target predictions')
    for (evaluation, direction), p in predictions.groupby(['evaluation', 'direction']):
        natural = p[p.condition.eq('natural')].set_index('content_uid')
        masked = p[p.condition.eq('source_masked')].set_index('content_uid').loc[natural.index]
        require(set(natural.index) == set(masked.index) and np.array_equal(natural.true_binary_label, masked.true_binary_label), 'Mask row mismatch')
    require(within.threshold.eq(.5).all() and external.threshold.eq(.5).all(), 'Threshold drift')


def figures(within, external, changes):
    directory = ROOT / 'outputs/figures/tfidf_logistic_title'
    directory.mkdir(parents=True, exist_ok=True)
    natural = external[external.condition.eq('natural') & external.scope.eq('same_target_holdout')]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(natural))
    w = [within[within.dataset.eq(r.target_dataset) & within.condition.eq('natural')].iloc[0].macro_f1 for r in natural.itertuples()]
    ax.bar(x-.18, w, .36, label='Within target title')
    ax.bar(x+.18, natural.macro_f1, .36, label='Transfer title')
    ax.set(xticks=x, xticklabels=natural.direction, ylim=(0, 1), ylabel='Macro F1', title='Same title-eligible target holdout')
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'within_vs_transfer_macro_f1.png', dpi=180); plt.close(fig)
    selected = changes[changes.evaluation.eq('transfer') & changes.condition.eq('natural') & changes.metric.eq('macro_f1')]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(selected))
    ax.bar(x-.18, selected.combined_same_rows, .36, label='Combined text')
    ax.bar(x+.18, selected.title, .36, label='Title only')
    ax.set(xticks=x, xticklabels=[r.direction+'\n'+('all target' if r.scope == 'all_eligible' else 'target holdout') for r in selected.itertuples()],
           ylim=(0, 1), ylabel='Macro F1', title='Transfer: identical target rows per comparison')
    ax.legend(); fig.tight_layout(); fig.savefig(directory / 'title_vs_combined_transfer_macro_f1.png', dpi=180); plt.close(fig)


def write_report(cohorts, selection, within, external, changes, degradation, diagnostics, fits, cv):
    md = base.md
    selected = selection[selection.selected]
    natural = external[external.condition.eq('natural')]
    change = changes[changes.evaluation.eq('transfer') & changes.condition.eq('natural')]
    drop = degradation[degradation.condition.eq('natural')]
    findings = []
    for direction in DIRECTIONS:
        part = change[change.direction.eq(direction)]
        hold = part[part.scope.eq('same_target_holdout')].set_index('metric')
        all_rows = part[part.scope.eq('all_eligible')].set_index('metric')
        decline = drop[drop.direction.eq(direction) & drop.metric.eq('macro_f1')].iloc[0]
        findings.append(f'{direction}: title transfer macro F1 is {hold.loc["macro_f1", "title"]:.4f} on target holdout '
            f'and {all_rows.loc["macro_f1", "title"]:.4f} on all eligible target rows. Same-row title-minus-combined changes '
            f'are {hold.loc["macro_f1", "title_minus_combined"]:+.4f} and {all_rows.loc["macro_f1", "title_minus_combined"]:+.4f}, respectively. '
            f'On holdout, Fake F1 changes by {hold.loc["fake_f1", "title_minus_combined"]:+.4f} '
            f'and ROC-AUC by {hold.loc["roc_auc", "title_minus_combined"]:+.4f}; transfer-minus-within title macro F1 is {decline.transfer_drop:+.4f}.')
    within_changes = changes[changes.evaluation.eq('within') & changes.condition.eq('natural') & changes.metric.eq('macro_f1')]
    findings += [f'{r.target_dataset} within-domain title macro F1 is {r.title:.4f}, versus {r.combined_same_rows:.4f} '
                 f'for combined text on identical holdout rows ({r.title_minus_combined:+.4f}).' for r in within_changes.itertuples()]
    lines = ['# Title-only TF-IDF + Logistic Regression sensitivity', '',
        '## Protocol and sample', '',
        'Only `title_text` enters the natural predictor pipeline, and only the already-audited `title_text_source_masked` field enters masking sensitivity. Eligibility is the existing `title_only_modeling_eligible` flag; no groups, labels or assignments are regenerated. Fake=1, Real=0; seed=42; threshold=0.5.', '',
        md(cohorts[cohorts.partition.isin(['all', 'training_pool', 'holdout'])][['dataset', 'partition', 'N', 'fake_N', 'real_N', 'leakage_groups', 'title_source_cue_N']]), '',
        'All fold-level counts and training/validation class-support checks are saved in the cohort summary. Every eligible natural and masked title contains usable analyzer tokens. Existing UID/group assignments match both split manifests exactly. No group crosses partitions or transfer datasets.', '',
        'The unchanged 24-configuration grid (unigrams/unigrams+bigrams; min_df 2/5; C 0.1/1/10; none/balanced) uses sublinear_tf=True, max_df=1.0, lowercase=True, L2/liblinear, max_iter=5000. Selection reuses the baseline implementation: mean grouped-CV macro F1 within 0.001 of best, then SD within 0.001 of lowest, then unigram, smaller C, unweighted, larger min_df, config ID. Holdout never enters selection.', '',
        md(selected[['dataset', 'config_id', 'ngram_max', 'min_df', 'C', 'class_weight', 'macro_f1_mean', 'macro_f1_sd']]), '',
        'Both natural configurations were locked before any title holdout or transfer evaluation. Within fits use only the source training pool. Transfer refits use all eligible source titles (including the former source holdout); target titles are transformed only. Both directions and both representations have unlabeled frozen prediction bundles before evaluation labels are joined. Dataset labels necessarily serve as source labels in the reverse direction, but never enter the opposite source fit or selection. Masked titles use natural-title hyperparameters without retuning.', '',
        '## Within-domain title holdout', '',
        md(within[['dataset', 'condition', 'N', 'macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']]), '',
        '## Title cross-context transfer', '',
        md(natural[['direction', 'scope', 'N', 'macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']]), '',
        '## Transfer minus within-target performance', '',
        'These drops compare exactly the same title-eligible target holdout UIDs. Negative is degradation.', '',
        md(drop[['direction', 'metric', 'within_target', 'cross_context', 'transfer_drop']]), '',
        '## Title versus combined-text transfer', '',
        'Primary differences below use saved combined-text predictions restricted to exactly the title-eligible evaluation UIDs. No combined model is refitted or re-predicted. The comparison CSV additionally preserves original published combined metrics, original cohort sizes and explicitly unpaired differences. Even with matched target rows, training cohorts differ slightly between representations, and hyperparameters are separately selected; this is a representation-protocol comparison, not an isolated manipulation of body length.', '',
        md(change[['direction', 'scope', 'metric', 'title_N', 'combined_original_N', 'title', 'combined_same_rows', 'title_minus_combined']]), '',
        '## Prediction bias', '',
        md(natural[['direction', 'scope', 'actual_fake_prevalence', 'predicted_fake_prevalence', 'mean_fake_probability', 'tn', 'fp', 'fn', 'tp', 'near_single_class_prediction']]), '',
        'Confusion counts treat Fake as positive: TN/FP/FN/TP. Near-single-class means predicted Fake prevalence <=5% or >=95%, matching the previous descriptive diagnostic. No calibration or threshold adjustment was made.', '',
        '## Vocabulary shift', '',
        md(diagnostics[diagnostics.condition.eq('natural')][['direction', 'vocabulary_size', 'target_token_occurrence_coverage_percent', 'zero_feature_rows', 'source_analyzer_unigrams_median', 'target_analyzer_unigrams_median']]), '',
        'Coverage counts occurrences of default analyzer unigram tokens, without counting bigrams twice. Vocabulary size includes bigrams when selected. Token lengths use that same analyzer. Coverage and zero-feature diagnostics use all eligible target titles.', '',
        '## Source-masked title sensitivity', '',
        'The existing audited mask dictionary/fields are unchanged. Masking uses cohort metadata from both datasets without labels (a declared transductive string-masking sensitivity). It can remove legitimate institutional mentions, retains the SOURCE token and URL paths, and does not eliminate publisher style or source selection. Cohorts and hyperparameters are identical to natural title fits.', '',
        md(external[external.condition.eq('source_masked')][['direction', 'scope', 'macro_f1', 'fake_f1', 'roc_auc', 'predicted_fake_prevalence']]), '',
        '## Interpretation', '',
        *[text for finding in findings for text in (finding, '')],
        'Title-only does not materially rescue transfer in a consistent way across the prespecified metrics: holdout macro-F1 improvements are minimal, both directions have below-0.5 title ROC-AUC, and ROC-AUC declines relative to combined text in both scopes. FakeHealth-to-CoAID Fake F1 also declines, while the reverse direction improves Fake F1 but retains poor ranking. This is a descriptive judgment without a formal significance test. The smaller CoAID-to-FakeHealth within-versus-transfer gap largely reflects a substantially weaker within-FakeHealth title baseline, rather than successful transfer.', '',
        'Natural transfer predictions are no longer near-single-class by the prespecified 5%/95% rule, but still overpredict Fake for FakeHealth-to-CoAID and underpredict Fake in the reverse direction. Median analyzer title lengths are now similar (8 versus 9 tokens), yet vocabulary coverage remains asymmetric (about 39% versus 71%); 386 CoAID targets have zero features under the FakeHealth title model. Thus length comparability does not remove lexical/domain mismatch. These findings remain consistent with dataset/context dependence, without identifying which underlying difference causes it.', '',
        'Interpretation is based on the complete prespecified metrics and both evaluation scopes; no post-hoc significance or materiality cutoff is applied. A narrower transfer drop alone can reflect a weaker within-domain title model, so the absolute same-row transfer change must also be examined.', '',
        'Representation changes remove bodies and reduce document-length disparity, but also change available content, vocabulary/IDF, selected hyperparameters and slightly the source sample. Therefore improvement cannot prove that body length alone caused prior failure. Remaining dataset/context dependence may include topical mix, periods, collection and publisher/style differences. FakeHealth review-rating-derived labels and CoAID veracity labels have different operational definitions; title-only modeling does not harmonize those labels or identify a causal COVID effect.', '',
        '## Validation and artifacts', '',
        f'All {len(cv)} CV fits converged: {bool(cv.converged.all())}; all {len(fits)} final fits converged: {bool(fits.converged.all())}. CV failures, if any, remain documented and excluded by the unchanged selection rule. No grid expansion, threshold tuning or target fitting occurred.', '',
        'Validation covers unchanged split and input fingerprints, group separation, title-only predictor allowlisting, class support, locked configurations before transfer, source-only vocabulary/IDF fitting with immutable target transforms, target-label-free prediction interfaces, threshold 0.5, full target coverage, masking row parity and unique output keys. Final hashes are recorded alongside this report. CV checkpoints, selection/prediction locks and final models support reuse without repeating fits.', '',
        'Tables: `outputs/tables/tfidf_logistic_title_{cv_results,model_selection,holdout_results,transfer_results,transfer_degradation,vs_combined,vocabulary_diagnostics,cohort_summary,fit_manifest}.csv`. Predictions: `outputs/predictions/tfidf_logistic_title_predictions.csv` (no raw titles). Figures: `outputs/figures/tfidf_logistic_title/`. Models/locks: `outputs/models/tfidf_logistic_title/`. Script: `src/run_tfidf_logistic_title_sensitivity.py`. Progress: `logs/tfidf_logistic_title_sensitivity.log`.', '']
    REPORT.write_text('\n'.join(lines))


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/tfidf_logistic_title_sensitivity.log', level=logging.INFO,
                        format='%(asctime)s %(message)s')
    signature = input_signature()
    meta_path = REPORT.with_suffix('.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        require(meta['signature'] == signature, 'Completed experiment signature changed; do not silently rerun')
        require(meta['validation'] == 'passed' and all(sha(ROOT / p) == h for p, h in meta['outputs'].items()), 'Saved result fingerprint failed')
        print('Validated title sensitivity reused; no fits or evaluations repeated.')
        return
    signature_path = STATE / 'run_signature.json'
    if signature_path.exists():
        require(json.loads(signature_path.read_text()) == signature, 'Checkpoint signature changed; stop before refitting')
    else:
        write_json(signature_path, signature)
    views = {ds: load_features(ds) for ds in base.PREFIXES}
    cohorts = cohort_summary(views)
    cohorts.to_csv(TABLES / 'tfidf_logistic_title_cohort_summary.csv', index=False)
    cv = pd.DataFrame([row for ds, d in views.items() for row in run_cv(ds, d)])
    selection = base.aggregate_cv(cv)
    configs = lock_selections(selection)
    selection['selected'] = [r.config_id == configs[r.dataset]['config_id'] for r in selection.itertuples()]
    bundles = freeze_predictions(views, configs)
    predictions = join_evaluation_labels(bundles)
    within, external = evaluate(predictions)
    changes, degradation = comparisons(within, external, predictions)
    diagnostics = pd.DataFrame([b['diagnostic'] for b in bundles.values() if b['diagnostic'] is not None])
    fits = pd.DataFrame([dict(bundle=key, **b['fit']) for key, b in bundles.items()])
    validate_outputs(cv, selection, within, external, predictions, changes, degradation, views, fits)
    tables = dict(cv_results=cv, model_selection=selection, holdout_results=within, transfer_results=external,
                  transfer_degradation=degradation, vs_combined=changes, vocabulary_diagnostics=diagnostics,
                  cohort_summary=cohorts, fit_manifest=fits)
    paths = []
    for name, frame in tables.items():
        path = TABLES / f'tfidf_logistic_title_{name}.csv'
        frame.to_csv(path, index=False); paths.append(path)
    pred_path = ROOT / 'outputs/predictions/tfidf_logistic_title_predictions.csv'
    predictions.to_csv(pred_path, index=False); paths.append(pred_path)
    figures(within, external, changes)
    write_report(cohorts, selection, within, external, changes, degradation, diagnostics, fits, cv)
    require(input_signature() == signature, 'Original inputs changed during experiment')
    paths += [REPORT, *STATE.glob('*.json'), *STATE.glob('*.joblib'), *(ROOT / 'outputs/figures/tfidf_logistic_title').glob('*.png')]
    write_json(meta_path, dict(signature=signature, validation='passed',
        outputs={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}, cv_fit_count=len(cv), final_fit_count=len(fits),
        all_fits_converged=bool(cv.converged.all() and fits.converged.all()), threshold=.5,
        configurations_locked_before_transfer=True, target_labels_joined_after_predictions_frozen=True,
        source_only_fit=True, no_split_reassignment=True))
    logging.info('Title sensitivity complete; all output and input validations passed')
    print('Title sensitivity complete: 240 CV fits, 8 final fits. Validation passed. See outputs/reports/tfidf_logistic_title_sensitivity.md')


if __name__ == '__main__':
    main()
