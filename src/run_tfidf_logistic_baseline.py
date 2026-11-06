"""Frozen within-dataset TF-IDF -> Logistic Regression baseline; no transfer.

Run with .venv/bin/python src/run_tfidf_logistic_baseline.py.
Per-fit checkpoints prevent repeated evaluation on restart. A completed run is
reused only after checking input, protocol, version and output fingerprints.
"""
from pathlib import Path
import hashlib
import itertools
import json
import logging
import os
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = {'FakeHealth': 'fakehealth', 'CoAID': 'coaid'}
FIELDS = {'natural': 'combined_text', 'source_masked': 'combined_text_source_masked'}
TABLES = ROOT / 'outputs/tables'
STATE = ROOT / 'outputs/models/tfidf_logistic'
REPORT = ROOT / 'outputs/reports/tfidf_logistic_baseline.md'
META_PATH = REPORT.with_suffix('.json')
SEED = 42
TOLERANCE = .001
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'fake_precision', 'fake_recall',
           'fake_f1', 'real_precision', 'real_recall', 'real_f1', 'roc_auc', 'average_precision']
GRID = [dict(config_id=f'cfg_{i:02d}', ngram_max=n, min_df=m, C=c, class_weight=w)
        for i, (n, m, c, w) in enumerate(itertools.product([1, 2], [2, 5], [.1, 1., 10.], ['none', 'balanced']))]
PROTOCOL = dict(seed=SEED, grid=GRID, selection='unweighted mean macro F1', tie_tolerance=TOLERANCE,
    tie_order=['mean macro F1 within .001 of best', 'SD within .001 of lowest in tied set',
               'unigram', 'smaller C', 'class_weight none', 'larger min_df', 'config_id'],
    solver='liblinear', penalty='l2', sklearn_penalty_parameter='l1_ratio=0.0',
    max_iter=5000, tol=1e-4, threshold=.5, vectorizer=dict(lowercase=True, strip_accents=None,
    analyzer='word', dtype='float64', sublinear_tf=True, max_df=1., norm='l2', use_idf=True,
    smooth_idf=True, token_pattern=r'(?u)\b\w\w+\b'),
    conditions=FIELDS, masked_retuning=False, cv_standard_deviation_ddof=1)
COLUMNS = ['dataset', 'content_uid', 'combined_text', 'combined_text_source_masked',
           'binary_analysis_label', 'binary_analysis_target', 'original_review_rating',
           'veracity_original', 'label_definition', 'primary_modeling_eligible',
           'leakage_group', 'cv_fold', 'holdout_assignment']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(json.dumps(sorted(map(str, values)), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def dump(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    joblib.dump(value, temp, compress=3)
    temp.replace(path)


def input_paths():
    paths = [ROOT / f'data/processed/{p}_text_modeling_ready.csv' for p in PREFIXES.values()]
    paths += [ROOT / f'outputs/splits/{p}_text_{s}.csv' for p in PREFIXES.values() for s in ['grouped_cv', 'holdout']]
    paths += [TABLES / name for name in ['text_modeling_ready_summary.csv', 'text_source_cue_audit.csv', 'text_source_mask_dictionary.csv']]
    return paths


def get_signature():
    return dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in input_paths()},
                script_sha256=sha(__file__), protocol=PROTOCOL,
                versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                              numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__))


def load_cohort(dataset):
    prefix = PREFIXES[dataset]
    d = pd.read_csv(ROOT / f'data/processed/{prefix}_text_modeling_ready.csv', usecols=COLUMNS, keep_default_na=False)
    require(d.primary_modeling_eligible.dtype == bool, 'Eligibility must be boolean')
    d = d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
    require(d.content_uid.is_unique and d.dataset.eq(dataset).all(), 'Cohort identity error')
    require(d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all(), 'Target coding error')
    require(set(d.binary_analysis_label) == {'Fake', 'Real'}, 'Missing class')
    require(d.cv_fold.isin([-1, 0, 1, 2, 3, 4]).all(), 'Unexpected fold')
    require((d.holdout_assignment.eq('test') == d.cv_fold.eq(-1)).all(), 'Holdout mismatch')
    for suffix, col, target in [('grouped_cv', 'fold', 'cv_fold'), ('holdout', 'assignment', 'holdout_assignment')]:
        old = pd.read_csv(ROOT / f'outputs/splits/{prefix}_text_{suffix}.csv').set_index('content_uid')
        require(old.index.is_unique and set(d.content_uid) <= set(old.index), 'Missing/duplicate split UID')
        aligned = old.loc[d.content_uid]
        require(aligned[col].tolist() == d[target].tolist(), 'Split reassignment')
        require(aligned.leakage_group.tolist() == d.leakage_group.tolist(), 'Group reassignment')
        require(aligned.binary_analysis_target.tolist() == d.binary_analysis_target.tolist(), 'Split target mismatch')
    require(d.groupby('leakage_group').cv_fold.nunique().le(1).all(), 'Group crosses partitions')
    summary = pd.read_csv(TABLES / 'text_modeling_ready_summary.csv')
    row = summary[summary.dataset.eq(dataset) & summary.condition.eq('primary_natural') & summary.partition.eq('all')].iloc[0]
    require(len(d) == row.N and int(d.binary_analysis_target.sum()) == row.fake_N, 'Frozen cohort count mismatch')
    for train, valid in cv_partitions(d):
        assert_partition(train, valid, d[d.cv_fold.eq(-1)])
    return d


def texts(frame, condition):
    """The sole predictor boundary: one allowlisted text column, without joins."""
    return frame[FIELDS[condition]].to_numpy(dtype=str, copy=True)


def cv_partitions(d):
    for fold in range(5):
        yield d[(d.cv_fold >= 0) & d.cv_fold.ne(fold)], d[d.cv_fold.eq(fold)]


def assert_partition(train, validation, holdout=None):
    require(train.cv_fold.ge(0).all(), 'Holdout entered training')
    require(set(train.binary_analysis_target) == {0, 1}, 'Training lost class support')
    require(set(validation.binary_analysis_target) == {0, 1}, 'Evaluation lost class support')
    require(not set(train.content_uid) & set(validation.content_uid), 'UID overlap')
    require(not set(train.leakage_group) & set(validation.leakage_group), 'Leakage-group overlap')
    if holdout is not None:
        require(not set(train.leakage_group) & set(holdout.leakage_group), 'Training/holdout group overlap')
        require(not set(validation.leakage_group) & set(holdout.leakage_group), 'Validation/holdout group overlap')


def make_pipeline(config):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, int(config['ngram_max'])), min_df=int(config['min_df']),
            lowercase=True, strip_accents=None, analyzer='word', dtype=np.float64,
            sublinear_tf=True, max_df=1.)),
        # sklearn >=1.8 spells pure L2 as l1_ratio=0; penalty is deprecated.
        ('logistic', LogisticRegression(l1_ratio=0., solver='liblinear', C=float(config['C']),
            class_weight=None if config['class_weight'] == 'none' else 'balanced',
            max_iter=5000, tol=1e-4, random_state=SEED))])


def fit_pipeline(train, condition, config):
    model = make_pipeline(config)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(texts(train, condition), train.binary_analysis_target.to_numpy())
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    iterations = int(model['logistic'].n_iter_.max())
    converged = converged and iterations < model['logistic'].max_iter
    return model, dict(converged=converged, n_iter=iterations, fit_seconds=time.perf_counter() - start,
                       warnings='; '.join(str(w.message) for w in caught),
                       vocabulary_size=len(model['tfidf'].vocabulary_),
                       vocabulary_sha256=digest(model['tfidf'].vocabulary_), train_uid_sha256=digest(train.content_uid),
                       train_group_sha256=digest(train.leakage_group.unique()))


def metrics(y, probability, prediction=None):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    prediction = (probability >= .5).astype(int) if prediction is None else np.asarray(prediction, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(N=len(y), fake_N=int(y.sum()), real_N=int((y == 0).sum()),
        accuracy=float(accuracy_score(y, prediction)), balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, average='macro', labels=[0, 1], zero_division=0)),
        fake_precision=float(precision[1]), fake_recall=float(recall[1]), fake_f1=float(f1[1]),
        real_precision=float(precision[0]), real_recall=float(recall[0]), real_f1=float(f1[0]),
        roc_auc=float(roc_auc_score(y, probability)), average_precision=float(average_precision_score(y, probability)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def evaluate_cv(dataset, d, condition, config):
    rows = []
    for fold, (train, valid) in enumerate(cv_partitions(d)):
        path = STATE / f'cv_{PREFIXES[dataset]}_{condition}_{config["config_id"]}_{fold}.json'
        if path.exists():
            row = json.loads(path.read_text())
            require(row['train_uid_sha256'] == digest(train.content_uid) and row['validation_uid_sha256'] == digest(valid.content_uid), 'Stale CV cache')
        else:
            assert_partition(train, valid, d[d.cv_fold.eq(-1)])
            model, fit = fit_pipeline(train, condition, config)
            probability = model.predict_proba(texts(valid, condition))[:, 1]
            row = dict(dataset=dataset, condition=condition, **config, fold=fold, train_N=len(train), validation_N=len(valid),
                       **fit, **metrics(valid.binary_analysis_target, probability),
                       validation_uid_sha256=digest(valid.content_uid), holdout_used=False)
            write_json(path, row)
        rows.append(row)
    logging.info('%s %s %s CV macro F1 %.4f', dataset, condition, config['config_id'], np.mean([r['macro_f1'] for r in rows]))
    return rows


def aggregate_cv(cv):
    rows = []
    keys = ['dataset', 'condition', 'config_id']
    for key, part in cv.groupby(keys, sort=True):
        require(len(part) == 5 and set(part.fold) == set(range(5)), 'Incomplete CV configuration')
        config = next(c for c in GRID if c['config_id'] == key[2])
        row = dict(zip(keys, key)) | config | dict(all_converged=bool(part.converged.all()))
        for metric in METRICS:
            row[metric + '_mean'] = float(part[metric].mean())
            row[metric + '_sd'] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def select_config(summary):
    candidates = summary[summary.all_converged].copy()
    require(len(candidates) > 0, 'All proposed configurations failed convergence; no automatic grid expansion')
    candidates = candidates[candidates.macro_f1_mean >= candidates.macro_f1_mean.max() - TOLERANCE]
    candidates = candidates[candidates.macro_f1_sd <= candidates.macro_f1_sd.min() + TOLERANCE]
    # Remaining differences are explicitly treated as numerical ties.
    candidates['weight_priority'] = candidates.class_weight.ne('none').astype(int)
    chosen = candidates.sort_values(['ngram_max', 'C', 'weight_priority', 'min_df', 'config_id'], ascending=[True, True, True, False, True]).iloc[0]
    return next(dict(c) for c in GRID if c['config_id'] == chosen.config_id)


def lock_selections(selections, summary):
    path = STATE / 'selection_lock.json'
    payload = dict(selections=selections, protocol=PROTOCOL, selection_scope='natural training-pool grouped CV only',
                   natural_cv_sha256=hashlib.sha256(summary.to_json(orient='records').encode()).hexdigest())
    if path.exists():
        require(json.loads(path.read_text()) == payload, 'Selections differ from locked configuration')
    else:
        write_json(path, payload)
    return path


def feature_rows(dataset, model, train):
    vectorizer, logistic = model['tfidf'], model['logistic']
    names = vectorizer.get_feature_names_out()
    coefficients = logistic.coef_[0]
    frequencies = np.asarray((vectorizer.transform(texts(train, 'natural')) > 0).sum(axis=0)).ravel()
    dictionary = pd.read_csv(TABLES / 'text_source_mask_dictionary.csv', keep_default_na=False)
    analyzer = vectorizer.build_analyzer()
    identifiers = []
    for row in dictionary.itertuples():
        # TF-IDF lowercases and splits domains; reproduce the same representation.
        represented = set(analyzer(row.identifier))
        identifiers.append((row.identifier, represented, row.category, row.source_identity))
    artifact_terms = {'access denied', 'page not', 'not found', 'forbidden', 'javascript', 'cookie',
                      'cookies', 'browser', 'subscribe', 'log in', 'sign up', 'privacy policy', 'robot',
                      'mod_security', 'twenty four', 'saving lives', 'protecting people'}
    rows = []
    for label, indices in [('Fake', np.argsort(-coefficients, kind='stable')), ('Real', np.argsort(coefficients, kind='stable'))]:
        indices = [i for i in indices if (coefficients[i] > 0 if label == 'Fake' else coefficients[i] < 0)][:30]
        for rank, idx in enumerate(indices, 1):
            feature = names[idx]
            # Conservative: exact full identifier after tokenization, or >=3-char
            # multiword/domain fragments; generic component words can be ambiguous.
            matched = [(name, category, identity) for name, representations, category, identity in identifiers
                       if feature in representations and len(feature) >= 3]
            matches = [name for name, _, _ in matched]
            exact = [name for name in matches if ' '.join(vectorizer.build_tokenizer()(name.lower())) == feature]
            rows.append(dict(dataset=dataset, associated_class=label, rank=rank, feature=feature,
                coefficient=float(coefficients[idx]), training_document_frequency=int(frequencies[idx]),
                training_document_percent=100 * int(frequencies[idx]) / len(train),
                source_identifier_match=bool(exact), source_identifier_fragment_match=bool(matches),
                publisher_name_or_fragment_cue=any(category == 'publisher_name' for _, category, _ in matched),
                domain_or_fragment_cue=any(category == 'domain' for _, category, _ in matched),
                institutional_identifier_or_fragment_cue=any(identity in {'cdc.gov', 'who.int', 'nih.gov'} for _, _, identity in matched),
                source_matches=json.dumps(sorted(set(matches))),
                crawl_artifact_candidate=feature in artifact_terms,
                interpretation='Candidate lexical cue, not causal; partial identifier matches may be ordinary words.'))
    return rows


def evaluate_final(dataset, d, condition, config, selection_path):
    # Guard remains active even when reading a checkpoint.
    require(selection_path.exists(), 'Holdout evaluation attempted before configuration lock')
    locked = json.loads(selection_path.read_text())
    require(locked['selections'].get(dataset) == config, 'Configuration was not selected/locked')
    path = STATE / f'final_{PREFIXES[dataset]}_{condition}.joblib'
    if path.exists():
        return joblib.load(path)
    train, test = d[d.cv_fold.ge(0)], d[d.cv_fold.eq(-1)]
    assert_partition(train, test)
    model, fit = fit_pipeline(train, condition, config)
    require(fit['converged'], f'{dataset} {condition}: final model failed convergence; holdout not evaluated')
    # Exactly one holdout predict_proba call; all metrics/figures reuse it.
    probability = model.predict_proba(texts(test, condition))[:, 1]
    prediction = (probability >= .5).astype(int)
    row = dict(dataset=dataset, condition=condition, model='tfidf_logistic', train_N=len(train), **config, **fit,
               **metrics(test.binary_analysis_target, probability), threshold=.5,
               evaluation_uid_sha256=digest(test.content_uid), selection_lock_sha256=sha(selection_path))
    preds = test[['dataset', 'content_uid', 'binary_analysis_target', 'binary_analysis_label',
                  'original_review_rating', 'veracity_original', 'label_definition', 'leakage_group', 'cv_fold']].copy()
    preds = preds.rename(columns={'binary_analysis_target': 'true_binary_label'})
    preds['predicted_label'] = prediction
    preds['fake_probability'] = probability
    preds['condition'] = condition
    result = dict(model=model, result=row, predictions=preds,
                  top_features=feature_rows(dataset, model, train) if condition == 'natural' else [])
    dump(path, result)
    logging.info('%s %s holdout evaluated once: macro F1 %.4f', dataset, condition, row['macro_f1'])
    return result


def evaluate_dummies(dataset, d, selection_path):
    require(selection_path.exists(), 'Dummy evaluation before selection lock')
    path = STATE / f'dummy_{PREFIXES[dataset]}.json'
    if path.exists():
        return json.loads(path.read_text())
    train, test = d[d.cv_fold.ge(0)], d[d.cv_fold.eq(-1)]
    rows = []
    for strategy in ['most_frequent', 'stratified']:
        dummy = DummyClassifier(strategy=strategy, random_state=SEED)
        dummy.fit(np.zeros((len(train), 1)), train.binary_analysis_target)
        proba = dummy.predict_proba(np.zeros((len(test), 1)))[:, 1]
        # Stratified probabilities are one-hot draws; use the same realization.
        rows.append(dict(dataset=dataset, condition='dummy', model='dummy_' + strategy,
                         **metrics(test.binary_analysis_target, proba), threshold=.5,
                         fitted_on='training_pool_only', train_N=len(train)))
    write_json(path, rows)
    return rows


def sensitivity(cv, holdout, selections):
    rows = []
    for ds, config in selections.items():
        natural = cv[cv.dataset.eq(ds) & cv.condition.eq('natural') & cv.config_id.eq(config['config_id'])].set_index('fold')
        masked = cv[cv.dataset.eq(ds) & cv.condition.eq('source_masked')].set_index('fold')
        for metric in METRICS:
            delta = masked[metric] - natural[metric]
            for fold in range(5):
                rows.append(dict(dataset=ds, scope='cv_fold', fold=fold, metric=metric,
                    natural=float(natural.loc[fold, metric]), masked=float(masked.loc[fold, metric]),
                    masked_minus_natural=float(delta.loc[fold]), delta_sd=None))
            rows.append(dict(dataset=ds, scope='cv_mean', fold=-2, metric=metric,
                natural=float(natural[metric].mean()), masked=float(masked[metric].mean()),
                masked_minus_natural=float(delta.mean()), delta_sd=float(delta.std(ddof=1))))
            h = holdout[holdout.dataset.eq(ds) & holdout.model.eq('tfidf_logistic')].set_index('condition')
            rows.append(dict(dataset=ds, scope='holdout', fold=-1, metric=metric,
                natural=float(h.loc['natural', metric]), masked=float(h.loc['source_masked', metric]),
                masked_minus_natural=float(h.loc['source_masked', metric] - h.loc['natural', metric]), delta_sd=None))
    return pd.DataFrame(rows)


def plots(predictions):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'logs/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT / 'outputs/figures/tfidf_logistic'
    directory.mkdir(parents=True, exist_ok=True)
    for ds, prefix in PREFIXES.items():
        x = predictions[predictions.dataset.eq(ds) & predictions.condition.eq('natural')]
        matrix = confusion_matrix(x.true_binary_label, x.predicted_label, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        ax.imshow(matrix, cmap='Blues')
        for i, j in itertools.product(range(2), repeat=2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               xlabel='Predicted', ylabel='True', title=f'{ds}: natural-text holdout')
        fig.tight_layout(); fig.savefig(directory / f'{prefix}_confusion_matrix.png', dpi=180); plt.close(fig)
        precision, recall, _ = precision_recall_curve(x.true_binary_label, x.fake_probability)
        ap = average_precision_score(x.true_binary_label, x.fake_probability)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.step(recall, precision, where='post', label=f'Fake AP = {ap:.3f}')
        ax.axhline(x.true_binary_label.mean(), linestyle='--', color='grey', label='Holdout Fake prevalence')
        ax.set(xlabel='Fake recall', ylabel='Fake precision', xlim=(0, 1), ylim=(0, 1.02), title=f'{ds}: natural-text holdout')
        ax.legend(); fig.tight_layout(); fig.savefig(directory / f'{prefix}_precision_recall.png', dpi=180); plt.close(fig)


def md(frame):
    def format_cell(v):
        if isinstance(v, float):
            return f'{v:.4f}'
        return str(v).replace('|', '\\|')
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(format_cell, row)) + ' |' for row in frame.itertuples(index=False, name=None))


def report(selection, holdout, differences, features, cv, cohorts):
    lines = ['# First TF-IDF + Logistic Regression baseline', '',
        'Within-dataset classification only: FakeHealth HealthStory and frozen CoAID News Specification A. No transfer, alternative classifier, threshold tuning, title/body-only experiment or publisher-held-out split was run.', '',
        '## Frozen protocol', '',
        'Fake=1, Real=0. Only primary_modeling_eligible rows and combined_text enter the primary Pipeline. Source-masked sensitivity uses combined_text_source_masked on identical rows/splits. Targets, ratings, source/review metadata, identifiers, engagement, provenance and quality flags are never concatenated into predictors. Original target/rating definitions are retained in the prediction output for reporting.', '',
        'The 24 prespecified configurations cross ngram_range=(1,1)/(1,2), min_df=2/5, C=0.1/1/10, and class_weight=None/balanced. TF-IDF uses lowercase=True, strip_accents=None, word analysis, sublinear_tf=True, max_df=1.0, float64 sparse data, L2 normalization and default smoothed IDF. Logistic Regression uses the stable sparse-compatible binary liblinear solver, pure L2 regularization, max_iter=5000, tol=1e-4 and seed 42. In installed sklearn, l1_ratio=0.0 specifies L2 without the deprecated penalty parameter. Liblinear regularizes its synthetic intercept with default intercept_scaling=1.', '',
        'Each Pipeline fits a fresh vectorizer exclusively inside its training fold. Selection uses unweighted mean macro F1 across the five existing grouped folds. Configurations within 0.001 of the maximum mean are tied; among those, SD within 0.001 of the minimum is tied. Then prefer unigram, smaller C, class_weight=None, larger min_df, and stable configuration ID, in that order. SD is the sample SD (ddof=1), not an uncertainty interval; unequal fold sizes remain unchanged. Nonconverged configurations cannot win.', '',
        'Both datasets’ natural-text configurations were locked before any holdout prediction. Final models fit the complete training pool and evaluate the reserved holdout once at probability >=0.5. Source masking is never retuned. Masked CV and holdout use the frozen natural-text hyperparameters. All curves, confusion matrices and diagnostics reuse saved predictions; restart checkpoints avoid reevaluation. Selected CV performance is a tuning estimate, not nested-CV generalization performance.', '',
        '## Results', '']
    for ds in PREFIXES:
        selected = selection[selection.dataset.eq(ds) & selection.selected].iloc[0]
        config = {key: selected[key] for key in ['ngram_max', 'min_df', 'C', 'class_weight']}
        h = holdout[holdout.dataset.eq(ds)]
        scores = pd.DataFrame([{'metric': metric, 'CV mean': selected[metric + '_mean'], 'CV SD': selected[metric + '_sd']} for metric in METRICS])
        d = differences[differences.dataset.eq(ds) & differences.scope.isin(['cv_mean', 'holdout'])]
        f = features[features.dataset.eq(ds)]
        counts = cohorts[ds].groupby('holdout_assignment').binary_analysis_target.agg(['size', 'sum']).reset_index().rename(columns={'size': 'N', 'sum': 'Fake N'})
        counts['Real N'] = counts.N - counts['Fake N']
        lines += [f'### {ds}', '', f'Selected: ngram_range=(1,{config["ngram_max"]}), min_df={config["min_df"]}, C={config["C"]}, class_weight={config["class_weight"]}.', '',
            md(counts), '', md(scores), '',
            md(h[['condition', 'model', 'N', 'fake_N', 'real_N'] + METRICS]), '',
            'Confusion matrices below use rows=true Real/Fake, columns=predicted Real/Fake: [[TN, FP], [FN, TP]].', '',
            md(h[['condition', 'model', 'tn', 'fp', 'fn', 'tp']]), '',
            'Masked minus natural (paired by fold for CV; sample SD of paired differences recorded):', '',
            md(d[['scope', 'metric', 'natural', 'masked', 'masked_minus_natural', 'delta_sd']]), '',
            f'Of the 60 signed top features, {int(f.source_identifier_match.sum())} match full tokenized source identifiers, {int(f.source_identifier_fragment_match.sum())} match identifier components, and {int(f.crawl_artifact_candidate.sum())} match the fixed crawl-artifact candidate list. Component matches are deliberately broad and can be ordinary words; they are not proof of publisher identity.', '',
            md(f.groupby('associated_class', sort=False).head(8)[['associated_class', 'rank', 'feature', 'coefficient', 'training_document_frequency', 'source_identifier_match', 'crawl_artifact_candidate']]), '']
    lines += ['## Interpretation and limitations', '',
        'The ground truths differ: FakeHealth uses author review rating <3 versus >=3, whereas CoAID uses its original Fake/Real dataset labels. Scores are not directly interchangeable measurements of factual truth. These are resource/text-grouped splits, not unseen-publisher tests. Resource/group leakage checks do not remove publisher, topic, collection or time-period confounding.', '',
        'CoAID has a strong source/class association: the frozen modeling-ready audit reports 2,930 rows from class-exclusive source categories, 604 from mixed-class categories (561 Fake and 43 Real), and 43 without source metadata. FakeHealth also has publisher associations. High CoAID scores may reflect these collection/source regularities rather than general misinformation language. CoAID bodies are generally truncated near 500 characters; FakeHealth often contains much longer article bodies. Differences in available text, labeling and selection affect cross-dataset comparisons.', '',
        'Explicit source masking removes only known string identifiers and URL hosts. Institutional mentions may be legitimate content; stylistic cues, URL paths, topic and the [SOURCE] placeholder itself remain informative. The existing dictionary was derived label-blind from both cohorts’ metadata, including held-out metadata; this is a declared transductive sensitivity transform, not a training-only source-discovery system. Masked-minus-natural differences are descriptive paired sensitivity results, not estimates of the entire publisher-confounding effect. No significance claim or arbitrary materiality cutoff is applied.', '',
        'Top coefficients describe conditional associations under the fitted vocabulary and regularization. They neither establish causal linguistic mechanisms nor prove that misinformation spreads differently. The explanatory engagement findings remain frozen.', '',
        'Dummy most-frequent and seed-42 stratified baselines were fitted only to each training pool; the majority baseline reflects its training majority, not an oracle choice based on held-out labels. Stratified probabilities are a single one-hot random realization. Average precision is the non-interpolated AP definition, not trapezoidal PR area. ROC-AUC/AP use Fake probabilities, and every split contains both classes.', '',
        'Cross-context transfer can be a separately authorized next experiment after locking its source-only selection/refit protocol in both directions and reviewing current cross-dataset overlap safeguards. Neither dataset’s held-out result was used to choose the other dataset’s model. Differences in labels, sources and text length limit transfer interpretation; no transfer was performed here.', '',
        '## Reproducibility and validation', '',
        f'All {len(cv)} CV fits and four final Logistic Regression fits converged: {bool(cv.converged.all() and holdout.loc[holdout.model.eq("tfidf_logistic"), "converged"].all())}. Original cohort/split fingerprints are unchanged; pipeline training UID digests, vocabulary digests, convergence and selected-configuration locks are saved. Automated tests cover training-only vocabulary fitting, predictor allowlisting, split identity and group separation, masking cohort parity, holdout timing, unique output keys, predictions and metric consistency.', '',
        'Run `.venv/bin/python src/run_tfidf_logistic_baseline.py` (completed runs are verified and reused). Tests: `.venv/bin/python -m unittest discover -s src -p "test_tfidf_logistic_baseline.py"`. Progress: logs/tfidf_logistic_baseline.log. Primary tables: outputs/tables/tfidf_logistic_{cv_results,model_selection,holdout_results,source_mask_sensitivity,top_features}.csv. Predictions: outputs/predictions/tfidf_logistic_holdout_predictions.csv. Four figures: outputs/figures/tfidf_logistic/. Fitted models, per-fit checkpoints and protocol locks: outputs/models/tfidf_logistic/. Metadata: outputs/reports/tfidf_logistic_baseline.json.', '']
    REPORT.write_text('\n'.join(lines))


def validate_outputs(cv, selection, holdout, predictions, features, differences, cohorts):
    for frame, keys in [(cv, ['dataset', 'condition', 'config_id', 'fold']),
                        (selection, ['dataset', 'condition', 'config_id']),
                        (holdout, ['dataset', 'condition', 'model']),
                        (predictions, ['dataset', 'condition', 'content_uid']),
                        (features, ['dataset', 'associated_class', 'rank']),
                        (differences, ['dataset', 'scope', 'fold', 'metric'])]:
        require(not frame.duplicated(keys).any(), f'Duplicate output key: {keys}')
    require(len(cv) == 250 and cv.converged.all(), 'Missing/nonconverged CV fits')
    require(len(holdout) == 8, 'Missing final/dummy results')
    for ds, d in cohorts.items():
        require(selection[selection.dataset.eq(ds)].selected.sum() == 1, 'Selection count')
        for condition in FIELDS:
            p = predictions[predictions.dataset.eq(ds) & predictions.condition.eq(condition)]
            test = d[d.cv_fold.eq(-1)]
            require(set(p.content_uid) == set(test.content_uid) and len(p) == len(test), 'Prediction coverage mismatch')
            require(set(p.true_binary_label) == {0, 1}, 'Holdout missing a class')
            row = holdout[holdout.dataset.eq(ds) & holdout.condition.eq(condition)].iloc[0]
            require(row.converged, 'Final model failed convergence')
            recomputed = metrics(p.true_binary_label, p.fake_probability)
            for key, value in recomputed.items():
                require(np.isclose(row[key], value), 'Saved prediction metric mismatch')


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/tfidf_logistic_baseline.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', force=True)
    signature = get_signature()
    if META_PATH.exists():
        old = json.loads(META_PATH.read_text())
        require(old['signature'] == signature, 'Completed run inputs/code/protocol changed; refusing automatic holdout reevaluation')
        for path, expected in old['outputs'].items():
            require(sha(ROOT / path) == expected, 'Saved artifact changed: ' + path)
        print('Validated completed baseline reused; no models refitted or holdouts reevaluated.')
        return
    start_lock = STATE / 'run_signature.json'
    if start_lock.exists():
        require(json.loads(start_lock.read_text()) == signature, 'Incomplete run signature changed; inspect before resuming')
    else:
        write_json(start_lock, signature)
    cohorts = {ds: load_cohort(ds) for ds in PREFIXES}
    logging.info('Protocol frozen; %s', {ds: len(d) for ds, d in cohorts.items()})
    cv_rows = []
    for ds, d in cohorts.items():
        for config in GRID:
            cv_rows += evaluate_cv(ds, d, 'natural', config)
    natural_summary = aggregate_cv(pd.DataFrame(cv_rows))
    selections = {ds: select_config(natural_summary[natural_summary.dataset.eq(ds)]) for ds in PREFIXES}
    selection_path = lock_selections(selections, natural_summary)
    logging.info('Both natural configurations locked before any holdout: %s', selections)
    print('Natural-text configurations selected and locked:', json.dumps(selections), flush=True)
    for ds, d in cohorts.items():
        cv_rows += evaluate_cv(ds, d, 'source_masked', selections[ds])
    cv = pd.DataFrame(cv_rows)
    summary = aggregate_cv(cv)
    summary['selected'] = [r.condition == 'natural' and r.config_id == selections[r.dataset]['config_id'] for r in summary.itertuples()]
    summary['frozen_sensitivity_configuration'] = summary.condition.eq('source_masked')
    holdout_rows, preds, features = [], [], []
    for ds, d in cohorts.items():
        for condition in FIELDS:
            final = evaluate_final(ds, d, condition, selections[ds], selection_path)
            holdout_rows.append(final['result']); preds.append(final['predictions']); features += final['top_features']
        holdout_rows += evaluate_dummies(ds, d, selection_path)
    holdout, predictions, top = pd.DataFrame(holdout_rows), pd.concat(preds, ignore_index=True), pd.DataFrame(features)
    differences = sensitivity(cv, holdout, selections)
    validate_outputs(cv, summary, holdout, predictions, top, differences, cohorts)
    for name, table in [('cv_results', cv), ('model_selection', summary), ('holdout_results', holdout),
                        ('source_mask_sensitivity', differences), ('top_features', top)]:
        table.to_csv(TABLES / f'tfidf_logistic_{name}.csv', index=False)
    prediction_path = ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv'
    prediction_path.parent.mkdir(exist_ok=True, parents=True)
    predictions.to_csv(prediction_path, index=False)
    plots(predictions)
    report(summary, holdout, differences, top, cv, cohorts)
    require(signature == get_signature(), 'Inputs or code changed during execution')
    artifacts = [*TABLES.glob('tfidf_logistic_*.csv'), prediction_path, REPORT,
                 *(ROOT / 'outputs/figures/tfidf_logistic').glob('*.png'), *STATE.glob('*.json'), *STATE.glob('*.joblib')]
    write_json(META_PATH, dict(signature=signature, outputs={str(p.relative_to(ROOT)): sha(p) for p in sorted(artifacts)},
        validation='passed', no_split_reassignment=True, selection_before_holdout=True,
        all_pipelines_converged=True, cv_fit_count=len(cv), final_logistic_fit_count=4,
        holdout_evaluations_per_dataset_condition=1, selection_lock_sha256=sha(selection_path)))
    logging.info('Finished; validation passed and all outputs fingerprinted')
    print(holdout[holdout.model.eq('tfidf_logistic')][['dataset', 'condition', 'macro_f1', 'roc_auc', 'average_precision']].to_json(orient='records'))
    print('Validation passed. Report: outputs/reports/tfidf_logistic_baseline.md', flush=True)


if __name__ == '__main__':
    main()
