"""Locked, bidirectional external TF-IDF/Logistic Regression evaluation.

Uses existing baseline utilities without rerunning tuning or target-domain fits.
All four unlabeled prediction sets are frozen before target labels are joined.
"""
from pathlib import Path
import hashlib
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

import run_tfidf_logistic_baseline as baseline

ROOT = baseline.ROOT
TABLES = ROOT / 'outputs/tables'
STATE = ROOT / 'outputs/models/tfidf_logistic_transfer'
REPORT = ROOT / 'outputs/reports/tfidf_logistic_cross_context_transfer.md'
META = REPORT.with_suffix('.json')
FIELDS = dict(baseline.FIELDS)
DIRECTIONS = {'FH_to_CA': ('FakeHealth', 'CoAID'), 'CA_to_FH': ('CoAID', 'FakeHealth')}
CONFIGS = {
    'FakeHealth': dict(config_id='cfg_21', ngram_max=2, min_df=5, C=1., class_weight='balanced'),
    'CoAID': dict(config_id='cfg_05', ngram_max=1, min_df=2, C=10., class_weight='balanced'),
}
FEATURE_COLUMNS = ['dataset', 'content_uid', 'primary_modeling_eligible', 'combined_text',
                   'combined_text_source_masked', 'leakage_group', 'cv_fold', 'holdout_assignment',
                   'cross_dataset_overlap_flag', 'cross_dataset_overlap_group_flag']
TARGET_COLUMNS = ['binary_analysis_target', 'binary_analysis_label', 'veracity_original',
                  'original_review_rating', 'label_definition']
METRICS = baseline.METRICS
COMPARE = ['macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']
PROTOCOL = dict(seed=42, directions=DIRECTIONS, source_configurations=CONFIGS,
    text_fields=FIELDS, source_training='all primary_modeling_eligible rows, including former source holdout',
    target_evaluation=['all_eligible', 'same_target_holdout'], threshold=.5,
    target_labels='joined only after all four prediction sets are frozen',
    no_retuning=True, no_threshold_selection=True, no_recalibration=True,
    numerical_pipeline=baseline.PROTOCOL,
    diagnostic_token_definition='Occurrences of preprocessed default word-tokenizer unigrams; no bigram double counting',
    descriptive_calibration_gap_warning_threshold=.10, near_single_class_prediction_threshold=.95)
require = baseline.require
sha = baseline.sha
digest = baseline.digest
write_json = baseline.write_json


def paths():
    result = [ROOT / f'data/processed/{prefix}_text_modeling_ready.csv' for prefix in baseline.PREFIXES.values()]
    result += [ROOT / f'outputs/splits/{prefix}_text_holdout.csv' for prefix in baseline.PREFIXES.values()]
    result += [TABLES / f'tfidf_logistic_{name}.csv' for name in ['model_selection', 'holdout_results', 'source_mask_sensitivity', 'top_features']]
    result += [TABLES / 'text_cross_dataset_overlap_audit.csv', baseline.META_PATH,
               baseline.STATE / 'selection_lock.json', baseline.STATE / 'run_signature.json',
               ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv', Path(baseline.__file__)]
    return result


def signature():
    return dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths()},
                script_sha256=sha(__file__), protocol=json.loads(json.dumps(PROTOCOL)),
                versions=baseline.get_signature()['versions'])


def validate_frozen_baseline():
    """Read locked configs and verify hashes, not fit or select any model."""
    metadata = json.loads(baseline.META_PATH.read_text())
    require(metadata['validation'] == 'passed', 'Baseline was not validated')
    require(metadata['signature'] == baseline.get_signature(), 'Baseline inputs, code or environment changed')
    # Check only artifacts needed for this experiment; hashes do not expose labels.
    for path in paths():
        relative = str(path.relative_to(ROOT))
        expected = metadata['outputs'].get(relative)
        if expected:
            require(sha(path) == expected, 'Baseline artifact changed: ' + relative)
    locked = json.loads((baseline.STATE / 'selection_lock.json').read_text())
    require(locked['selections'] == CONFIGS, 'Source hyperparameters differ from frozen baseline')
    require(locked['protocol'] == baseline.PROTOCOL, 'Frozen pipeline protocol changed')
    selection = pd.read_csv(TABLES / 'tfidf_logistic_model_selection.csv')
    for ds, config in CONFIGS.items():
        selected = selection[selection.dataset.eq(ds) & selection.condition.eq('natural') & selection.selected]
        require(len(selected) == 1, 'Expected exactly one frozen natural-text configuration')
        for key, value in config.items():
            require(selected.iloc[0][key] == value, 'Selection-table configuration mismatch')


def load_features(dataset):
    """Target-facing loader deliberately excludes every label/rating column."""
    prefix = baseline.PREFIXES[dataset]
    d = pd.read_csv(ROOT / f'data/processed/{prefix}_text_modeling_ready.csv', usecols=FEATURE_COLUMNS, keep_default_na=False)
    require(d.primary_modeling_eligible.dtype == bool, 'Eligibility is not boolean')
    d = d[d.primary_modeling_eligible].sort_values('content_uid').reset_index(drop=True)
    require(d.content_uid.is_unique and d.dataset.eq(dataset).all(), 'Dataset identity error')
    require(not d.cross_dataset_overlap_flag.any() and not d.cross_dataset_overlap_group_flag.any(), 'Unresolved cross-dataset overlap')
    require(d.groupby('leakage_group').cv_fold.nunique().le(1).all(), 'Existing grouping integrity failure')
    require((d.holdout_assignment.eq('test') == d.cv_fold.eq(-1)).all(), 'Existing holdout mismatch')
    # Only identity/assignment columns are read from the original holdout manifest.
    old = pd.read_csv(ROOT / f'outputs/splits/{prefix}_text_holdout.csv',
                     usecols=['content_uid', 'leakage_group', 'assignment']).set_index('content_uid')
    require(old.index.is_unique and set(d.content_uid) <= set(old.index), 'Holdout UID mismatch')
    old = old.loc[d.content_uid]
    require(old.assignment.tolist() == d.holdout_assignment.tolist(), 'Holdout reassignment')
    require(old.leakage_group.tolist() == d.leakage_group.tolist(), 'Leakage-group reassignment')
    require(not set(TARGET_COLUMNS) & set(d.columns), 'Target labels loaded into feature view')
    return d


def read_labels(dataset, eligible_uids):
    """Called for source fitting, or after the prediction lock for evaluation."""
    d = pd.read_csv(ROOT / f'data/processed/{baseline.PREFIXES[dataset]}_text_modeling_ready.csv',
                    usecols=['content_uid'] + TARGET_COLUMNS, keep_default_na=False).set_index('content_uid')
    require(d.index.is_unique, 'Duplicate label UID')
    d = d.loc[list(eligible_uids)]
    require(set(d.binary_analysis_target) == {0, 1}, 'Missing class support')
    require(d.binary_analysis_target.eq(d.binary_analysis_label.eq('Fake').astype(int)).all(), 'Label coding mismatch')
    return d


def source_training_view(dataset, feature_view):
    require(feature_view.dataset.eq(dataset).all(), 'Source dataset mixing')
    labels = read_labels(dataset, feature_view.content_uid)
    source = feature_view.copy()
    source['binary_analysis_target'] = labels.binary_analysis_target.to_numpy()
    return source


def vocabulary_diagnostics(model, source_text, target_text, target_matrix):
    v = model['tfidf']
    preprocessor, tokenizer = v.build_preprocessor(), v.build_tokenizer()
    # Unigram membership is well defined because both locked ranges start at 1.
    source_vocabulary = set(v.vocabulary_)
    counts, known, fractions, source_tokens = [], [], [], []
    for text in target_text:
        tokens = tokenizer(preprocessor(text))
        n, k = len(tokens), sum(token in source_vocabulary for token in tokens)
        counts.append(n); known.append(k); fractions.append(k / n if n else 0.)
    for text in source_text:
        source_tokens.append(len(tokenizer(preprocessor(text))))
    nnz = np.asarray(target_matrix.getnnz(axis=1)).ravel()
    cells = int(target_matrix.shape[0]) * int(target_matrix.shape[1])
    row = dict(vocabulary_size=len(source_vocabulary), source_N=len(source_text), target_N=len(target_text),
        target_documents_with_known_token_percent=100 * float(np.mean(np.asarray(known) > 0)),
        target_token_occurrence_coverage_percent=100 * sum(known) / sum(counts) if sum(counts) else 0.,
        target_median_known_token_fraction=float(np.median(fractions)),
        target_empty_token_document_N=int(sum(n == 0 for n in counts)),
        zero_feature_rows=int((nnz == 0).sum()), zero_feature_percent=100 * float(np.mean(nnz == 0)),
        target_nonzero_features=int(target_matrix.nnz), target_total_feature_cells=cells,
        target_feature_sparsity=1 - target_matrix.nnz / cells,
        median_target_nonzero_features=float(np.median(nnz)),
        total_target_token_occurrences=int(sum(counts)), known_target_token_occurrences=int(sum(known)),
        token_definition=PROTOCOL['diagnostic_token_definition'])
    for name, texts, tokens in [('source', source_text, source_tokens), ('target', target_text, counts)]:
        for measure, values in [('characters', [len(t) for t in texts]),
                                ('whitespace_tokens', [len(t.split()) for t in texts]),
                                ('analyzer_unigrams', tokens)]:
            for q, label in [(.05, 'p05'), (.5, 'median'), (.95, 'p95')]:
                row[f'{name}_{measure}_{label}'] = float(np.quantile(values, q))
    return row


def fit_and_predict(source, target_features, direction, condition, config):
    """Pure source fit. Target schema has no labels, and only transform is used."""
    source_name, target_name = DIRECTIONS[direction]
    require(source.dataset.eq(source_name).all() and target_features.dataset.eq(target_name).all(), 'Cross-dataset row mixing')
    require(not set(TARGET_COLUMNS) & set(target_features.columns), 'Target labels entered fitting/prediction interface')
    require(config == CONFIGS[source_name], 'Unfrozen source configuration')
    require(not set(source.content_uid) & set(target_features.content_uid), 'Cross-dataset UID overlap')
    require(not set(source.leakage_group) & set(target_features.leakage_group), 'Cross-dataset group overlap')
    require(source.primary_modeling_eligible.all() and target_features.primary_modeling_eligible.all(), 'Ineligible row in transfer')
    model, fit = baseline.fit_pipeline(source, condition, config)
    require(fit['converged'], 'Source fit did not converge')
    source_text = baseline.texts(source, condition)
    target_text = baseline.texts(target_features, condition)
    before_vocabulary = dict(model['tfidf'].vocabulary_)
    before_idf = model['tfidf'].idf_.copy()
    matrix = model['tfidf'].transform(target_text)
    probability = model['logistic'].predict_proba(matrix)[:, 1]
    require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), 'Invalid transfer probabilities')
    require(before_vocabulary == model['tfidf'].vocabulary_ and np.array_equal(before_idf, model['tfidf'].idf_), 'Target changed source vectorizer')
    p = target_features[['content_uid', 'leakage_group', 'cv_fold', 'holdout_assignment']].copy()
    p['direction'], p['source_dataset'], p['target_dataset'], p['condition'] = direction, source_name, target_name, condition
    p['fake_probability'] = probability
    p['predicted_label'] = (probability >= .5).astype(int)
    fit.update(source_dataset=source_name, target_dataset=target_name, source_N=len(source), target_N=len(target_features),
               source_uid_sha256=digest(source.content_uid), target_uid_sha256=digest(target_features.content_uid),
               target_text_fitted=False, target_labels_available=False, threshold=.5, source_config=config)
    diagnostic = dict(direction=direction, source_dataset=source_name, target_dataset=target_name, condition=condition,
                      **vocabulary_diagnostics(model, source_text, target_text, matrix))
    return dict(model=model, predictions=p, fit=fit, vocabulary_diagnostics=diagnostic)


def prediction_path(direction, condition):
    return STATE / f'frozen_{direction}_{condition}.joblib'


def freeze_all_predictions(feature_views):
    result = {}
    for direction, (source_name, target_name) in DIRECTIONS.items():
        for condition in FIELDS:
            path = prediction_path(direction, condition)
            if path.exists():
                bundle = joblib.load(path)
            else:
                source = source_training_view(source_name, feature_views[source_name])
                bundle = fit_and_predict(source, feature_views[target_name], direction, condition, CONFIGS[source_name])
                baseline.dump(path, bundle)
                logging.info('Frozen %s %s: source N=%s, target N=%s; no target labels joined',
                             direction, condition, len(source), len(feature_views[target_name]))
            require(not set(TARGET_COLUMNS + ['true_binary_label']) & set(bundle['predictions'].columns), 'Unlabeled checkpoint contains target labels')
            require(bundle['fit']['source_uid_sha256'] == digest(feature_views[source_name].content_uid), 'Source membership changed')
            require(bundle['fit']['target_uid_sha256'] == digest(feature_views[target_name].content_uid), 'Target membership changed')
            result[(direction, condition)] = bundle
    lock = dict(protocol_sha256=sha(STATE / 'protocol_lock.json'),
                predictions={str(prediction_path(d, c).relative_to(ROOT)): sha(prediction_path(d, c)) for d, c in result},
                all_predictions_frozen_before_target_label_join=True)
    lock_path = STATE / 'prediction_lock.json'
    if lock_path.exists():
        require(json.loads(lock_path.read_text()) == lock, 'Frozen predictions changed')
    else:
        write_json(lock_path, lock)
    return result, lock_path


def labeled_predictions(bundles, lock_path):
    require(lock_path.exists(), 'Target evaluation attempted before all predictions were frozen')
    lock = json.loads(lock_path.read_text())
    require(lock['all_predictions_frozen_before_target_label_join'], 'Missing prediction freeze guarantee')
    require(len(lock['predictions']) == 4 and len(bundles) == 4, 'Incomplete bidirectional prediction freeze')
    for path, expected in lock['predictions'].items():
        require(sha(ROOT / path) == expected, 'Frozen prediction artifact changed')
    pieces = []
    for (direction, condition), bundle in bundles.items():
        p = bundle['predictions'].copy()
        target = DIRECTIONS[direction][1]
        labels = read_labels(target, p.content_uid)
        p = p.merge(labels.reset_index(), on='content_uid', how='left', validate='one_to_one')
        p = p.rename(columns={'binary_analysis_target': 'true_binary_label'})
        pieces.append(p)
    return pd.concat(pieces, ignore_index=True)


def evaluation_results(predictions):
    rows = []
    for (direction, condition), all_target in predictions.groupby(['direction', 'condition'], sort=True):
        for scope, p in [('all_eligible', all_target), ('same_target_holdout', all_target[all_target.holdout_assignment.eq('test')])]:
            require(set(p.true_binary_label) == {0, 1}, 'Target evaluation partition lacks a class')
            measured = baseline.metrics(p.true_binary_label, p.fake_probability)
            actual, predicted, mean_probability = p.true_binary_label.mean(), p.predicted_label.mean(), p.fake_probability.mean()
            gap = mean_probability - actual
            warning = ('Mean Fake probability differs from actual prevalence by at least 10 percentage points; '
                       'descriptive marginal calibration warning, not a full calibration assessment.') if abs(gap) >= .10 else 'No >=10-point marginal gap; this does not establish calibration.'
            rows.append(dict(direction=direction, source_dataset=DIRECTIONS[direction][0], target_dataset=DIRECTIONS[direction][1],
                condition=condition, scope=scope, **measured, threshold=.5,
                actual_fake_prevalence=float(actual), predicted_fake_prevalence=float(predicted),
                mean_fake_probability=float(mean_probability), mean_probability_minus_actual=float(gap),
                predicted_minus_actual_prevalence=float(predicted - actual),
                brier_score=float(brier_score_loss(p.true_binary_label, p.fake_probability)),
                marginal_calibration_warning=warning,
                near_single_class_prediction=bool(predicted >= .95 or predicted <= .05),
                evaluation_uid_sha256=digest(p.content_uid)))
    return pd.DataFrame(rows)


def comparisons(results, predictions):
    # These baseline labels/results are not opened until every transfer prediction is frozen.
    within = pd.read_csv(TABLES / 'tfidf_logistic_holdout_results.csv')
    within_predictions = pd.read_csv(ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv')
    degradation, overlaps = [], []
    for (direction, condition), part in predictions.groupby(['direction', 'condition'], sort=True):
        target = DIRECTIONS[direction][1]
        p = part[part.holdout_assignment.eq('test')].set_index('content_uid')
        w = within_predictions[within_predictions.dataset.eq(target) & within_predictions.condition.eq(condition)].set_index('content_uid')
        require(p.index.is_unique and w.index.is_unique and set(p.index) == set(w.index), 'Same-target-holdout UID mismatch')
        w = w.loc[p.index]
        require(np.array_equal(p.true_binary_label, w.true_binary_label), 'Saved target label mismatch')
        require(np.array_equal(p.leakage_group, w.leakage_group), 'Saved target group mismatch')
        saved = within[within.dataset.eq(target) & within.condition.eq(condition) & within.model.eq('tfidf_logistic')]
        require(len(saved) == 1, 'Missing/duplicate within-target baseline')
        wr = saved.iloc[0]
        # Metrics below are reused, not recomputed by fitting a target model.
        tr = results[results.direction.eq(direction) & results.condition.eq(condition) & results.scope.eq('same_target_holdout')].iloc[0]
        for metric in METRICS:
            degradation.append(dict(direction=direction, target_dataset=target, condition=condition, metric=metric,
                target_holdout_N=len(p), within_target=float(wr[metric]), cross_context=float(tr[metric]),
                transfer_drop=float(tr[metric] - wr[metric]), target_uid_sha256=digest(p.index)))
        tc, wc = p.predicted_label.eq(p.true_binary_label), w.predicted_label.eq(w.true_binary_label)
        categories = np.select([tc & wc, ~tc & ~wc, ~tc & wc, tc & ~wc],
                               ['both_correct', 'both_incorrect', 'only_transfer_wrong', 'only_within_wrong'], default='INVALID')
        for uid, category in zip(p.index, categories):
            overlaps.append(dict(direction=direction, target_dataset=target, condition=condition, content_uid=uid,
                error_category=category, true_binary_label=int(p.loc[uid, 'true_binary_label']),
                transfer_predicted_label=int(p.loc[uid, 'predicted_label']), within_predicted_label=int(w.loc[uid, 'predicted_label']),
                leakage_group=p.loc[uid, 'leakage_group']))
    return pd.DataFrame(degradation), pd.DataFrame(overlaps)


def masking_sensitivity(results):
    rows = []
    for (direction, scope), part in results.groupby(['direction', 'scope'], sort=True):
        part = part.set_index('condition')
        require(part.loc['natural', 'evaluation_uid_sha256'] == part.loc['source_masked', 'evaluation_uid_sha256'], 'Masking cohort mismatch')
        for metric in METRICS:
            natural, masked = float(part.loc['natural', metric]), float(part.loc['source_masked', metric])
            rows.append(dict(direction=direction, scope=scope, metric=metric, natural=natural, masked=masked, masked_minus_natural=masked - natural))
    return pd.DataFrame(rows)


def figures(results, degradation):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'logs/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = ROOT / 'outputs/figures/tfidf_logistic_transfer'
    directory.mkdir(parents=True, exist_ok=True)
    for metric, label in [('macro_f1', 'Macro F1'), ('fake_f1', 'Fake F1')]:
        d = degradation[degradation.metric.eq(metric) & degradation.condition.eq('natural')].set_index('target_dataset').loc[['FakeHealth', 'CoAID']]
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        x = np.arange(2)
        left = ax.bar(x - .18, d.within_target, .36, label='Within target')
        right = ax.bar(x + .18, d.cross_context, .36, label='Cross context')
        ax.bar_label(left, fmt='%.3f', padding=3); ax.bar_label(right, fmt='%.3f', padding=3)
        ax.set(xticks=x, xticklabels=d.index, ylabel=label, ylim=(0, 1.08), title=f'{label}: identical target holdout rows')
        ax.legend(); fig.tight_layout(); fig.savefig(directory / f'within_vs_transfer_{metric}.png', dpi=180); plt.close(fig)
    for direction in DIRECTIONS:
        r = results[results.direction.eq(direction) & results.condition.eq('natural') & results.scope.eq('all_eligible')].iloc[0]
        matrix = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=int)
        fig, ax = plt.subplots(figsize=(4.9, 4.3)); ax.imshow(matrix, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
               xlabel='Predicted', ylabel='True', title=direction.replace('_to_', ' → ') + ': all eligible target rows')
        fig.tight_layout(); fig.savefig(directory / f'{direction}_confusion_matrix.png', dpi=180); plt.close(fig)


def write_report(results, degradation, diagnostics, sensitivity, errors, fit_manifest):
    md = baseline.md
    overlap_counts = errors.groupby(['direction', 'condition', 'error_category']).size().rename('N').reset_index()
    lines = ['# Frozen TF-IDF + Logistic Regression cross-context transfer', '',
        'Only FakeHealth → CoAID and CoAID → FakeHealth were run, each on natural text and the previously frozen source-masked text. No tuning, new classifier, recalibration, threshold change, target-domain fit or engagement analysis was performed.', '',
        '## Protocol and scope', '',
        'Source configurations match the validated baseline: FakeHealth uses word uni/bigrams, min_df=5, C=1; CoAID uses unigrams, min_df=2, C=10. Both use sublinear TF, balanced class weights, L2/liblinear, max_iter=5000, tol=1e-4, float64 TF-IDF and seed 42. The threshold is fixed at probability >=0.5.', '',
        'Each source pipeline was refitted on ALL modeling-ready source rows, including its former within-domain holdout: 1,647 FakeHealth or 3,577 CoAID rows. Within-domain tuning and holdout evaluation had already ended. Source vocabulary and IDF were fitted only on source text. Target feature views exclude labels; the target only passed through transform/predict. Both directions and both conditions were locked before fitting; all four prediction sets were frozen to disk before evaluation labels were joined. A dataset’s labels are used only in its source role for fitting and its target role after prediction freezing for evaluation. Reverse-direction results never alter the other model.', '',
        'Existing cohorts, masking, CV and holdout assignments are unchanged. The current cross-dataset overlap audit is empty and all eligible cross-dataset overlap flags are false. This reuses the existing exact/high-similarity/resource audit; it does not prove absence of every paraphrase or latent dependence. No raw datasets were accessed.', '',
        'WITHIN-CONTEXT means FakeHealth → FakeHealth or CoAID → CoAID, using saved training-pool-to-holdout results. CROSS-CONTEXT means FakeHealth → CoAID or CoAID → FakeHealth, with a full-source refit. External results use the entire eligible target cohort; paired degradation comparisons separately use exactly the existing target holdout UIDs. Those saved target metrics and predictions were reused without retraining or reevaluating target-domain models. The target cohorts are external to the source fits, but are previously studied benchmark datasets, not newly collected blind test sets.', '',
        '## External target performance', '', md(results[['direction', 'condition', 'scope', 'N', 'fake_N', 'real_N'] + METRICS]), '',
        'Confusion matrices use rows=true Real/Fake, columns=predicted Real/Fake: [[TN, FP], [FN, TP]].', '',
        md(results[['direction', 'condition', 'scope', 'tn', 'fp', 'fn', 'tp']]), '',
        '## Transfer degradation on identical target holdout rows', '',
        'transfer_drop = cross_context − within_target. Negative values mean poorer external transfer. Natural transfer is compared to the natural within-target model; masked transfer is compared to the saved masked within-target sensitivity model.', '',
        md(degradation[degradation.metric.isin(COMPARE)][['direction', 'condition', 'metric', 'target_holdout_N', 'within_target', 'cross_context', 'transfer_drop']]), '',
        '## Explicit source masking', '',
        md(sensitivity[sensitivity.metric.isin(COMPARE)]), '',
        'Masking uses the unchanged combined_text_source_masked field and the same source hyperparameters and rows. The prior dictionary used metadata from both cohorts without labels: this is a declared transductive string-masking sensitivity. It removes known explicit identifiers, not publisher style, topic, selection, URL paths, text length or other dataset cues. Legitimate institutional mentions may be removed and the SOURCE token remains. These differences are not estimates of the entire publisher-confounding effect.', '',
        '## Vocabulary and document-length shift', '',
        md(diagnostics[['direction', 'condition', 'vocabulary_size', 'target_documents_with_known_token_percent', 'target_token_occurrence_coverage_percent',
                        'target_median_known_token_fraction', 'zero_feature_percent', 'target_feature_sparsity']]), '',
        md(diagnostics[['direction', 'condition', 'source_characters_median', 'target_characters_median', 'source_analyzer_unigrams_median', 'target_analyzer_unigrams_median']]), '',
        'Token coverage counts occurrences of lowercased default word-tokenizer unigrams (length >=2) in the fitted source vocabulary, rather than mixing unigram and bigram counts. Fractions use each document’s tokenizer output; empty-token documents have fraction zero and are counted separately. Known-token presence and actual nonzero TF-IDF rows are reported separately. Sparsity is 1 − nnz/(target N × source vocabulary size); high sparsity is normal in text models. Vocabulary/IDF were not updated in response to diagnostics. Full p05/median/p95 length summaries and nonzero counts are saved in the diagnostics CSV.', '',
        '## Prediction bias and errors', '',
        md(results[['direction', 'condition', 'scope', 'actual_fake_prevalence', 'predicted_fake_prevalence', 'mean_fake_probability', 'mean_probability_minus_actual', 'brier_score', 'near_single_class_prediction']]), '',
        'A prespecified descriptive marginal-calibration warning flags |mean predicted Fake probability − actual Fake prevalence| >=0.10. This is not a full calibration assessment or statistical test. Near-single-class prediction means >=95% predicted as either class. No target recalibration or threshold repair was performed. Brier scores are evaluation diagnostics only; class weighting can shift probability calibration.', '',
        md(overlap_counts), '',
        'UIDs and the four mutually exclusive error categories are saved for later qualitative review, without article text.', '',
        '## Interpretation and limitations', '']
    for direction in DIRECTIONS:
        r = results[results.direction.eq(direction) & results.condition.eq('natural') & results.scope.eq('all_eligible')].iloc[0]
        drops = degradation[degradation.direction.eq(direction) & degradation.condition.eq('natural')].set_index('metric')
        mask = sensitivity[sensitivity.direction.eq(direction) & sensitivity.scope.eq('all_eligible') & sensitivity.metric.eq('macro_f1')].iloc[0]
        lines += [f'{direction.replace("_to_", " → ")}: external macro F1 {r.macro_f1:.4f}, balanced accuracy {r.balanced_accuracy:.4f}, Fake F1 {r.fake_f1:.4f}; identical-holdout macro-F1 change {drops.loc["macro_f1", "transfer_drop"]:+.4f}. Predicted Fake prevalence is {r.predicted_fake_prevalence:.1%} versus actual {r.actual_fake_prevalence:.1%}. Masked external macro-F1 change is {mask.masked_minus_natural:+.4f}. {r.marginal_calibration_warning}', '']
    natural_drops = degradation[degradation.condition.eq('natural') & degradation.metric.eq('macro_f1')]
    if natural_drops.transfer_drop.lt(-.1).all():
        lines += ['Both directions lose more than 0.10 macro F1 on identical target holdout rows. Together with class-bias and vocabulary/length diagnostics, this supports substantial dependence on dataset/context/domain in these models. This descriptive result does not identify which component caused the decline.', '']
    else:
        lines += ['Read the paired metric changes direction by direction; transfer is not assumed to fail equally or through the same mechanism. These point estimates do not isolate causal mechanisms or provide uncertainty intervals.', '']
    lines += [
        'FakeHealth and CoAID have different ground-truth definitions: author review rating <3 versus >=3 in FakeHealth, and original dataset Fake/Real labels in CoAID. Publisher/source composition differs, and CoAID source/class association is extremely strong. Within-target performance can rely on recurring source and collection cues that do not travel across datasets.', '',
        'FakeHealth generally has much longer article bodies; CoAID bodies are generally truncated near 500 characters. Topics and collection processes also differ. Transfer therefore combines period, dataset, source, label-definition, text-length and topic shifts. A degradation is not automatically a COVID-period effect, evidence about sensitive periods generally, or a causal explanation of misinformation diffusion.', '',
        'The full-source transfer models and training-pool-only within-target baselines have different training populations and sample sizes; only evaluation rows are matched. Natural and masked differences are descriptive, not significance tests. No new bootstrap or model search was performed. The primary engagement results remain frozen.', '',
        '## Validation and artifacts', '',
        f'All four source refits converged: {bool(fit_manifest.converged.all())}. Source-only vocabulary/IDF fitting, immutable target transforms, label-free prediction checkpoints, fixed threshold, full target coverage, identical holdout UIDs and unique output keys are checked. Original inputs and baseline artifacts are fingerprinted. Completed runs verify saved hashes and reuse results.', '',
        'Reproduce: `.venv/bin/python src/run_tfidf_logistic_cross_context_transfer.py`. Tests: `.venv/bin/python -m unittest discover -s src -p "test_tfidf_logistic_cross_context_transfer.py"`. Progress is written to logs/tfidf_logistic_cross_context_transfer.log.', '',
        'Tables: outputs/tables/tfidf_logistic_transfer_{results,degradation,vocabulary_diagnostics,source_mask_sensitivity,error_overlap,error_overlap_summary,fit_manifest}.csv. Predictions: outputs/predictions/tfidf_logistic_cross_context_predictions.csv. Figures: outputs/figures/tfidf_logistic_transfer/. Fitted source models and unlabeled prediction locks: outputs/models/tfidf_logistic_transfer/. Adjacent report JSON stores fingerprints and validation status.', '']
    REPORT.write_text('\n'.join(lines))


def validate_outputs(results, degradation, diagnostics, sensitivity, errors, predictions, feature_views):
    for frame, keys in [(results, ['direction', 'condition', 'scope']), (degradation, ['direction', 'condition', 'metric']),
                        (diagnostics, ['direction', 'condition']), (sensitivity, ['direction', 'scope', 'metric']),
                        (errors, ['direction', 'condition', 'content_uid']), (predictions, ['direction', 'condition', 'content_uid'])]:
        require(not frame.duplicated(keys).any(), 'Duplicate output keys: ' + str(keys))
    require(len(results) == 8 and len(diagnostics) == 4, 'Missing transfer conditions')
    require(np.isfinite(predictions.fake_probability).all(), 'Nonfinite probabilities')
    require(predictions.fake_probability.between(0, 1).all(), 'Probability out of bounds')
    require(np.array_equal(predictions.predicted_label, (predictions.fake_probability >= .5).astype(int)), 'Threshold changed')
    for direction, (_, target) in DIRECTIONS.items():
        eligible = feature_views[target]
        for condition in FIELDS:
            p = predictions[predictions.direction.eq(direction) & predictions.condition.eq(condition)]
            require(len(p) == len(eligible) and set(p.content_uid) == set(eligible.content_uid), 'Target coverage mismatch')
            e = errors[errors.direction.eq(direction) & errors.condition.eq(condition)]
            require(set(e.content_uid) == set(eligible.loc[eligible.holdout_assignment.eq('test'), 'content_uid']), 'Error-overlap holdout mismatch')
            require(set(e.error_category) <= {'both_correct', 'both_incorrect', 'only_transfer_wrong', 'only_within_wrong'}, 'Invalid error category')
            for scope, subset in [('all_eligible', p), ('same_target_holdout', p[p.holdout_assignment.eq('test')])]:
                r = results[results.direction.eq(direction) & results.condition.eq(condition) & results.scope.eq(scope)].iloc[0]
                for key, value in baseline.metrics(subset.true_binary_label, subset.fake_probability).items():
                    require(np.isclose(r[key], value), 'Saved metrics do not match predictions')


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / 'logs/tfidf_logistic_cross_context_transfer.log', level=logging.INFO,
                        format='%(asctime)s %(message)s', force=True)
    validate_frozen_baseline()
    sig = signature()
    if META.exists():
        old = json.loads(META.read_text())
        require(old['signature'] == sig, 'Completed transfer inputs/code/protocol changed; refusing automatic rerun')
        for path, expected in old['outputs'].items():
            require(sha(ROOT / path) == expected, 'Transfer artifact changed: ' + path)
        print('Validated transfer outputs reused; no models refitted or predictions repeated.')
        return
    lock_path = STATE / 'protocol_lock.json'
    if lock_path.exists():
        require(json.loads(lock_path.read_text()) == sig, 'Incomplete transfer protocol changed')
    else:
        write_json(lock_path, sig)
    overlap = pd.read_csv(TABLES / 'text_cross_dataset_overlap_audit.csv')
    require(overlap.empty, 'Cross-dataset overlaps require review before transfer')
    feature_views = {ds: load_features(ds) for ds in baseline.PREFIXES}
    logging.info('Both directions and conditions locked; no target-label feedback permitted')
    bundles, prediction_lock = freeze_all_predictions(feature_views)
    logging.info('All four prediction sets frozen; opening evaluation labels')
    predictions = labeled_predictions(bundles, prediction_lock)
    results = evaluation_results(predictions)
    degradation, errors = comparisons(results, predictions)
    sensitivity = masking_sensitivity(results)
    diagnostics = pd.DataFrame([b['vocabulary_diagnostics'] for b in bundles.values()])
    fit_manifest = pd.DataFrame([dict(direction=d, condition=c, **b['fit']) for (d, c), b in bundles.items()])
    validate_outputs(results, degradation, diagnostics, sensitivity, errors, predictions, feature_views)
    error_summary = errors.groupby(['direction', 'target_dataset', 'condition', 'error_category']).size().rename('N').reset_index()
    for name, frame in [('results', results), ('degradation', degradation), ('vocabulary_diagnostics', diagnostics),
                        ('source_mask_sensitivity', sensitivity), ('error_overlap', errors),
                        ('error_overlap_summary', error_summary), ('fit_manifest', fit_manifest)]:
        frame.to_csv(TABLES / f'tfidf_logistic_transfer_{name}.csv', index=False)
    pred_path = ROOT / 'outputs/predictions/tfidf_logistic_cross_context_predictions.csv'
    predictions.to_csv(pred_path, index=False)
    figures(results, degradation)
    write_report(results, degradation, diagnostics, sensitivity, errors, fit_manifest)
    require(sig == signature(), 'Inputs/code changed during transfer')
    artifacts = [*TABLES.glob('tfidf_logistic_transfer_*.csv'), pred_path, REPORT,
                 *STATE.glob('*.json'), *STATE.glob('*.joblib'), *(ROOT / 'outputs/figures/tfidf_logistic_transfer').glob('*.png')]
    write_json(META, dict(signature=sig, outputs={str(p.relative_to(ROOT)): sha(p) for p in sorted(artifacts)},
        validation='passed', source_only_fit=True, target_labels_joined_after_predictions_frozen=True,
        all_refits_converged=bool(fit_manifest.converged.all()), refit_count=4, threshold=.5,
        prediction_lock_sha256=sha(prediction_lock), existing_baseline_tuning_reused=True))
    logging.info('Transfer completed; validation passed; frozen baseline and cohorts unchanged')
    print(results[results.condition.eq('natural') & results.scope.eq('all_eligible')][['direction', 'macro_f1', 'balanced_accuracy', 'fake_f1', 'roc_auc', 'average_precision']].to_json(orient='records'))
    print('Validation passed. Report: outputs/reports/tfidf_logistic_cross_context_transfer.md')


if __name__ == '__main__':
    main()
