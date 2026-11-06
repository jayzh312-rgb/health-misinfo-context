"""Source-only training tests and checks of saved transfer artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import run_tfidf_logistic_cross_context_transfer as transfer


def toy_frames():
    source = pd.DataFrame(dict(dataset=['FakeHealth'] * 12, content_uid=[f'source{i}' for i in range(12)],
        primary_modeling_eligible=[True] * 12, leakage_group=[f'sourcegroup{i}' for i in range(12)],
        combined_text=['sourcecommon clinical study'] * 6 + ['sourcecommon fake remedy'] * 6,
        combined_text_source_masked=['sourcecommon clinical study'] * 6 + ['sourcecommon fake remedy'] * 6,
        binary_analysis_target=[0] * 6 + [1] * 6, cv_fold=[-1] * 2 + [0] * 10,
        holdout_assignment=['test'] * 2 + ['train'] * 10,
        engagement=['poisonengagement'] * 12, publisher_source=['poisonpublisher'] * 12))
    target = pd.DataFrame(dict(dataset=['CoAID'] * 2, content_uid=['target0', 'target1'],
        primary_modeling_eligible=[True] * 2, leakage_group=['targetgroup0', 'targetgroup1'],
        combined_text=['targetonlytoken clinical study', 'targetonlytoken alienword'],
        combined_text_source_masked=['targetonlytoken clinical study', 'targetonlytoken alienword'],
        cv_fold=[-1, 0], holdout_assignment=['test', 'train'], engagement=['poisonengagement'] * 2))
    return source, target


class SourceOnlyTests(unittest.TestCase):
    def test_frozen_baseline_and_label_free_target_views(self):
        transfer.validate_frozen_baseline()
        for dataset in transfer.baseline.PREFIXES:
            d = transfer.load_features(dataset)
            self.assertFalse(set(transfer.TARGET_COLUMNS) & set(d.columns))
            self.assertTrue(d.primary_modeling_eligible.all())
        self.assertEqual(transfer.PROTOCOL['threshold'], .5)

    def test_target_never_enters_vectorizer_fit(self):
        source, target = toy_frames()
        fitted_texts = []
        original = TfidfVectorizer.fit_transform
        def record_fit(estimator, raw_documents, y=None):
            fitted_texts.append(list(raw_documents))
            return original(estimator, raw_documents, y)
        with patch.object(TfidfVectorizer, 'fit_transform', record_fit):
            bundle = transfer.fit_and_predict(source, target, 'FH_to_CA', 'natural', transfer.CONFIGS['FakeHealth'])
        self.assertEqual(fitted_texts, [source.combined_text.tolist()])
        vocabulary = set(bundle['model']['tfidf'].vocabulary_)
        self.assertNotIn('targetonlytoken', vocabulary)
        self.assertNotIn('poisonpublisher', vocabulary)
        self.assertNotIn('poisonengagement', vocabulary)
        self.assertFalse(set(transfer.TARGET_COLUMNS) & set(bundle['predictions'].columns))
        self.assertFalse(bundle['fit']['target_labels_available'])
        self.assertFalse(bundle['fit']['target_text_fitted'])
        self.assertEqual(bundle['fit']['source_N'], 12)  # former source holdout is included
        self.assertEqual(bundle['fit']['threshold'], .5)
        self.assertEqual(bundle['vocabulary_diagnostics']['zero_feature_rows'], 1)
        self.assertAlmostEqual(bundle['vocabulary_diagnostics']['target_token_occurrence_coverage_percent'], 40.)

    def test_target_labels_and_wrong_configuration_rejected(self):
        source, target = toy_frames()
        invalid = target.assign(binary_analysis_target=[1, 0])
        with patch.object(transfer.baseline, 'fit_pipeline') as fitter:
            with self.assertRaisesRegex(ValueError, 'Target labels entered'):
                transfer.fit_and_predict(source, invalid, 'FH_to_CA', 'natural', transfer.CONFIGS['FakeHealth'])
            fitter.assert_not_called()
        config = dict(transfer.CONFIGS['FakeHealth'], C=10.)
        with self.assertRaisesRegex(ValueError, 'Unfrozen source configuration'):
            transfer.fit_and_predict(source, target, 'FH_to_CA', 'natural', config)
        source.loc[0, 'dataset'] = 'CoAID'
        with self.assertRaisesRegex(ValueError, 'row mixing'):
            transfer.fit_and_predict(source, target, 'FH_to_CA', 'natural', transfer.CONFIGS['FakeHealth'])

    def test_target_text_changes_cannot_change_source_model(self):
        source, target = toy_frames()
        a = transfer.fit_and_predict(source, target, 'FH_to_CA', 'natural', transfer.CONFIGS['FakeHealth'])
        changed = target.copy()
        changed['combined_text'] = ['wholly different target words', 'unseen completely']
        b = transfer.fit_and_predict(source, changed, 'FH_to_CA', 'natural', transfer.CONFIGS['FakeHealth'])
        self.assertEqual(a['model']['tfidf'].vocabulary_, b['model']['tfidf'].vocabulary_)
        np.testing.assert_array_equal(a['model']['tfidf'].idf_, b['model']['tfidf'].idf_)
        np.testing.assert_allclose(a['model']['logistic'].coef_, b['model']['logistic'].coef_, rtol=0, atol=0)

    def test_evaluation_labels_not_opened_without_complete_prediction_lock(self):
        with TemporaryDirectory() as directory, patch.object(transfer, 'read_labels') as labels:
            with self.assertRaisesRegex(ValueError, 'before all predictions'):
                transfer.labeled_predictions({}, Path(directory) / 'missing.json')
            labels.assert_not_called()


class SavedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not transfer.META.exists():
            raise unittest.SkipTest('Transfer not yet run; preflight only')
        cls.meta = json.loads(transfer.META.read_text())
        cls.features = {ds: transfer.load_features(ds) for ds in transfer.baseline.PREFIXES}
        cls.results, cls.degradation, cls.diagnostics, cls.sensitivity, cls.errors = [
            pd.read_csv(transfer.TABLES / f'tfidf_logistic_transfer_{name}.csv')
            for name in ['results', 'degradation', 'vocabulary_diagnostics', 'source_mask_sensitivity', 'error_overlap']]
        cls.predictions = pd.read_csv(transfer.ROOT / 'outputs/predictions/tfidf_logistic_cross_context_predictions.csv')

    def test_unique_keys_coverage_metrics_and_input_fingerprints(self):
        self.assertEqual(self.meta['signature'], transfer.signature())
        for path, expected in self.meta['outputs'].items():
            self.assertEqual(transfer.sha(transfer.ROOT / path), expected, path)
        transfer.validate_outputs(self.results, self.degradation, self.diagnostics, self.sensitivity,
                                  self.errors, self.predictions, self.features)
        self.assertEqual(len(self.predictions), 2 * (1647 + 3577))
        self.assertEqual(len(self.errors), 2 * (330 + 702))
        self.assertFalse(set(transfer.FIELDS.values()) & set(self.predictions.columns))
        self.assertTrue(np.isfinite(self.predictions.fake_probability).all())
        self.assertTrue(self.results.threshold.eq(.5).all())

    def test_source_only_vocabulary_idf_and_frozen_hyperparameters(self):
        for direction, (source, target) in transfer.DIRECTIONS.items():
            d = self.features[source]
            for condition in transfer.FIELDS:
                bundle = joblib.load(transfer.prediction_path(direction, condition))
                fit, model = bundle['fit'], bundle['model']
                self.assertEqual(fit['source_uid_sha256'], transfer.digest(d.content_uid))
                self.assertEqual(fit['target_uid_sha256'], transfer.digest(self.features[target].content_uid))
                self.assertEqual(fit['source_config'], transfer.CONFIGS[source])
                self.assertEqual(model['logistic'].C, transfer.CONFIGS[source]['C'])
                self.assertEqual(model['logistic'].class_weight, 'balanced')
                self.assertEqual(model['logistic'].solver, 'liblinear')
                self.assertEqual(model['logistic'].l1_ratio, 0)
                self.assertTrue(fit['converged'])
                self.assertFalse(fit['target_labels_available'])
                self.assertFalse(set(transfer.TARGET_COLUMNS) & set(bundle['predictions'].columns))
                vectorizer = model['tfidf']
                analyzer = vectorizer.build_analyzer()
                df = {}
                for text in transfer.baseline.texts(d, condition):
                    for term in set(analyzer(text)):
                        df[term] = df.get(term, 0) + 1
                expected = {term for term, count in df.items() if count >= vectorizer.min_df}
                self.assertEqual(expected, set(vectorizer.vocabulary_))
                expected_idf = [1 + np.log((1 + len(d)) / (1 + df[term])) for term in vectorizer.get_feature_names_out()]
                np.testing.assert_allclose(vectorizer.idf_, expected_idf, rtol=1e-12)

    def test_masked_target_parity_and_identical_holdout_comparison(self):
        saved = pd.read_csv(transfer.ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv')
        for direction, (_, target) in transfer.DIRECTIONS.items():
            p = self.predictions[self.predictions.direction.eq(direction)]
            natural = p[p.condition.eq('natural')].set_index('content_uid')
            masked = p[p.condition.eq('source_masked')].set_index('content_uid').loc[natural.index]
            for col in ['true_binary_label', 'leakage_group', 'holdout_assignment']:
                self.assertEqual(natural[col].tolist(), masked[col].tolist())
            for condition in transfer.FIELDS:
                within = saved[saved.dataset.eq(target) & saved.condition.eq(condition)].set_index('content_uid')
                external = p[p.condition.eq(condition) & p.holdout_assignment.eq('test')].set_index('content_uid')
                self.assertEqual(set(within.index), set(external.index))
                overlaps = self.errors[self.errors.direction.eq(direction) & self.errors.condition.eq(condition)].set_index('content_uid')
                self.assertEqual(set(overlaps.index), set(within.index))
                for row in overlaps.itertuples():
                    tc = row.transfer_predicted_label == row.true_binary_label
                    wc = row.within_predicted_label == row.true_binary_label
                    expected = {(True, True): 'both_correct', (False, False): 'both_incorrect',
                                (False, True): 'only_transfer_wrong', (True, False): 'only_within_wrong'}[tc, wc]
                    self.assertEqual(row.error_category, expected)

    def test_prediction_lock_and_metric_differences(self):
        self.assertTrue(self.meta['target_labels_joined_after_predictions_frozen'])
        self.assertEqual(self.meta['refit_count'], 4)
        lock = json.loads((transfer.STATE / 'prediction_lock.json').read_text())
        self.assertEqual(len(lock['predictions']), 4)
        self.assertTrue(lock['all_predictions_frozen_before_target_label_join'])
        for row in self.degradation.itertuples():
            self.assertAlmostEqual(row.transfer_drop, row.cross_context - row.within_target, places=12)
        for row in self.sensitivity.itertuples():
            self.assertAlmostEqual(row.masked_minus_natural, row.masked - row.natural, places=12)


if __name__ == '__main__':
    unittest.main()
