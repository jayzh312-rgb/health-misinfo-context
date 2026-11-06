"""Focused title sensitivity tests; reuse final model/prediction checkpoints."""
import ast
import inspect
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import run_tfidf_logistic_title_sensitivity as title


class TitleSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views = {ds: title.load_features(ds) for ds in title.base.PREFIXES}

    def test_frozen_splits_groups_and_classes(self):
        summary = title.cohort_summary(self.views)
        self.assertTrue(summary.fake_N.gt(0).all() and summary.real_N.gt(0).all())
        for frame in self.views.values():
            self.assertTrue(frame.groupby('leakage_group').cv_fold.nunique().eq(1).all())
        self.assertFalse(set(self.views['FakeHealth'].leakage_group) & set(self.views['CoAID'].leakage_group))
        title.input_signature()  # Includes original split SHA-256 verification.

    def test_only_title_predictor_even_with_poisoned_metadata(self):
        frame = pd.DataFrame({'title_text': ['safe title'], 'title_text_source_masked': ['masked title'],
                              'body_text': ['BODY_POISON'], 'combined_text': ['COMBINED_POISON'],
                              'publisher_source': ['PUBLISHER_POISON'], 'binary_analysis_target': [1]})
        self.assertEqual(title.texts(frame, 'natural').tolist(), ['safe title'])
        self.assertEqual(title.texts(frame, 'source_masked').tolist(), ['masked title'])
        tree = ast.parse(inspect.getsource(title.fit_source))
        fits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit']
        self.assertEqual(len(fits), 1)
        self.assertEqual(ast.unparse(fits[0].args[0]), 'texts(source, condition)')
        self.assertFalse(set(title.FEATURE_COLUMNS) & {'body_text', 'combined_text', 'binary_analysis_target'})

    def test_target_transform_only_and_label_rejection(self):
        model = joblib.load(title.STATE / 'transfer_fakehealth_natural.joblib')['model']
        target = self.views['CoAID'].head(3)
        before_vocabulary = dict(model['tfidf'].vocabulary_)
        before_idf = model['tfidf'].idf_.copy()
        with patch.object(Pipeline, 'fit', side_effect=AssertionError('Unexpected fitting')), \
             patch.object(TfidfVectorizer, 'fit', side_effect=AssertionError('Unexpected fitting')), \
             patch.object(TfidfVectorizer, 'fit_transform', side_effect=AssertionError('Unexpected fitting')), \
             patch.object(LogisticRegression, 'fit', side_effect=AssertionError('Unexpected fitting')):
            prediction, _ = title.predict_features(model, target, 'natural')
        self.assertEqual(len(prediction), 3)
        self.assertEqual(before_vocabulary, model['tfidf'].vocabulary_)
        np.testing.assert_array_equal(before_idf, model['tfidf'].idf_)
        with self.assertRaisesRegex(ValueError, 'Target labels'):
            title.predict_features(model, target.assign(binary_analysis_target=0), 'natural')

    def test_source_only_vocabulary_from_checkpoint(self):
        for prefix, source in [('fakehealth', 'FakeHealth'), ('coaid', 'CoAID')]:
            bundle = joblib.load(title.STATE / f'transfer_{prefix}_natural.joblib')
            vec = bundle['model']['tfidf']
            counts = {}
            analyzer = vec.build_analyzer()
            for text in title.texts(self.views[source], 'natural'):
                for token in set(analyzer(text)):
                    counts[token] = counts.get(token, 0) + 1
            expected = {token for token, count in counts.items() if count >= vec.min_df}
            self.assertEqual(set(vec.vocabulary_), expected)
            expected_idf = np.array([np.log((1+len(self.views[source]))/(1+counts[t]))+1 for t in vec.get_feature_names_out()])
            np.testing.assert_allclose(vec.idf_, expected_idf, rtol=1e-12)
            self.assertEqual(bundle['fit']['train_uid_sha256'], title.digest(self.views[source].content_uid))

    def test_selection_uses_only_source_cv_and_locked_configs(self):
        cv = pd.read_csv(title.TABLES / 'tfidf_logistic_title_cv_results.csv')
        self.assertFalse(cv.holdout_used.any())
        locked = json.loads((title.STATE / 'selection_lock.json').read_text())['selections']
        summary = title.base.aggregate_cv(cv)
        for ds in self.views:
            self.assertEqual(title.base.select_config(summary[summary.dataset.eq(ds)]), locked[ds])
            title.verify_lock(ds, locked[ds])
            other = summary[summary.dataset.ne(ds)].copy()
            other['macro_f1_mean'] = 999.
            altered = pd.concat([summary[summary.dataset.eq(ds)], other])
            self.assertEqual(title.base.select_config(altered[altered.dataset.eq(ds)]), locked[ds])
        with self.assertRaises(ValueError):
            title.verify_lock('FakeHealth', dict(config_id='unlocked'))
        tree = ast.parse(inspect.getsource(title.run_cv))
        self.assertNotIn('holdout_results', ast.unparse(tree))

    def test_prediction_lock_labels_and_threshold(self):
        lock = json.loads((title.STATE / 'prediction_lock.json').read_text())
        self.assertFalse(lock['target_evaluation_labels_joined'])
        self.assertEqual(lock['threshold'], .5)
        self.assertEqual(len(lock['prediction_bundles']), 8)
        for path, expected in lock['prediction_bundles'].items():
            self.assertEqual(title.sha(title.ROOT / path), expected)
            bundle = joblib.load(title.ROOT / path)
            self.assertEqual(bundle['fit']['selection_lock_sha256'], title.sha(title.STATE / 'selection_lock.json'))
            self.assertFalse(bundle['fit']['target_text_fitted'])
            self.assertFalse(bundle['fit']['target_labels_available'])
            p = bundle['predictions']
            self.assertNotIn('true_binary_label', p)
            self.assertNotIn('binary_analysis_target', p)
            np.testing.assert_array_equal(p.predicted_label, (p.fake_probability >= .5).astype(int))
        with patch.object(title, 'STATE', title.ROOT / 'outputs/models/nonexistent_title_lock_test'):
            with self.assertRaisesRegex(ValueError, 'before predictions freeze'):
                title.join_evaluation_labels({})

    def test_complete_unique_outputs_convergence_and_exact_pairing(self):
        read = lambda name: pd.read_csv(title.TABLES / f'tfidf_logistic_title_{name}.csv')
        predictions = pd.read_csv(title.ROOT / 'outputs/predictions/tfidf_logistic_title_predictions.csv')
        title.validate_outputs(read('cv_results'), read('model_selection'), read('holdout_results'),
            read('transfer_results'), predictions, read('vs_combined'), read('transfer_degradation'), self.views, read('fit_manifest'))
        self.assertTrue(read('cv_results').converged.all())
        self.assertEqual(len(read('fit_manifest')), 8)
        change = read('vs_combined')
        self.assertTrue(change.exact_same_target_rows.all())
        np.testing.assert_allclose(change.title-change.combined_same_rows, change.title_minus_combined, atol=1e-14)
        self.assertFalse(any('text' in c or 'title' in c for c in predictions.columns))
        old = pd.read_csv(title.ROOT / 'outputs/predictions/tfidf_logistic_cross_context_predictions.csv')
        for r in change[change.evaluation.eq('transfer')].itertuples():
            current = predictions[predictions.evaluation.eq('transfer') & predictions.direction.eq(r.direction) & predictions.condition.eq(r.condition)]
            if r.scope == 'same_target_holdout':
                current = current[current.cv_fold.eq(-1)]
            prior = old[old.direction.eq(r.direction) & old.condition.eq(r.condition)].set_index('content_uid').loc[current.content_uid]
            value = title.base.metrics(prior.true_binary_label, prior.fake_probability)[r.metric]
            self.assertAlmostEqual(value, r.combined_same_rows, places=12)


if __name__ == '__main__':
    unittest.main()
