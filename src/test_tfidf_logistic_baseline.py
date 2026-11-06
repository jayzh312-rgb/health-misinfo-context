"""Unit leakage guards plus integration checks against saved baseline artifacts."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import run_tfidf_logistic_baseline as baseline


class ProtocolTests(unittest.TestCase):
    def test_grid_and_pipeline(self):
        self.assertEqual(len(baseline.GRID), 24)
        model = baseline.make_pipeline(baseline.GRID[0])
        self.assertEqual(list(model.named_steps), ['tfidf', 'logistic'])
        self.assertEqual(model['logistic'].solver, 'liblinear')
        self.assertEqual(model['logistic'].l1_ratio, 0)
        self.assertEqual(model['logistic'].random_state, 42)
        self.assertEqual(model['tfidf'].min_df, 2)

    def test_vocabulary_training_only_and_predictor_allowlist(self):
        train = pd.DataFrame(dict(
            combined_text=['clinical study benefit'] * 4 + ['fabricated claim miracle'] * 4,
            combined_text_source_masked=['clinical study benefit'] * 4 + ['fabricated claim miracle'] * 4,
            binary_analysis_target=[0] * 4 + [1] * 4,
            content_uid=[f'train{i}' for i in range(8)], leakage_group=[f'g{i}' for i in range(8)],
            publisher_source=['poisonpublisher'] * 8, original_review_rating=['poisonrating'] * 8,
            engagement=['poisonengagement'] * 8, provenance=['poisonprovenance'] * 8))
        model, fit = baseline.fit_pipeline(train, 'natural', baseline.GRID[0])
        self.assertTrue(fit['converged'])
        vocabulary = dict(model['tfidf'].vocabulary_)
        model.predict_proba(['holdoutonlytoken clinical study', 'validationonlytoken fabricated claim'])
        self.assertEqual(vocabulary, model['tfidf'].vocabulary_)
        self.assertEqual(set(vocabulary), {'clinical', 'study', 'benefit', 'fabricated', 'claim', 'miracle'})
        np.testing.assert_array_equal(baseline.texts(train, 'natural'), train.combined_text)
        poisoned = train.copy()
        for column in set(train.columns) - set(baseline.FIELDS.values()):
            poisoned[column] = 'POISON'
        np.testing.assert_array_equal(baseline.texts(train, 'natural'), baseline.texts(poisoned, 'natural'))
        with self.assertRaises(KeyError):
            baseline.texts(train, 'publisher_source')

    def test_tie_break_is_deterministic(self):
        rows = [dict(c, dataset='FakeHealth', condition='natural', all_converged=True,
                     macro_f1_mean=.7, macro_f1_sd=.03) for c in baseline.GRID]
        table = pd.DataFrame(rows)
        chosen = baseline.select_config(table)
        self.assertEqual((chosen['ngram_max'], chosen['C'], chosen['class_weight'], chosen['min_df']), (1, .1, 'none', 5))
        self.assertEqual(chosen, baseline.select_config(table.sample(frac=1, random_state=8)))
        table.loc[23, 'macro_f1_mean'] = .702
        self.assertEqual(baseline.select_config(table)['config_id'], 'cfg_23')
        table.loc[23, 'macro_f1_mean'] = .7005
        table.loc[23, 'macro_f1_sd'] = .02
        self.assertEqual(baseline.select_config(table)['config_id'], 'cfg_23')
        table.loc[23, 'all_converged'] = False
        self.assertNotEqual(baseline.select_config(table)['config_id'], 'cfg_23')

    def test_holdout_requires_selection_before_fit(self):
        with TemporaryDirectory() as directory, patch.object(baseline, 'fit_pipeline') as fit:
            with self.assertRaisesRegex(ValueError, 'before configuration lock'):
                baseline.evaluate_final('FakeHealth', pd.DataFrame(), 'natural', baseline.GRID[0], Path(directory) / 'absent.json')
            fit.assert_not_called()

    def test_frozen_partition_guards(self):
        for ds in baseline.PREFIXES:
            d = baseline.load_cohort(ds)
            holdout = d[d.cv_fold.eq(-1)]
            for train, valid in baseline.cv_partitions(d):
                baseline.assert_partition(train, valid, holdout)
                self.assertTrue(train.cv_fold.ge(0).all())
                self.assertEqual(set(train.binary_analysis_target), {0, 1})
                self.assertEqual(set(valid.binary_analysis_target), {0, 1})
                for condition in baseline.FIELDS:
                    self.assertEqual(len(baseline.texts(train, condition)), len(train))
                    self.assertEqual(len(baseline.texts(valid, condition)), len(valid))
            with self.assertRaisesRegex(ValueError, 'Holdout entered training'):
                baseline.assert_partition(holdout, d[d.cv_fold.eq(0)])
            train, valid = next(baseline.cv_partitions(d))
            invalid = valid.copy()
            invalid.loc[invalid.index[0], 'leakage_group'] = train.leakage_group.iloc[0]
            with self.assertRaisesRegex(ValueError, 'Leakage-group overlap'):
                baseline.assert_partition(train, invalid)


class SavedBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not baseline.META_PATH.exists():
            raise unittest.SkipTest('Baseline not yet run; unit/preflight checks only')
        cls.meta = json.loads(baseline.META_PATH.read_text())
        cls.cohorts = {ds: baseline.load_cohort(ds) for ds in baseline.PREFIXES}
        cls.cv, cls.selection, cls.holdout, cls.differences, cls.features = [
            pd.read_csv(baseline.TABLES / f'tfidf_logistic_{name}.csv')
            for name in ['cv_results', 'model_selection', 'holdout_results', 'source_mask_sensitivity', 'top_features']]
        cls.predictions = pd.read_csv(baseline.ROOT / 'outputs/predictions/tfidf_logistic_holdout_predictions.csv')

    def test_fingerprints_and_output_keys(self):
        self.assertEqual(self.meta['signature'], baseline.get_signature())
        for path, digest in self.meta['outputs'].items():
            self.assertEqual(baseline.sha(baseline.ROOT / path), digest, path)
        baseline.validate_outputs(self.cv, self.selection, self.holdout, self.predictions, self.features, self.differences, self.cohorts)
        self.assertEqual(len(self.features), 120)
        self.assertFalse(set(baseline.FIELDS.values()) & set(self.predictions.columns))
        self.assertEqual(len(self.predictions), 2 * (330 + 702))

    def test_all_fit_memberships_and_masked_configuration_parity(self):
        for ds, d in self.cohorts.items():
            selected = self.selection[self.selection.dataset.eq(ds) & self.selection.selected].iloc[0]
            for fold, (train, valid) in enumerate(baseline.cv_partitions(d)):
                rows = self.cv[self.cv.dataset.eq(ds) & self.cv.fold.eq(fold)]
                self.assertEqual(len(rows), 25)
                self.assertTrue(rows.train_uid_sha256.eq(baseline.digest(train.content_uid)).all())
                self.assertTrue(rows.validation_uid_sha256.eq(baseline.digest(valid.content_uid)).all())
                self.assertFalse(rows.holdout_used.any())
                masked = rows[rows.condition.eq('source_masked')].iloc[0]
                for key in ['config_id', 'ngram_max', 'min_df', 'C', 'class_weight']:
                    self.assertEqual(masked[key], selected[key])
            for condition in baseline.FIELDS:
                row = self.holdout[self.holdout.dataset.eq(ds) & self.holdout.condition.eq(condition)].iloc[0]
                self.assertEqual(row.train_uid_sha256, baseline.digest(d.loc[d.cv_fold.ge(0), 'content_uid']))
                self.assertEqual(row.evaluation_uid_sha256, baseline.digest(d.loc[d.cv_fold.eq(-1), 'content_uid']))
                self.assertEqual(row.selection_lock_sha256, self.meta['selection_lock_sha256'])
                self.assertTrue(row.converged)
                self.assertLess(row.n_iter, 5000)

    def test_selection_reconstructed_from_natural_cv_only(self):
        locked = json.loads((baseline.STATE / 'selection_lock.json').read_text())
        for ds in baseline.PREFIXES:
            natural = self.selection[self.selection.dataset.eq(ds) & self.selection.condition.eq('natural')]
            self.assertEqual(baseline.select_config(natural), locked['selections'][ds])
        self.assertTrue(self.meta['selection_before_holdout'])
        self.assertEqual(self.meta['holdout_evaluations_per_dataset_condition'], 1)

    def test_final_vocabulary_excludes_tokens_seen_only_in_holdout(self):
        # Inspect fitted state; never call predict again on held-out data.
        for ds, prefix in baseline.PREFIXES.items():
            d = self.cohorts[ds]
            for condition in baseline.FIELDS:
                final = joblib.load(baseline.STATE / f'final_{prefix}_{condition}.joblib')
                vectorizer = final['model']['tfidf']
                analyzer = vectorizer.build_analyzer()
                train = d[d.cv_fold.ge(0)]
                counts = {}
                for text in baseline.texts(train, condition):
                    for term in set(analyzer(text)):
                        counts[term] = counts.get(term, 0) + 1
                expected = {term for term, count in counts.items() if count >= vectorizer.min_df}
                self.assertEqual(set(vectorizer.vocabulary_), expected)
                expected_idf = [1 + np.log((1 + len(train)) / (1 + counts[term]))
                                for term in vectorizer.get_feature_names_out()]
                np.testing.assert_allclose(vectorizer.idf_, expected_idf, rtol=1e-12)
                self.assertTrue(final['result']['converged'])
                self.assertEqual(final['model']['logistic'].classes_.tolist(), [0, 1])

    def test_sensitivity_deltas_and_saved_probabilities(self):
        self.assertTrue(self.predictions.fake_probability.between(0, 1).all())
        np.testing.assert_array_equal(self.predictions.predicted_label, (self.predictions.fake_probability >= .5).astype(int))
        for row in self.differences.itertuples():
            self.assertAlmostEqual(row.masked_minus_natural, row.masked - row.natural, places=12)
        for ds in baseline.PREFIXES:
            p = self.predictions[self.predictions.dataset.eq(ds)]
            natural = p[p.condition.eq('natural')].set_index('content_uid')
            masked = p[p.condition.eq('source_masked')].set_index('content_uid').loc[natural.index]
            self.assertEqual(natural.true_binary_label.tolist(), masked.true_binary_label.tolist())
            self.assertEqual(natural.leakage_group.tolist(), masked.leakage_group.tolist())
            self.assertEqual(natural.cv_fold.tolist(), masked.cv_fold.tolist())


if __name__ == '__main__':
    unittest.main()
