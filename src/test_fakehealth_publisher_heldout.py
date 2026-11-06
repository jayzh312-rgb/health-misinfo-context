"""Leakage, provenance and metric checks for the completed nested experiment.

No model fitting; final fitted artifacts are reused and fitting is prohibited
where predictions are checked. Run after the experiment completes.
"""
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import run_fakehealth_publisher_heldout as run


class PublisherHeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(run.audit, 'run_audit', side_effect=AssertionError('Audit must not rerun')), \
             patch.object(run.audit, 'build_family_map', side_effect=AssertionError('Ontology must not change')):
            cls.d = run.load_cohort()
        cls.assignments = pd.read_csv(run.OUTER_PATH, keep_default_na=False)
        cls.predictions = pd.read_csv(run.PRED_PATH)
        cls.read = staticmethod(lambda name: pd.read_csv(run.TABLES / f'fakehealth_publisher_heldout_{name}.csv'))

    def test_outer_splits_exact_audited_design_and_barriers(self):
        pd.testing.assert_frame_equal(run.outer_assignments(self.d), self.assignments)
        self.assertEqual(len(self.assignments), 1647)
        self.assertTrue(self.assignments.content_uid.is_unique)
        for fold in range(3):
            tr = self.d[self.assignments.outer_fold.ne(fold)]
            te = self.d[self.assignments.outer_fold.eq(fold)]
            run.check_partition(tr, te, minimum=20)
            self.assertEqual(len(te), 549)
        for key in ['publisher_family','constraint_component','leakage_group']:
            self.assertTrue(self.assignments.groupby(key).outer_fold.nunique().eq(1).all())

    def test_inner_membership_training_only_and_class_support(self):
        cv = self.read('inner_cv')
        for outer in range(3):
            train = self.d[self.assignments.outer_fold.ne(outer)].reset_index(drop=True)
            test_ids = set(self.d[self.assignments.outer_fold.eq(outer)].content_uid)
            parts, _ = run.choose_inner_splits(train)
            self.assertEqual(len(parts), 3)
            for fold,(tr,va) in enumerate(parts):
                run.check_partition(train.iloc[tr], train.iloc[va], minimum=20)
                self.assertFalse(set(train.iloc[tr].content_uid) & test_ids)
                self.assertFalse(set(train.iloc[va].content_uid) & test_ids)
                saved = cv[cv.outer_fold.eq(outer) & cv.inner_fold.eq(fold)]
                self.assertEqual(len(saved), 24)
                self.assertTrue(saved.train_uid_sha256.eq(run.digest(train.iloc[tr].content_uid)).all())
                self.assertTrue(saved.validation_uid_sha256.eq(run.digest(train.iloc[va].content_uid)).all())

    def test_outer_labels_cannot_affect_inner_selection(self):
        cv = self.read('inner_cv')
        summary = run.summarize_inner(cv)
        lock = json.loads((run.STATE/'selection_lock.json').read_text())
        for fold in range(3):
            clean_train = self.d[self.assignments.outer_fold.ne(fold)].reset_index(drop=True)
            poisoned = self.d.copy()
            poisoned.loc[self.assignments.outer_fold.eq(fold),'binary_analysis_target'] = 7
            poisoned_train = poisoned[self.assignments.outer_fold.ne(fold)].reset_index(drop=True)
            pd.testing.assert_frame_equal(clean_train, poisoned_train)
            a,_ = run.choose_inner_splits(clean_train); b,_ = run.choose_inner_splits(poisoned_train)
            for (atr,ava),(btr,bva) in zip(a,b):
                np.testing.assert_array_equal(atr,btr); np.testing.assert_array_equal(ava,bva)
            self.assertEqual(run.base.select_config(summary[summary.outer_fold.eq(fold)]), lock['configs'][str(fold)])
        self.assertFalse(cv.outer_test_used.any())
        self.assertEqual(list(inspect.signature(run.inner_cv).parameters), ['outer_train','outer_fold','splits'])

    def test_inner_fit_interface_uses_only_inner_training_text(self):
        outer = 0
        train = self.d[self.assignments.outer_fold.ne(outer)].reset_index(drop=True)
        parts,_ = run.choose_inner_splits(train)
        seen = []
        def fake_fit(frame, condition, config):
            seen.append(tuple(frame.content_uid))
            return object(), dict(converged=True, train_uid_sha256=run.digest(frame.content_uid))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, 'STATE', Path(directory)), \
             patch.object(run.base, 'GRID', [run.base.GRID[0]]), \
             patch.object(run, 'fit_training', side_effect=fake_fit), \
             patch.object(run, 'predict_unlabeled', side_effect=lambda model,frame,condition: np.full(len(frame),.4)):
            records = run.inner_cv(train, outer, parts)
        self.assertEqual(len(records), 3)
        self.assertEqual(seen, [tuple(train.iloc[tr].content_uid) for tr,_ in parts])
        fit_ast = ast.parse(inspect.getsource(run.fit_training))
        calls = [n for n in ast.walk(fit_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[0]),'texts(train, condition)')

    def test_predictor_allowlist_and_no_source_metadata(self):
        frame = pd.DataFrame({'combined_text':['natural'], 'combined_text_source_masked':['masked'],
            'publisher_family':['SOURCE_POISON'], 'leakage_group':['ID_POISON'],
            'publisher_source':['SOURCE_POISON'], 'title_text':['TITLE_POISON'],
            'binary_analysis_target':[1]})
        self.assertEqual(run.texts(frame,'natural').tolist(),['natural'])
        self.assertEqual(run.texts(frame,'source_masked').tolist(),['masked'])
        self.assertEqual(run.FIELDS, {'natural':'combined_text','source_masked':'combined_text_source_masked'})

    def test_saved_vectorizer_vocabulary_and_idf_are_training_only(self):
        # Verify all six fitted vocabularies/IDFs from document frequencies in
        # outer-training text, with no vectorizer fit and no article output.
        for outer in range(3):
            train = self.d[self.assignments.outer_fold.ne(outer)]
            for condition in run.FIELDS:
                bundle = joblib.load(run.STATE/f'outer{outer}_{condition}.joblib')
                vec = bundle['model']['tfidf']
                analyzer = vec.build_analyzer()
                counts = {}
                for text in run.texts(train,condition):
                    for token in set(analyzer(text)):
                        counts[token] = counts.get(token,0)+1
                expected = {x for x,n in counts.items() if n>=vec.min_df}
                self.assertEqual(set(vec.vocabulary_),expected)
                expected_idf = np.array([np.log((1+len(train))/(1+counts[t]))+1 for t in vec.get_feature_names_out()])
                np.testing.assert_allclose(vec.idf_,expected_idf,rtol=1e-12)
                self.assertEqual(bundle['fit']['train_uid_sha256'],run.digest(train.content_uid))

    def test_target_transform_only_locks_and_threshold(self):
        lock_hash=run.sha(run.STATE/'selection_lock.json')
        lock=json.loads((run.STATE/'selection_lock.json').read_text())
        for outer in range(3):
            for condition in run.FIELDS:
                b=joblib.load(run.STATE/f'outer{outer}_{condition}.joblib')
                self.assertEqual(b['selection_lock_sha256'],lock_hash)
                self.assertEqual(b['selected_config'],lock['configs'][str(outer)])
                self.assertEqual(b['threshold'],.5)
                self.assertNotIn('true_binary_label',b['predictions'])
                self.assertNotIn('binary_analysis_target',b['predictions'])
                np.testing.assert_array_equal(b['predictions'].predicted_label,(b['predictions'].fake_probability>=.5).astype(int))
        b=joblib.load(run.STATE/'outer0_natural.joblib')
        target=self.d[self.assignments.outer_fold.eq(0)].head(2).drop(columns=['binary_analysis_target','binary_analysis_label'])
        with patch.object(Pipeline,'fit',side_effect=AssertionError('target fit')), \
             patch.object(TfidfVectorizer,'fit_transform',side_effect=AssertionError('target fit')), \
             patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('target fit')), \
             patch.object(LogisticRegression,'fit',side_effect=AssertionError('target fit')):
            p=run.predict_unlabeled(b['model'],target,'natural')
        np.testing.assert_allclose(p,b['predictions'].set_index('content_uid').loc[target.content_uid].fake_probability)
        with self.assertRaisesRegex(ValueError,'Labels in prediction'):
            run.predict_unlabeled(b['model'],target.assign(binary_analysis_target=0),'natural')

    def test_one_prediction_mask_parity_convergence_and_metric_accounting(self):
        diagnostics=pd.read_csv(run.TABLES/'fakehealth_publisher_level_diagnostics.csv')
        results=self.read('results')
        run.validate_results(self.d,self.assignments,self.read('inner_cv'),self.read('model_selection'),
            self.predictions,results,self.read('fit_manifest'),diagnostics)
        self.assertTrue(self.read('inner_cv').converged.all())
        for r in results.itertuples():
            p=self.predictions[self.predictions.condition.eq(r.condition)]
            if r.scope=='outer_fold':p=p[p.outer_fold.eq(r.outer_fold)]
            got=run.base.metrics(p.true_binary_label,p.fake_probability)
            for metric in run.METRICS:
                self.assertAlmostEqual(got[metric],getattr(r,metric),places=12)
        s=self.read('summary')
        pd.testing.assert_frame_equal(s,run.summarize_results(results),check_exact=False,rtol=1e-12,atol=1e-12)
        sensitivity=self.read('source_mask_sensitivity')
        np.testing.assert_allclose(sensitivity.source_masked-sensitivity.natural,sensitivity.masked_minus_natural,atol=1e-14)

    def test_previous_artifacts_unchanged_and_complete_outputs(self):
        meta=json.loads(run.REPORT.with_suffix('.json').read_text())
        self.assertEqual(meta['signature'],run.signature())
        for path,expected in {**meta['signature']['inputs'],**meta['outputs']}.items():
            self.assertEqual(run.sha(run.ROOT/path),expected)
        self.assertEqual(meta['inner_fit_count'],216)
        self.assertEqual(meta['outer_fit_count'],6)
        self.assertTrue(meta['previous_inputs_unchanged'])


if __name__=='__main__':
    unittest.main()
