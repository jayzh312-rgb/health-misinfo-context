"""Leakage, decision-score, comparator and reproducibility checks; no new fits."""
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import run_tfidf_linear_svm_replication as run


class LinearSVMReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.views={ds:run.load_features(ds) for ds in run.PREFIXES}
        cls.read=staticmethod(lambda name:pd.read_csv(run.TABLES/f'tfidf_linear_svm_{name}.csv'))
        cls.predictions=pd.read_csv(run.PRED_PATH)

    def test_original_splits_and_resource_barriers(self):
        summary=run.preflight(self.views)
        self.assertTrue(summary.fake_N.gt(0).all() and summary.real_N.gt(0).all())
        for frame in self.views.values():
            self.assertTrue(frame.groupby('leakage_group').cv_fold.nunique().eq(1).all())
        run.signature()  # Fingerprints both original CV and holdout manifests.

    def test_decision_threshold_zero_and_ranking_scores(self):
        y=np.array([0,1,1,0]);scores=np.array([-.1,0.,.1,2.])
        got=run.metrics(y,scores)
        self.assertEqual((got['tn'],got['fp'],got['fn'],got['tp']),(1,1,0,2))
        self.assertAlmostEqual(got['accuracy'],.75)
        self.assertAlmostEqual(got['roc_auc'],roc_auc_score(y,scores))
        self.assertAlmostEqual(got['average_precision'],average_precision_score(y,scores))
        self.assertFalse(any('probability' in c or 'calibration' in c for c in self.predictions.columns))

    def test_predictor_allowlist_and_svm_only(self):
        frame=pd.DataFrame({'combined_text':['intended text'],'combined_text_source_masked':['MASKED_POISON'],
            'title_text':['TITLE_POISON'],'publisher_source':['SOURCE_POISON'],'engagement':[999],
            'binary_analysis_target':[1],'cv_fold':[0]})
        self.assertEqual(run.texts(frame).tolist(),['intended text'])
        model=run.make_pipeline(run.GRID[0])
        self.assertIsInstance(model['svm'],LinearSVC)
        self.assertEqual(model['svm'].max_iter,20000)
        self.assertNotIn('combined_text_source_masked',run.FEATURES)
        self.assertFalse(set(run.LABELS)&set(run.FEATURES))
        with patch.object(run,'run',side_effect=lambda:LogisticRegression().fit([[0],[1]],[0,1])):
            with self.assertRaisesRegex(RuntimeError,'LR refitting prohibited'):
                run.main()

    def test_cv_fit_and_label_reads_restricted_to_training_pool(self):
        dataset='FakeHealth';features=self.views[dataset]
        training=features[features.cv_fold.ge(0)]
        original_reader=run.read_labels;read_uids=[];seen=[]
        def spy_labels(ds,uids):
            read_uids.append(set(uids));return original_reader(ds,uids)
        def fake_fit(source,config):
            seen.append(tuple(source.content_uid))
            return object(),dict(converged=True,train_uid_sha256=run.digest(source.content_uid))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run,'STATE',Path(directory)),patch.object(run,'GRID',[run.GRID[0]]), \
             patch.object(run,'read_labels',side_effect=spy_labels),patch.object(run,'fit_source',side_effect=fake_fit), \
             patch.object(run,'predict_unlabeled',side_effect=lambda model,target:np.full(len(target),-.2)):
            rows=run.run_cv(dataset,features)
        self.assertEqual(read_uids,[set(training.content_uid)])
        self.assertEqual(seen,[tuple(training[training.cv_fold.ne(f)].content_uid) for f in range(5)])
        self.assertEqual(len(rows),5)
        fit_ast=ast.parse(inspect.getsource(run.fit_source))
        calls=[n for n in ast.walk(fit_ast) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[0]),'texts(source)')

    def test_selection_ties_and_other_dataset_outcomes_do_not_influence(self):
        rows=[]
        for config in [run.GRID[0],run.GRID[8]]:
            rows.append(dict(**config,all_converged=True,macro_f1_mean=.6,macro_f1_sd=.02))
        # Explicit SVM final config-ID tie-break, not LR's larger-min_df rule.
        self.assertEqual(run.select_config(pd.DataFrame(rows)),run.GRID[0])
        cv=self.read('cv_results');summary=run.summarize_cv(cv)
        lock=json.loads((run.STATE/'selection_lock.json').read_text())
        self.assertEqual(set(lock['configs']),set(run.PREFIXES))
        for dataset in run.PREFIXES:
            altered=summary.copy();altered.loc[altered.dataset.ne(dataset),'macro_f1_mean']=999
            self.assertEqual(run.select_config(altered[altered.dataset.eq(dataset)]),lock['configs'][dataset])
        self.assertFalse(cv.holdout_used.any())

    def test_source_only_vocabulary_and_idf_in_all_final_models(self):
        for evaluation in ['within','transfer']:
            for dataset,prefix in run.PREFIXES.items():
                bundle=joblib.load(run.STATE/f'{evaluation}_{prefix}.joblib')
                source=self.views[dataset]
                if evaluation=='within':source=source[source.cv_fold.ge(0)]
                vec=bundle['model']['tfidf'];analyzer=vec.build_analyzer();counts={}
                for text in run.texts(source):
                    for token in set(analyzer(text)):counts[token]=counts.get(token,0)+1
                self.assertEqual(set(vec.vocabulary_),{t for t,n in counts.items() if n>=vec.min_df})
                expected=np.array([np.log((1+len(source))/(1+counts[t]))+1 for t in vec.get_feature_names_out()])
                np.testing.assert_allclose(vec.idf_,expected,rtol=1e-12)
                self.assertEqual(bundle['fit']['train_uid_sha256'],run.digest(source.content_uid))
                self.assertFalse(bundle['target_text_fitted'])

    def test_target_transform_only_frozen_configs_and_predictions(self):
        lock=json.loads((run.STATE/'prediction_lock.json').read_text())
        self.assertEqual(len(lock['bundles']),4);self.assertFalse(lock['evaluation_labels_joined'])
        self.assertEqual(lock['threshold'],0.)
        for path,expected in lock['bundles'].items():
            self.assertEqual(run.sha(run.ROOT/path),expected)
            b=joblib.load(run.ROOT/path)
            self.assertEqual(b['selection_lock_sha256'],run.sha(run.STATE/'selection_lock.json'))
            self.assertFalse(b['target_labels_in_prediction_interface'])
            self.assertNotIn('true_binary_label',b['predictions'])
            np.testing.assert_array_equal(b['predictions'].predicted_label,(b['predictions'].decision_score>=0).astype(int))
        b=joblib.load(run.STATE/'transfer_fakehealth.joblib');target=self.views['CoAID'].head(2)
        with patch.object(Pipeline,'fit',side_effect=AssertionError('target fitting')), \
             patch.object(TfidfVectorizer,'fit_transform',side_effect=AssertionError('target fitting')), \
             patch.object(TfidfVectorizer,'fit',side_effect=AssertionError('target fitting')), \
             patch.object(LinearSVC,'fit',side_effect=AssertionError('target fitting')):
            score=run.predict_unlabeled(b['model'],target)
        np.testing.assert_allclose(score,b['predictions'].set_index('content_uid').loc[target.content_uid].decision_score)
        with self.assertRaisesRegex(ValueError,'Target labels'):
            run.predict_unlabeled(b['model'],target.assign(binary_analysis_target=0))
        with patch.object(run,'STATE',run.ROOT/'outputs/models/nonexistent_svm_lock_test'):
            with self.assertRaisesRegex(ValueError,'Missing prediction freeze'):
                run.evaluate({})

    def test_complete_unique_outputs_metrics_and_comparator_rows(self):
        within,external=self.read('holdout_results'),self.read('transfer_results')
        comparison,drop=self.read('vs_logistic'),self.read('transfer_degradation')
        run.validate_outputs(self.read('cv_results'),self.read('model_selection'),within,external,self.predictions,
                             self.views,self.read('fit_manifest'),comparison,drop)
        self.assertTrue(self.read('cv_results').converged.all())
        for frame in [within,external]:
            for row in frame.itertuples():
                p=self.predictions[self.predictions.direction.eq(row.direction)]
                if row.scope=='same_target_holdout':p=p[p.cv_fold.eq(-1)]
                measured=run.metrics(p.true_binary_label,p.decision_score)
                for metric in run.COMPARE:self.assertAlmostEqual(measured[metric],getattr(row,metric),places=12)
        rebuilt,rebuilt_drop=run.comparisons(within,external)
        pd.testing.assert_frame_equal(comparison,rebuilt,check_exact=False,rtol=1e-12,atol=1e-12)
        pd.testing.assert_frame_equal(drop,rebuilt_drop,check_exact=False,rtol=1e-12,atol=1e-12)
        self.assertTrue(comparison.same_evaluation_uids.all())

    def test_lr_cohorts_splits_and_saved_artifacts_unchanged(self):
        meta=json.loads(run.REPORT.with_suffix('.json').read_text())
        self.assertEqual(meta['signature'],run.signature())
        for path,expected in {**meta['signature']['inputs'],**meta['outputs']}.items():
            self.assertEqual(run.sha(run.ROOT/path),expected)
        self.assertEqual(meta['cv_fit_count'],320);self.assertEqual(meta['final_fit_count'],4)
        self.assertEqual(meta['logistic_fit_count'],0);self.assertFalse(meta['probability_calibration'])


if __name__=='__main__':
    unittest.main()
