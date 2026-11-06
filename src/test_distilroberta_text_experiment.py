"""Source isolation, selection, packing, metrics, and completed-run checks."""
import ast
import inspect
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import run_distilroberta_text_experiment as exp


class ProtocolTests(unittest.TestCase):
    def test_frozen_classical_evaluation_rows(self):
        cohorts,_=exp.audit.load_assignments()
        old=pd.read_csv(exp.TABLES/'tfidf_linear_svm_vs_logistic.csv')
        for r in old.itertuples():
            c=cohorts[r.target_dataset]
            if r.scope=='same_target_holdout':c=c[c.cv_fold.eq(-1)]
            self.assertEqual(r.N,len(c))
            self.assertEqual(r.evaluation_uid_sha256,exp.uid_hash(c.content_uid))

    def test_pinned_revision_and_packer(self):
        self.assertEqual(exp.REVISION,'fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b')
        t=exp.audit.load_tokenizer()
        self.assertEqual(t.model_max_length,512)
        title=t('A health headline',add_special_tokens=False)['input_ids']
        body=t('Body text '*600,add_special_tokens=False,truncation=False)['input_ids']
        for cap in (256,512):
            ids=exp.audit.pack_title_body(t,title,body,cap)
            self.assertEqual(ids,exp.audit.pack_title_body(t,title,body,cap))
            self.assertEqual(ids[1:len(title)+1],title)
            self.assertEqual(ids,t('A health headline','Body text '*600,max_length=cap,truncation='only_second')['input_ids'])
        with self.assertRaises(ValueError):exp.audit.pack_title_body(t,[10]*509,body,512)

    def test_training_only_weights_and_source_only_fit(self):
        np.testing.assert_allclose(exp.weights_from_training([0,0,0,1]),[2/3,2])
        with self.assertRaises(ValueError):exp.weights_from_training([0,0])
        source=inspect.getsource(exp.fit)
        self.assertIn('y=read_labels(dataset,train_uids)',source)
        self.assertIn('weights=weights_from_training(y)',source)
        self.assertNotIn('target_dataset',source)
        self.assertNotIn('holdout_results',source)
        features=inspect.getsource(exp.pack_features)
        self.assertIn("usecols=['content_uid','title_text','body_text']",features)
        self.assertNotIn('read_labels',features)
        self.assertEqual(exp.SETTINGS['threshold'],.5)

    def test_accumulated_weighted_ce_matches_full_effective_batch(self):
        x=torch.tensor([[.1,.4],[.8,.2],[.3,.9]],requires_grad=True)
        y=torch.tensor([0,0,1]);w=torch.tensor([.75,1.5])
        full=torch.nn.functional.cross_entropy(x,y,weight=w);full.backward();expected=x.grad.clone()
        x.grad=None
        for i in range(3):
            (torch.nn.functional.cross_entropy(x[i:i+1],y[i:i+1],weight=w,reduction='sum')/w[y].sum()).backward()
        torch.testing.assert_close(x.grad,expected)

    def test_metrics_threshold_and_ranking_ties(self):
        m=exp.metrics([0,0,1,1],[.1,.4,.35,.8])
        self.assertEqual(m['roc_auc'],.75)
        self.assertAlmostEqual(m['average_precision'],5/6)
        self.assertEqual((m['tn'],m['fp'],m['fn'],m['tp']),(2,0,1,1))
        tied=exp.metrics([0,1],[.5,.5])
        self.assertEqual(tied['roc_auc'],.5);self.assertEqual(tied['average_precision'],.5)
        self.assertEqual(tied['predicted_fake_prevalence'],1.)

    def test_selection_uses_mean_tolerance_then_sd_epochs_lr(self):
        rows=[]
        for dataset in exp.PREFIXES:
            for lr,epoch,values in [(2e-5,2,[.8]*5),(5e-5,2,[.8]*5),(2e-5,3,[.8005]*5),(5e-5,3,[.79,.79,.8,.81,.812])]:
                for fold,value in enumerate(values):rows.append(dict(dataset=dataset,learning_rate=lr,epoch=epoch,fold=fold,macro_f1=value))
        s=exp.select_configs(pd.DataFrame(rows))
        self.assertTrue(all(r['learning_rate']==2e-5 and r['epochs']==2 for r in s))
        self.assertNotIn('read_labels',inspect.getsource(exp.select_configs))

    def test_freeze_before_observed_label_join(self):
        freeze=inspect.getsource(exp.frozen_prediction)
        self.assertNotIn('read_labels',freeze)
        scoring=inspect.getsource(exp.score_frozen)
        self.assertLess(scoring.index("audit.sha(path)"),scoring.index('read_labels('))
        run=inspect.getsource(exp.run_experiment)
        self.assertLess(run.index('locked_configurations.json'),run.index("'transfer_refit'"))
        self.assertLess(run.index('for cap in (512,256)'),run.index('cap==256'))
        self.assertIn("config['learning_rate'],config['epochs']",run)

    def test_pilot_has_no_optimizer_updates(self):
        source=ast.parse(inspect.getsource(exp.memory_pilot))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(source)))
        p=json.loads((exp.MODELS/'memory_pilot.json').read_text())
        self.assertTrue(p['passed']);self.assertEqual(p['microbatch']*p['accumulation'],16)
        self.assertTrue(all(r['parameters_unchanged'] for r in p['rows'] if r['backward_succeeded']))


class CompletedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (exp.ROOT/'outputs/reports/distilroberta_text_experiment.json').exists():
            raise unittest.SkipTest('Full experiment not finished yet')
        cls.cohorts,cls.inputs=exp.audit.load_assignments()
        cls.meta=json.loads((exp.ROOT/'outputs/reports/distilroberta_text_experiment.json').read_text())
        cls.manifest=pd.read_csv(exp.TABLES/'distilroberta_fit_manifest.csv')
        cls.pred=pd.read_csv(exp.ROOT/'outputs/predictions/distilroberta_predictions.csv')

    def test_inputs_groups_and_previous_outputs_unchanged(self):
        state=json.loads((exp.MODELS/'experiment_state.json').read_text())
        self.assertEqual(self.inputs,state['signature']['inputs'])
        self.assertEqual(exp.old_environment_fingerprint(),state['original_environment_fingerprint'])
        self.assertEqual(exp.prior_output_snapshot(),state['prior_outputs'])
        for c in self.cohorts.values():self.assertTrue(c.groupby('leakage_group').cv_fold.nunique().eq(1).all())

    def test_unique_fits_weights_no_holdout_in_cv_and_sensitivity_reuse(self):
        self.assertTrue(self.manifest.fingerprint.is_unique)
        selection=pd.read_csv(exp.TABLES/'distilroberta_model_selection.csv').set_index('dataset')
        for dataset,c in self.cohorts.items():
            labels=dict(zip(c.content_uid,exp.read_labels(dataset,c.content_uid)))
            for _,part in self.manifest[self.manifest.dataset.eq(dataset)].groupby('fit_fingerprint'):
                r=part.iloc[0]
                train=c[(c.cv_fold.ge(0))&(c.cv_fold.ne(r.fold))] if r.role=='cv' else c[c.cv_fold.ge(0)] if r.role=='within_refit' else c
                self.assertEqual(r.train_uid_sha256,exp.uid_hash(train.content_uid))
                expected=exp.weights_from_training([labels[u] for u in train.content_uid])
                np.testing.assert_allclose([r.class_weight_real,r.class_weight_fake],expected)
                self.assertFalse(part.holdout_used_for_selection.any())
                if r.max_length==256:
                    self.assertEqual(r.learning_rate,selection.loc[dataset,'learning_rate'])
                    self.assertEqual(part.epoch.max(),selection.loc[dataset,'epochs'])
                for checkpoint in part.checkpoint:self.assertTrue((exp.ROOT/checkpoint).exists())
        cv=pd.read_csv(exp.TABLES/'distilroberta_cv_results.csv')
        self.assertEqual(len(cv),40);self.assertFalse(cv.holdout_used.any())
        self.assertEqual(self.manifest[self.manifest.role.eq('cv')].fit_fingerprint.nunique(),20)

    def test_prediction_freezes_coverage_and_recomputed_metrics(self):
        self.assertFalse(self.pred.duplicated(['max_length','direction','content_uid']).any())
        for (cap,direction),p in self.pred.groupby(['max_length','direction']):
            c=self.cohorts[p.target_dataset.iloc[0]]
            target=c[c.cv_fold.eq(-1)] if p.evaluation.iloc[0]=='within' else c
            self.assertEqual(set(p.content_uid),set(target.content_uid))
            self.assertTrue(p.predicted_label.eq(p.fake_probability.ge(.5).astype(int)).all())
        self.assertEqual(len(self.pred),2*(330+702+1647+3577))
        for path in exp.MODELS.glob('fits/*/predictions-*-unlabeled.csv'):
            meta=json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(exp.audit.sha(path),meta['prediction_sha256'])
            self.assertNotIn('true_label',pd.read_csv(path,nrows=0).columns)
            self.assertEqual(meta['expected']['config_lock_sha256'],self.meta['config_lock_sha256'])
        for name in ['holdout_results','transfer_results']:
            table=pd.read_csv(exp.TABLES/f'distilroberta_{name}.csv')
            for r in table.itertuples():
                p=self.pred[self.pred.max_length.eq(r.max_length)&self.pred.direction.eq(r.direction)]
                if r.scope=='same_target_holdout':p=p[p.cv_fold.eq(-1)]
                self.assertEqual(r.evaluation_uid_sha256,exp.uid_hash(p.content_uid))
                measured=exp.metrics(p.true_label,p.fake_probability)
                for metric in exp.METRICS:self.assertAlmostEqual(getattr(r,metric),measured[metric],places=12)

    def test_classical_uid_identity_and_degradation(self):
        compare=pd.read_csv(exp.TABLES/'distilroberta_vs_classical.csv')
        old=pd.read_csv(exp.TABLES/'tfidf_linear_svm_vs_logistic.csv')
        for r in compare.itertuples():
            c=old[old.direction.eq(r.direction)&old.scope.eq(r.scope)&old.metric.eq(r.metric)].iloc[0]
            self.assertEqual(r.evaluation_uid_sha256,c.evaluation_uid_sha256)
            self.assertEqual(r.logistic,c.logistic);self.assertEqual(r.linear_svm,c.linear_svm)
        drops=pd.read_csv(exp.TABLES/'distilroberta_transfer_degradation.csv')
        np.testing.assert_allclose(drops.transfer_minus_within,drops.cross_context-drops.within_target,atol=1e-14)


if __name__=='__main__':unittest.main()
