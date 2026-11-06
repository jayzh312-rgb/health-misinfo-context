"""Checks for the fixed four-fit scope and paired-seed aggregation."""
import inspect
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import run_distilroberta_budget_seed_stability as check


class ProtocolTests(unittest.TestCase):
    def test_installed_versions_match_pinned_environment(self):
        from importlib.metadata import version
        for line in (check.ROOT/'requirements-transformer.txt').read_text().splitlines():
            if '==' in line and not line.startswith('#'):
                package,pin=line.split('==',1)
                self.assertEqual(version(package),pin)

    def test_target_labels_rejected_by_fit_guard(self):
        with patch.object(check,'READ_LABELS') as labels:
            with self.assertRaises(ValueError):check.source_labels('CoAID',['target'])
            labels.assert_not_called()
            check.source_labels('FakeHealth',['source'])
            labels.assert_called_once_with('FakeHealth',['source'])

    def test_fixed_new_seed_scope_and_freeze_order(self):
        run=inspect.getsource(check.main)
        self.assertIn('for seed in (123,2026):',run)
        self.assertIn("base.fit('FakeHealth',source.content_uid.tolist(),features",run)
        self.assertNotIn('base.run_experiment(',run)
        score=inspect.getsource(check.evaluate)
        self.assertLess(score.index("sha(path)!=meta['prediction_sha256']"),score.index("READ_LABELS('CoAID'"))
        self.assertEqual(check.SEEDS,(42,123,2026));self.assertEqual(check.BUDGETS,(512,256))

    def test_paired_summary_keeps_seed_identity_and_sample_sd(self):
        rows=[];pred=[]
        expected=[.2,-.1,.3]
        for seed,change in zip(check.SEEDS,expected):
            for cap,value in [(512,.4),(256,.4+change)]:
                for scope in ('all_eligible','same_target_holdout'):
                    rows.append(dict(seed=seed,max_length=cap,scope=scope,**{m:value for m in check.PAIRED_METRICS}))
                for uid,prob in [('a',.2),('b',.8)]:
                    pred.append(dict(seed=seed,max_length=cap,content_uid=uid,cv_fold=-1,fake_probability=prob,predicted_label=int(prob>=.5)))
        with patch.object(check,'table'),patch.object(check.base,'write_json'):
            deltas,summary,agreement=check.summarize(pd.DataFrame(rows),pd.DataFrame(pred))
        d=deltas[deltas.scope.eq('same_target_holdout')&deltas.metric.eq('macro_f1')].sort_values('seed')
        np.testing.assert_allclose(d.delta_256_minus_512,expected,atol=1e-15)
        s=summary[summary.kind.eq('paired_delta')&summary.scope.eq('same_target_holdout')&summary.metric.eq('macro_f1')].iloc[0]
        self.assertAlmostEqual(s['mean'],sum(expected)/3)
        self.assertAlmostEqual(s.sample_sd,np.std(expected,ddof=1))
        self.assertEqual(s.positive_seeds,2)
        self.assertTrue(all(a['identical_class_pct']==100 for a in agreement))

    def test_empirical_target_identity_and_prior_protocol_lock(self):
        audit=json.loads((check.MODELS/'target_content_audit.json').read_text())
        self.assertTrue(audit['identical_unpadded_content'])
        self.assertEqual(audit['packed_256_sha256'],audit['packed_512_sha256'])
        self.assertEqual(audit['target_N'],3577);self.assertLessEqual(audit['maximum_unpadded_length'],256)
        state=json.loads((check.MODELS/'protocol_lock.json').read_text())
        self.assertEqual(check.sha(check.base.__file__),state['signature']['original_script_sha256'])
        self.assertEqual(state['signature']['settings'],check.ORIGINAL_SETTINGS)
        self.assertEqual(state['signature']['config'],{'learning_rate':2e-5,'epochs':3})


class CompletedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not check.REPORT.with_suffix('.json').exists():raise unittest.SkipTest('Fits still running')
        cls.meta=json.loads(check.REPORT.with_suffix('.json').read_text())

    def test_exact_seed42_reuse_and_all_seed_coverage(self):
        r=pd.read_csv(check.TABLES/'distilroberta_budget_seed_results.csv',float_precision='round_trip')
        old=pd.read_csv(check.TABLES/'distilroberta_transfer_results.csv',float_precision='round_trip')
        self.assertEqual(len(r),12)
        for row in r[r.seed.eq(42)].itertuples():
            original=old[old.direction.eq('FH_to_CA')&old.max_length.eq(row.max_length)&old.scope.eq(row.scope)].iloc[0]
            for metric in check.PAIRED_METRICS:self.assertEqual(getattr(row,metric),original[metric])
        p=pd.read_csv(check.PREDICTIONS)
        self.assertEqual(len(p),21462);self.assertFalse(p.duplicated(['seed','max_length','content_uid']).any())
        target=pd.read_csv(check.ROOT/'data/processed/coaid_text_modeling_ready.csv',usecols=['content_uid','primary_modeling_eligible','cv_fold'])
        target=target[target.primary_modeling_eligible].set_index('content_uid')
        for _,part in p.groupby(['seed','max_length']):
            self.assertEqual(set(part.content_uid),set(target.index))
            self.assertEqual(part.cv_fold.tolist(),target.loc[part.content_uid,'cv_fold'].tolist())
            self.assertTrue(part.predicted_label.eq(part.fake_probability.ge(.5).astype(int)).all())

    def test_four_new_fits_and_unchanged_prior_outputs(self):
        manifest=pd.read_csv(check.TABLES/'distilroberta_budget_seed_fit_manifest.csv')
        self.assertEqual(len(manifest),6);self.assertEqual((~manifest.reused).sum(),4)
        self.assertEqual(set(manifest[~manifest.reused].seed),{123,2026})
        self.assertTrue(manifest.fit_fingerprint.is_unique)
        self.assertTrue(manifest.source_uid_sha256.nunique()==1)
        state=json.loads((check.MODELS/'protocol_lock.json').read_text())
        self.assertEqual(check.prior_snapshot(),state['prior_outputs'])
        self.assertEqual(check.base.old_environment_fingerprint(),state['classical_environment'])
        for path,value in self.meta['outputs'].items():self.assertEqual(check.sha(check.ROOT/path),value)
        self.assertTrue(self.meta['validation']['passed'])


if __name__=='__main__':unittest.main()
