"""Protocol and saved-result checks; no source fitting or output mutation."""
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import run_covid_rumor_external_transfer as run


class ExternalTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target=pd.read_csv(run.COHORT,keep_default_na=False)
        cls.p=pd.read_csv(run.PRED)
        cls.meta=json.loads(run.META.read_text())
        cls.results=pd.read_csv(run.ROOT/'outputs/tables/covid_rumor_external_transfer_results.csv')

    def test_news_binary_item_identity(self):
        t=self.target
        self.assertTrue(t.subset.eq('NEWS').all())
        self.assertTrue(t.target_uid.is_unique and t.rumor_id.is_unique)
        labels=pd.read_csv(run.NEWS/'news.csv',header=None,names=['rumor_id','label','content','sentiment'],usecols=['rumor_id','label']).set_index('rumor_id')
        self.assertTrue(labels.loc[t.rumor_id,'label'].isin(['T','F']).all())
        expected='covid_rumor:NEWS:'+t.rumor_id.astype(str)
        self.assertEqual(t.target_uid.tolist(),expected.tolist())
        self.assertTrue(t.quality_exclusion_reason.eq('').all())

    def test_quality_boundary_and_rules(self):
        d=pd.DataFrame({'predictor_text':['','https://example.com/item','Access denied','A substantive English claim exists.','這是一個中文測試']})
        result=run.quality_view(d)
        self.assertTrue(result.quality_exclusion_reason.iloc[:3].ne('').all())
        self.assertEqual(result.quality_exclusion_reason.iloc[3],'')
        self.assertNotEqual(result.quality_exclusion_reason.iloc[4],'')
        for field in ['label','binary_target']:
            with self.assertRaises(ValueError): run.quality_view(d.assign(**{field:0}))
            with self.assertRaises(ValueError): run.overlap_audit(d.assign(**{field:0}),{})
            with self.assertRaises(ValueError): run.predict_all(d.assign(**{field:0}),{}, {})

    def test_feature_loader_requests_news_text_only(self):
        original=pd.read_csv
        calls=[]
        def reader(path,*args,**kwargs):
            calls.append((path,kwargs.get('usecols')))
            return original(path,*args,**kwargs)
        with patch.object(run.pd,'read_csv',side_effect=reader):
            features=run.read_features()
        self.assertEqual(calls,[(run.NEWS/'news.csv',['rumor_id','content'])])
        self.assertNotIn('label',features)
        self.assertNotIn('sentiment',features)

    def test_exact_overlap_primary_and_approximate_flags(self):
        e=pd.read_csv(run.ROOT/'outputs/tables/covid_rumor_external_overlap_evidence.csv')
        exact=set(e.loc[e.automatic_exclusion,'target_uid'])
        approx=set(e.loc[e.overlap_type.eq('high_similarity'),'target_uid'])
        self.assertEqual(set(self.target.loc[~self.target.clean_primary,'target_uid']),set(self.target.target_uid)&exact)
        self.assertEqual(set(self.target.loc[self.target.high_similarity_flag,'target_uid']),set(self.target.target_uid)&approx)
        self.assertTrue(self.target.loc[~self.target.clean_primary,'exclusion_reason'].ne('').all())
        self.assertTrue(e.loc[e.overlap_type.eq('high_similarity'),'automatic_exclusion'].eq(False).all())

    def test_prediction_freeze_precedes_label_join(self):
        self.assertEqual(self.meta['events'],['label_blind_quality_and_overlap_complete','target_cohort_frozen',
                                             'all_predictions_frozen','target_labels_joined_for_evaluation'])
        lock=json.loads((run.STATE/'prediction_lock.json').read_text())
        self.assertEqual(lock['predictions_sha256'],run.sha(run.PRED))
        self.assertEqual(lock['target_sha256'],run.sha(run.COHORT))
        self.assertFalse(lock['target_labels_joined'])
        self.assertFalse({'label','binary_target','predictor_text','content'}&set(self.p))
        with patch.object(run,'EVENTS',['target_cohort_frozen']):
            with self.assertRaises(ValueError): run.join_labels(self.p)

    def test_thresholds_complete_identical_rows_unique_keys(self):
        self.assertFalse(self.p.duplicated(['target_uid','classifier','source_dataset']).any())
        groups=list(self.p.groupby(['classifier','source_dataset']))
        self.assertEqual(len(groups),4)
        for (classifier,_),p in groups:
            threshold=.5 if classifier=='LR' else 0.
            self.assertTrue(p.threshold.eq(threshold).all())
            np.testing.assert_array_equal(p.predicted_label,(p.score>=threshold).astype(int))
            self.assertEqual(set(p.target_uid),set(self.target.target_uid))
            self.assertEqual(set(p.loc[p.clean_primary,'target_uid']),set(self.target.loc[self.target.clean_primary,'target_uid']))

    def test_source_reuse_no_target_fit_unchanged_fingerprints(self):
        self.assertEqual(self.meta['source_refits'],0)
        fits=pd.read_csv(run.ROOT/'outputs/tables/covid_rumor_external_fit_manifest.csv')
        self.assertTrue(fits.action.eq('reused').all())
        self.assertFalse(fits.target_text_fitted.any() or fits.target_labels_available.any())
        self.assertTrue(fits.source_only_vocabulary.all())
        self.assertEqual(self.meta['protected_artifacts'],run.protected_files())
        self.assertEqual(self.meta['inputs'],run.input_hashes())
        models=run.validate_sources(run.source_views())
        self.assertEqual(len(models),4)

    def test_metrics_recompute_and_comparison(self):
        labels=pd.read_csv(run.NEWS/'news.csv',header=None,names=['rumor_id','label','content','sentiment'],usecols=['rumor_id','label'])
        lookup=dict(zip('covid_rumor:NEWS:'+labels.rumor_id.astype(str),labels.label.map({'F':1,'T':0})))
        for r in self.results[self.results.scope.eq('clean_primary')].itertuples():
            p=self.p[self.p.classifier.eq(r.classifier)&self.p.source_dataset.eq(r.source_dataset)&self.p.clean_primary]
            m=run.base.metrics(p.target_uid.map(lookup),p.score,p.predicted_label)
            for name,value in m.items(): self.assertAlmostEqual(value,getattr(r,name),places=12)
        comparison=pd.read_csv(run.ROOT/'outputs/tables/covid_rumor_external_source_comparison.csv')
        np.testing.assert_allclose(comparison.coaid_minus_fakehealth,comparison.coaid_source-comparison.fakehealth_source,atol=1e-14)

    def test_all_output_hashes_and_keys(self):
        for path,digest in self.meta['outputs'].items(): self.assertEqual(run.sha(run.ROOT/path),digest,path)
        self.assertFalse(self.results.duplicated(['classifier','source_dataset','scope']).any())
        diag=pd.read_csv(run.ROOT/'outputs/tables/covid_rumor_external_vocabulary_diagnostics.csv')
        self.assertTrue(diag.target_token_occurrence_coverage_percent.between(0,100).all())
        self.assertTrue(diag.zero_feature_rows.between(0,len(self.target)).all())


if __name__=='__main__': unittest.main()
