"""Metadata identity and count-only split feasibility tests. No model training."""
import ast
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import audit_publisher_heldout_feasibility as audit


def metadata(records):
    return pd.DataFrame([dict(dataset='Example', content_uid=str(i), publisher_source=name,
                              article_host=h, article_url=url) for i, (name, h, url) in enumerate(records)])


class PublisherFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = audit.load_rows()
        cls.mapping = audit.build_family_map(cls.raw)
        joined = cls.raw.merge(cls.mapping[['dataset', 'content_uid', 'publisher_family', 'identity_status', 'identity_resolved']],
                              on=['dataset', 'content_uid'], validate='one_to_one')
        cls.rows = pd.concat([audit.resource_components(g)[0] for _, g in joined.groupby('dataset')], ignore_index=True)
        cls.support = audit.support_table(cls.rows)

    def test_normalization_label_blind_and_order_independent(self):
        poisoned = self.raw.copy()
        poisoned['binary_analysis_target'] = 1-poisoned.binary_analysis_target
        poisoned['binary_analysis_label'] = 'POISON'
        poisoned['title_text'] = 'POISON'
        poisoned['body_text'] = 'POISON'
        actual = audit.build_family_map(poisoned.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(actual, self.mapping)
        source = inspect.getsource(audit.build_family_map)
        self.assertNotIn('binary_analysis', source)

    def test_aliases_host_precedence_and_uncertain_names(self):
        frame = metadata([('CNN', 'www.cnn.com', ''), ('CNN Health', 'edition.cnn.com', ''),
                          ('Example News', 'example.com', ''), ('Other Attribution', 'example.com', ''),
                          ('Tenant', 'one.blogspot.com', ''), ('Tenant Two', 'two.blogspot.com', ''),
                          ('ABC News', 'abcnews.go.com', ''), ('Container', 'go.com', '')])
        result = audit.build_family_map(frame).set_index('content_uid')
        self.assertEqual(result.loc['0', 'publisher_family'], result.loc['1', 'publisher_family'])
        self.assertEqual(result.loc['2', 'publisher_family'], result.loc['3', 'publisher_family'])
        self.assertNotEqual(result.loc['4', 'publisher_family'], result.loc['5', 'publisher_family'])
        self.assertNotEqual(result.loc['6', 'publisher_family'], result.loc['7', 'publisher_family'])

    def test_missing_sources_and_platform_users_are_not_merged(self):
        frame = metadata([('', '', ''), ('', '', ''), ('youtube.com', 'youtube.com', 'https://youtube.com/watch?v=123'),
                          ('twitter.com', 'twitter.com', 'https://twitter.com/account_a/status/1'),
                          ('twitter.com', 'twitter.com', 'https://twitter.com/account_b/status/2')])
        result = audit.build_family_map(frame).set_index('content_uid')
        self.assertEqual(result.loc['0', 'identity_status'], 'missing_source_metadata')
        self.assertEqual(result.loc['2', 'identity_status'], 'unresolved_platform_publisher')
        self.assertTrue(result.loc[['0', '1', '2'], 'publisher_family'].eq('').all())
        self.assertNotEqual(result.loc['3', 'publisher_family'], result.loc['4', 'publisher_family'])

    def test_resource_components_keep_both_barriers(self):
        rows = pd.DataFrame(dict(dataset=['E']*5, content_uid=list('abcde'), leakage_group=['r1','r1','r2','r3','r3'],
            publisher_family=['A','B','B','C',''], identity_resolved=[True,True,True,True,False],
            binary_analysis_target=[0,1,0,1,0]))
        result, conflicts = audit.resource_components(rows)
        self.assertEqual(result.loc[:2, 'constraint_component'].nunique(), 1)
        self.assertTrue(result.loc[:2, 'strict_publisher_eligible'].all())
        self.assertFalse(result.loc[3:, 'strict_publisher_eligible'].any())
        self.assertEqual(len(conflicts), 2)
        bad_train, bad_test = result.iloc[[0]], result.iloc[[1,2]]
        with self.assertRaisesRegex(ValueError, 'Resource crosses'):
            audit.validate_partition(result.iloc[:3], bad_train, bad_test)

    def test_no_model_fitting(self):
        tree = ast.parse(inspect.getsource(audit))
        calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)
                 and x.func.attr in {'fit', 'fit_transform', 'partial_fit'}]
        self.assertEqual(calls, [])
        with patch.object(audit, 'run_audit', side_effect=lambda: LogisticRegression().fit([[0], [1]], [0,1])):
            with self.assertRaisesRegex(RuntimeError, 'forbids model fitting'):
                audit.main()

    def test_counts_and_partitions_reproduce_deterministically(self):
        expected = pd.read_csv(audit.TABLES / 'publisher_heldout_feasibility.csv', keep_default_na=False)
        actual = audit.evaluate_designs(self.rows.sample(frac=1, random_state=12), self.support)
        # CSV roundtrip standardizes nullable mixed boolean/numeric columns.
        from io import StringIO
        actual = pd.read_csv(StringIO(actual.to_csv(index=False)), keep_default_na=False)
        pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-12, atol=1e-12)
        self.assertFalse(actual.duplicated(['dataset','design','algorithm','folds','fold','partition']).any())

    def test_all_rows_accounted_and_missing_separate(self):
        self.assertEqual(len(self.mapping), len(self.raw))
        self.assertFalse(self.mapping.duplicated(['dataset','content_uid']).any())
        for ds, frame in self.raw.groupby('dataset'):
            support = self.support[self.support.dataset.eq(ds)]
            self.assertEqual(int(support.N.sum()), len(frame))
            self.assertEqual(int(support.fake_N.sum()), int(frame.binary_analysis_target.sum()))
            self.assertTrue(support.loc[support.row_type.eq('unresolved_source'), 'publisher_family'].eq('').all())
            self.assertTrue(support.loc[support.row_type.eq('publisher_family'), 'publisher_family'].ne('').all())
        saved = pd.read_csv(audit.TABLES / 'publisher_mixed_source_cohort_sizes.csv')
        self.assertTrue(saved.N.add(saved.lost_N).eq(saved.original_N).all())
        self.assertTrue(saved.fake_N.add(saved.real_N).eq(saved.N).all())

    def test_preserved_inputs_splits_and_output_fingerprints(self):
        meta = json.loads(audit.REPORT.with_suffix('.json').read_text())
        for path, expected in {**meta['inputs'], **meta['unchanged_splits'], **meta['outputs']}.items():
            self.assertEqual(audit.sha(audit.ROOT / path), expected)
        self.assertEqual(meta['script_sha256'], audit.sha(audit.__file__))
        self.assertEqual(meta['model_fit_count'], 0)
        self.assertFalse(meta['final_split_assignments_written'])
        mapping = pd.read_csv(audit.TABLES / 'publisher_family_map.csv', keep_default_na=False)
        self.assertFalse(set(mapping.columns) & {'fold','assignment','cv_fold','holdout_assignment'})


if __name__ == '__main__':
    unittest.main()
