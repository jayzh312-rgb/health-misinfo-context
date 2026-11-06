"""Tests for the frozen-model lexical audit; no training or transfer evaluation."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import audit_cross_context_lexical_alignment as audit


class LexicalAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with audit.prohibit_fitting():
            cls.shared, cls.sizes, cls.models, cls.fingerprints = audit.load_frozen()
            cls.shared = audit.flag_features(cls.shared, cls.models['CoAID'])

    def test_unigram_intersection_and_determinism(self):
        expected = sorted(set(x for x in self.models['FakeHealth']['tfidf'].get_feature_names_out() if ' ' not in x)
                          & set(self.models['CoAID']['tfidf'].get_feature_names_out()))
        self.assertEqual(self.shared.feature.tolist(), expected)
        self.assertTrue(self.shared.feature.str.split().str.len().eq(1).all())
        a = audit.importance_subsets(self.shared)
        b = audit.importance_subsets(self.shared.sample(frac=1, random_state=9))
        for (name, left), (_, right) in zip(a, b):
            self.assertEqual(set(left.feature), set(right.feature))
            if name.startswith('top'):
                self.assertEqual(left.feature.tolist(), right.feature.tolist())

    def test_fake_positive_orientation(self):
        for model in self.models.values():
            np.testing.assert_array_equal(model['logistic'].classes_, [0, 1])
        for prefix in ('fakehealth', 'coaid'):
            np.testing.assert_array_equal(self.shared[f'{prefix}_sign'], np.sign(self.shared[f'{prefix}_coefficient']))
        categories = audit.ranked_categories(self.shared)
        self.assertTrue((categories['stable_fake_features'][['fakehealth_coefficient', 'coaid_coefficient']] > 0).all().all())
        self.assertTrue((categories['stable_real_features'][['fakehealth_coefficient', 'coaid_coefficient']] < 0).all().all())

    def test_fingerprints_and_fail_closed(self):
        self.assertEqual(audit.verify_inputs(), self.fingerprints)
        with patch.object(audit, 'sha', return_value='mismatch'):
            with self.assertRaises(ValueError):
                audit.verify_inputs()

    def test_no_fitting(self):
        with audit.prohibit_fitting():
            with self.assertRaisesRegex(RuntimeError, 'prohibits fitting'):
                LogisticRegression().fit([[0], [1]], [0, 1])
        tree = ast.parse(inspect.getsource(audit))
        forbidden = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr in {'fit', 'fit_transform', 'partial_fit'}]
        self.assertEqual(forbidden, [])

    def test_cutoffs_cannot_depend_on_labels(self):
        cols = ['feature', 'fakehealth_abs_coefficient', 'coaid_abs_coefficient', 'max_abs_coefficient']
        restricted = self.shared[cols]
        clean = audit.importance_subsets(restricted)
        poisoned = audit.importance_subsets(restricted.assign(target_label=np.arange(len(restricted)) % 2))
        self.assertEqual([name for name, _ in clean], ['all', 'above_median_either', 'top1000', 'top500', 'top100'])
        for (_, left), (_, right) in zip(clean, poisoned):
            self.assertEqual(left.feature.tolist(), right.feature.tolist())
        median = restricted[(restricted.fakehealth_abs_coefficient > restricted.fakehealth_abs_coefficient.median()) |
                            (restricted.coaid_abs_coefficient > restricted.coaid_abs_coefficient.median())]
        self.assertEqual(set(clean[1][1].feature), set(median.feature))

    def test_summary_and_unique_output_keys(self):
        summary = pd.DataFrame([audit.summarize(n, f) for n, f in audit.importance_subsets(self.shared)])
        categories = audit.ranked_categories(self.shared)
        stability, details = audit.transfer_stability(self.shared)
        audit.validate_tables(self.shared, summary, categories, stability, details)
        flags = audit.enrichment(self.shared, categories)
        self.assertFalse(flags.duplicated(['group', 'flag']).any())
        self.assertTrue(summary.sign_agreement_N.add(summary.sign_reversal_N).eq(summary.N).all())

    def test_near_zero_precedes_sign(self):
        frame = pd.DataFrame({'feature': ['a', 'b', 'c'],
            'fakehealth_coefficient': [1., 2., 3.], 'coaid_coefficient': [-.049, -.05, .05]})
        for prefix in ('fakehealth', 'coaid'):
            frame[f'{prefix}_sign'] = np.sign(frame[f'{prefix}_coefficient'])
            frame[f'{prefix}_abs_coefficient'] = frame[f'{prefix}_coefficient'].abs()
        _, details = audit.transfer_stability(frame)
        got = details[details.source_dataset.eq('fakehealth')].set_index('feature').target_status.to_dict()
        self.assertEqual(got, {'a': 'near_zero', 'b': 'opposite_direction', 'c': 'same_direction'})


if __name__ == '__main__':
    unittest.main()
