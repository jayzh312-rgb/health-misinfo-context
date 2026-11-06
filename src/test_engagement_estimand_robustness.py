"""Checks specific to the common estimand and deterministic sensitivity rules."""
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from audit_engagement_estimand_robustness import (
    add_release_columns, change_flags, design, standardized_counts,
    support_fraction, tail_sets, verify_full_hurdle_mean,
)
from run_primary_engagement_models import Fit, moments


class EstimandAuditChecks(unittest.TestCase):
    def test_tail_ties_are_deterministic_and_explicit(self):
        frame = pd.DataFrame({"content_uid": [f"id{i:03d}" for i in range(200)],
                              "unique_post_count": [50, 50, 50, 10, 9] + [0] * 195})
        sets = tail_sets(frame.sample(frac=1, random_state=42))
        self.assertEqual(sets["exclude_top_1pct"], {"id000", "id001"})
        self.assertEqual(sets["exclude_top_1pct_with_boundary_ties"], {"id000", "id001", "id002"})
        self.assertEqual(len(sets["exclude_top_5_items"]), 5)
        self.assertFalse(sets["all_retained"])

    def test_hurdle_standardization_is_all_item_mean(self):
        x = np.array([[1., 0.], [1., 1.], [1., 0.]])
        b = np.array([-.3, .7, 1.5, -.4, 2.])
        f = Fit("Hurdle_Logit_ZTNB2", b, np.eye(5), np.eye(5) * .01,
                np.zeros((3, 5)), -10., True, "", "[]", [], [], False, 0., True, 2, x, np.array([0., 1., 5.]))
        estimates = standardized_counts(f, x)
        r, a = x.copy(), x.copy()
        r[:, 1], a[:, 1] = 0, 1
        mr, ma = moments(f, r), moments(f, a)
        assert_allclose(estimates["standardized_fake_mean"], ma[0].mean())
        assert_allclose(estimates["standardized_real_mean"], mr[0].mean())
        assert_allclose(estimates["mean_ratio"], ma[0].mean()/mr[0].mean())
        self.assertNotAlmostEqual(estimates["mean_ratio"], ma[3].mean()/mr[3].mean())
        self.assertNotAlmostEqual(estimates["mean_ratio"], np.exp(b[3]))
        self.assertLess(estimates["CI_lower"], estimates["mean_ratio"])
        self.assertGreater(estimates["CI_upper"], estimates["mean_ratio"])
        verify_full_hurdle_mean(f)

    def test_membership_overlap_and_release_order(self):
        frame = pd.DataFrame({"member_release_periods_json": [
            '["11-01-2020", "05-01-2020"]', '["05-01-2020"]',
            '["07-01-2020"]', '["09-01-2020"]', '["11-01-2020"]',
            '["05-01-2020"]', '["07-01-2020"]', '["09-01-2020"]', '["11-01-2020"]'],
            "analysis_group": ["real"] * 5 + ["fake"] * 4})
        frame = add_release_columns(frame)
        self.assertEqual(frame.iloc[0].first_release, "05-01-2020")
        self.assertEqual(frame.iloc[0].last_release, "11-01-2020")
        self.assertEqual(frame.multiple_releases.sum(), 1)
        self.assertAlmostEqual(support_fraction(design(frame, "membership_indicators")), 1/9)
        self.assertEqual(support_fraction(design(frame, "first_release")), 0)

    def test_magnitude_change_separate_from_direction(self):
        a = change_flags(.36, .237)
        self.assertTrue(a["same_direction"])
        self.assertEqual(a["change_classification"], "substantial")
        self.assertFalse(a["point_estimate_crosses_1"])
        self.assertTrue(change_flags(.93, 1.003)["point_estimate_crosses_1"])


if __name__ == "__main__":
    unittest.main()
