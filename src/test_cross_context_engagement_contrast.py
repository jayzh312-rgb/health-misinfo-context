"""Independent-product arithmetic and saved cross-context integrity checks."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

import run_cross_context_engagement_contrast as cross


class ProductTests(unittest.TestCase):
    def test_cartesian_product_not_matched_replicates(self):
        theta, ratios = cross.independent_product([1., 2.], [1., 2.])
        assert_allclose(ratios, [1., .5, 2., 1.])
        self.assertGreater(ratios.var(), 0.)
        assert_allclose(np.exp(theta), ratios)

    def test_reordering_either_marginal_is_irrelevant(self):
        a, f = np.array([.2, .3, .5]), np.array([.7, .8, 1.])
        t, c = cross.independent_product(a, f)
        u, d = cross.independent_product(a[::-1], f[[1, 2, 0]])
        assert_allclose(np.sort(t), np.sort(u))
        assert_allclose(np.sort(c), np.sort(d))

    def test_swap_contexts_negates_log_and_reciprocates_ratio(self):
        t, c = cross.independent_product([.2, .4, .5], [.7, 1.])
        u, d = cross.independent_product([.7, 1.], [.2, .4, .5])
        assert_allclose(np.sort(-t), np.sort(u))
        assert_allclose(np.sort(1 / c), np.sort(d))

    def test_invalid_ratios_rejected(self):
        for a in ([0., .3], [-1., .3], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                cross.independent_product(a, [.8])

    def test_point_estimate_is_not_bootstrap_median(self):
        row, theta, c = cross.contrast_row("test", .2, .8, [.1, .2, .6], [.7, .8, .9], "test")
        self.assertEqual(row["ratio_of_ratios"], .25)
        self.assertEqual(row["cartesian_combinations"], 9)
        self.assertEqual(row["bootstrap_median"], np.median(c))
        assert_allclose([row["CI_lower_ratio"], row["CI_upper_ratio"]], np.quantile(c, [.025, .975]))
        assert_allclose([row["CI_lower_log"], row["CI_upper_log"]], np.quantile(theta, [.025, .975]))


class SavedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = cross.TABLES / "cross_context_engagement_metadata.json"
        if not path.exists():
            raise unittest.SkipTest("Run the cross-context analysis before integration tests")
        cls.meta = json.loads(path.read_text())
        cls.results = pd.read_csv(cross.TABLES / "cross_context_engagement_contrast.csv")

    def test_original_inputs_and_analysis_script_unchanged(self):
        self.assertTrue(self.meta["complete"])
        self.assertEqual(self.meta["original_bootstrap_models_refitted"], 0)
        self.assertEqual(self.meta["original_successful_draws_reused"], 4000)
        for relative, fingerprint in self.meta["input_sha256"].items():
            self.assertEqual(cross.sha(cross.ROOT / relative), fingerprint)
        self.assertEqual(cross.sha(Path(cross.__file__)), self.meta["cross_context_script_sha256"])

    def test_million_pairs_and_saved_quantiles(self):
        draws = pd.read_csv(cross.TABLES / "version_aware_bootstrap_refits.csv")
        f = draws[draws.specification.eq(cross.FH) & draws.success].mean_ratio.to_numpy()
        for row in self.results.itertuples():
            a = draws[draws.specification.eq(row.coaid_specification) & draws.success].mean_ratio.to_numpy()
            theta, values = cross.independent_product(a, f)
            self.assertEqual(len(values), 1000000)
            self.assertEqual(row.cartesian_combinations, len(values))
            assert_allclose([row.CI_lower_ratio, row.CI_upper_ratio], np.quantile(values, [.025, .975]))
            assert_allclose([row.CI_lower_log, row.CI_upper_log], np.quantile(theta, [.025, .975]))
            self.assertAlmostEqual(row.ratio_of_ratios, row.coaid_ratio / row.fakehealth_ratio)
            self.assertEqual(row.ratio_interval_contains_1, row.CI_lower_ratio <= 1 <= row.CI_upper_ratio)

    def test_dominant_classes_partition_original_draws(self):
        frame = pd.read_csv(cross.TABLES / "cross_context_dominant_component_diagnostic.csv")
        self.assertEqual(set(frame.dominant_component_multiplicity), {"0", "1", "2_or_more"})
        self.assertEqual(frame.bootstrap_draws.sum(), 1000)
        self.assertEqual(frame.cartesian_pairs.sum(), 1000000)
        self.assertEqual(frame.bootstrap_draws.tolist(), [363, 372, 265])
        assert_allclose(frame.cross_context_C_IQR, frame.cross_context_C_q75-frame.cross_context_C_q25)

    def test_exclusion_bootstrap_and_absent_tail_intervals(self):
        meta = json.loads(cross.CACHE_META.read_text())
        draws = pd.read_csv(cross.CACHE)
        self.assertEqual(meta["removed_N"], 187)
        self.assertEqual(meta["retained_N"], 5150)
        self.assertEqual(meta["retained_clusters"], 5055)
        self.assertEqual(draws.success.sum(), 1000)
        self.assertTrue(draws.attempt.is_unique)
        self.assertEqual(meta["bootstrap_csv_sha256"], cross.sha(cross.CACHE))
        sens = pd.read_csv(cross.TABLES / "cross_context_sensitivity_results.csv")
        trimmed = sens[sens.sensitivity.isin(["exclude_top_1pct", "exclude_top_5_items"])]
        self.assertEqual(len(trimmed), 2)
        self.assertTrue(trimmed[["CI_lower_ratio", "CI_upper_ratio"]].isna().all().all())
        self.assertTrue(sens[sens.sensitivity.eq("exclude_dominant_component")].CI_lower_ratio.notna().all())


if __name__ == "__main__":
    unittest.main()
