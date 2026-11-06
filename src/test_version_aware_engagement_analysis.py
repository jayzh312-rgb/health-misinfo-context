"""Tests of component resampling and corrected within-dataset outputs."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

import run_version_aware_engagement_analysis as analysis
from run_primary_engagement_models import Fit, fit_hurdle


class OverlapAndEstimandTests(unittest.TestCase):
    def test_transitive_graph_singletons_and_typed_degrees(self):
        frame = pd.DataFrame(dict(content_uid=["a", "b", "c", "d"],
                                  analysis_group=["fake", "real", "real", "fake"],
                                  unique_post_count=[1, 2, 3, 0]))
        spec = "FakeHealth_test"
        overlaps = pd.DataFrame([
            dict(specification=spec, social_type="post", final_entity_uids_json='["a","b"]'),
            dict(specification=spec, social_type="post", final_entity_uids_json='["a","b"]'),
            dict(specification=spec, social_type="reply", final_entity_uids_json='["b","c"]')])
        _, assn, comp, summary = analysis.overlap_graph(spec, frame, overlaps)
        self.assertEqual(summary["overlap_components"], 2)
        self.assertEqual(summary["singleton_components"], 1)
        self.assertEqual(summary["mixed_components"], 1)
        rows = assn.set_index("entity_uid")
        self.assertEqual(rows.loc["a", "shared_post_degree"], 1)
        self.assertEqual(rows.loc["a", "shared_post_id_count"], 2)
        self.assertEqual(rows.loc["b", "shared_reply_degree"], 1)
        self.assertEqual(rows.loc["a", "overlap_cluster_uid"], rows.loc["c", "overlap_cluster_uid"])
        self.assertEqual(rows.loc["d", "cluster_size"], 1)
        _, reversed_assn, _, _ = analysis.overlap_graph(spec, frame.iloc[::-1], overlaps.iloc[::-1])
        self.assertEqual(rows.overlap_cluster_uid.to_dict(), reversed_assn.set_index("entity_uid").overlap_cluster_uid.to_dict())

    def test_resampling_keeps_whole_repeated_clusters(self):
        class FixedDraw:
            def integers(self, low, high, size):
                return np.array([1, 1, 0])
        idx, selected = analysis.resample_indices([np.array([0]), np.array([1, 2]), np.array([3])], FixedDraw())
        assert_array_equal(selected, [1, 1, 0])
        assert_array_equal(idx, [1, 2, 1, 2, 0])

    def test_percentile_explicitly_excludes_failed_refits(self):
        draws = pd.DataFrame({"mean_ratio": [1., 2., 10000.], "success": [True, True, False]})
        assert_allclose(analysis.percentile(draws, "mean_ratio"), [1.025, 1.975])

    def test_common_estimand_uses_full_hurdle(self):
        x = np.array([[1., 0.], [1., 1.], [1., 0.]])
        b = np.array([-.3, .7, 1.5, -.4, 2.])
        fit = Fit("Hurdle_Logit_ZTNB2", b, np.eye(5), np.eye(5), np.zeros((3, 5)),
                  -10, True, "", "[]", [], [], False, 0, True, 2, x, np.array([0., 1., 5.]))
        result = analysis.point_metrics(fit, x)
        reference = analysis.standardized_counts(fit, x)
        self.assertAlmostEqual(result["mean_ratio"], reference["mean_ratio"])
        self.assertNotAlmostEqual(result["mean_ratio"], result["positive_mean_ratio"])
        self.assertNotAlmostEqual(result["mean_ratio"], np.exp(b[3]))

    def test_warm_refit_agrees_with_tested_multistart_likelihood(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame({"Intercept": np.ones(400), "fake_vs_real": np.repeat([0., 1.], 200)})
        y = rng.negative_binomial(2, .2, len(x)).astype(float)
        y[rng.random(len(x)) < .4] = 0
        baseline = fit_hurdle(y, x)
        warm = analysis.warm_fit(y, x, baseline.name, baseline.params)
        self.assertTrue(warm.stable)
        assert_allclose(warm.ll, baseline.ll, atol=1e-5)
        assert_allclose(analysis.point_metrics(warm, x)["mean_ratio"],
                        analysis.point_metrics(baseline, x)["mean_ratio"], rtol=1e-5)


class SavedOutputTests(unittest.TestCase):
    def test_graph_coverage_and_known_component_counts(self):
        path = analysis.TABLES / "coaid_social_overlap_clusters.csv"
        if not path.exists():
            self.skipTest("Run analysis before integration checks")
        graph = pd.read_csv(path)
        expected = {"A_primary_strict": 5056, "B_resource_sensitivity": 4938, "C_high_certainty": 4905}
        for spec, k in expected.items():
            rows = graph[graph.specification.eq(spec)]
            frame = pd.read_csv(analysis.ROOT / "data/processed" / analysis.SAMPLES[spec][0])
            self.assertEqual(set(rows.entity_uid), set(frame.content_uid))
            self.assertTrue(rows.entity_uid.is_unique)
            self.assertEqual(rows.overlap_cluster_uid.nunique(), k)
            self.assertTrue((rows.groupby("overlap_cluster_uid").size() == rows.groupby("overlap_cluster_uid").cluster_size.first()).all())

    def test_result_counts_and_bootstrap_intervals(self):
        path = analysis.TABLES / "version_aware_engagement_results.csv"
        if not path.exists():
            self.skipTest("Run analysis before integration checks")
        results = pd.read_csv(path)
        draws = pd.read_csv(analysis.BOOT_PATH)
        for r in results.itertuples():
            self.assertEqual((r.fake_N, r.real_N), analysis.SAMPLES[r.specification][2:])
            self.assertAlmostEqual(r.standardized_ratio, r.standardized_fake_mean / r.standardized_real_mean)
            part = draws[draws.specification.eq(r.specification)]
            column, flag = ("mean_ratio", "success") if r.outcome == "unique_post_count" else ("reply_mean_ratio", "reply_success")
            lo, hi = analysis.percentile(part, column, flag)
            assert_allclose([r.primary_CI_lower, r.primary_CI_upper], [lo, hi])
        self.assertFalse(draws.duplicated(["specification", "attempt"]).any())

    def test_no_input_changes(self):
        if not analysis.RUN_META.exists():
            self.skipTest("Run analysis before integration checks")
        meta = json.loads(analysis.RUN_META.read_text())
        for relative, expected in meta["input_sha256"].items():
            self.assertEqual(analysis.sha(analysis.ROOT / relative), expected)

    def test_completed_full_run_reaches_target_and_preserves_old_code(self):
        if not analysis.RUN_META.exists():
            self.skipTest("Run analysis before integration checks")
        meta = json.loads(analysis.RUN_META.read_text())
        if not meta.get("complete") or meta.get("pilot_only"):
            self.skipTest("Full analysis has not completed")
        summary = pd.read_csv(analysis.TABLES / "version_aware_bootstrap_summary.csv")
        self.assertEqual(set(summary.specification), set(analysis.SAMPLES))
        self.assertTrue(summary.successful_post_refits.ge(1000).all())
        self.assertTrue(summary.target_met.all())
        self.assertEqual(meta["script_sha256"], analysis.sha(Path(analysis.__file__)))
        self.assertIn("src/run_primary_engagement_models.py", meta["input_sha256"])
        self.assertIn("outputs/tables/common_engagement_estimand.csv", meta["input_sha256"])


if __name__ == "__main__":
    unittest.main()
