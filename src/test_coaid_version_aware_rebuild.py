"""Rebuild integrity tests only; no EDA or model execution."""
import hashlib
import json
import unittest
from collections import defaultdict

import pandas as pd

import rebuild_coaid_version_aware_samples as rebuild


class VersionAwareRebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.news, cls.old, cls.old_groups, _, cls.raw,
         cls.tables, cls.hashes) = rebuild.load_inputs()
        cls.plans, cls.exclusions = rebuild.build_plans(cls.old_groups, cls.raw, cls.tables)
        cls.frames = {s: pd.read_csv(rebuild.PROCESSED / p, keep_default_na=False)
                      for s, p in rebuild.SPECS.items()}
        cls.lineage = rebuild.read_table("coaid_version_aware_entity_lineage")
        cls.overlap = pd.read_csv(rebuild.TABLES / "coaid_cross_entity_social_overlap.csv",
                                  dtype={"social_id": str}, keep_default_na=False)

    def test_uid_order_invariance_and_capture_identity(self):
        self.assertEqual(rebuild.entity_uid("host/item", "version", ["b", "a"]),
                         rebuild.entity_uid("host/item", "version", ["a", "b"]))
        self.assertNotEqual(rebuild.entity_uid("host/item", "capture", ["a"]),
                            rebuild.entity_uid("host/item", "capture", ["b"]))
        self.assertFalse(rebuild.specific_url("cdc.gov"))
        self.assertTrue(rebuild.specific_url("cdc.gov/specific/page.html"))

    def test_partition_rejects_duplicate_or_missing_assignment(self):
        with self.assertRaises(ValueError):
            rebuild.partition_check([["a"], ["a"]], {"a"}, "duplicate")
        with self.assertRaises(ValueError):
            rebuild.partition_check([["a"]], {"a", "b"}, "missing")

    def test_uncertain_edge_cannot_create_transitive_version(self):
        changed = {k: v.copy() for k, v in self.tables.items()}
        pairs = changed["coaid_content_version_pair_evidence"]
        group = next(g for g in self.plans["A_primary_strict"]
                     if g["mode"] == "validated_content_version" and len(g["members"]) > 1)
        members = set(group["members"])
        mask = (pairs.resource_uid.eq(group["resource_audit"]) &
                pairs.left_member_uid.isin(members) & pairs.right_member_uid.isin(members))
        self.assertTrue(mask.any())
        pairs.loc[mask, "classification"] = "E. INSUFFICIENT EVIDENCE"
        with self.assertRaisesRegex(ValueError, "Uncertain/transitive"):
            rebuild.build_plans(self.old_groups, self.raw, changed)

    def test_no_new_merges_and_all_source_lineage(self):
        excluded = set(self.exclusions.member_source_uid)
        for spec, frame in self.frames.items():
            self.assertTrue(frame.content_uid.is_unique)
            lin = self.lineage[self.lineage.specification.eq(spec)]
            lin_index = lin.set_index("final_entity_uid")
            self.assertEqual(set(lin.final_entity_uid), set(frame.content_uid))
            included = set(self.raw) - (excluded if spec.startswith("C") else set())
            rebuild.partition_check([json.loads(x) for x in frame.member_original_uids_json], included, spec)
            self.assertEqual(set(frame.content_uid), {g["uid"] for g in self.plans[spec]})
            for r in frame.itertuples():
                members = json.loads(r.member_original_uids_json)
                self.assertEqual(members, json.loads(lin_index.loc[r.content_uid, "member_source_uids"]))
                self.assertLessEqual(set(members), set(self.old_groups[r.previous_entity_uid]))
                if len(members) > 1:
                    self.assertEqual({self.raw[m]["news_url_key"] for m in members}, {r.canonical_resource_key})
                    self.assertTrue(rebuild.specific_url(r.canonical_resource_key))

    def test_counts_presence_and_overlap_from_raw_id_sets(self):
        for spec, frame in self.frames.items():
            owners = {"post": defaultdict(set), "reply": defaultdict(set)}
            for r in frame.itertuples():
                members = json.loads(r.member_original_uids_json)
                sets = {kind: set().union(*(self.raw[m][kind] for m in members)) for kind in ("posts", "replies")}
                self.assertEqual(r.unique_post_count, len(sets["posts"]))
                self.assertEqual(r.unique_reply_count, len(sets["replies"]))
                self.assertEqual(r.engagement_present, bool(sets["posts"] or sets["replies"]))
                for plural, singular in (("posts", "post"), ("replies", "reply")):
                    for identifier in sets[plural]:
                        owners[singular][identifier].add(r.content_uid)
            audit = self.overlap[self.overlap.specification.eq(spec)]
            for singular in owners:
                shared = {sid: ids for sid, ids in owners[singular].items() if len(ids) > 1}
                observed = audit[audit.social_type.eq(singular)]
                self.assertEqual(set(observed.social_id), set(shared))
                for r in observed.itertuples():
                    self.assertEqual(set(json.loads(r.final_entity_uids_json)), shared[r.social_id])
                    self.assertEqual(r.excess_memberships, len(shared[r.social_id]) - 1)
                flag_counts = defaultdict(int)
                for entities in shared.values():
                    for eid in entities:
                        flag_counts[eid] += 1
                for r in frame.itertuples():
                    self.assertEqual(getattr(r, f"cross_entity_{singular}_overlap_count"), flag_counts[r.content_uid])
                    self.assertEqual(getattr(r, f"cross_entity_{singular}_overlap_flag"), flag_counts[r.content_uid] > 0)

    def test_all_five_material_and_nine_overmerged_parents(self):
        checks = rebuild.read_table("coaid_version_aware_component_validation")
        self.assertEqual(len(checks), 42)
        self.assertTrue(checks.passed.all())
        for spec, groups in self.plans.items():
            socials = {g["uid"]: {kind: set().union(*(self.raw[m][kind] for m in g["members"]))
                                   for kind in ("posts", "replies")} for g in groups}
            self.assertEqual(len(rebuild.validate_components(spec, groups, socials, self.old_groups, self.tables)), 14)
            giant = [g for g in groups if g["parent"] == rebuild.GIANT]
            self.assertGreater(len(giant), 1)
            self.assertTrue(all(len(g["members"]) < 110 for g in giant))

    def test_specification_policies_and_sizes(self):
        expected = {"A_primary_strict": (5337, 922, 4415),
                    "B_resource_sensitivity": (5205, 921, 4284),
                    "C_high_certainty": (5082, 920, 4162)}
        for spec, frame in self.frames.items():
            self.assertEqual((len(frame), sum(frame.coaid_within_dataset_analysis_group.eq("fake")),
                              sum(frame.coaid_within_dataset_analysis_group.eq("real"))), expected[spec])
            uncertain = frame[frame.aggregation_mode.eq("insufficient_version_capture")]
            if spec.startswith("A"):
                self.assertEqual(len(uncertain), 241)
                self.assertTrue(uncertain.release_observation_count.eq(1).all())
            if spec.startswith("B"):
                assumed = frame[frame.aggregation_mode.eq("assumed_stable_resource")]
                self.assertEqual(len(assumed), 109)
                self.assertTrue(assumed.canonical_resource_key.map(rebuild.specific_url).all())
            if spec.startswith("C"):
                self.assertFalse(frame.version_classification.str.startswith("E.").any())
                self.assertFalse(frame.unresolved_identity_flag.any())
        self.assertEqual(len(self.exclusions), 255)

    def test_existing_pipeline_schema_without_running_pipeline(self):
        for frame in self.frames.values():
            self.assertLessEqual(set(self.old.columns), set(frame.columns))
            self.assertEqual(set(frame.coaid_within_dataset_analysis_group), {"fake", "real"})
            for col in ("unique_post_count", "unique_reply_count"):
                self.assertTrue(frame[col].notna().all())
                self.assertTrue(frame[col].ge(0).all())
                self.assertTrue(frame[col].mod(1).eq(0).all())
            self.assertTrue(frame.timestamp_role.eq("secondary_sensitivity_only").all())

    def test_inputs_unchanged(self):
        metadata = json.loads((rebuild.TABLES / "coaid_version_aware_rebuild_metadata.json").read_text())
        for relative, sha in metadata["input_sha256"].items():
            self.assertEqual(hashlib.sha256((rebuild.ROOT / relative).read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
