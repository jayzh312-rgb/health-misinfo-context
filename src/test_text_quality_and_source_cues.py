"""Regression checks for frozen audit artifacts and predictor boundaries."""
import json
import unittest
import pandas as pd
from audit_text_quality_and_source_cues import (
    ROOT, PREFIXES, FIELDS, META, read, fingerprint, build_dictionary,
    compile_dictionary, cues_and_mask, quality_decision, title_decision,
)


class TextAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = {ds: read(ROOT / f'data/processed/{p}_text_primary.csv') for ds, p in PREFIXES.items()}
        cls.ready = {ds: read(ROOT / f'data/processed/{p}_text_modeling_ready.csv') for ds, p in PREFIXES.items()}
        cls.dictionary = read(ROOT / 'outputs/tables/text_source_mask_dictionary.csv')
        cls.compiled = compile_dictionary(cls.dictionary)

    def test_originals_and_split_fingerprints_unchanged(self):
        metadata = json.loads((ROOT / 'outputs/reports/text_quality_and_source_cue_audit_metadata.json').read_text())
        for path, digest in metadata['input_sha256'].items():
            self.assertEqual(fingerprint(ROOT / path), digest)
        for ds, original in self.original.items():
            pd.testing.assert_frame_equal(original, self.ready[ds][original.columns])

    def test_no_reassignment_or_group_leakage(self):
        for ds, prefix in PREFIXES.items():
            d = self.ready[ds]
            h = read(ROOT / f'outputs/splits/{prefix}_text_holdout.csv').set_index('content_uid')
            c = read(ROOT / f'outputs/splits/{prefix}_text_grouped_cv.csv').set_index('content_uid')
            for flag in ['primary_modeling_eligible', 'title_only_modeling_eligible', 'body_only_modeling_eligible']:
                x = d[d[flag]].set_index('content_uid')
                self.assertTrue(set(x.index) <= set(h.index) & set(c.index))
                self.assertEqual(x.cv_fold.tolist(), c.loc[x.index, 'fold'].tolist())
                self.assertEqual(x.holdout_assignment.tolist(), h.loc[x.index, 'assignment'].tolist())
                self.assertEqual(x.leakage_group.tolist(), c.loc[x.index, 'leakage_group'].tolist())
                self.assertTrue(x.groupby('leakage_group').cv_fold.nunique().le(1).all())
                train = x[x.cv_fold >= 0]
                held = x[x.cv_fold == -1]
                self.assertFalse(set(train.leakage_group) & set(held.leakage_group))
                self.assertEqual(set(held.binary_analysis_label), {'Fake', 'Real'})
                for fold in range(5):
                    for part in [train[train.cv_fold == fold], train[train.cv_fold != fold]]:
                        self.assertEqual(set(part.binary_analysis_label), {'Fake', 'Real'})

    def test_dictionary_is_label_and_order_invariant(self):
        full = pd.concat(self.original.values(), ignore_index=True)
        expected = build_dictionary(full[list(META)])
        changed = full.copy()
        for col in set(full.columns) - set(META):
            changed[col] = 'POISON_LABEL_ENGAGEMENT_TEXT'
        pd.testing.assert_frame_equal(expected, build_dictionary(changed.sample(frac=1, random_state=12)))
        pd.testing.assert_frame_equal(expected, self.dictionary)
        self.assertFalse(expected.identifier.str.contains('POISON').any())

    def test_mask_is_deterministic_and_preserves_unmatched_language(self):
        sample = 'Reuters reports at https://cnn.com/health?x=1; ordinary time and who remain.'
        result, categories, _ = cues_and_mask(sample, self.compiled)
        self.assertEqual(result, '[SOURCE] reports at https://[SOURCE]/health?x=1; ordinary time and who remain.')
        self.assertIn('explicit_url_host', categories)
        self.assertEqual(cues_and_mask(result, self.compiled)[0], result)
        self.assertEqual(cues_and_mask('A clinical study found benefits.', self.compiled)[0], 'A clinical study found benefits.')
        self.assertEqual(cues_and_mask(sample, list(reversed(self.compiled)))[0], result)

    def test_exclusions_and_review_coverage(self):
        audit = read(ROOT / 'outputs/tables/text_quality_audit.csv')
        for ds, d in self.ready.items():
            flagged = d[d.primary_text_eligible & d.mixed_artifact_component_flag]
            self.assertTrue(set(flagged.content_uid) <= set(audit.content_uid))
            self.assertFalse(flagged.artifact_audit_class.isin(['NOT_FLAGGED', 'UNCERTAIN']).any())
            self.assertTrue(d.loc[~d.primary_modeling_eligible, 'modeling_exclusion_reason'].ne('').all())
            self.assertTrue(d.loc[d.primary_modeling_eligible, 'modeling_exclusion_reason'].eq('').all())
            self.assertFalse(d.loc[d.primary_modeling_eligible, 'artifact_audit_class'].eq('ARTIFACT_DOMINATED').any())
            if ds == 'FakeHealth':
                fallback = d[d.primary_text_eligible & d.title_origin.ne('fetched_content')]
                self.assertEqual(len(fallback), 86)
                self.assertTrue(set(fallback.content_uid) <= set(audit.content_uid))

    def test_label_blind_quality_and_text_only_predictors(self):
        for ds, d in self.ready.items():
            original = self.original[ds]
            selected = original[original.primary_text_eligible & (original.mixed_artifact_component_flag | original.title_origin.ne('fetched_content'))]
            # Decisions must not vary under target/metadata changes.
            changed = selected.copy()
            for col in ['binary_analysis_target', 'binary_analysis_label', 'original_review_rating', 'veracity_original']:
                changed[col] = 'POISON'
            self.assertEqual([quality_decision(r) for r in selected.itertuples()], [quality_decision(r) for r in changed.itertuples()])
            self.assertEqual([title_decision(r) for r in selected.itertuples()], [title_decision(r) for r in changed.itertuples()])
            for row in d.itertuples():
                self.assertEqual(row.combined_text, '\n\n'.join(s for s in (row.title_text, row.body_text) if s))
            # Representative source-rich cases and all eligible reviewed cases verify saved transform.
            check = d[d.primary_text_eligible & (d.mixed_artifact_component_flag | d.title_fallback_reviewed)]
            check = pd.concat([check, d.sample(n=25, random_state=42)]).drop_duplicates('content_uid')
            for field in FIELDS:
                self.assertEqual([cues_and_mask(t, self.compiled)[0] for t in check[field]], check[field + '_source_masked'].tolist())


if __name__ == '__main__':
    unittest.main()
