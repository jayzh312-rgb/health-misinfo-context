"""Cohort normalization, transparent duplicate joins, and split isolation."""
import itertools
import json
import re
import unittest

import numpy as np
import pandas as pd

import prepare_text_analysis_cohorts as prep


class TextRuleTests(unittest.TestCase):
    def test_minimal_normalization_preserves_lexical_content(self):
        self.assertEqual(prep.normalize('  HELLO,\n World! https://Example.org/A?x=1  '), 'HELLO, World! https://Example.org/A?x=1')
        self.assertEqual(prep.normalize('"A Real Title!"'), '"A Real Title!"')
        for value in ('', '  ', '""', "''", 'null', 'NaN', None):
            self.assertEqual(prep.normalize(value), '')
        self.assertEqual(prep.combine('Title', 'Body'), 'Title\n\nBody')
        self.assertEqual(prep.combine('', 'Body'), 'Body')

    def test_quality_flags_do_not_rewrite_text(self):
        text = '"Access denied | www.webmd.com used Cloudflare to restrict access"'
        self.assertEqual(prep.quality(text, 'title'), 'clear_crawl_artifact')
        self.assertEqual(prep.normalize(text), text)
        self.assertEqual(prep.quality('Coronavirus Disease 2019 (COVID-19)', 'title'), 'not_flagged')

    def test_high_similarity_join_matches_brute_force(self):
        base = ' '.join('Word'+str(i) for i in range(90))
        texts = {base, base+' Extra', base.replace('Word43','Changed'), base.replace('Word12','Other'),
                 ' '.join('Else'+str(i) for i in range(90)), 'Short title'}
        matches, _ = prep.near_pairs(texts, width=5, threshold=.90, minimum_tokens=30)
        found = {frozenset((a,b)) for a,b,_ in matches}
        sets = {}
        for text in texts:
            tokens = re.findall(r'\w+',text)
            if len(tokens)>=30:
                sets[text] = {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}
        expected = {frozenset((a,b)) for a,b in itertools.combinations(sets,2)
                    if len(sets[a]&sets[b])/len(sets[a]|sets[b])>=.90}
        self.assertEqual(found,expected)
        self.assertTrue(found)

    def test_conservative_url_key_does_not_merge_domain(self):
        self.assertEqual(prep.url_key('https://www.example.com/Article?utm_source=x'), prep.url_key('http://example.com/Article'))
        self.assertNotEqual(prep.url_key('https://example.com/Article'), prep.url_key('https://example.com/Other'))
        self.assertFalse(prep.specific_url('example.com'))


class CohortAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((prep.TABLES/'text_cohort_preparation_metadata.json').read_text())
        cls.frames = {name: pd.read_csv(prep.ROOT/'data/processed'/f'{prefix}_text_primary.csv', keep_default_na=False)
                      for name,prefix in prep.PREFIXES.items()}

    def test_input_rows_and_original_strings_preserved(self):
        for name,frame in self.frames.items():
            original = pd.read_csv(prep.ROOT/'data/processed'/prep.INPUTS[name],keep_default_na=False).set_index('content_uid')
            self.assertEqual(set(frame.content_uid),set(original.index))
            for row in frame.itertuples():
                self.assertEqual(row.title_original,original.loc[row.content_uid,'title'])
                self.assertEqual(row.body_original,original.loc[row.content_uid,'text'])
                self.assertEqual(row.title_text,prep.normalize(row.title_original))
                self.assertEqual(row.body_text,prep.normalize(row.body_original))
                self.assertEqual(row.combined_text,prep.combine(row.title_text,row.body_text))
                self.assertEqual(row.binary_analysis_target,int(row.binary_analysis_label=='Fake'))

    def test_no_engagement_features_and_explicit_allowlist(self):
        self.assertEqual(self.meta['feature_allowlist'],['title_text','body_text','combined_text'])
        self.assertFalse(self.meta['model_or_vectorizer_fitted'])
        self.assertFalse(self.meta['engagement_features_imported'])
        for frame in self.frames.values():
            self.assertFalse(any('engagement' in c or 'post_count' in c or 'reply_count' in c or 'retweet' in c for c in frame.columns))
            self.assertFalse(frame.lineage_metadata_json.str.contains('raw_engagement|raw_tweet|raw_reply',regex=True).any())

    def test_no_group_or_row_leakage_and_complete_eligibility(self):
        for name,frame in self.frames.items():
            prefix = prep.PREFIXES[name]
            held = pd.read_csv(prep.SPLITS/f'{prefix}_text_holdout.csv')
            cv = pd.read_csv(prep.SPLITS/f'{prefix}_text_grouped_cv.csv')
            eligible = frame[frame.primary_text_eligible]
            prep.validate_splits(eligible,held,cv)
            self.assertEqual(set(cv.fold),{-1,0,1,2,3,4})
            for fold in range(5):
                self.assertEqual(cv[cv.fold.eq(fold)].binary_analysis_label.nunique(),2)
            self.assertTrue(held.assignment.eq('test').mean()>.15)
            self.assertTrue(held.assignment.eq('test').mean()<.25)

    def test_split_determinism_with_reordered_rows(self):
        for name,frame in self.frames.items():
            expected = prep.split_plan(name,frame)
            shuffled = prep.split_plan(name,frame.sample(frac=1,random_state=81))
            pd.testing.assert_frame_equal(expected[0],shuffled[0])
            pd.testing.assert_frame_equal(expected[1],shuffled[1])
            stored = pd.read_csv(prep.SPLITS/f'{prep.PREFIXES[name]}_text_grouped_cv.csv')
            pd.testing.assert_frame_equal(expected[1],stored)

    def test_resource_lineage_and_exact_text_barriers(self):
        for name,frame in self.frames.items():
            for field in ['canonical_resource_key','title_text','body_text']:
                nonempty = frame[frame[field].ne('')]
                self.assertTrue(nonempty.groupby(field).leakage_group.nunique().le(1).all())
            if name=='CoAID':
                nonempty = frame[frame.audited_resource_uid.ne('')]
                self.assertTrue(nonempty.groupby('audited_resource_uid').leakage_group.nunique().le(1).all())
        edges = pd.read_csv(prep.TABLES/'text_leakage_edge_audit.csv')
        for name,part in edges.groupby('dataset'):
            groups = self.frames[name].set_index('content_uid').leakage_group
            self.assertTrue((part.left_uid.map(groups)==part.right_uid.map(groups)).all())

    def test_all_inputs_unchanged(self):
        for relative,expected in self.meta['input_sha256'].items():
            self.assertEqual(prep.sha(prep.ROOT/relative),expected)
        self.assertEqual(prep.sha(prep.Path(prep.__file__)),self.meta['script_sha256'])

    def test_transfer_directions_do_not_mix_datasets(self):
        plans = pd.read_csv(prep.SPLITS/'text_cross_context_transfer_assignments.csv')
        for experiment,source,target in [('FH_to_CA','FakeHealth','CoAID'),('CA_to_FH','CoAID','FakeHealth')]:
            p = plans[plans.experiment.eq(experiment)]
            self.assertEqual(set(p[p.role.eq('train_source')].dataset),{source})
            self.assertEqual(set(p[p.role.eq('test_target')].dataset),{target})
            for name in [source,target]:
                self.assertEqual(set(p[p.dataset.eq(name)].content_uid),set(self.frames[name].loc[self.frames[name].primary_text_eligible,'content_uid']))
            self.assertFalse(p.automatic_exclusion.any())


if __name__=='__main__':
    unittest.main()
