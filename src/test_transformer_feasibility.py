"""Integrity, real-tokenizer, and inference-only checks for the completed audit."""
import ast
import inspect
import json
import unittest

import numpy as np
import pandas as pd

import audit_transformer_feasibility as audit


class TransformerFeasibilityTests(unittest.TestCase):
    def test_original_inputs_and_splits_unchanged(self):
        cohorts,current=audit.load_assignments()
        meta=json.loads(audit.REPORT.with_suffix('.json').read_text())
        self.assertEqual(current,meta['inputs'])
        self.assertEqual({ds:len(d) for ds,d in cohorts.items()},{'FakeHealth':1647,'CoAID':3577})
        for d in cohorts.values():
            self.assertTrue(d.groupby('leakage_group').cv_fold.nunique().eq(1).all())
        for path,expected in meta['outputs'].items():
            self.assertEqual(audit.sha(audit.ROOT/path),expected)

    def test_environment_and_dependency_guard(self):
        meta=json.loads(audit.REPORT.with_suffix('.json').read_text())
        self.assertEqual(meta['missing_packages'],[])
        self.assertTrue(meta['official_tokenizer_audited'])
        self.assertTrue(meta['environment']['python'].startswith('3.12.'))
        self.assertTrue(meta['environment']['torch_import'])
        self.assertTrue(meta['environment']['transformers_import'])
        self.assertTrue(meta['environment']['cpu_available'])
        tree=ast.parse(inspect.getsource(audit.main))
        gated=[node for node in ast.walk(tree) if isinstance(node,ast.If) and ast.unparse(node.test)=='missing']
        self.assertEqual(len(gated),1)
        self.assertIsInstance(gated[0].body[0],ast.Raise)
        self.assertLess(gated[0].lineno,next(n.lineno for n in ast.walk(tree)
            if isinstance(n,ast.Call) and ast.unparse(n.func)=='tokenizer_measurements'))

    def test_no_training_or_installation_execution(self):
        tree=ast.parse(inspect.getsource(audit))
        forbidden={'fit','train','backward','step','install','system','Popen','check_call'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                # platform.system only reports OS; it is not os.system execution.
                if ast.unparse(node.func)=='platform.system':continue
                self.assertNotIn(node.func.attr,forbidden)
                if ast.unparse(node.func)=='subprocess.run':
                    self.assertEqual(ast.literal_eval(node.args[0]),['/usr/sbin/sysctl','-n','hw.memsize'])
        meta=json.loads(audit.REPORT.with_suffix('.json').read_text())
        self.assertEqual(meta['packages_installed'],0);self.assertEqual(meta['training_runs'],0)
        self.assertEqual(meta['packages'],audit.package_status())

    def test_predictor_only_and_label_independent_packing(self):
        source=inspect.getsource(audit.tokenizer_measurements)
        self.assertIn("usecols=['content_uid','combined_text','title_text','body_text']",source)
        self.assertNotIn('binary_analysis',source)
        self.assertNotIn('publisher',source)
        self.assertNotIn('engagement',source)
        self.assertEqual(list(inspect.signature(audit.title_body_allocation).parameters),
                         ['title_tokens','body_tokens','budget','pair_special_tokens'])
        got=audit.title_body_allocation(20,1000,512)
        self.assertEqual(got,{'feasible':True,'title_kept':20,'body_kept':488})
        self.assertFalse(audit.title_body_allocation(509,1000,512)['feasible'])
        self.assertEqual(audit.title_body_allocation(20,1,512)['body_kept'],1)

    def test_deterministic_count_summary_and_retention_arithmetic(self):
        lengths=[100,200,600,1000]
        first=audit.summarize_lengths(lengths)
        self.assertEqual(first,audit.summarize_lengths(list(reversed(lengths))))
        self.assertEqual(first['percent_gt_512'],50.)
        self.assertEqual(first['median'],400.)
        got=audit.retention(lengths,512)
        expected=np.array([1.,1.,512/600,512/1000])
        self.assertAlmostEqual(got['mean_retained'],expected.mean())
        self.assertAlmostEqual(got['p10_retained'],np.quantile(expected,.1))

    def test_proposals_not_training_and_artifacts_unique(self):
        proposal=audit.proposals()
        self.assertEqual(proposal.max_length.tolist(),[512,256])
        self.assertTrue(proposal.status.eq('supported by completed token audit').all())
        token=pd.read_csv(audit.TABLES/'transformer_token_length_audit.csv')
        trunc=pd.read_csv(audit.TABLES/'transformer_truncation_audit.csv')
        self.assertTrue(token.dataset.is_unique)
        self.assertFalse(trunc.duplicated(['dataset','max_length']).any())
        self.assertEqual(len(trunc),8)
        meta=json.loads(audit.REPORT.with_suffix('.json').read_text())
        self.assertEqual(meta['script_sha256'],audit.sha(audit.__file__))
        self.assertEqual(meta['requirements_sha256'],audit.sha(audit.ROOT/'requirements-transformer.txt'))
        self.assertEqual(audit.unchanged_output_stats(),meta['unrelated_output_stat_checks'])
        self.assertTrue(meta['original_environment_unchanged'])

    def test_official_tokenizer_count_determinism(self):
        t=audit.load_tokenizer()
        self.assertEqual(t.model_max_length,512)
        self.assertEqual(len(audit.checkpoint_metadata()['revision']),40)
        a=t('A short synthetic health headline.',truncation=False)['input_ids']
        b=t('A short synthetic health headline.',truncation=False)['input_ids']
        self.assertEqual(a,b)
        rows=pd.read_csv(audit.TABLES/'transformer_token_length_audit.csv')
        self.assertEqual(rows.N.tolist(),rows.determinism_verified_N.tolist())

    def test_official_title_packing_boundaries(self):
        t=audit.load_tokenizer()
        title=t('Health headline',add_special_tokens=False)['input_ids']
        body=t('body text '*800,add_special_tokens=False,truncation=False)['input_ids']
        for budget in (128,256,384,512):
            packed=audit.pack_title_body(t,title,body,budget)
            expected=t('Health headline','body text '*800,truncation='only_second',max_length=budget)['input_ids']
            self.assertEqual(packed,expected)
            self.assertEqual(len(packed),budget)
        self.assertEqual(len(audit.pack_title_body(t,[10]*508,body,512)),512)
        with self.assertRaises(ValueError):audit.pack_title_body(t,[10]*509,body,512)
        self.assertEqual(len(audit.pack_title_body(t,[],[],512)),4)
        rows=pd.read_csv(audit.TABLES/'transformer_truncation_audit.csv')
        self.assertTrue((rows.packed_full_title_verified_N+rows.titles_over_usable_pair_budget_N).eq(rows.N).all())
        self.assertTrue(np.allclose(rows.naive_combined_title_loss_percent,100*rows.naive_combined_title_loss_N/rows.N))

    def test_inference_only_pilot(self):
        meta=json.loads(audit.REPORT.with_suffix('.json').read_text())
        p=meta['pilot']
        self.assertEqual(p['status'],'completed')
        self.assertEqual(p['examples_per_dataset'],{'FakeHealth':8,'CoAID':8})
        self.assertEqual(p['unique_examples'],16)
        self.assertEqual([r['batch_size'] for r in p['rows']],[1,2])
        self.assertFalse(p['gradients_created'])
        self.assertTrue(p['parameter_versions_unchanged'])
        self.assertEqual(p['backward_passes'],0)
        self.assertEqual(p['optimizer_steps'],0)
        self.assertTrue(all(r['examples_per_second']>0 for r in p['rows']))


if __name__=='__main__':
    unittest.main()
