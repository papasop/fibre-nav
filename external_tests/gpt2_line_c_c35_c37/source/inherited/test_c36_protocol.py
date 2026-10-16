"""Standard-library protocol, packaging and import regressions."""
import sys,tempfile,unittest,zipfile,subprocess,contextlib,io,ast
from pathlib import Path
from c36_paths import configure
ROOT=Path(__file__).resolve().parent;configure(ROOT)
from c36_protocol import *
from c36_parent import probe,resume_score
from c35_protocol import specs as c35_specs
class Tests(unittest.TestCase):
    def test_32_unique_fixed_states(self):
        s=specs();self.assertEqual(len(s),32);self.assertEqual(len({t['id'] for t in s}),32);self.assertEqual(len({t['seed'] for t in s}),32)
    def test_four_routes_equal(self):
        self.assertEqual([sum(t['case']==c for t in specs()) for c in CASES],[8]*4)
    def test_seed_separation(self):
        self.assertFalse({s['seed'] for s in specs()}&{s['seed'] for s in c35_specs()})
    def test_centres_and_norms_unchanged(self):
        self.assertEqual({s['norm'] for s in specs()},{.375,.75})
        self.assertTrue(all(s['base_id']==s['case']+'__c31_train%02d'%(int(s['id'][-2:])%3) for s in specs()))
    def test_replay_fixed_not_selected(self):self.assertEqual(replay_ids(),[c+'__c35_new%02d'%i for c in CASES for i in (0,1)])
    def test_budgets_fixed(self):self.assertEqual([SETTINGS[k] for k in ('response_budget','kl_budget','margin_gate')],[.002,.005,.05])
    def test_no_fit_or_head_selection_in_worker(self):
        tree=ast.parse((ROOT/'worker_c36.py').read_text());calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(calls&{'fit_all','select','fit','compile_affine'})
    def test_heavy_pipeline_selftest_has_no_300_second_limit(self):
        tree=ast.parse((ROOT/'launch_c36.py').read_text())
        found=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='visible' and any(isinstance(x,ast.Constant) and x.value=='C36 archive/tensor/seal/resume tests' for x in n.args)]
        self.assertEqual(len(found),1)
        kw={k.arg:k.value for k in found[0].keywords}
        self.assertIsInstance(kw['timeout'],ast.Constant);self.assertIsNone(kw['timeout'].value)
    def test_root_only_clean_launcher_import(self):
        code='import sys;sys.path.insert(0,sys.argv[1]);import launch_c36;import c30_core;assert launch_c36.ROOT.name==sys.argv[2]'
        r=subprocess.run([sys.executable,'-I','-c',code,str(ROOT),ROOT.name],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
    def test_pinned_C35_source_unchanged(self):check_sources(ROOT)
    def test_parent_freeze_is_specific(self):self.assertEqual(SETTINGS['expected_parent_freeze_sha256'],'d95a6d84f312261d355bf2d584c878eff2caeb2b7d19d38d92e0bc8f1c1faec8')
    def test_source_zip_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('launch.py','pass')
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):probe(z,ROOT)
    def test_archive_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('x','changed');z.writestr('MANIFEST.json',json.dumps({'x':'0'*64}))
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):verify(z,True)
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../x','x');z.writestr('MANIFEST.json','{}')
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):verify(z,True)
    def test_duplicate_zip_rejected(self):
        import warnings
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(p,'w') as z:z.writestr('x','x');z.writestr('x','y');z.writestr('MANIFEST.json','{}')
            with zipfile.ZipFile(p) as z:
                with self.assertRaises(ValueError):verify(z,True)
    def test_nested_manifest_and_last3_backups(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'session/run';dump(out/'identity.json',{'protocol':PROTOCOL});dump(out/'nested/MANIFEST.json',{'evidence':True});drive=Path(d)/'drive'
            for i in range(4):dump(out/'progress.json',{'i':i});p=snapshot(out,drive)
            with zipfile.ZipFile(p) as z:self.assertIn('nested/MANIFEST.json',verify(z,True))
            self.assertEqual(len(list(drive.glob('*.zip'))),3)
    def test_online_and_scope_flags(self):
        self.assertTrue(SETTINGS['no_new_training']);self.assertTrue(SETTINGS['no_branch_selection']);self.assertFalse(SETTINGS['fresh_evaluation']);self.assertFalse(SETTINGS['independent_model_or_chart_seed'])
        self.assertTrue(all(SETTINGS[k]==0 for k in ('online_write_Jacobians','online_write_SVD_calls','online_write_correction_steps','online_write_backtracking')))
if __name__=='__main__':unittest.main()
