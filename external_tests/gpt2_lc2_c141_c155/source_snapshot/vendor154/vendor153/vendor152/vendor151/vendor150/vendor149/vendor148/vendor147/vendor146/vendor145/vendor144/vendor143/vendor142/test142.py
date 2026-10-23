import copy,contextlib,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core142 import *
from fixture142 import make_fixture,pack
from audit142 import audit
from worker142 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.paths,cls.c=make_fixture(cls.root/'fixture');cls.cached=None
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,paths=None,c=None):
        with contextlib.redirect_stdout(io.StringIO()):return audit(paths or self.paths,self.fresh(),c or self.c)
    def changed(self,member,fn,kind='C141'):
        folder=self.fresh()
        with zipfile.ZipFile(self.paths[kind]) as z:z.extractall(folder)
        p=folder/member
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as f:a={k:f[k] for k in f.files}
            np.savez_compressed(p,**fn(a))
        elif p.suffix=='.json':dump(p,fn(clean_json(p.read_bytes())))
        else:p.write_bytes(fn(p.read_bytes()))
        if member=='ALL288_CONFIRMATION_ENDPOINTS.json':
            records=clean_json(p.read_bytes());dump(folder/'ENDPOINT_PROGRESS.json',records)
            for i,r in enumerate(records):dump(folder/f'endpoints/record{i:03d}.json',r)
        if member=='ALL96_NEW_PREPARATIONS.json':dump(folder/'PREPARATION_PROGRESS.json',clean_json(p.read_bytes()))
        sealpath=folder/'PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json'
        if sealpath.is_file():
            seal=clean_json(sealpath.read_bytes());seal['artifact_seals']={n:sha(folder/n) for n in seal['artifact_seals']};dump(sealpath,seal)
        z=pack(folder,self.fresh()/'changed.zip');paths=dict(self.paths,**{kind:z});c=copy.deepcopy(self.c);c['pins'][kind]['sha256']=sha(z)
        # Rebind the C140 link only when testing a C140 input to reach its tensor checks.
        return paths,c
    def blocks(self,member,fn):
        paths,c=self.changed(member,fn)
        with self.assertRaises((ValueError,KeyError)):self.execute(paths,c)
    def test_full_cpu_readonly_archive_all48_96_288_without_model_or_solve(self):
        with safe.no_solvers():s=self.execute()
        self.assertEqual(s['states_audited'],48);self.assertEqual(s['endpoint_records_audited'],288);self.assertEqual(s['parent_decision'],PARENT_DECISION);self.assertFalse(s['local_confirmation_passed']);self.assertEqual(s['confirmation_summaries']['all48']['0'][ARMS[2]]['passed'],42)
    def test_whole_hash_mismatch_blocks_before_payload(self):
        c=copy.deepcopy(self.c);c['pins']['C141']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA256'):self.execute(c=c)
    def test_production_pin_rejects_synthetic_archive(self):
        with self.assertRaises(ValueError):self.execute(c=contract())
    def test_source_tamper_blocks_even_rebound_manifest(self):self.blocks('source/worker141.py',lambda b:b+b'\n# changed\n')
    def test_fixed_threshold_change_blocks(self):self.blocks('DESIGN_FROZEN_BEFORE_MODEL.json',lambda d:dict(d,acceptance_budgets=dict(d['acceptance_budgets'],response=.003)))
    def test_frozen_control_change_blocks_even_rebound(self):
        def f(a):a['C140_a'][0]+=.001;return a
        self.blocks('FROZEN_SHARED_CONTROLS.npz',f)
    def test_missing_endpoint_retains_full_denominator_and_blocks(self):self.blocks('ALL288_CONFIRMATION_ENDPOINTS.json',lambda r:r[:-1])
    def test_permuted_endpoint_labels_block(self):self.blocks('ALL288_CONFIRMATION_ENDPOINTS.json',lambda r:[r[1],r[0]]+r[2:])
    def test_endpoint_gate_forgery_blocks(self):
        def f(r):r[0]['recomputed_gates']['complete_pass']=not r[0]['recomputed_gates']['complete_pass'];return r
        self.blocks('ALL288_CONFIRMATION_ENDPOINTS.json',f)
    def test_response_scalar_forgery_blocks(self):
        def f(r):r[0]['metrics']['response_drift']+=.001;return r
        self.blocks('ALL288_CONFIRMATION_ENDPOINTS.json',f)
    def test_summary_forgery_blocks(self):
        def f(d):d['confirmation_summaries']['all48']['0'][ARMS[2]]['passed']=48;return d
        self.blocks('SUMMARY.json',f)
    def test_paired_destruction_forgery_blocks(self):
        def f(d):d['all48']['0'][ARMS[2]][ARMS[0]]['destroyed_ids']=['invented'];return d
        self.blocks('PAIRED_VS_RAW_AND_C137.json',f)
    def test_missing_preparation_blocks(self):self.blocks('ALL96_NEW_PREPARATIONS.json',lambda r:r[:-1])
    def test_round2_repeat_forgery_blocks(self):
        def f(r):r[144]['repeat_comparison']['observed_error_Linf']=.1;return r
        self.blocks('ALL288_CONFIRMATION_ENDPOINTS.json',f)
    def test_wrong_capture_shape_blocks(self):
        sid=prior.registry()[0]['state_id']
        def f(a):a['capture3147']=a['capture3147'][:-1];return a
        self.blocks('prewrite/'+sid+'.npz',f)
    def test_full3211_tail_change_blocks(self):
        sid=prior.registry()[0]['state_id']
        def f(a):a['write_features3211'][-1]+=.001;return a
        self.blocks('prewrite/'+sid+'.npz',f)
    def test_nonfinite_tensor_blocks(self):
        sid=prior.registry()[0]['state_id']
        def f(a):a['observed'][0]=np.nan;return a
        self.blocks('prewrite/'+sid+'.npz',f)
    def test_unsafe_archive_paths_block(self):
        p=self.fresh()/'unsafe.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('../bad','bad')
        with self.assertRaises(ValueError):Archive(p)
    def test_duplicate_zip_names_block(self):
        p=self.fresh()/'duplicate.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('x','a');z.writestr('x','b')
        with self.assertRaises(ValueError):Archive(p)
    def test_independent_gates_detect_signed_margin_change(self):
        sp=prior.registry()[0];point=np.zeros(DIM,np.float32)
        from fixture138 import metric
        m=metric(point,sp['case'],False);m['exact_margins'][0]+=.1
        with self.assertRaises(ValueError):native_gates(m,sp['case'],point,False)
    def test_response_effect_sign_and_scale(self):
        d=residual_diagnostic([.004,0,0,0,0,0],[.001,0,0,0,0,0],np.zeros(DIM));self.assertAlmostEqual(d['cancellation_projection_fraction'],.75);self.assertAlmostEqual(d['cancellation_cosine'],1);self.assertAlmostEqual(d['measured_finite_response_effect'][0],-.003)
    def test_response_effect_zero_denominators_not_invented(self):
        d=residual_diagnostic(np.zeros(6),np.zeros(6),np.zeros(DIM));self.assertIsNone(d['cancellation_cosine']);self.assertIsNone(d['cancellation_projection_fraction'])
    def test_response_effect_not_declared_operator_error_or_bound(self):
        d=residual_diagnostic(np.ones(6),np.zeros(6),np.zeros(DIM));self.assertFalse(d['causal_derivative_or_control_accuracy_bound_established'])
    def test_solver_guard_blocks_decompositions(self):
        with safe.no_solvers():
            with self.assertRaises(RuntimeError):np.linalg.svd(np.eye(2))
    def test_worker_packages_failure_without_model_or_closure(self):
        c=copy.deepcopy(self.c);c['pins']['C141']['sha256']='0'*64
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.fresh(),c)
        self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['new_model_forwards'],0)
    def test_kernel_gpu_environment_not_mutated_by_launcher(self):
        import os,launch142
        before=dict(os.environ);env=launch142.child_environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(dict(os.environ),before)
    def test_all_source_imports_are_numpy_only(self):
        import sys
        self.assertFalse(any(k in sys.modules for k in ('torch','transformers','peft','huggingface_hub')))

    def test_signed_float32_anchor_KL_rounding_is_retained(self):
        from fixture138 import metric
        sp=prior.registry()[0];p=np.zeros(DIM,np.float32);m=metric(p,sp['case'],False);m['per_anchor_KL']=[-1e-9]*12
        native_gates(m,sp['case'],p,False)
    def test_launcher_runs_C142_test_and_worker_not_parent(self):
        import inspect,launch142
        source=inspect.getsource(launch142.start)
        self.assertIn("ROOT/'test142.py'",source);self.assertIn("ROOT/'worker142.py'",source);self.assertNotIn("ROOT/'worker141.py'",source)
    def test_C140_actual_step12_tamper_blocks_after_link_rebinding(self):
        def change(a):a['after_a'][0]+=.001;return a
        paths,c=self.changed('training/step12/UPDATE.npz',change,'C140')
        folder=self.fresh()
        with zipfile.ZipFile(self.paths['C141']) as z:z.extractall(folder)
        link=clean_json((folder/'INPUTS_FROZEN.json').read_bytes());link['C140']['sha256']=c['pins']['C140']['sha256'];dump(folder/'INPUTS_FROZEN.json',link)
        seals=clean_json((folder/'PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json').read_bytes());seals['artifact_seals']['INPUTS_FROZEN.json']=sha(folder/'INPUTS_FROZEN.json');dump(folder/'PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json',seals)
        paths['C141']=pack(folder,self.fresh()/'C141_rebound.zip');c['pins']['C141']['sha256']=sha(paths['C141'])
        with self.assertRaisesRegex(ValueError,'actual update12'):self.execute(paths,c)

if __name__=='__main__':unittest.main(verbosity=2)
