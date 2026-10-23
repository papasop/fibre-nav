"""NumPy archival/phase/algebra tests. Actual CPU Torch smoke runs separately."""
import contextlib,copy,io,tempfile,unittest,zipfile
from unittest.mock import patch
from core143 import *
from fixture143 import worker_data,dummy_inputs,emit_bindings,Toy,Analytic,archive_inputs,pack
from inputs143 import load
from worker143 import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.data=worker_data();cls.paths,cls.c=dummy_inputs(cls.root/'dummy');cls.cache={};cls.real_fixture=None
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'run'
            def loader(paths,out,c):d=copy.deepcopy(self.data);emit_bindings(out,d,paths);return d
            rt=lambda out,d,ph,calls:Toy(out,d,ph,calls,mode)
            backend=lambda rt,d,ph,calls:Analytic(rt,d,ph,calls,mode)
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,rt,backend,loader)
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def archives(self):
        if self.real_fixture is None:
            self.__class__.real_fixture=archive_inputs(self.fresh()/'archives')
        return self.real_fixture
    def test_full84_worker12_updates1008_backwards504_endpoints3865_forwards(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertTrue(s['fixed_final_training_qualification_passed']);self.assertEqual(s['completed_training_steps'],12);self.assertEqual(s['new_backward_calls'],1008);self.assertEqual(s['final_endpoint_records'],504);self.assertEqual(s['new_model_forwards'],3865);self.assertEqual(s['trainable_shared_coordinates'],6)
        self.assertFalse(s['LC2_closed']);self.assertEqual(s['independent_confirmation_states'],0);self.assertEqual(s['final_training_summaries']['all84']['0'][EVAL_ARMS[2]]['passed'],84)
        need(set(np.load(out/'ALL252_FROZEN_FINAL_POINTS.npz').files)=={r['state_id']+'__'+a for r in registry() for a in EVAL_ARMS},'Actual all252 final coverage')
        final=np.load(out/'FINAL_SHARED_CONTROL_STEP12.npz')['a'];last=np.load(out/'training/step12/UPDATE.npz')['after_a'];exact(final,last,'Fixed last update actual tensor')
    def test_registry_all84_explicit_training_no_unseen_or_excluded(self):
        r=registry();self.assertEqual(len(r),84);self.assertEqual(len({x['state_id'] for x in r}),84);self.assertTrue(all(x['dataset_role']=='training' and x['used_for_training'] and not x['independent_validation_eligible'] for x in r));self.assertEqual(Counter(x['case'] for x in r),dict(xor_00=28,xor_01=28,xor_11=28));self.assertEqual(Counter(x['training_origin'] for x in r)['C141_exposed_confirmation48'],48)
    def test_all48_exposed_included_not_only_six_failures(self):self.assertEqual([r['state_id'] for r in registry()[36:]],[r['state_id'] for r in prior.registry()])
    def test_preflight_drift_blocks_before_any_backward(self):
        s,_=self.execute('preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0)
    def test_zero_parity_drift_blocks_before_any_backward(self):
        s,_=self.execute('zero_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertFalse(s['functional_zero_parity_complete_before_gradients'])
    def test_warm_parity_drift_blocks_before_any_backward(self):
        s,_=self.execute('warm_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertFalse(s['functional_warm_parity_complete_before_gradients'])
    def test_partial_update_never_becomes_qualified_checkpoint(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],1);self.assertFalse((out/'FINAL_SHARED_CONTROL_STEP12.npz').exists());self.assertEqual(s['requested_final_endpoint_records'],504)
    def test_missing_gradient_state_blocks(self):
        s,_=self.execute('missing_state');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_gradient_weight_blocks(self):
        s,_=self.execute('wrong_weight');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_shared_vector_digest_blocks(self):
        s,_=self.execute('wrong_digest');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_nonfinite_gradient_blocks(self):
        s,_=self.execute('nonfinite');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_frozen_basis_mutation_blocks(self):
        s,_=self.execute('mutate_basis');self.assertFalse(s['complete'])
    def test_complete_failed_final_retained_without_checkpoint_substitution(self):
        s,_=self.execute('failed_final');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['fixed_final_training_qualification_passed']);self.assertEqual(s['final_endpoint_records'],504);self.assertEqual(s['final_training_summaries']['all84']['0'][EVAL_ARMS[2]]['passed'],83);self.assertEqual(s['decision'],'C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE')
    def test_early_final_freeze_and_missing_full_batch_block(self):
        d=copy.deepcopy(self.data);phase=Phase(d)
        with self.assertRaises(ValueError):phase.freeze({})
        with self.assertRaises(ValueError):phase.begin_zero()
    def test_84_weighted_aggregate_gradient_matches_central_difference(self):
        rng=np.random.default_rng(143);M=rng.normal(size=(84,3));a=np.array([.02,-.01,.03]);b=np.linspace(.01,.3,84)
        def f(x):loss=.01+(M@x+b)**2;grad=2*(M@x+b)[:,None]*M;return aggregate(loss,grad)
        g,rec=f(a);fd=np.array([(f(a+np.eye(3)[j]*1e-6)[1]['objective']-f(a-np.eye(3)[j]*1e-6)[1]['objective'])/2e-6 for j in range(3)]);np.testing.assert_allclose(g,fd,atol=1e-10,rtol=1e-8);self.assertAlmostEqual(sum(rec['weights']),1.)
    def test_Adam_clipped_direction_sign_scale_and_noop(self):
        a=np.array([.02,-.03]);g=np.array([3.,-4.]);after,m,v,rec=adam_step(a,g,np.zeros(2),np.zeros(2),1);clipped=g/5;expected=a-RATE*clipped/(abs(clipped)+ADAM_EPS);np.testing.assert_allclose(after,expected,atol=1e-15,rtol=1e-12);self.assertAlmostEqual(rec['gradient_clip_factor'],.2)
        zero,_,_,_=adam_step(a,np.zeros(2),np.zeros(2),np.zeros(2),1);exact(zero,a,'Noop zero gradient')
    def test_unregistered_update_and_nonfinite_loss_block(self):
        with self.assertRaises(ValueError):adam_step(np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2),13)
        with self.assertRaises(ValueError):aggregate(np.r_[np.nan,np.zeros(83)],np.zeros((84,2)))
    def test_zero_control_raw_identity_and_no_online_radius_clipping(self):
        p=np.zeros(DIM,np.float32);r=p.copy();r[0]=21.;V=np.zeros((1,DIM));V[0,0]=1;out,diag=proposal(p,r,V,np.zeros(1));exact(out,r,'Zero control retains raw');self.assertFalse(diag['radial_clipping_performed']);self.assertGreater(diag['actual_delta_norm'],20.)
    def test_three_original_hashes_block_before_model(self):
        for k in INPUT_KEYS:
            c=copy.deepcopy(self.c);c['pins'][k]['sha256']='0'*64
            with self.assertRaises(ValueError):load(self.paths,self.fresh(),c)
    def test_original_archive_reader_reconstructs84_and_resets_moments(self):
        paths,c=self.archives();out=self.fresh()
        with contextlib.redirect_stdout(io.StringIO()):d=load(paths,out,c)
        self.assertEqual(len(d['points']),84);self.assertEqual(d['C141_parent_decision'],'C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE');f=np.load(out/'INITIAL_C140_STEP12_CONTROL.npz');exact(f['m'],np.zeros_like(f['a']),'Reset first moment');exact(f['v'],np.zeros_like(f['a']),'Reset second moment')
    def test_C142_rebound_source_change_blocks(self):
        paths,c=self.archives();folder=self.fresh()
        with zipfile.ZipFile(paths['C142']) as z:z.extractall(folder)
        with (folder/'source/worker142.py').open('a') as f:f.write('\n# tampered\n')
        bad=pack(folder,self.fresh()/'bad.zip');pp=dict(paths,C142=bad);cc=copy.deepcopy(c);cc['pins']['C142']['sha256']=sha(bad)
        with self.assertRaisesRegex(ValueError,'trusted source'):load(pp,self.fresh(),cc)
    def test_C142_rebound_parent_decision_forgery_blocks(self):
        paths,c=self.archives();folder=self.fresh()
        with zipfile.ZipFile(paths['C142']) as z:z.extractall(folder)
        s=clean_json((folder/'SUMMARY.json').read_bytes());s['parent_decision']='C141_PASSED';dump(folder/'SUMMARY.json',s);bad=pack(folder,self.fresh()/'bad.zip');pp=dict(paths,C142=bad);cc=copy.deepcopy(c);cc['pins']['C142']['sha256']=sha(bad)
        with self.assertRaises(ValueError):load(pp,self.fresh(),cc)
    def test_production_contract_rejects_fixture_before_payload(self):
        with self.assertRaises(ValueError):load(self.paths,self.fresh(),contract())
    def test_launcher_GPU_mask_is_child_only_and_correct_worker(self):
        import os,inspect,launch143
        before=dict(os.environ);env=launch143.child_environment();self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(before,dict(os.environ));source=inspect.getsource(launch143.start);self.assertIn("ROOT/'worker143.py'",source);self.assertIn("ROOT/'test143.py'",source);self.assertIn("ROOT/'torch_smoke143.py'",source)

if __name__=='__main__':unittest.main(verbosity=2)
