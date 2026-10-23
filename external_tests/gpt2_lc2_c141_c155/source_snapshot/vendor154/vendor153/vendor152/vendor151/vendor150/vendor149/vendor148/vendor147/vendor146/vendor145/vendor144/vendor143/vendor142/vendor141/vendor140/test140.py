import contextlib,copy,io,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from core140 import *
from fixture140 import *
from inputs140 import audit139
from worker140 import run,final_summaries

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.paths,cls.c=dummy_inputs(cls.root);cls.cache={};cls.z139,cls.olddata,cls.oldc=archive139(cls.root/'parent')
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fresh(self):return Path(tempfile.mkdtemp(dir=self.root))
    def execute(self,mode='normal'):
        if mode not in self.cache:
            out=self.fresh()/'out'
            with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,self.c,lambda *a:Toy(*a,mode),lambda *a:Analytic(*a,mode),loader)
            self.cache[mode]=(s,out)
        return self.cache[mode]
    def audit(self,path=None):
        out=self.fresh()
        with contextlib.closing(Archive(path or self.z139)) as z:audit139(z,copy.deepcopy(self.olddata),out,self.oldc)
        return clean_json((out/'C139_BOUND_RECEIPT.json').read_bytes())
    def changed(self,n,fn):
        folder=self.fresh()
        with zipfile.ZipFile(self.z139) as z:z.extractall(folder)
        p=folder/n
        if p.suffix=='.json':dump(p,fn(clean_json(p.read_bytes())))
        else:
            with np.load(p,allow_pickle=False) as f:d={k:f[k] for k in f.files}
            np.savez_compressed(p,**fn(d))
        return pack(folder,self.fresh()/'changed.zip')
    def test_full_fixed12_full36_432_backward216_endpoints1657_forwards(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['completed_training_steps'],12);self.assertEqual(s['new_backward_calls'],432);self.assertEqual(s['final_endpoint_records'],216);self.assertEqual(s['new_model_forwards'],1657);self.assertTrue(s['fixed_final_training_qualification_passed']);self.assertTrue((out/'FINAL_SHARED_CONTROL_STEP12.npz').is_file())
    def test_all36_are_training_without_independent_or_excluded_credit(self):
        rr=registry();self.assertEqual(len(rr),36);self.assertEqual(Counter(r['training_origin'] for r in rr),dict(historical12=12,C139_exposed_kernel24=24));self.assertEqual(Counter(r['case'] for r in rr),dict(xor_00=12,xor_01=12,xor_11=12));self.assertTrue(all(r['C140_membership']=='training' and not r['independent_validation_eligible'] for r in rr));self.assertEqual(design()['independent_confirmation_states'],0)
    def test_tighter_training_safety_does_not_change_acceptance(self):self.assertEqual(RESPONSE_SAFETY,.00125);self.assertEqual(design()['acceptance_budgets']['response'],.002);self.assertEqual(design()['acceptance_budgets']['KL'],.005)
    def test_C139_audit_reconstructs24_coordinates48_preps96_endpoints(self):
        r=self.audit();self.assertEqual(r['kernel_coordinates_reconstructed'],24);self.assertEqual(r['preparations_checked'],48);self.assertEqual(r['endpoints_checked'],96);self.assertFalse(r['parent_decision_reclassified'])
    def test_C139_rebound_coordinate_tamper_blocks(self):
        def f(d):d[next(iter(d))][0]+=.01;return d
        with self.assertRaises(ValueError):self.audit(self.changed('ALL24_KERNEL_COORDINATES_BEFORE_MODEL.npz',f))
    def test_C139_rebound_preparation_gate_tamper_blocks(self):
        def f(d):d[0]['gates']['complete_pass']=False;return d
        with self.assertRaises(ValueError):self.audit(self.changed('ALL48_KERNEL_PREPARATIONS.json',f))
    def test_C139_rebound_candidate_tamper_blocks(self):
        def f(d):d['a'][0]+=.01;return d
        with self.assertRaises(ValueError):self.audit(self.changed('FROZEN_C137_CONTROL.npz',f))
    def test_C139_rebound_source_tamper_blocks(self):
        with self.assertRaises(ValueError):self.audit(self.changed('source/PROTOCOL.json',lambda d:dict(d,bad=True)))
    def test_C139_rebound_feature_tail_tamper_blocks(self):
        sid=previous.registry()[0]['state_id']
        def f(d):d['write_features3211'][-1]+=.01;return d
        with self.assertRaises(ValueError):self.audit(self.changed('prewrite/'+sid+'.npz',f))
    def test_C139_rebound_endpoint_gate_tamper_blocks(self):
        def f(d):d[0]['recomputed_gates']['complete_pass']=False;return d
        with self.assertRaises(ValueError):self.audit(self.changed('ALL96_KERNEL_ENDPOINTS.json',f))
    def test_aggregate_matches_independent_central_difference(self):
        rng=np.random.default_rng(42);mat=rng.normal(size=(36,4));a=np.array([.02,-.03,.04,.01]);b=np.linspace(.01,.3,36)
        def value(x):return aggregate((b+mat@x)**2,2*(b+mat@x)[:,None]*mat)[1]['objective']
        g,_=aggregate((b+mat@a)**2,2*(b+mat@a)[:,None]*mat);fd=np.array([(value(a+np.eye(4)[j]*1e-6)-value(a-np.eye(4)[j]*1e-6))/2e-6 for j in range(4)]);np.testing.assert_allclose(g,fd,atol=1e-9,rtol=1e-7)
    def test_aggregate_uniform_floor_and_no_row_deletion(self):
        g,r=aggregate(np.r_[np.zeros(35),1000.],np.ones((36,3)));self.assertTrue(np.all(np.array(r['weights'])>=.1/36));self.assertAlmostEqual(sum(r['weights']),1)
        with self.assertRaises(ValueError):aggregate(np.zeros(35),np.zeros((35,3)))
    def test_nonfinite_gradient_or_loss_blocks(self):
        for l,g in ((np.r_[np.nan,np.zeros(35)],np.ones((36,3))),(np.zeros(36),np.full((36,3),np.inf))):
            with self.assertRaises(ValueError):aggregate(l,g)
    def test_Adam_sign_scale_projection_and_noop(self):
        a=np.array([.1249,-.1249,.01]);g=np.array([-1.,1.,0.]);n,mm,vv,r=adam_step(a,g,np.zeros(3),np.zeros(3),1);self.assertEqual(n[0],.125);self.assertEqual(n[1],-.125);self.assertEqual(n[2],a[2]);self.assertEqual(r['projected_coordinates'],2)
        z=adam_step(a,np.zeros(3),np.zeros(3),np.zeros(3),1)[0];exact(z,a,'Zero update')
    def test_unregistered_step_blocks(self):
        for k in (0,13,True):
            with self.assertRaises(ValueError):adam_step(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),k)
    def test_zero_is_exact_raw_and_no_radial_clipping(self):
        d=data140();p=next(iter(d['points'].values()));point,diag=proposal(p['prepared'],p['raw'],d['V'],np.zeros(len(d['V'])));exact(point,p['raw'],'Zero control');self.assertFalse(diag['radial_clipping_performed'])
    def test_preflight_drift_blocks_all_gradients(self):
        s,_=self.execute('preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_zero_parity_drift_blocks_all_gradients(self):
        s,_=self.execute('zero_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_warm_parity_drift_blocks_all_gradients(self):
        s,_=self.execute('warm_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_missing_training_state_blocks_before_update(self):
        s,_=self.execute('missing_state');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_gradient_weight_blocks_before_update(self):
        s,_=self.execute('wrong_weight');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_control_digest_blocks_before_update(self):
        s,_=self.execute('wrong_digest');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_nonfinite_gradient_blocks_before_update(self):
        s,_=self.execute('nonfinite');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_frozen_basis_mutation_blocks_final_evaluation(self):
        s,_=self.execute('mutate_basis');self.assertFalse(s['complete']);self.assertEqual(s['final_endpoint_records'],0)
    def test_partial_gradient_interruption_keeps_fixed12_denominator(self):
        s,out=self.execute('interrupt');self.assertFalse(s['complete']);self.assertEqual(s['requested_training_steps'],12);self.assertEqual(s['completed_training_steps'],1);self.assertEqual(s['requested_scalar_backward_calls'],432);self.assertEqual(s['final_endpoint_records'],0);self.assertFalse((out/'FINAL_SHARED_CONTROL_STEP12.npz').exists())
    def test_final_failure_retained_no_checkpoint_selection(self):
        s,out=self.execute('failed_final');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['fixed_final_training_qualification_passed']);self.assertFalse(s['best_checkpoint_selected']);self.assertEqual(s['final_training_summaries']['all36']['0'][EVAL_ARMS[2]]['passed'],35);self.assertEqual(s['final_endpoint_records'],216)
    def test_repeat_drift_retained_blocks_qualification(self):
        s,_=self.execute('repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['fixed_final_training_qualification_passed'])
    def test_final_vector_is_exact_step12_and_all_proposals_presealed(self):
        s,out=self.execute();f=np.load(out/'FINAL_SHARED_CONTROL_STEP12.npz')['a'];last=np.load(out/'training/step12/UPDATE.npz')['after_a'];exact(f,last,'Fixed final12');freeze=clean_json((out/'STEP12_FREEZE_BEFORE_EVALUATION.json').read_bytes());self.assertEqual(freeze['completed_steps'],12);self.assertTrue(freeze['no_best_or_recovery_selection']);self.assertEqual(len(np.load(out/'ALL108_FROZEN_FINAL_POINTS.npz').files),108)
    def phase(self):d=data140();return Phase(d),d
    def test_early_training_and_final_freeze_block(self):
        p,d=self.phase()
        with self.assertRaises(ValueError):p.begin_zero()
        with self.assertRaises(ValueError):p.begin_training()
        with self.assertRaises(ValueError):p.freeze({})
    def test_unauthorized_native_and_excluded_functional_blocks(self):
        p,d=self.phase()
        with self.assertRaises(ValueError):p.consume(next(iter(d['points'].values()))['prepared'],'xor_00',False)
        with self.assertRaises(ValueError):p.functional('xor_10__excluded',d['a'],np.zeros(DIM,np.float32))
    def test_wrong_original_blocks_before_model(self):
        c=copy.deepcopy(self.c);c['pins']['C139']['sha256']='0'*64
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.fresh()/'out',c,Toy,Analytic)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_no_solver_svd_or_fitting_used_by_worker(self):
        with safe.no_solvers():s,_=self.execute('no_solvers')
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_Jacobians'],0);self.assertEqual(s['new_SVD_calls'],0)
    def test_success_does_not_close_LC2_or_claim_real_model(self):
        s,_=self.execute();self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['writer_deployed']);self.assertFalse(s['real_model_execution_claim']);self.assertEqual(s['confirmation_states_evaluated'],0)

if __name__=='__main__':unittest.main(verbosity=2)
