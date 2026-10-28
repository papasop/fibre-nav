import contextlib,io,tempfile,unittest
from unittest.mock import patch
from core136 import *
from worker136 import run
from fixture136 import make_fixture,Toy,Analytic

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.paths,cls.c=make_fixture(cls.root/'fixture');cls.serial=0
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def fresh(self):type(self).serial+=1;p=self.root/('case'+str(self.serial));p.mkdir();return p
    def execute(self,mode='normal',rt=Toy,c=None):
        out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,out,c or self.c,rt,lambda *args:Analytic(*args,mode=mode))
        return s,out
    def data(self):
        v=np.zeros((2,DIM));v[0,0]=1;v[1,1]=1;theta=np.zeros(DIM,np.float32);raw=theta.copy();raw[0]=.003;raw[1]=.004
        return dict(V=v,points={r['state_id']:dict(prepared=theta.copy(),raw=raw.copy()) for r in registry()},maps={(r['state_id'],arm):dict(point=raw.copy() if arm==ARMS[0] else theta.copy()) for r in registry() for arm in ARMS[:2]})
    def ready_zero(self):
        d=self.data();p=Phase(d)
        for key,(h,case,write) in p.preflight.items():
            sid=key[1];point=d['points'][sid]['prepared'] if key[0]=='prepared' else d['maps'][(sid,key[2])]['point'];p.authorize(key,point,case,write);p.consume(point,case,write)
        p.begin_zero();return d,p
    def ready_train(self):
        d,p=self.ready_zero();a=np.zeros(2)
        for spec in registry():p.functional(spec['state_id'],a,d['points'][spec['state_id']]['raw'])
        p.begin_training();return d,p
    def test_full16_training_and_fixed_final48_native_endpoints(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['completed_training_steps'],16);self.assertEqual(s['new_backward_calls'],192);self.assertEqual(s['new_model_forwards'],577);self.assertEqual(s['final_endpoint_records'],48);self.assertEqual(s['new_head_fits'],1);self.assertTrue(s['step16_frozen_before_final_evaluation']);self.assertTrue(s['functional_zero_parity_complete_before_gradients']);self.assertFalse(s['best_checkpoint_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.assertNotIn('torch',sys.modules)
        with np.load(out/'FINAL_SHARED_CONTROL_STEP16.npz',allow_pickle=False) as f,np.load(out/'training/step16/UPDATE.npz',allow_pickle=False) as last:exact(f['a'],last['after_a'],'Not final16')
        rows=clean_json((out/'ALL48_FINAL_TRAINING_ENDPOINTS.json').read_bytes());self.assertEqual(len(rows),48);self.assertTrue(all(r['C136_membership']=='training' and not r['independent_validation_eligible'] for r in rows));self.assertEqual(sum(r['role']=='historical_seen_center_probe' for r in rows),24)
        for step in range(1,17):
            with np.load(out/f'training/step{step:02d}/UPDATE.npz',allow_pickle=False) as f:
                a,m,v,_=adam_step(f['before_a'],f['full_mean_gradient'],f['before_m'],f['before_v'],step);close(a,f['after_a'],'Update arithmetic',1e-15);close(m,f['after_m'],'Moment arithmetic',1e-15)
    def test_registry_promotes_all12_to_explicit_training(self):
        self.assertEqual(len(registry()),12);self.assertEqual(design()['independent_confirmation_states'],0);self.assertEqual(STEPS,16);self.assertTrue(all('xor_10' not in r['state_id'] for r in registry()))
    def test_zero_vector_exactly_reproduces_raw_bytes(self):
        d=self.data();p=next(iter(d['points'].values()));point,_=proposal(p['prepared'],p['raw'],d['V'],np.zeros(2));self.assertEqual(safe.tensor_hash(point),safe.tensor_hash(p['raw']))
    def test_diagonal_vector_controls_components_independently(self):
        d=self.data();p=next(iter(d['points'].values()));point,_=proposal(p['prepared'],p['raw'],d['V'],np.array([.1,-.1]));self.assertAlmostEqual(float(point[0]),1.1*float(p['raw'][0]),8);self.assertAlmostEqual(float(point[1]),.9*float(p['raw'][1]),8)
    def test_radius_is_not_silently_clipped(self):
        d=self.data();p=next(iter(d['points'].values()));raw=p['raw'].copy();raw[0]=20.;point,diag=proposal(p['prepared'],raw,d['V'],np.array([.1,0.]));self.assertGreater(diag['actual_delta_norm'],20);self.assertFalse(diag['radial_clipping_performed']);self.assertAlmostEqual(float(point[0]),22.)
    def test_gain_bound_violation_blocks(self):
        d=self.data();p=next(iter(d['points'].values()))
        with self.assertRaises(ValueError):proposal(p['prepared'],p['raw'],d['V'],np.array([.126,0.]))
    def test_nonfinite_control_blocks(self):
        d=self.data();p=next(iter(d['points'].values()))
        with self.assertRaises(ValueError):proposal(p['prepared'],p['raw'],d['V'],np.array([np.nan,0.]))
    def test_manual_first_Adam_has_descent_sign_and_fixed_scale(self):
        a=np.zeros(2);g=np.array([2.,-3.]);new,m,v,r=adam_step(a,g,a.copy(),a.copy(),1);self.assertLess(new[0],0);self.assertGreater(new[1],0);self.assertAlmostEqual(abs(new[0]),RATE,7);self.assertLess(r['gradient_clip_factor'],1.)
    def test_initial_zero_gradient_is_noop_without_invented_direction(self):
        a=np.zeros(3);new,m,v,r=adam_step(a,a.copy(),a.copy(),a.copy(),1);exact(new,a,'Zero update');self.assertEqual(r['update_Linf'],0)
    def test_Adam_projection_is_reported(self):
        a=np.array([.1249]);new,m,v,r=adam_step(a,np.array([-1.]),np.zeros(1),np.zeros(1),1);self.assertEqual(new[0],GAIN_BOUND);self.assertEqual(r['projected_coordinates'],1)
    def test_response_read_KL_and_radius_losses_have_declared_units(self):
        theta=np.zeros(DIM,np.float32);point=theta.copy();point[0]=40;o=np.r_[np.full(6,.003),np.full(16,-.075)];parts=objective(o,.009,point,theta,np.zeros(2),'xor_00');self.assertAlmostEqual(parts['response_loss'],2.);self.assertAlmostEqual(parts['KL_loss'],1.);self.assertAlmostEqual(parts['radius_loss'],1.)
    def test_zero_parity_required_before_training(self):
        d,p=self.ready_zero()
        with self.assertRaises(ValueError):p.begin_training()
    def test_nonzero_control_forbidden_during_zero_parity(self):
        d,p=self.ready_zero();sid=registry()[0]['state_id']
        with self.assertRaises(ValueError):p.functional(sid,np.array([.01,0.]),d['points'][sid]['raw'])
    def test_duplicate_functional_zero_state_blocks(self):
        d,p=self.ready_zero();sid=registry()[0]['state_id'];a=np.zeros(2);p.functional(sid,a,d['points'][sid]['raw'])
        with self.assertRaises(ValueError):p.functional(sid,a,d['points'][sid]['raw'])
    def test_unknown_training_state_or_wrong_vector_blocks(self):
        d,p=self.ready_train();a=np.zeros(2);p.begin_batch(0,a)
        with self.assertRaises(ValueError):p.functional('xor_10__forbidden',a,np.zeros(DIM,np.float32))
        sid=registry()[0]['state_id']
        with self.assertRaises(ValueError):p.functional(sid,np.array([.01,0.]),d['points'][sid]['raw'])
    def test_missing_training_state_blocks_update_and_early_freeze(self):
        d,p=self.ready_train();p.begin_batch(0,np.zeros(2))
        with self.assertRaises(ValueError):p.end_batch()
        with self.assertRaises(ValueError):p.freeze({})
    def test_step_skipping_or_extra_step_rejected(self):
        d,p=self.ready_train()
        with self.assertRaises(ValueError):p.begin_batch(1,np.zeros(2))
        with self.assertRaises(ValueError):p.begin_batch(16,np.zeros(2))
    def test_wrong_original_pin_blocks_before_model(self):
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C135']['sha256']='0'*64;s,_=self.execute(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('Wrong fixed original C135',' '.join(s['errors']))
    def test_source_identity_change_blocks_before_model(self):
        c=dict(self.c,source135=dict(self.c['source135']));c['source135'][next(iter(c['source135']))]='0'*64;s,_=self.execute(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertIn('C135 delivered source changed',' '.join(s['errors']))
    def test_raw_replay_drift_blocks_before_any_functional_gradient(self):
        s,_=self.execute(rt=lambda *args:Toy(*args,preflight_drift=.0003));self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['completed_training_steps'],0)
    def test_functional_zero_drift_blocks_before_first_update(self):
        s,_=self.execute(mode='zero_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['completed_training_steps'],0)
    def test_nonfinite_gradient_blocks_before_first_update(self):
        s,_=self.execute(mode='nonfinite');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_missing_gradient_state_blocks_before_first_update(self):
        s,_=self.execute(mode='missing_state');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_gradient_vector_digest_blocks_before_update(self):
        s,_=self.execute(mode='wrong_digest');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_zero_gradient_retains_all16_steps_without_success_reclassification(self):
        s,out=self.execute(mode='zero_gradient');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['completed_training_steps'],16);self.assertFalse(s['fixed_final_training_qualification_passed']);self.assertFalse(s['LC2_closed'])
        with np.load(out/'FINAL_SHARED_CONTROL_STEP16.npz',allow_pickle=False) as f:self.assertTrue(np.all(f['a']==0))
    def test_interrupted_training_never_qualifies_partial_checkpoint(self):
        s,out=self.execute(mode='interrupt');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],2);self.assertFalse(s['step16_frozen_before_final_evaluation']);self.assertEqual(s['final_endpoint_records'],0);self.assertFalse((out/'FINAL_SHARED_CONTROL_STEP16.npz').exists());self.assertTrue((out/'LATEST_PARTIAL_CONTROL.npz').is_file())
    def test_full_worker_uses_no_new_solver_or_SVD(self):
        with patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')),patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')):s,_=self.execute()
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_Jacobians'],0);self.assertEqual(s['new_SVD_calls'],0)

if __name__=='__main__':unittest.main()
