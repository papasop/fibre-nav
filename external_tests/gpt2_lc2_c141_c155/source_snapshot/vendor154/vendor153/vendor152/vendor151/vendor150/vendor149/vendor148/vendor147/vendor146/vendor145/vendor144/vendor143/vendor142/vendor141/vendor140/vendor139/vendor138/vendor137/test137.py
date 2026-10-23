import contextlib,io,json,tempfile,unittest
from unittest.mock import patch
from core137 import *
from worker137 import run
from fixture137 import make_fixture,Toy,Analytic

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
        return dict(V=v,initial_a=np.array([.01,-.02]),points={r['state_id']:dict(prepared=theta.copy(),raw=raw.copy()) for r in registry()},maps={(r['state_id'],arm):dict(point=raw.copy() if arm==ARMS[0] else theta.copy()) for r in registry() for arm in ARMS[:2]})
    def ready_zero(self):
        d=self.data();p=Phase(d)
        for key,(h,case,write) in p.preflight.items():
            sid=key[1];point=d['points'][sid]['prepared'] if key[0]=='prepared' else (proposal(d['points'][sid]['prepared'],d['points'][sid]['raw'],d['V'],d['initial_a'])[0] if key[0]=='warm' else d['maps'][(sid,key[2])]['point']);p.authorize(key,point,case,write);p.consume(point,case,write)
        p.begin_zero();return d,p
    def ready_train(self):
        d,p=self.ready_zero();a=np.zeros(2)
        for spec in registry():p.functional(spec['state_id'],a,d['points'][spec['state_id']]['raw'])
        p.begin_warm()
        for spec in registry():
            sid=spec['state_id'];p.functional(sid,d['initial_a'],proposal(d['points'][sid]['prepared'],d['points'][sid]['raw'],d['V'],d['initial_a'])[0])
        p.begin_training();return d,p
    def test_full24_training_and_fixed_final48_native_endpoints(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['completed_training_steps'],24);self.assertEqual(s['new_backward_calls'],288);self.assertEqual(s['new_model_forwards'],817);self.assertEqual(s['final_endpoint_records'],48);self.assertEqual(s['new_head_fits'],1);self.assertTrue(s['step24_frozen_before_final_evaluation']);self.assertTrue(s['functional_zero_parity_complete_before_gradients']);self.assertFalse(s['best_checkpoint_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.assertNotIn('torch',sys.modules)
        with np.load(out/'FINAL_SHARED_CONTROL_STEP24.npz',allow_pickle=False) as f,np.load(out/'training/step24/UPDATE.npz',allow_pickle=False) as last:exact(f['a'],last['after_a'],'Not final24')
        rows=clean_json((out/'ALL48_FINAL_TRAINING_ENDPOINTS.json').read_bytes());self.assertEqual(len(rows),48);self.assertTrue(all(r['C137_membership']=='training' and not r['independent_validation_eligible'] for r in rows));self.assertEqual(sum(r['role']=='historical_seen_center_probe' for r in rows),24)
        for step in range(1,25):
            with np.load(out/f'training/step{step:02d}/UPDATE.npz',allow_pickle=False) as f:
                a,m,v,_=adam_step(f['before_a'],f['full_weighted_gradient'],f['before_m'],f['before_v'],step);close(a,f['after_a'],'Update arithmetic',1e-15);close(m,f['after_m'],'Moment arithmetic',1e-15)
    def test_registry_promotes_all12_to_explicit_training(self):
        self.assertEqual(len(registry()),12);self.assertEqual(design()['independent_confirmation_states'],0);self.assertEqual(STEPS,24);self.assertTrue(all('xor_10' not in r['state_id'] for r in registry()))
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
        with self.assertRaises(ValueError):p.functional(sid,np.array([.011,0.]),d['points'][sid]['raw'])
    def test_duplicate_functional_zero_state_blocks(self):
        d,p=self.ready_zero();sid=registry()[0]['state_id'];a=np.zeros(2);p.functional(sid,a,d['points'][sid]['raw'])
        with self.assertRaises(ValueError):p.functional(sid,a,d['points'][sid]['raw'])
    def test_unknown_training_state_or_wrong_vector_blocks(self):
        d,p=self.ready_train();a=d['initial_a'];p.begin_batch(0,a)
        with self.assertRaises(ValueError):p.functional('xor_10__forbidden',a,np.zeros(DIM,np.float32))
        sid=registry()[0]['state_id']
        with self.assertRaises(ValueError):p.functional(sid,np.array([.011,0.]),d['points'][sid]['raw'])
    def test_missing_training_state_blocks_update_and_early_freeze(self):
        d,p=self.ready_train();p.begin_batch(0,np.zeros(2))
        with self.assertRaises(ValueError):p.end_batch()
        with self.assertRaises(ValueError):p.freeze({})
    def test_step_skipping_or_extra_step_rejected(self):
        d,p=self.ready_train()
        with self.assertRaises(ValueError):p.begin_batch(1,np.zeros(2))
        with self.assertRaises(ValueError):p.begin_batch(24,np.zeros(2))
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
    def test_zero_gradient_retains_all24_steps_without_success_reclassification(self):
        s,out=self.execute(mode='zero_gradient');self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['completed_training_steps'],24);self.assertFalse(s['LC2_closed'])
        for pair in s['paired_vs_C136_step16'].values():self.assertEqual(pair,dict(rescued_ids=[],destroyed_ids=[]))
        with np.load(out/'FINAL_SHARED_CONTROL_STEP24.npz',allow_pickle=False) as f,np.load(out/'INITIAL_C136_STEP16_CONTROL.npz',allow_pickle=False) as initial:exact(f['a'],initial['a'],'Zero gradient must retain initial warm vector')
    def test_interrupted_training_never_qualifies_partial_checkpoint(self):
        s,out=self.execute(mode='interrupt');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],2);self.assertFalse(s['step24_frozen_before_final_evaluation']);self.assertEqual(s['final_endpoint_records'],0);self.assertFalse((out/'FINAL_SHARED_CONTROL_STEP24.npz').exists());self.assertTrue((out/'LATEST_PARTIAL_CONTROL.npz').is_file())
    def test_full_worker_uses_no_new_solver_or_SVD(self):
        with patch('numpy.linalg.svd',side_effect=AssertionError('SVD forbidden')),patch('numpy.linalg.solve',side_effect=AssertionError('solve forbidden')):s,_=self.execute()
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_Jacobians'],0);self.assertEqual(s['new_SVD_calls'],0)

    def test_logmeanexp_weights_and_value_match_central_difference(self):
        losses=np.linspace(.01,.3,12);grads=np.arange(36,dtype=float).reshape(12,3)/13
        g,rec=aggregate(losses,grads);eps=1e-6
        for j in range(3):
            plus=aggregate(losses+eps*grads[:,j],grads)[1]['objective'];minus=aggregate(losses-eps*grads[:,j],grads)[1]['objective']
            self.assertAlmostEqual(g[j],(plus-minus)/(2*eps),8)
        weights=np.array(rec['weights']);self.assertTrue(np.all(weights>=.1/12));self.assertAlmostEqual(weights.sum(),1.);self.assertGreater(weights[-1],weights[0])
    def test_equal_losses_give_uniform_weights_and_mean_gradient(self):
        gradients=np.arange(24,dtype=float).reshape(12,2);g,rec=aggregate(np.full(12,.2),gradients)
        np.testing.assert_allclose(rec['weights'],np.full(12,1/12),rtol=0,atol=1e-15);np.testing.assert_allclose(g,gradients.mean(0),atol=1e-14);self.assertAlmostEqual(rec['objective'],.2)
    def test_aggregate_remains_finite_for_extreme_loss(self):
        losses=np.zeros(12);losses[4]=1e6;g,rec=aggregate(losses,np.ones((12,3)))
        self.assertTrue(np.isfinite(rec['objective']));np.testing.assert_allclose(g,np.ones(3));self.assertGreater(rec['weights'][4],.9)
    def test_aggregate_rejects_missing_nonfinite_or_negative_losses(self):
        for losses in (np.zeros(11),np.r_[np.zeros(11),np.nan],np.r_[np.zeros(11),-.1]):
            with self.assertRaises(ValueError):aggregate(losses,np.ones((12,3)))
        with self.assertRaises(ValueError):aggregate(np.zeros(12),np.ones((11,3)))
    def test_initial_control_is_exact_C136_step16_and_moments_reset(self):
        s,out=self.execute();self.assertTrue(s['complete'],s['errors'])
        with contextlib.closing(Archive(self.paths['C136'])) as z:parent=z.arrays('FINAL_SHARED_CONTROL_STEP16.npz')['a']
        with np.load(out/'INITIAL_C136_STEP16_CONTROL.npz',allow_pickle=False) as f,np.load(out/'training/step01/UPDATE.npz',allow_pickle=False) as u:
            exact(f['a'],parent,'Wrong parent');exact(u['before_a'],parent,'Wrong first vector');self.assertTrue(np.all(u['before_m']==0) and np.all(u['before_v']==0))
        receipt=clean_json((out/'C136_BOUND_RECEIPT.json').read_bytes());self.assertEqual(receipt['training_rows_checked'],192);self.assertFalse(receipt['parent_decision_reclassified']);self.assertTrue(s['functional_warm_parity_complete_before_gradients'])
    def test_warm_drift_blocks_before_first_gradient(self):
        s,_=self.execute(mode='warm_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['completed_training_steps'],0)
    def test_wrong_aggregate_weight_blocks_before_update(self):
        s,_=self.execute(mode='wrong_weight');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_changed_unweighted_gradient_blocks_before_update(self):
        s,_=self.execute(mode='wrong_state_gradient');self.assertFalse(s['complete']);self.assertEqual(s['completed_training_steps'],0)
    def test_training_requires_complete_warm_parity(self):
        d,p=self.ready_zero()
        for spec in registry():p.functional(spec['state_id'],np.zeros(2),d['points'][spec['state_id']]['raw'])
        with self.assertRaises(ValueError):p.begin_training()
        p.begin_warm()
        with self.assertRaises(ValueError):p.begin_training()
    def test_wrong_C136_hash_blocks_before_model(self):
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C136']['sha256']='0'*64;s,_=self.execute(c=c);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)

    def rebound_parent(self,name,edit):
        import zipfile
        with zipfile.ZipFile(self.paths['C136']) as z:payload={n:z.read(n) for n in z.namelist()}
        payload[name]=edit(payload[name]);payload['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in payload.items() if n!='RESULT_MANIFEST.json'})+'\n').encode()
        path=self.fresh()/'rebound.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for n,b in payload.items():z.writestr(n,b)
        c=dict(self.c,pins={k:dict(v) for k,v in self.c['pins'].items()});c['pins']['C136']['sha256']=sha(path)
        out=self.fresh()/'out'
        with contextlib.redirect_stdout(io.StringIO()):s=run(dict(self.paths,C136=path),out,c,Toy,Analytic)
        return s
    def test_C136_final_vector_change_blocks_even_rebound_manifest(self):
        def change(b):
            with np.load(io.BytesIO(b),allow_pickle=False) as f:a=f['a'].copy()
            a[0]+=.001;out=io.BytesIO();np.savez_compressed(out,a=a);return out.getvalue()
        s=self.rebound_parent('FINAL_SHARED_CONTROL_STEP16.npz',change);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_C136_objective_forgery_blocks_even_rebound_manifest(self):
        def change(b):
            rows=json.loads(b);rows[0]['objective']+=1;return json.dumps(rows).encode()
        s=self.rebound_parent('training/step01/OBSERVATIONS.json',change);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_C136_source_change_blocks_even_rebound_manifest(self):
        s=self.rebound_parent('source/core136.py',lambda b:b+b'\n# changed\n');self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_C136_gate_forgery_blocks_even_rebound_manifest(self):
        def change(b):
            rows=json.loads(b);rows[0]['recomputed_gates']['complete_pass']=not rows[0]['recomputed_gates']['complete_pass'];return json.dumps(rows).encode()
        s=self.rebound_parent('ALL48_FINAL_TRAINING_ENDPOINTS.json',change);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)

if __name__=='__main__':unittest.main()
