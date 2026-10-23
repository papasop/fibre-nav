"""CPU-only corruption and independent arithmetic tests; no model or fitting."""
import copy,contextlib,io,tempfile,unittest
from unittest.mock import patch
from core153 import *
from audit153 import audit
from fixture153 import base_fixture,mutate,json_change,npz_change,loader

class MathChecks(unittest.TestCase):
    def test_two_rates_same_initial_moments_and_gradient_give_fivefold_first_update(self):
        p=np.zeros((3,2,3));g=np.arange(-9,9,dtype=float).reshape(p.shape);x,m,v,i=update(p,g,p,p,1,TRAIN_ARMS[0]);y,n,w,j=update(p,g,p,p,1,TRAIN_ARMS[1]);exact(m,n,'Matched first moment');exact(v,w,'Matched second moment');np.testing.assert_allclose(x,5*y,atol=1e-18);self.assertEqual([i['learning_rate'],j['learning_rate']],[.0005,.0001])
    def test_independent_arithmetic_does_not_call_parent_loss_or_optimizer(self):
        p=np.zeros((3,2,3));w=np.ones(3)/3;theta=np.zeros(DIM,dtype=np.float32);V=np.zeros((3,DIM));V[:,:3]=np.eye(3)
        with patch.object(parent,'effective',side_effect=AssertionError('parent effective')),patch.object(parent,'update',side_effect=AssertionError('parent update')),patch.object(parent,'loss',side_effect=AssertionError('parent loss')):
            e=effective(p,w,TRAIN_ARMS[0]);q=proposal(theta,theta,V,e);r=loss(np.zeros(22),0.,q,theta,e,'xor_00');update(p,p,p,p,1,TRAIN_ARMS[0]);self.assertEqual(r['response_loss'],0.)
    def test_unregistered_rate_arm_and_excluded_case_block(self):
        p=np.zeros((3,2,3))
        with self.assertRaises(ValueError):update(p,p,p,p,1,'routed_cyclic_coupling')
        with self.assertRaises(ValueError):loss(np.zeros(22),0.,np.zeros(DIM,dtype=np.float32),np.zeros(DIM,dtype=np.float32),np.zeros((2,3)),'xor_10')
    def test_positive_cyclic_shift_matches_dense_operator(self):
        V=np.zeros((3,DIM));V[:,:3]=np.eye(3);p=np.zeros(DIM,dtype=np.float32);raw=p.copy();raw[:3]=[1.,2.,3.];e=np.array([[.01,-.02,.03],[.02,.03,-.01]]);P=np.array([[0,0,1],[1,0,0],[0,1,0]],float);want=(raw.astype(float)+V.T@(np.diag(e[0])+np.diag(e[1])@P)@raw[:3]).astype(np.float32);exact(proposal(p,raw,V,e),want,'Fixed positive shift')
    def test_effective_mixing_orders_accept_only_exact_known_digest(self):
        p=np.arange(18,dtype=float).reshape(3,2,3)/1000;w=np.array([.2,.3,.5]);e=effective(p,w,TRAIN_ARMS[1]);actual,order=archived_effective(p,w,TRAIN_ARMS[1],row_digest(e));exact(actual,e,'Exact digest reproduction');self.assertEqual(order,'explicit_left_sum')
        with self.assertRaises(ValueError):archived_effective(p,w,TRAIN_ARMS[1],'0'*64)
    def test_pullback_matches_independent_central_difference(self):
        p=np.arange(18,dtype=float).reshape(3,2,3)/1000;w=np.array([.2,.3,.5]);e=effective(p,w,TRAIN_ARMS[1]);g=2*e;want=pullback(g,w,TRAIN_ARMS[1]);fd=np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            a=p.copy();b=p.copy();a[idx]+=1e-6;b[idx]-=1e-6;fd[idx]=((effective(a,w,TRAIN_ARMS[1])**2).sum()-(effective(b,w,TRAIN_ARMS[1])**2).sum())/2e-6
        np.testing.assert_allclose(fd,want,atol=1e-12,rtol=1e-8)
    def test_first_adam_direction_clipping_and_noop(self):
        p=np.zeros((3,2,3));g=np.arange(-9,9,dtype=float).reshape(p.shape);n,m,v,info=update(p,g,p,p,1,TRAIN_ARMS[0]);cl=g/np.linalg.norm(g);np.testing.assert_allclose(n,-.0005*cl/(abs(cl)+1e-8),atol=1e-15);self.assertLess(info['gradient_clip_factor'],1.);z,_,_,_=update(p,p,p,p,1,TRAIN_ARMS[0]);exact(z,p,'No arbitrary zero-gradient direction')
    def test_box_projection_and_unregistered_step_block(self):
        p=np.full((3,2,3),.125);g=-np.ones_like(p);n,_,_,i=update(p,g,np.zeros_like(p),np.zeros_like(p),1,TRAIN_ARMS[0]);self.assertEqual(i['projected_coordinates'],p.size);exact(n,p,'Fixed box')
        with self.assertRaises(ValueError):update(p,g,p,p,13,TRAIN_ARMS[0])
    def test_exact_max_ties_average_all_states(self):
        rows=[dict(state_id=sp['state_id'],objective_parts=dict(response_loss=1.,read_loss=0.),response_gradient=np.full((3,2,2),i).tolist(),guard_gradient=np.zeros((3,2,2)).tolist()) for i,sp in enumerate(registry())];g,a=aggregate(rows,(3,2,2));np.testing.assert_allclose(g,41.5);self.assertEqual(a['exact_max_tie_count'],84)
        with self.assertRaises(ValueError):aggregate(rows[:-1],(3,2,2))
        with self.assertRaises(ValueError):aggregate(rows[::-1],(3,2,2))
    def test_nonfinite_and_bad_shapes_block(self):
        with self.assertRaises(ValueError):effective(np.full((3,2,3),np.nan),np.ones(3)/3,TRAIN_ARMS[1])
        with self.assertRaises(ValueError):effective(np.zeros((3,3,3)),np.ones(3)/3,TRAIN_ARMS[1])
    def test_zero_cosine_is_undefined(self):self.assertIsNone(cosine(np.zeros(3),np.ones(3)))
    def test_model_import_and_solvers_block(self):
        with no_models(),self.assertRaises(ValueError):__import__('torch')
        with safe.no_solvers(),self.assertRaises(Exception):np.linalg.svd(np.eye(2))

class ArchiveChecks(unittest.TestCase):
    def check(self,member=None,fn=None,rebind=False,passes=False):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            if member:paths,c=mutate(root,member,fn,rebind)
            else:_,paths,c=base_fixture()
            with no_models(),safe.no_solvers(),contextlib.redirect_stdout(io.StringIO()):
                if not passes:
                    with self.assertRaises((ValueError,KeyError,TypeError,AssertionError)):audit(paths,root/'out',c,input_loader=loader)
                    return
                result=audit(paths,root/'out',c,input_loader=loader)
            return result
    def test_full_archive_no_models_all_denominators_and_parent_retained(self):
        s=self.check(passes=True);self.assertTrue(s['complete']);self.assertEqual(s['training_updates_audited'],24);self.assertEqual(s['training_records_audited'],2016);self.assertEqual(s['frozen_proposals_recomputed'],336);self.assertEqual(s['native_endpoint_records_recomputed'],672);self.assertEqual(sum(s['archived_effective_digest_matches_by_registered_order'].values()),2016);self.assertFalse(s['LC2_closed']);self.assertFalse(s['parent_decision_reclassified']);self.assertEqual(s['new_model_forwards'],0)
    def test_actual_final_head_change_blocks_even_rebound_seal(self):
        self.check('FINAL_'+TRAIN_ARMS[1]+'_STEP12.npz',npz_change(lambda a:a['parameters'].__setitem__((0,1,0),.0123)),True)
    def test_source_change_blocks_even_rebound_manifest(self):self.check('source/core152.py',lambda b:b+b'\n#changed\n')
    def test_actual_initial_cross_block_change_blocks_even_rebound_seal(self):self.check('INITIAL_C150_COUPLED_CONTROL.npz',npz_change(lambda a:a['parameters'].__setitem__((0,1,0),.01)),True)
    def test_nonzero_initial_Adam_moment_blocks(self):self.check('training/'+TRAIN_ARMS[0]+'/step01/UPDATE.npz',npz_change(lambda a:a['before_m'].__setitem__((0,0,0),.01)))
    def test_wrong_recorded_rate_blocks(self):self.check('TRAINING_PROGRESS.json',json_change(lambda r:r[1].__setitem__('learning_rate',.0005)))
    def test_changed_registry_order_blocks_even_rebound_seal(self):self.check('ALL84_TRAINING_REGISTRY_FROZEN.json',json_change(lambda r:r.reverse()),True)
    def test_feature_tail_change_blocks_even_rebound_seal(self):self.check('ALL84_TRAINING_INPUTS.npz',npz_change(lambda a:a[registry()[0]['state_id']+'__features'].__setitem__(-1,.12345)),True)
    def test_fixed_final_is_not_intermediate_checkpoint(self):self.check('FINAL_'+TRAIN_ARMS[0]+'_STEP12.npz',npz_change(lambda a:a['parameters'].__setitem__((0,0,0),.009)),True)
    def test_incomplete_parent_never_accepted(self):self.check('SUMMARY.json',json_change(lambda s:s.__setitem__('complete',False)))
    def test_candidate_selection_flag_blocks(self):self.check('SUMMARY.json',json_change(lambda s:s.__setitem__('candidate_selected',True)))
    def test_unverified_derivative_flag_is_not_promoted(self):
        s=self.check(passes=True);self.assertFalse(s['gradients_independently_recomputed']);self.assertFalse(s['causal_failure_explanation_established']);self.assertFalse(s['rate_diagnostics']['true_gradient_or_failure_cause_established']);self.assertEqual(s['rate_diagnostics']['all84_trajectories_retained'],84);self.assertEqual(s['rate_diagnostics']['all2016_training_observations_retained'],2016)
    def test_actual_worker_default_audit_path_complete_without_model_import(self):
        from worker153 import run
        _,paths,c=base_fixture()
        with tempfile.TemporaryDirectory() as tmp,patch('audit153.load152',loader),contextlib.redirect_stdout(io.StringIO()):
            s=run(paths,Path(tmp)/'out',c)
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['LC2_closed']);self.assertTrue(all(s['original_inputs_unchanged'].values()))
    def test_failed_native_state_retained_in_full84_trajectories(self):
        _,paths,c=base_fixture('failed_final152')
        with tempfile.TemporaryDirectory() as tmp,no_models(),safe.no_solvers(),contextlib.redirect_stdout(io.StringIO()):
            out=Path(tmp)/'out';s=audit(paths,out,c,input_loader=loader);rows=clean_json((out/'ALL84_TWO_RATE_RESPONSE_TRAJECTORIES.json').read_bytes())
        sid=registry()[0]['state_id'];self.assertFalse(s['training_qualified'][TRAIN_ARMS[1]]);self.assertIn(sid,s['rate_diagnostics']['focus_failure_union_ids']);self.assertEqual(len(rows),84);self.assertFalse(rows[0]['arms'][TRAIN_ARMS[1]]['native_final_rounds'][0]['native_gates']['complete_pass']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['excluded_input_states_evaluated'],0)
    def test_frozen_point_change_blocks_even_rebound_seal(self):
        self.check('ALL336_FROZEN_FINAL_POINTS.npz',npz_change(lambda a:a[registry()[0]['state_id']+'__'+TRAIN_ARMS[1]].__setitem__(100,.1)),True)
    def test_missing_state_blocks(self):self.check('training/'+TRAIN_ARMS[1]+'/step01/OBSERVATIONS.json',json_change(lambda r:r.pop()))
    def test_nonfinite_effective_gradient_blocks(self):self.check('training/'+TRAIN_ARMS[1]+'/step01/OBSERVATIONS.json',json_change(lambda r:r[0]['effective_response_gradient'][0].__setitem__(0,float('nan'))))
    def test_wrong_pullback_blocks(self):self.check('training/'+TRAIN_ARMS[1]+'/step01/OBSERVATIONS.json',json_change(lambda r:r[0]['response_gradient'][0][1].__setitem__(0,1.)))
    def test_digest_forgery_blocks(self):self.check('training/'+TRAIN_ARMS[1]+'/step01/OBSERVATIONS.json',json_change(lambda r:r[0].__setitem__('effective_control_digest','0'*64)))
    def test_changed_update_blocks(self):self.check('training/'+TRAIN_ARMS[1]+'/step01/UPDATE.npz',npz_change(lambda a:a['after'].__setitem__((0,1,0),.01)))
    def test_gate_forgery_blocks(self):self.check('ALL672_FINAL_TRAINING_ENDPOINTS.json',json_change(lambda r:r[0]['gates'].__setitem__('complete_pass',not r[0]['gates']['complete_pass'])))
    def test_summary_change_blocks(self):self.check('SUMMARY.json',json_change(lambda s:s.__setitem__('training_qualified',{a:not s['training_qualified'][a] for a in TRAIN_ARMS})))
    def test_threshold_change_blocks(self):self.check('DESIGN_FROZEN_BEFORE_MODEL.json',json_change(lambda s:s['acceptance_budgets'].__setitem__('response',.003)),True)
    def test_missing_repeat_blocks(self):self.check('ALL672_FINAL_TRAINING_ENDPOINTS.json',json_change(lambda r:r.pop()))
    def test_wrong_registered_cyclic_shift_blocks(self):self.check('DESIGN_FROZEN_BEFORE_MODEL.json',json_change(lambda s:s.__setitem__('proposal',s['proposal'].replace('roll(u,1)','roll(u,2)'))),True)
    def test_wrong_production_pin_blocks_before_payload(self):
        _,paths,_=base_fixture()
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):audit(paths,Path(tmp),contract(),input_loader=loader)
if __name__=='__main__':unittest.main(verbosity=2)
