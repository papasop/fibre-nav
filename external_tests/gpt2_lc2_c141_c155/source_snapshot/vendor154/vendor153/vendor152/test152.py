"""CPU arithmetic, all84 synthetic workflow and failure blocking; not model evidence."""
import contextlib,io,tempfile,unittest,copy
from unittest.mock import patch
from core152 import *
from worker152 import run
from fixture152 import Native,Analytic,dummy_inputs,loader,worker_data

class MathChecks(unittest.TestCase):
    def test_common_initial_full_cyclic_head_has_nonzero_cross(self):
        d,_=worker_data();a=initial_parameters(d,TRAIN_ARMS[0]);b=initial_parameters(d,TRAIN_ARMS[1]);exact(a,b,'Same complete initial head');self.assertTrue(np.any(a[:,1,:]!=0));a.flat[0]+=.001;self.assertNotEqual(row_digest(a),row_digest(d['initial_P']))
    def test_parameter_counts_matched_and_registered_rates(self):
        d=design();self.assertEqual(d['trainable_parameters'],{a:216 for a in TRAIN_ARMS});self.assertEqual(RATES,{TRAIN_ARMS[0]:5e-4,TRAIN_ARMS[1]:1e-4});self.assertEqual(d['offline_training_steps_per_arm'],12)
    def test_same_gradient_same_moments_only_update_rate_differs(self):
        p=np.zeros((3,2,4));g=np.arange(-12,12,dtype=float).reshape(p.shape);a,m,v,info=update(p,g,p,p,1,TRAIN_ARMS[0]);b,n,u,other=update(p,g,p,p,1,TRAIN_ARMS[1]);exact(m,n,'Same first moments');exact(v,u,'Same second moments');np.testing.assert_allclose(a,5*b,atol=1e-18);self.assertEqual(info['learning_rate'],.0005);self.assertEqual(other['learning_rate'],.0001)
    def test_Adam_zero_gradient_noop_and_projection(self):
        p=np.zeros((3,2,4));a,_,_,_=update(p,p,p,p,1,TRAIN_ARMS[1]);exact(a,p,'No arbitrary zero-gradient direction');q=np.full_like(p,.125);a,_,_,info=update(q,-np.ones_like(p),p,p,1,TRAIN_ARMS[1]);exact(a,q,'Fixed box');self.assertEqual(info['projected_coordinates'],q.size)
    def test_unregistered_arm_or_step_block(self):
        p=np.zeros((3,2,4))
        for arm,step in [('unknown',1),(TRAIN_ARMS[0],0),(TRAIN_ARMS[1],13)]:
            with self.assertRaises(ValueError):update(p,p,p,p,step,arm)
    def test_cyclic_map_matches_independent_dense_operator(self):
        V=np.zeros((3,DIM));V[:,:3]=np.eye(3);p=np.zeros(DIM,dtype=np.float32);raw=p.copy();raw[:3]=[1.,2.,3.];e=np.array([[.01,.02,.03],[.02,-.01,.03]]);P=np.eye(3)[[2,0,1]];want=(raw.astype(float)+V.T@(np.diag(e[0])+np.diag(e[1])@P)@raw[:3]).astype(np.float32);exact(proposal(p,raw,V,e)[0],want,'Fixed positive shift')
    def test_pullback_matches_independent_central_difference(self):
        p=np.arange(18,dtype=float).reshape(3,2,3)/1000;w=np.array([.2,.3,.5]);want=pullback(2*effective(p,w,TRAIN_ARMS[0]),w,TRAIN_ARMS[0]);fd=np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            a=p.copy();b=p.copy();a[idx]+=1e-6;b[idx]-=1e-6;fd[idx]=((effective(a,w,TRAIN_ARMS[0])**2).sum()-(effective(b,w,TRAIN_ARMS[0])**2).sum())/2e-6
        np.testing.assert_allclose(fd,want,atol=1e-12,rtol=1e-8)
    def test_nonfinite_out_of_box_bad_shapes_and_weights_block(self):
        for p,w in [(np.full((3,2,3),np.nan),np.ones(3)/3),(np.full((3,2,3),.126),np.ones(3)/3),(np.zeros((3,3)),np.ones(3)/3),(np.zeros((3,2,3)),np.array([-1.,1.,1.]))]:
            with self.assertRaises(ValueError):effective(p,w,TRAIN_ARMS[0])
    def test_max_ties_and_missing_rows_are_not_selected_away(self):
        rows=[dict(state_id=sp['state_id'],objective_parts=dict(response_loss=1.,read_loss=0.),response_gradient=np.full((3,2,3),i).tolist(),guard_gradient=np.zeros((3,2,3)).tolist()) for i,sp in enumerate(registry())];g,a=aggregate(rows,(3,2,3));np.testing.assert_allclose(g,41.5);self.assertEqual(a['exact_max_tie_count'],84)
        for bad in (rows[:-1],rows[::-1]):
            with self.assertRaises(ValueError):aggregate(bad,(3,2,3))
    def test_initial_map_matches_both_arms_for_all84(self):
        d,_=worker_data()
        for sp in registry():
            sid=sp['state_id'];q=d['points'][sid];a=proposal(q['prepared'],q['raw'],d['V'],effective(initial_parameters(d,TRAIN_ARMS[0]),d['weights'][sid],TRAIN_ARMS[0]))[0];b=proposal(q['prepared'],q['raw'],d['V'],effective(initial_parameters(d,TRAIN_ARMS[1]),d['weights'][sid],TRAIN_ARMS[1]))[0];exact(a,b,'Both initial physical points')
    def test_phase_blocks_partial_zero_warm_batches_or_final_freeze(self):
        d,_=worker_data();ph=Phase(d)
        with self.assertRaises(ValueError):ph.begin_zero()
        with self.assertRaises(ValueError):ph.begin_training()
        with self.assertRaises(ValueError):ph.freeze({})

class WorkflowChecks(unittest.TestCase):
    def test_worker_default_binding_failure_stops_before_input_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,c=dummy_inputs(root/'inputs')
            with patch('worker152.resolve_default_factories',side_effect=ValueError('DEFAULT_BINDING_SENTINEL')) as resolve,patch('worker152.load') as read,contextlib.redirect_stdout(io.StringIO()):
                s=run(paths,root/'out',c)
            resolve.assert_called_once_with();read.assert_not_called();self.assertFalse(s['complete']);self.assertEqual(s['stage'],'production_factory_binding');self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_backward_calls'],0);self.assertTrue(any('DEFAULT_BINDING_SENTINEL' in e for e in s['errors']))
    def test_worker_defaults_are_resolved_before_input_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,c=dummy_inputs(root/'inputs');order=[]
            def resolve():order.append('resolve');return lambda *a:self.fail('model called'),lambda *a:self.fail('backend called')
            def read(*a):order.append('input');raise ValueError('INPUT_SENTINEL')
            with patch('worker152.resolve_default_factories',side_effect=resolve),contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',c,input_loader=read)
            self.assertEqual(order,['resolve','input']);self.assertTrue(s['production_factory_contract_passed']);self.assertEqual(s['stage'],'original_C150_audit_and_common_initialization');self.assertEqual(s['new_model_forwards'],0);self.assertTrue(any('INPUT_SENTINEL' in e for e in s['errors']))
    def workflow(self,mode='normal',native='normal'):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);paths,c=dummy_inputs(root/'inputs');rates=dict(RATES)
        try:
            with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',c,lambda o,d,p,cs:Native(o,d,p,cs,native),lambda rt,d,p,cs:Analytic(rt,d,p,cs,mode),loader)
        finally:RATES.clear();RATES.update(rates)
        return s,root/'out'
    def test_full_two_rate_workflow_complete_counts_and_fixed_final_seals(self):
        s,out=self.workflow();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],24);self.assertEqual(s['new_model_forwards'],6553);self.assertEqual(s['new_backward_calls'],4032);self.assertEqual(s['preflight_records'],336);self.assertEqual(s['zero_parity_records'],84);self.assertEqual(s['warm_parity_records'],168);self.assertEqual(s['final_endpoint_records'],672);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed']);self.assertTrue(s['parameter_counts_matched']);freeze=clean_json((out/'BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json').read_bytes());self.assertEqual(freeze['native_final_measurements_so_far'],0)
        for n,h in freeze['artifact_seals'].items():self.assertEqual(sha(out/n),h)
        states=[]
        for arm in TRAIN_ARMS:
            u=dict(np.load(out/'training'/arm/'step01/UPDATE.npz'));states.append(u);self.assertTrue(np.all(u['before_m']==0));self.assertTrue(np.all(u['before_v']==0));self.assertEqual(s['per_arm_training'][arm]['steps'],12)
        exact(states[0]['before'],states[1]['before'],'Same full initial head');exact(states[0]['gradient'],states[1]['gradient'],'Matched first full84 gradient');np.testing.assert_allclose(states[0]['after']-states[0]['before'],5*(states[1]['after']-states[1]['before']),atol=2e-17)
    def test_final_failure_retained_and_small_rate_not_automatically_qualified(self):
        s,_=self.workflow(native='failed_final152');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['training_qualified'][TRAIN_ARMS[1]]);self.assertEqual(s['final_training_summaries']['all84']['0'][TRAIN_ARMS[1]]['failed_gates']['response'],1)
    def test_repeat_drift_blocks_both_arms(self):
        s,_=self.workflow(native='repeat_drift152');self.assertTrue(s['complete']);self.assertFalse(s['repeatable']);self.assertFalse(any(s['training_qualified'].values()))
    def test_native_replay_drift_blocks_before_backward(self):
        s,_=self.workflow(native='preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_zero_functional_drift_blocks_before_backward(self):
        s,_=self.workflow('zero_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_warm_functional_drift_blocks_before_backward(self):
        s,_=self.workflow('warm_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_interruption_never_qualifies_either_arm(self):
        s,_=self.workflow('interrupt');self.assertFalse(s['complete']);self.assertFalse(any(s['training_qualified'].values()))
    def test_missing_gradient_state_blocks(self):
        s,_=self.workflow('missing_row');self.assertFalse(s['complete'])
    def test_wrong_pullback_blocks(self):
        s,_=self.workflow('wrong_pullback');self.assertFalse(s['complete'])
    def test_nonfinite_gradient_blocks(self):
        s,_=self.workflow('nonfinite');self.assertFalse(s['complete'])
    def test_router_mutation_blocks(self):
        s,_=self.workflow('mutate_router');self.assertFalse(s['complete'])
    def test_registered_rate_mutation_blocks(self):
        s,_=self.workflow('mutate_rate');self.assertFalse(s['complete']);self.assertTrue(any('Registered learning rates changed' in e for e in s['errors']))
if __name__=='__main__':unittest.main(verbosity=2)
