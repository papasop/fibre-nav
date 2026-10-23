"""CPU protocol tests. Full analytic workflow is synthetic, not model evidence."""
import copy,contextlib,io,tempfile,unittest
from core150 import *
from fixture150 import Native,Analytic,loader,dummy_inputs,worker_data
from worker150 import run

class Checks(unittest.TestCase):
    def test_router_deterministic_and_convex(self):
        U=np.arange(84*4,dtype=float).reshape(84,4)/100;r,W,m=old.fit_router(U);q,X,n=old.fit_router(U);exact(W,X,'Deterministic routing');compare(m,n);np.testing.assert_allclose(W.sum(1),1,atol=1e-14);self.assertTrue(np.all(W>=0));self.assertEqual(m['lloyd_iterations'],8)
    def test_identical_contexts_keep_empty_clusters_without_search(self):
        r,W,m=old.fit_router(np.ones((84,3)));np.testing.assert_allclose(W,np.full((84,3),1/3));self.assertTrue(m['empty_cluster_events']);self.assertEqual(m['seed_registry_indices'],[0,1,2])
    def test_router_unseen_context_needs_no_case_or_state_id(self):
        r,_,_=old.fit_router(np.arange(168,dtype=float).reshape(84,2));w=auditcore.route(np.array([123.,-456.]),r);self.assertEqual(w.shape,(3,));self.assertAlmostEqual(sum(w),1.)
    def test_router_rejects_wrong_training_denominator(self):
        with self.assertRaises(ValueError):old.fit_router(np.zeros((83,3)))
    def test_router_rejects_nonfinite(self):
        x=np.zeros((84,2));x[1,0]=np.nan
        with self.assertRaises(ValueError):old.fit_router(x)
    def test_invalid_scale_blocks(self):
        r,_,_=old.fit_router(np.zeros((84,2)));r['scale'][0]=0
        with self.assertRaises(ValueError):auditcore.route(np.zeros(2),r)
    def test_duplicated_initial_experts_match_global(self):
        a=np.array([.01,-.02]);w=np.array([.2,.3,.5]);np.testing.assert_allclose(effective(np.tile(a,(3,1)),w,TRAIN_ARMS[0]),a,atol=1e-17)
    def test_convex_routing_keeps_coefficient_box(self):
        A=np.array([[.125,-.125],[-.125,.125],[0.,.02]]);a=effective(A,np.array([.2,.4,.4]),TRAIN_ARMS[0]);self.assertTrue(np.all(abs(a)<=.125))
    def test_out_of_box_local_head_blocks(self):
        with self.assertRaises(ValueError):effective(np.full((3,2),.126),np.ones(3)/3,TRAIN_ARMS[0])
    def test_negative_routing_weight_blocks(self):
        with self.assertRaises(ValueError):effective(np.zeros((3,2)),np.array([-1.,1.,1.]),TRAIN_ARMS[0])
    def test_routing_pullback_matches_independent_central_difference(self):
        w=np.array([.2,.3,.5]);A=np.arange(9,dtype=float).reshape(3,3)/100;M=np.array([[1.,2.,-1.],[2.,0.,3.]]);t=np.array([.03,-.02]);a=w@A;g=2*M.T@(M@a-t);expected=pullback(g,w,TRAIN_ARMS[0]);fd=np.empty_like(A)
        for i in range(3):
            for j in range(3):
                p=A.copy();q=A.copy();p[i,j]+=1e-6;q[i,j]-=1e-6;fd[i,j]=((np.linalg.norm(M@(w@p)-t)**2)-(np.linalg.norm(M@(w@q)-t)**2))/2e-6
        np.testing.assert_allclose(fd,expected,atol=2e-10,rtol=1e-8)
    def test_adam_first_direction_and_projected_coordinates(self):
        a=np.zeros((3,2));g=np.array([[1.,-2.],[0.,3.],[-4.,0.]]);n,m,v,info=update(a,g,a,a,1);cl=g/np.linalg.norm(g);np.testing.assert_allclose(n,-RATE*cl/(abs(cl)+1e-8),atol=1e-15);self.assertEqual(info['projected_coordinates'],0)
    def test_objective_exact_ties_average_all_states(self):
        rows=[dict(state_id=sp['state_id'],objective_parts=dict(response_loss=1.,read_loss=0.),response_gradient=[[float(i),0.]]*3,guard_gradient=[[0.,0.]]*3) for i,sp in enumerate(registry())];g,agg=aggregate(rows,(3,2));np.testing.assert_allclose(g[:,0],41.5);self.assertEqual(agg['exact_max_tie_count'],84)
    def test_missing_or_permuted_rows_block(self):
        rows=[dict(state_id=sp['state_id'],objective_parts=dict(response_loss=0.,read_loss=0.),response_gradient=[0.],guard_gradient=[0.]) for sp in registry()]
        for bad in (rows[:-1],rows[::-1]):
            with self.assertRaises(ValueError):aggregate(bad,(1,))
    def test_new_context_routing_not_using_postwrite_labels(self):
        import inspect
        src=inspect.getsource(old.fit_router)+inspect.getsource(auditcore.route)+inspect.getsource(auditcore.projected_context)
        for forbidden in ('state_id','case','observed','response_loss','failed_ids'):self.assertNotIn(forbidden,src)
    def test_reusable_inference_accepts_new_prewrite_context_without_registry(self):
        d,_=worker_data();sp=registry()[0];p=d['points'][sp['state_id']];prepared=p['prepared'].copy();raw=p['raw'].copy();prepared[100]+=.003;raw[100]+=.003;A=np.tile(d['initial_a'],(K,1));point,diag=context_proposal(prepared,raw,d['V'],A,d['router'],TRAIN_ARMS[0]);self.assertEqual(point.shape,(DIM,));self.assertAlmostEqual(sum(diag['routing_weights']),1.);self.assertTrue(np.isfinite(point).all())
    def test_phase_blocks_endpoint_before_both_fixed_heads(self):
        d,_=worker_data();ph=Phase(d)
        with self.assertRaises(ValueError):ph.freeze({})
    def test_router_mutation_changes_frozen_digest(self):
        d,_=worker_data();before=fixed_digest(d);d['router']['centers'][0,0]+=.01;self.assertNotEqual(before,fixed_digest(d))
    def workflow(self,mode='normal',native='normal'):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);paths,c=dummy_inputs(root/'inputs')
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths,root/'out',c,lambda o,d,p,cs:Native(o,d,p,cs,native),lambda rt,d,p,cs:Analytic(rt,d,p,cs,mode),loader)
        return s,root/'out'
    def test_full_two_arm_synthetic_workflow_all_counts_seals_and_no_LC2(self):
        s,out=self.workflow();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_training_steps'],24);self.assertEqual(s['new_backward_calls'],4032);self.assertEqual(s['new_model_forwards'],6553);self.assertEqual(s['final_endpoint_records'],672);self.assertEqual(s['warm_parity_records'],168);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['LC2_single_fold_passed']);freeze=clean_json((out/'BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json').read_bytes());self.assertEqual(freeze['native_final_measurements_so_far'],0)
        for n,h in freeze['artifact_seals'].items():self.assertEqual(sha(out/n),h)
        self.assertEqual(s['per_arm_training'][TRAIN_ARMS[0]]['steps'],12);self.assertEqual(s['per_arm_training'][TRAIN_ARMS[1]]['steps'],12)
    def test_final_response_failure_retained(self):
        s,_=self.workflow(native='failed_final148');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['training_qualified'][TRAIN_ARMS[1]]);self.assertEqual(s['final_training_summaries']['all84']['0'][TRAIN_ARMS[1]]['failed_gates']['response'],1)
    def test_repeat_drift_blocks_qualification(self):
        s,_=self.workflow(native='repeat_drift148');self.assertTrue(s['complete']);self.assertFalse(s['repeatable']);self.assertFalse(any(s['training_qualified'].values()))
    def test_native_replay_drift_blocks_before_backward(self):
        s,_=self.workflow(native='preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_functional_zero_drift_blocks_before_backward(self):
        s,_=self.workflow('zero_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_functional_warm_drift_blocks_before_backward(self):
        s,_=self.workflow('warm_drift');self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0)
    def test_interruption_never_qualifies_either_arm(self):
        s,_=self.workflow('interrupt');self.assertFalse(s['complete']);self.assertFalse(any(s['training_qualified'].values()));self.assertFalse(s['LC2_closed'])
    def test_missing_gradient_state_blocks(self):
        s,_=self.workflow('missing_row');self.assertFalse(s['complete'])
    def test_wrong_routing_pullback_blocks(self):
        s,_=self.workflow('wrong_pullback');self.assertFalse(s['complete'])
    def test_nonfinite_aggregate_blocks(self):
        s,_=self.workflow('nonfinite');self.assertFalse(s['complete'])
    def test_frozen_router_mutation_blocks(self):
        s,_=self.workflow('mutate_router');self.assertFalse(s['complete'])


class CouplingMathChecks(unittest.TestCase):
    def test_initial_cross_zero_reproduces_diagonal_for_every_state(self):
        d,_=worker_data();A=initial_parameters(d,TRAIN_ARMS[0]);B=initial_parameters(d,TRAIN_ARMS[1]);self.assertTrue(np.all(B[:,1,:]==0))
        for sp in registry():
            p=d['points'][sp['state_id']];w=d['weights'][sp['state_id']];a=effective(A,w,TRAIN_ARMS[0]);b=effective(B,w,TRAIN_ARMS[1]);x,_=proposal(p['prepared'],p['raw'],d['V'],a);y,_=proposal(p['prepared'],p['raw'],d['V'],b);exact(x,y,'Common initial point');m=__import__('fixture138').metric(x,sp['case'],True);compare(loss(m['observed'],m['KL'],x,p['prepared'],a,sp['case']),loss(m['observed'],m['KL'],y,p['prepared'],b,sp['case']))
    def test_coupling_can_change_coordinate_with_zero_own_context(self):
        V=np.zeros((3,DIM));V[:,:3]=np.eye(3);p=np.zeros(DIM,dtype=np.float32);raw=p.copy();raw[0]=1.;a=np.zeros(3);b=np.zeros((2,3));b[1,1]=.1;x,_=proposal(p,raw,V,a);y,_=proposal(p,raw,V,b);self.assertEqual(x[1],0.);self.assertAlmostEqual(float(y[1]),.1,places=7)
    def test_cyclic_map_matches_independent_dense_operator(self):
        d,_=worker_data();p=d['points'][registry()[0]['state_id']];r=d['V'].shape[0];e=np.arange(2*r,dtype=float).reshape(2,r)/(100*r);u=context(p['prepared'],p['raw'],d['V']);P=np.eye(r)[np.mod(np.arange(r)-1,r)];expected=(p['raw'].astype(float)+d['V'].T@(np.diag(e[0])+np.diag(e[1])@P)@u).astype(np.float32);actual,_=proposal(p['prepared'],p['raw'],d['V'],e);exact(expected,actual,'Independent dense cyclic map')
    def test_coupled_pullback_matches_independent_finite_difference(self):
        A=np.arange(18,dtype=float).reshape(3,2,3)/1000;w=np.array([.2,.3,.5]);target=np.array([[.01,.02,-.01],[.03,-.02,.02]]);g=2*(effective(A,w,TRAIN_ARMS[1])-target);want=pullback(g,w,TRAIN_ARMS[1]);fd=np.zeros_like(A)
        for idx in np.ndindex(A.shape):
            x=A.copy();y=A.copy();x[idx]+=1e-6;y[idx]-=1e-6;fd[idx]=((effective(x,w,TRAIN_ARMS[1])-target)**2).sum()-((effective(y,w,TRAIN_ARMS[1])-target)**2).sum()
        np.testing.assert_allclose(fd/2e-6,want,atol=3e-12,rtol=1e-8)
    def test_coupled_nonfinite_and_wrong_blocks_block(self):
        for x in (np.zeros((3,3,4)),np.full((3,2,4),np.nan),np.full((3,2,4),.126)):
            with self.assertRaises(ValueError):effective(x,np.ones(3)/3,TRAIN_ARMS[1])
    def test_incomplete_batch_and_skipped_steps_do_not_freeze(self):
        d,_=worker_data();ph=Phase(d);ph.phase='training';ph.begin_batch(0,initial_parameters(d,TRAIN_ARMS[1]),TRAIN_ARMS[1])
        with self.assertRaises(ValueError):ph.end_batch()
        with self.assertRaises(ValueError):ph.freeze({})
    def test_diagonal_cross_initial_regularization_has_same_scale(self):
        a=np.array([.01,-.02,.03]);b=np.stack([a,np.zeros_like(a)]);self.assertEqual(.001*np.sum(a*a)/3,.001*np.sum(b*b)/3)
if __name__=='__main__':unittest.main(verbosity=2)
