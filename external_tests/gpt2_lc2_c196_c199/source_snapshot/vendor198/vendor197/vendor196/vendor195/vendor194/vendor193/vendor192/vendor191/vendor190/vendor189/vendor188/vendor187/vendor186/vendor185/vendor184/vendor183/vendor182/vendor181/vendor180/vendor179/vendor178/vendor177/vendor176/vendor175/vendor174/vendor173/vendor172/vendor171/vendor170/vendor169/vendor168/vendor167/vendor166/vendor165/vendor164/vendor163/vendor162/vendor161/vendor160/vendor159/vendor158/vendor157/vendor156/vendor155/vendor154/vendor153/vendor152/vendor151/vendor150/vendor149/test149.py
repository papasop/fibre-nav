import copy,contextlib,io,tempfile,unittest
from core149 import *
from audit149 import audit148
from inputs149 import bind
from fixture149 import make_fixture,rebound,npz_bytes
from worker149 import run

class MathChecks(unittest.TestCase):
    def test_registered_router_matches_fixed_parent_on_varied_contexts(self):
        U=np.sin(np.arange(84*4).reshape(84,4)/10);r,W,m=registered_router_arithmetic(U);rr,WW,mm=parent.fit_router(U)
        for k in r:close(r[k],rr[k],'Independent router arithmetic',1e-14)
        close(W,WW,'Independent fixed weights',1e-14);compare(m,mm)
    def test_identical_contexts_have_uniform_routes_and_retained_empty_clusters(self):
        r,W,m=registered_router_arithmetic(np.ones((84,3)));np.testing.assert_allclose(W,1/3);self.assertTrue(m['empty_cluster_events']);self.assertTrue(routing_diagnostics(W)['all_training_weights_exactly_identical'])
    def test_missing_training_row_and_nonfinite_context_block(self):
        with self.assertRaises(ValueError):registered_router_arithmetic(np.zeros((83,2)))
        a=np.zeros((84,2));a[0,0]=np.inf
        with self.assertRaises(ValueError):registered_router_arithmetic(a)
    def test_nonpositive_normalizer_blocks(self):
        r,_,_=registered_router_arithmetic(np.zeros((84,2)));r['scale'][0]=0
        with self.assertRaises(ValueError):route(np.zeros(2),r)
    def test_varying_weights_report_magnitudes_without_sufficiency_claim(self):
        W=np.tile([.2,.3,.5],(84,1));W[0]=[.1,.4,.5];d=routing_diagnostics(W);self.assertAlmostEqual(d['maximum_weight_range_across_states'],.1);self.assertFalse(d['all_training_weights_exactly_identical']);self.assertFalse(d['sufficiency_threshold_defined'])
    def test_identical_experts_are_not_mistaken_for_distinct_effective_controls(self):
        p=np.array([.01,.02]);A=np.tile(p,(3,1));W=np.tile([.2,.3,.5],(84,1));d=control_diagnostics(p,A,W);self.assertTrue(d['all_expert_controls_exactly_identical']);self.assertEqual(d['maximum_expert_coefficient_pairwise_Linf'],0.)
    def test_different_experts_can_cancel_under_uniform_routing(self):
        A=np.array([[.1,0.],[-.1,0.],[0.,0.]]);d=control_diagnostics(np.zeros(2),A,np.full((84,3),1/3));self.assertAlmostEqual(d['maximum_expert_coefficient_pairwise_Linf'],.2);self.assertLess(d['maximum_effective_control_vs_global_Linf'],1e-16)
    def test_same_argmax_routes_can_have_different_effective_controls(self):
        W=np.tile([.8,.1,.1],(84,1));W[1]=[.6,.2,.2];A=np.array([[.1,0.],[0.,.1],[0.,0.]]);d=control_diagnostics(np.zeros(2),A,W);self.assertGreater(d['maximum_effective_control_range_across_states'],0);self.assertEqual(routing_diagnostics(W)['hard_assignment_counts'],[84,0,0])
    def test_linear_pullback_matches_independent_central_difference(self):
        w=np.array([.2,.3,.5]);A=np.arange(9,dtype=float).reshape(3,3)/100;t=np.array([.01,.02,.03]);g=2*(w@A-t);expected=pullback(g,w,TRAIN_ARMS[1]);fd=np.zeros_like(A)
        for i in range(3):
            for j in range(3):
                p=A.copy();q=A.copy();p[i,j]+=1e-6;q[i,j]-=1e-6;fd[i,j]=(((w@p-t)**2).sum()-((w@q-t)**2).sum())/2e-6
        np.testing.assert_allclose(fd,expected,atol=1e-11)
    def test_first_adam_direction_scale_and_noop(self):
        p=np.zeros((3,2));g=np.arange(1,7,dtype=float).reshape(3,2);n,_,_,info=update(p,g,p,p,1);gc=g/np.linalg.norm(g);np.testing.assert_allclose(n,-.0005*gc/(abs(gc)+1e-8),atol=1e-15);same,_,_,_=update(p,p,p,p,1);exact(same,p,'Noop');self.assertEqual(info['projected_coordinates'],0)
    def test_out_of_box_coefficients_and_nonconvex_weights_block(self):
        with self.assertRaises(ValueError):effective(np.full((3,2),.126),np.ones(3)/3,TRAIN_ARMS[1])
        with self.assertRaises(ValueError):effective(np.zeros((3,2)),np.array([-1.,1.,1.]),TRAIN_ARMS[1])
    def test_different_numeric_margins_do_not_imply_different_gate_decisions(self):
        a=dict(metrics=dict(gates=dict(response=True)),gates=dict(proposal_radius_passed=True,failed_gates=[],complete_pass=True,response_drift=.001));b=copy.deepcopy(a);b['gates']['response_drift']=.0015;self.assertEqual(gate_signature(a),gate_signature(b));b['metrics']['gates']['response']=False;b['gates']['complete_pass']=False;b['gates']['failed_gates']=['response'];self.assertNotEqual(gate_signature(a),gate_signature(b))
    def test_zero_gradient_cosine_is_undefined_not_zero(self):self.assertIsNone(cosine(np.zeros(2),np.ones(2)))
    def test_model_import_and_solvers_are_blocked(self):
        with no_models(),self.assertRaises(ValueError):__import__('torch')
        with safe.no_solvers(),self.assertRaises(Exception):np.linalg.svd(np.eye(2))

class ArchiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()),no_models(),safe.no_solvers():cls.paths,cls.c=make_fixture(cls.root/'fixture');cls.d=bind(cls.paths,cls.root/'bindings',cls.c)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def audit_zip(self,p,out):
        with contextlib.redirect_stdout(io.StringIO()),no_models(),safe.no_solvers(),contextlib.closing(Archive(p)) as z:return audit148(z,copy.deepcopy(self.d),out,self.c)
    def j(self,name):
        with contextlib.closing(Archive(self.paths['C148'])) as z:return z.json(name)
    def arrays(self,name):
        with contextlib.closing(Archive(self.paths['C148'])) as z:return z.arrays(name)
    def reject(self,changes):
        target=self.root/(self._testMethodName+'.zip');rebound(self.paths['C148'],target,changes)
        with self.assertRaises((ValueError,KeyError,AssertionError)):self.audit_zip(target,self.root/(self._testMethodName+'_out'))
    def test_full_five_original_worker_no_models_and_parent_decision_retained(self):
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'full_worker',self.c)
        self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['training_records_audited'],2016);self.assertEqual(s['training_updates_audited'],24);self.assertEqual(s['native_endpoint_records_recomputed'],672);self.assertEqual(s['frozen_proposals_recomputed'],336);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['parent_decision_reclassified']);self.assertEqual(s['parent_decision'],self.j('SUMMARY.json')['decision']);self.assertEqual(s['training_qualified'],self.j('SUMMARY.json')['training_qualified'])
        for k in ('new_model_forwards','new_backward_calls','new_training_steps','new_head_fits','new_router_candidates','new_Jacobians','new_SVD_calls','linear_systems_solved'):self.assertEqual(s[k],0)
        rows=clean_json((self.root/'full_worker'/'ALL84_ROUTING_CONTROL_AND_ENDPOINT_DIAGNOSTICS.json').read_bytes());self.assertEqual(len(rows),84);self.assertTrue(any(not r['arms'][TRAIN_ARMS[1]]['complete_pass'] for r in rows));self.assertTrue(all(s['original_inputs_unchanged'].values()))
    def test_actual_final_head_change_blocks_with_rebound_hashes(self):
        name='FINAL_context_routed_STEP12.npz';a=self.arrays(name);a['parameters'][0,0]+=.0001;self.reject({name:npz_bytes(**a)})
    def test_router_center_change_blocks(self):
        name='FROZEN_TRAINING_ONLY_ROUTER.npz';a=self.arrays(name);a['centers'][0,0]+=.01;self.reject({name:npz_bytes(**a)})
    def test_cached_routing_weight_change_blocks(self):
        name='ALL84_PREWRITE_ROUTING.npz';a=self.arrays(name);a['weights'][0,0]+=.001;a['weights'][0,1]-=.001;self.reject({name:npz_bytes(**a)})
    def test_effective_gradient_pullback_change_blocks(self):
        folder='training/context_routed/step01/';r=self.j(folder+'OBSERVATIONS.json');r[0]['response_gradient'][0][0]+=.01;b=json.dumps(r).encode();self.reject({folder+'OBSERVATIONS.json':b,folder+'GRADIENT_PROGRESS.json':b})
    def test_optimizer_after_tensor_change_blocks(self):
        name='training/global_continuation/step01/UPDATE.npz';a=self.arrays(name);a['after'][0]+=.001;self.reject({name:npz_bytes(**a)})
    def test_missing_training_state_blocks(self):
        folder='training/global_continuation/step01/';r=self.j(folder+'OBSERVATIONS.json')[:-1];b=json.dumps(r).encode();self.reject({folder+'OBSERVATIONS.json':b,folder+'GRADIENT_PROGRESS.json':b})
    def test_nonfinite_gradient_blocks(self):
        name='training/global_continuation/step01/UPDATE.npz';a=self.arrays(name);a['gradient'][0]=np.nan;self.reject({name:npz_bytes(**a)})
    def test_frozen_point_change_blocks_even_with_rebound_seal(self):
        name='ALL336_FROZEN_FINAL_POINTS.npz';a=self.arrays(name);key=next(iter(a));a[key][0]+=.0001;b=npz_bytes(**a);seal=self.j('BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json');seal['artifact_seals'][name]=hashlib.sha256(b).hexdigest();self.reject({name:b,'BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json':json.dumps(seal).encode()})
    def test_gate_change_recomputed(self):
        name='ALL672_FINAL_TRAINING_ENDPOINTS.json';r=self.j(name);r[0]['gates']['complete_pass']=not r[0]['gates']['complete_pass'];b=json.dumps(r).encode();self.reject({name:b,'FINAL_ENDPOINT_PROGRESS.json':b,'final_measurements/record000.json':json.dumps(r[0]).encode()})
    def test_missing_repeat_record_blocks(self):
        r=self.j('ALL672_FINAL_TRAINING_ENDPOINTS.json')[:-1];b=json.dumps(r).encode();self.reject({'ALL672_FINAL_TRAINING_ENDPOINTS.json':b,'FINAL_ENDPOINT_PROGRESS.json':b})
    def test_summary_statistic_change_blocks(self):
        s=self.j('SUMMARY.json');s['final_training_summaries']['all84']['0'][TRAIN_ARMS[1]]['passed']+=1;self.reject({'SUMMARY.json':json.dumps(s).encode()})
    def test_source_pin_change_blocks_even_with_rebound_manifest(self):self.reject({'source/core148.py':b'raise AssertionError("archived source must not execute")\n'})
    def test_registered_threshold_change_blocks(self):
        a=self.j('DESIGN_FROZEN_BEFORE_MODEL.json');a['acceptance_budgets']['response']=.003;self.reject({'DESIGN_FROZEN_BEFORE_MODEL.json':json.dumps(a).encode()})
    def test_production_pin_rejects_synthetic_before_payload(self):
        with contextlib.redirect_stdout(io.StringIO()):s=run(self.paths,self.root/'wrongproduction',contract())
        self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['new_model_forwards'],0)
    def test_unsafe_archive_member_rejected(self):
        import zipfile
        p=self.root/'unsafe.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('../unsafe','x')
        with self.assertRaises(ValueError):Archive(p)

if __name__=='__main__':unittest.main(verbosity=2)
