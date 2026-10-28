"""Protocol / binding / actual worker wiring, with explicit full-size analytic fixtures."""
import copy,io,tempfile,unittest
from collections import Counter
from unittest.mock import patch
from core155 import *
from worker155 import run,validate_precision,resolve_default_runtime_factory
from fixture155 import inputs_fixture,bound,cached_loader,Native,Pair
class Checks(unittest.TestCase):
    def test_fixed_complete84_registry_request_order_and_denominators(self):
        r=requests();self.assertEqual(len(r),756);self.assertEqual(Counter(x[2] for x in r),dict(preflight=84,probe=504,repeat=168));self.assertEqual(len(set(x[:2] for x in r)),588);self.assertEqual(sum(sid in FOCUS for _,sid,_ in r),18);self.assertEqual(len(registry()),84);self.assertEqual(4*len(r)+2*18+2,3062)
    def test_two_focus_ids_are_registered_exposed_states(self):self.assertEqual(len(FOCUS),2);self.assertTrue(set(FOCUS)<=set(sp['state_id'] for sp in registry()));self.assertTrue(design()['all84_are_exposed_training'])
    def test_original_gates_and_safety_remain_unchanged(self):self.assertEqual(design()['acceptance_budgets'],prev.design()['acceptance_budgets']);self.assertFalse(design()['LC2_closed'])
    def test_coordinate_fd_sign_units_and_response_coordinate_switch(self):
        z=np.zeros(22);z[:2]=[.002,.001];d=np.arange(22)/1000;h=.5;r=coordinate_fd(z,z-h*d,z+h*d,h);np.testing.assert_allclose(r['central'],d,atol=1e-18);self.assertTrue(r['zero_response_hinge_active']);self.assertEqual(r['response_max_coordinate_zero'],[0])
    def test_bad_coordinate_shape_nan_or_scale_blocks(self):
        z=np.zeros(22)
        for a,b,c,h in [(z[:6],z,z,.25),(z,z,np.full(22,np.nan),.5),(z,z,z,.1)]:
            with self.assertRaises(ValueError):coordinate_fd(a,b,c,h)
    def test_precision_baseline_offset_kept_separate_from_state_change(self):
        p=np.zeros(DIM,np.float32);calls=dict(model_forwards=0,functional_FP64_forwards=0,FP64_baseline_forwards=0,FP64_focus_metrics=0);m=Pair(None,calls).precision64(p,'xor_00');validate_precision(m,p);self.assertFalse(m['acceptance_evaluated']);self.assertNotEqual(m['observed_local_reference'],m['observed_FP32_response_reference'])
    def test_forged_precision_offset_blocks(self):
        p=np.zeros(DIM,np.float32);cs=dict(model_forwards=0,functional_FP64_forwards=0,FP64_baseline_forwards=0,FP64_focus_metrics=0);m=Pair(None,cs).precision64(p,'xor_00');m['response_reference_offset'][0]+=.001
        with self.assertRaises(ValueError):validate_precision(m,p)
    def test_precision_cannot_claim_acceptance(self):
        p=np.zeros(DIM,np.float32);cs=dict(model_forwards=0,functional_FP64_forwards=0,FP64_baseline_forwards=0,FP64_focus_metrics=0);m=Pair(None,cs).precision64(p,'xor_00');m['acceptance_evaluated']=True
        with self.assertRaises(ValueError):validate_precision(m,p)
    def test_parity_does_not_hide_reference_or_coordinate_differences(self):
        p=np.zeros(DIM,np.float32);m=full_metric(np.zeros(22),0.,p,'xor_00');n=copy.deepcopy(m);n['observed'][0]+=.001;r=parity(m,n);self.assertAlmostEqual(r['observed_Linf'],.001);self.assertTrue(r['same_point'])
    def test_actual_default_runtime_route_is_current_and_pinned(self):
        f=resolve_default_runtime_factory();self.assertEqual(f.__module__,'runtime155')
        with patch('runtime155.resolve_pinned_factory',return_value=lambda *a:'C155_CURRENT'):self.assertEqual(f(None,None,None,None),'C155_CURRENT')
    def test_full_ten_archive_binding_reconstructs_parent_points_and_rows(self):
        d,h,p,r,pc=bound();self.assertEqual(len(p),1176);self.assertEqual(len(r),756);self.assertEqual(len(pc),14);self.assertEqual(set(h),set(a for a,_ in prev.ANCHORS));self.assertEqual(len(d['points']),84)
    def test_wrong_production_hash_blocks_before_model(self):
        paths,_=inputs_fixture()
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()):s=run(paths,Path(t),contract(),lambda *a:self.fail('No model'),lambda *a:self.fail('No pair'))
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def workflow(self,pairmode='normal',mutate=None):
        paths,c=inputs_fixture();tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)
        def read(*a):
            d,h,p,r,pc=cached_loader(*a)
            if mutate:mutate(d,h,p,r)
            return d,h,p,r,pc
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths,out,c,lambda o,d,p,cs:Native(o,d,p,cs),lambda rt,cs:Pair(rt,cs,pairmode),read)
        return s,out
    def test_full_actual_worker_1530_records_3062_forwards_and_sealed_points(self):
        s,out=self.workflow();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['metric_records'],1530);self.assertEqual(s['new_model_forwards'],3062);self.assertFalse(s['real_model_execution_claim']);self.assertTrue(s['C154_all756_native_replays_consistent']);self.assertTrue(s['all756_FP32_pairs_within_original_parity_tolerance']);self.assertTrue(s['all84_zero_pairs_checked_before_probes']);self.assertFalse(s['diagnostics']['FP64_full84_objective_available']);self.assertFalse(s['LC2_closed']);self.assertFalse(s['candidate_selected']);self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0)
        with np.load(out/'ALL588_FIXED_POINTS.npz') as z:self.assertEqual(len(z.files),588)
        self.assertEqual(len((out/'ALL_METRIC_RECORDS.jsonl').read_text().splitlines()),1530);self.assertEqual(len(clean_json((out/'ALL84_FP32_COORDINATE_TRAJECTORIES.json').read_bytes())),168)
        for n,h in clean_json((out/'SEALED_BEFORE_MODEL.json').read_bytes())['artifact_seals'].items():self.assertEqual(sha(out/n),h)
    def test_zero_native_functional_drift_stops_before_signed_probes(self):
        s,_=self.workflow('zero_parity_drift');self.assertFalse(s['complete']);self.assertEqual(s['runtime_calls']['path_metric_attempts'],0);self.assertFalse(s['candidate_selected'])
    def test_functional_failure_retains_completed_native_record(self):
        s,out=self.workflow('functional_failure');self.assertFalse(s['complete']);self.assertEqual(s['metric_records'],1);self.assertEqual(len((out/'ALL_METRIC_RECORDS.jsonl').read_text().splitlines()),1)
    def test_parent_point_change_blocks_before_model(self):
        def mutate(d,h,p,r):p[point_key('zero',registry()[0]['state_id'])][0]+=.01
        s,_=self.workflow(mutate=mutate);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_full_head_tail_change_cannot_hide_in_same_capture(self):
        def mutate(d,h,p,r):h[ANCHOR]['before'][-1,-1,-1]+=.0001
        s,_=self.workflow(mutate=mutate);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_zero_direction_blocks_before_model(self):
        def mutate(d,h,p,r):h[ANCHOR]['after']=h[ANCHOR]['before'].copy()
        s,_=self.workflow(mutate=mutate);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_mutated_future_physical_point_is_detected(self):
        points={('zero',s['state_id']):np.zeros(DIM,np.float32) for s in registry()};p=Phase(points);points[('zero',registry()[-1]['state_id'])][0]=1
        with self.assertRaises(ValueError):p.verify()
    def test_unregistered_native_call_and_missing_denominator_block(self):
        pts={('zero',s['state_id']):np.zeros(DIM,np.float32) for s in registry()};p=Phase(pts)
        with self.assertRaises(ValueError):p.consume(pts[('zero',registry()[0]['state_id'])],'xor_00',True)
        with self.assertRaises(ValueError):p.finish()
if __name__=='__main__':unittest.main(verbosity=2)
