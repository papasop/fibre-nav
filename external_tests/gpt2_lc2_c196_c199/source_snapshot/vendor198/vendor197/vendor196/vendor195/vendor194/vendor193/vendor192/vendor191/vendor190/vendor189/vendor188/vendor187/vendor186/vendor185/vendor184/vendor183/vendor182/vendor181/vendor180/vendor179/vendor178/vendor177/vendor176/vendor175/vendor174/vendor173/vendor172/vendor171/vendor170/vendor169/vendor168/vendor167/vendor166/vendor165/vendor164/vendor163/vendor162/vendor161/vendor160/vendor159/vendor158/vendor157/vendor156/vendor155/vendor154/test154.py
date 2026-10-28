"""Protocol math and actual worker/input routing, using full analytic training archives."""
import contextlib,io,tempfile,unittest,copy,zipfile
from unittest.mock import patch
from core154 import *
from worker154 import run,resolve_default_runtime_factory
from fixture154 import inputs_fixture,loader,Native

class MathChecks(unittest.TestCase):
    def test_fixed_registry_complete_order_and_all84(self):
        r=request_order();self.assertEqual(len(r),1512);self.assertEqual(Counter(x[3] for x in r),dict(preflight=168,probe=1008,repeat=336));self.assertEqual(len({x[:3] for x in r}),1176);self.assertEqual(len(registry()),84)
    def test_signed_scales_and_units_are_fixed(self):
        self.assertEqual(list(LABELS.values()),[0.,-.25,.25,-.5,.5,-1.,1.]);self.assertEqual(ANCHORS,(('small_step01',1),('small_step12',12)));self.assertEqual(ARM,'cyclic_rate_1e4')
    def test_exact_alpha_zero_and_one_restore_archived_bytes(self):
        b=np.arange(18,dtype=float).reshape(3,2,3)/1000;a=b+.0001;exact(interpolate(b,a,0.),b,'Exact before');exact(interpolate(b,a,1.),a,'Exact after')
    def test_unregistered_alpha_and_bool_block(self):
        b=np.zeros((3,2,3));a=b+.0001
        for h in [2.,True,float('nan')]:
            with self.assertRaises(ValueError):interpolate(b,a,h)
    def test_out_of_box_is_rejected_not_clipped(self):
        b=np.full((3,2,3),.125);a=b-.001
        with self.assertRaises(ValueError):interpolate(b,a,-1.)
    def test_nonfinite_wrong_shape_and_float32_coefficients_block(self):
        b=np.zeros((3,2,3));a=b+.0001
        for bad in [np.full_like(b,np.nan),np.zeros((3,3,3)),b.astype(np.float32)]:
            with self.assertRaises(ValueError):interpolate(bad,a,.5)
    def test_central_and_one_sided_sign_scale(self):
        # f(alpha)=7+3 alpha+2 alpha^2; slope is per actual update multiplier.
        for h in HALVES:
            r=finite_difference(7.,7-3*h+2*h*h,7+3*h+2*h*h,h,3.);self.assertAlmostEqual(r['central_slope'],3.);self.assertAlmostEqual(r['positive_slope'],3+2*h);self.assertAlmostEqual(r['negative_slope'],3-2*h);self.assertFalse(r['AD_accuracy_certified'])
    def test_disagreement_never_certifies_AD_error(self):
        r=finite_difference(0.,-1.,1.,1.,99.);self.assertIsNone(r['derivative_agreement_acceptance_rule']);self.assertFalse(r['AD_accuracy_certified'])
    def test_bad_FD_values_or_scale_block(self):
        for h in [.1,0.]:
            with self.assertRaises(ValueError):finite_difference(0.,-1.,1.,h,1.)
        with self.assertRaises(ValueError):finite_difference(float('nan'),-1.,1.,1.,1.)
    def test_actual_default_runtime_binding_has_current_entrypoint(self):
        f=resolve_default_runtime_factory();self.assertEqual(f.__module__,'runtime154')
        import runtime154
        with patch('runtime154.resolve_pinned_factory',return_value=lambda *a:('CURRENT',len(a))):self.assertEqual(f(1,2,3,4),('CURRENT',4))

class WorkflowChecks(unittest.TestCase):
    def workflow(self,mode='normal',mutate=None):
        paths,c=inputs_fixture();tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)/'out'
        def read(*a):
            d,h=loader(*a)
            if mutate:mutate(d,h)
            return d,h
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths,out,c,lambda o,d,p,cs:Native(o,d,p,cs,mode),read)
        return s,out
    def test_full_native_stencil_counts_seals_and_no_training(self):
        s,out=self.workflow();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],3025);self.assertEqual(s['native_metric_records'],1512);self.assertEqual((s['preflight_records'],s['probe_records'],s['repeat_records']),(168,1008,336));self.assertEqual(s['new_backward_calls'],0);self.assertEqual(s['new_training_steps'],0);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['candidate_selected']);self.assertFalse(s['LC2_closed']);self.assertTrue(s['both_zero_panels_replayed_before_probes']);self.assertTrue(s['repeatable']);self.assertEqual(len(s['diagnostics']),2)
        seal=clean_json((out/'STENCIL_FROZEN_BEFORE_MODEL.json').read_bytes());self.assertEqual(seal['model_forwards_so_far'],0)
        for n,h in seal['artifact_seals'].items():self.assertEqual(sha(out/n),h)
        with np.load(out/'ALL1176_STENCIL_POINTS.npz') as z:self.assertEqual(len(z.files),1176)
        self.assertEqual(len((out/'ALL_NATIVE_MEASUREMENTS.jsonl').read_text().splitlines()),1512)
    def test_repeat_drift_retained_without_derivative_conclusion(self):
        s,_=self.workflow('repeat_drift');self.assertTrue(s['complete'],s['errors']);self.assertFalse(s['repeatable']);self.assertFalse(s['AD_accuracy_certified']);self.assertIn('REPLAY_DRIFT',s['decision']);self.assertEqual(s['native_metric_records'],1512)
    def test_actual_future_point_mutation_blocks(self):
        s,_=self.workflow('mutate_future_point');self.assertFalse(s['complete']);self.assertFalse(s['candidate_selected']);self.assertTrue(any('Actual frozen point' in e for e in s['errors']))
    def test_seal_change_stops_before_first_native_metric(self):
        s,_=self.workflow('mutate_seal');self.assertFalse(s['complete']);self.assertEqual(s['native_metric_records'],0);self.assertEqual(s['new_model_forwards'],1)
    def test_zero_replay_drift_stops_before_probes(self):
        s,_=self.workflow('preflight_drift');self.assertFalse(s['complete']);self.assertEqual(s['probe_records'],0);self.assertEqual(s['new_backward_calls'],0)
    def test_interruption_retains_attempt_and_no_conclusion(self):
        s,out=self.workflow('interrupt');self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertEqual(s['runtime_calls']['path_metric_attempts'],1);self.assertEqual(s['probe_records'],0);self.assertEqual(len((out/'ALL_NATIVE_ATTEMPTS.jsonl').read_text().splitlines()),169)
    def test_out_of_box_registry_stops_before_model(self):
        def changed(d,h):
            for v in h.values():v['before'][:]=.125;v['after'][:]=.124
        s,_=self.workflow(mutate=changed);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_zero_direction_stops_before_model(self):
        def changed(d,h):
            for v in h.values():v['after']=v['before'].copy()
        s,_=self.workflow(mutate=changed);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_default_factory_failure_blocks_before_input_or_model(self):
        paths,c=inputs_fixture()
        with tempfile.TemporaryDirectory() as t,patch('worker154.resolve_default_runtime_factory',side_effect=ValueError('DEFAULT_ENTRY_SENTINEL')),patch('worker154.load') as read,contextlib.redirect_stdout(io.StringIO()):s=run(paths,Path(t),c)
        read.assert_not_called();self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0);self.assertTrue(any('DEFAULT_ENTRY_SENTINEL' in e for e in s['errors']))
    def test_wrong_production_pin_blocks_before_archive_or_model(self):
        paths,_=inputs_fixture()
        with tempfile.TemporaryDirectory() as t,contextlib.redirect_stdout(io.StringIO()):s=run(paths,Path(t),contract(),lambda *a:self.fail('No model'),None)
        self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)

if __name__=='__main__':unittest.main(verbosity=2)
