"""Archive/math/gate/baseline/precision tests, complete analytic all84 producer and reader."""
import copy,io,tempfile,unittest,zipfile
from core156 import *
from audit156 import bind,baseline
from worker156 import run
from fixture156 import fixture,changed_archive,json_change,array_change
class MathChecks(unittest.TestCase):
    def test_fixed1530_order_all84_and_two_focus(self):self.assertEqual(len(ordered_records()),1530);self.assertEqual(Counter(r[3] for r in ordered_records()),dict(native_FP32=756,functional_FP32=756,functional_FP64=18));self.assertEqual(len(registry()),84);self.assertEqual(len(FOCUS),2)
    def test_fd_coordinate_sign_and_scale(self):
        z=np.arange(22,dtype=float);d=np.linspace(-1,1,22)
        for h,_,_ in SCALES:np.testing.assert_allclose(fd(z,z-h*d,z+h*d,h)['central'],d,atol=1e-14)
    def test_fd_scalar_quadratic_positive_and_negative_differ(self):
        for h,_,_ in SCALES:r=fd(7.,7-3*h+2*h*h,7+3*h+2*h*h,h);self.assertAlmostEqual(r['central'],3.);self.assertAlmostEqual(r['positive'],3+2*h);self.assertAlmostEqual(r['negative'],3-2*h)
    def test_bad_fd_scale_bool_shape_or_nonfinite_blocks(self):
        for args in [(0.,1.,2.,True),(0.,1.,2.,.1),([0.],[1.,2.],[3.],.5),(0.,np.nan,2.,1.)]:
            with self.assertRaises(ValueError):fd(*args)
    def test_constant_reference_offset_cancels_signed_coordinate_slopes(self):
        z=np.linspace(-.003,.003,22);d=np.arange(22)*1e-6;o=np.r_[np.full(6,.0002),np.zeros(16)]
        for h,_,_ in SCALES:np.testing.assert_allclose(fd(z,z-h*d,z+h*d,h)['central'],fd(z+o,z-h*d+o,z+h*d+o,h)['central'],atol=1e-18)
    def test_reference_offset_need_not_cancel_nonlinear_response_loss(self):
        z=np.array([.002,0,0,0,0,0]);m=z.copy();m[0]-=.0001;p=z.copy();p[0]+=.0001;o=np.array([-.0003,0,0,0,0,0]);a=fd(response_loss(z),response_loss(m),response_loss(p),1.)['central'];b=fd(response_loss(z+o),response_loss(m+o),response_loss(p+o),1.)['central'];self.assertNotEqual(a,b)
    def test_maximum_response_coordinate_switch_is_recorded(self):
        a=np.zeros(22);a[:2]=[.002,.001];obs={n:a.copy() for n in LABELS};obs['plus_one'][1]=.003;r=trajectory(obs);self.assertTrue(r['max_coordinate_changes_at_sampled_panels']);self.assertEqual(r['panels']['plus_one']['response_max_coordinates'],[1])
    def test_hinge_switch_and_all_scales_are_retained(self):
        obs={n:np.r_[.0018+alpha*.0001,np.zeros(21)] for n,alpha in LABELS.items()};r=trajectory(obs);self.assertTrue(r['hinge_changes_at_sampled_panels']);self.assertEqual(len(r['scales']),3)
    def test_missing_panel_or_nonfinite_response_blocks(self):
        with self.assertRaises(ValueError):trajectory({'zero':np.zeros(22)})
        with self.assertRaises(ValueError):response_loss([np.nan]*6)
    def test_original_FP32_budgets_no_FP64_full_objective_or_new_threshold(self):self.assertEqual(design()['acceptance_budgets'],parent.design()['acceptance_budgets']);self.assertTrue(design()['no_FP64_full84_objective']);self.assertTrue(design()['no_new_stability_threshold']);self.assertFalse(design()['LC2_closed'])
class ArchiveChecks(unittest.TestCase):
    def audit(self,archive=None,c=None):
        if archive is None:archive,c=fixture()
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):s=run(archive,out,c)
        return s,out
    def reject(self,name,change,rebind=False):
        p,c=changed_archive(name,change,rebind);s,_=self.audit(p,c);self.assertFalse(s['complete'],s);self.assertEqual(s['new_model_forwards'],0);self.assertFalse(s['LC2_closed']);return s
    def test_full_readonly_workflow_1530_records_actual_points_and_no_model(self):
        s,out=self.audit();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['records_audited'],1530);self.assertEqual(s['training_states_audited'],84);self.assertEqual(s['actual_frozen_point_entries_verified'],588);self.assertEqual(s['actual_frozen_coefficient_tensors_verified'],7);self.assertTrue(s['saved_parent_diagnostics_recomputed']);self.assertTrue(s['original_C155_input_unchanged']);self.assertFalse(s['ten_predecessor_original_ZIP_bytes_rehashed']);self.assertEqual(s['new_model_forwards'],0);self.assertEqual(s['new_backward_calls'],0);self.assertFalse(s['candidate_selected']);self.assertFalse(s['AD_accuracy_certified']);self.assertEqual(s['trajectory_diagnostics']['all84_FP32_trajectory_records'],168);self.assertEqual(len(s['trajectory_diagnostics']['focus']),2);self.assertFalse(s['FP64_full84_objective_available'])
        self.assertEqual(len(clean_json((out/'ALL84_RECOMPUTED_COORDINATE_TRAJECTORIES.json').read_bytes())),168)
    def test_wrong_production_whole_hash_blocks_before_payload(self):
        p,_=fixture();s,_=self.audit(p,contract());self.assertFalse(s['complete']);self.assertEqual(s['records_audited'],0)
    def test_source_change_blocks_with_rebound_manifest(self):self.reject('source/core155.py',lambda b:b+b'\n# tampered\n')
    def test_actual_coefficient_tail_change_blocks_even_rebound_seal(self):self.reject('COEFFICIENTS.npz',array_change(lambda a:a['zero'].__setitem__((-1,-1,-1),a['zero'][-1,-1,-1]+.001)),True)
    def test_actual_frozen_point_change_blocks_even_rebound_seal(self):self.reject('ALL588_FIXED_POINTS.npz',array_change(lambda a:a[next(iter(a))].__setitem__(5,.123)),True)
    def test_permuted_state_row_binding_blocks(self):self.reject('ALL_METRIC_RECORDS.jsonl',lambda b:b'\n'.join([b.splitlines()[1],b.splitlines()[0]]+b.splitlines()[2:])+b'\n')
    def test_missing_repeat_record_blocks(self):self.reject('ALL_METRIC_RECORDS.jsonl',lambda b:b'\n'.join(b.splitlines()[:-1])+b'\n')
    def test_fp64_baseline_offset_forgery_blocks(self):self.reject('model_runtime/FP64_BASELINE_PROVENANCE.json',json_change(lambda a:a['response_reference_offset'].__setitem__(0,.001)))
    def test_stored_fp64_zero_logits_change_blocks(self):self.reject('model_runtime/FP64_BASELINE_LOGITS.npz',array_change(lambda a:a['FP64_logits'].__setitem__((0,0),.1)))
    def test_forged_saved_trajectory_is_recomputed(self):self.reject('ALL84_FP32_COORDINATE_TRAJECTORIES.json',json_change(lambda a:a[0]['scales'][0]['central'].__setitem__(0,99.)))
    def test_parent_objective_summary_change_blocks(self):self.reject('SUMMARY.json',json_change(lambda a:a['diagnostics']['all84_FP32_objectives']['native_FP32'].__setitem__('base_objective',99.)))
    def test_unsafe_and_duplicate_archive_paths_block(self):
        with tempfile.TemporaryDirectory() as t:
            for names in [['../outside'],['same','same']]:
                p=Path(t)/'bad.zip'
                with zipfile.ZipFile(p,'w') as z:
                    for n in names:z.writestr(n,b'x')
                with self.assertRaises(ValueError):Archive(p)
    def test_model_and_solver_guards_are_effective(self):
        with parent.prior.no_models(),safe.no_solvers():
            with self.assertRaises(Exception):__import__('torch')
            with self.assertRaises(Exception):np.linalg.solve(np.eye(2),np.ones(2))
if __name__=='__main__':unittest.main(verbosity=2)
