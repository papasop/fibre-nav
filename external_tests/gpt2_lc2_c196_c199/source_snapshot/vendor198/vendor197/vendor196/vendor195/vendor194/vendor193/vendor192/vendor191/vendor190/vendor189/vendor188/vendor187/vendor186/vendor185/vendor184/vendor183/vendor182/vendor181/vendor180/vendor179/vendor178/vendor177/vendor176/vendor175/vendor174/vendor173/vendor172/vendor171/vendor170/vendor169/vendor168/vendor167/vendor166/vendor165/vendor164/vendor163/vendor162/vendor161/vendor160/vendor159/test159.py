"""Fixed requests, protocol change, failure retention and same-run guards on analytic fixtures."""
import copy,io,tempfile,unittest
from core159 import *
from fixture159 import Analytic,inputs,loader,context
from worker159 import run
class Checks(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def workflow(self,factory=Analytic,load=loader):
        paths,c=inputs(self.root/'inputs')
        with contextlib.redirect_stdout(io.StringIO()):return run(paths,self.root/'out',c,factory,load)
    def test_full_synthetic_protocol_27_forwards18_backwards(self):
        s=self.workflow();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],27);self.assertEqual(s['new_backward_calls'],18);self.assertFalse(s['LC2_closed']);self.assertFalse(s['real_model_execution_claim']);self.assertEqual(s['C157_original_decision'],'C157_INCOMPLETE_NO_DIRECTIONAL_CONCLUSION')
    def test_large_cross_archive_drift_does_not_block_new_protocol(self):
        def changed(*args):
            c=loader(*args);c['replay_absolute']={k:v+.01 for k,v in c['replay_absolute'].items()};return c
        s=self.workflow(load=changed);self.assertTrue(s['complete'],s['errors']);self.assertTrue(all(not r['within_historical_tolerance'] for r in s['cross_run_endpoint_comparisons']));self.assertFalse(s['C157_decision_reclassified'])
    def test_within_run_forward_drift_blocks(self):
        class Changed(Analytic):
            def __init__(self,*a):super().__init__(*a);self.zeros={}
            def measure(self,sid,n):
                r=super().measure(sid,n)
                if n=='zero':
                    self.zeros[sid]=self.zeros.get(sid,0)+1
                    if self.zeros[sid]>1:r['absolute_response'][0]+=.01
                return r
        s=self.workflow(Changed);self.assertFalse(s['complete']);self.assertTrue(any('AD/no-grad center differs' in e for e in s['errors']))
    def test_actual_point_mutation_is_blocked(self):
        class Changed(Analytic):
            def ad(self,sid):r=super().ad(sid);self.points[(sid,'minus_one')][0]+=.1;return r
        s=self.workflow(Changed);self.assertFalse(s['complete']);self.assertTrue(any('Actual fixed point changed' in e for e in s['errors']))
    def test_reference_mutation_cannot_complete(self):
        class Changed(Analytic):
            def ad(self,sid):r=super().ad(sid);self.references['current_FP64'][0]+=.1;return r
        s=self.workflow(Changed);self.assertFalse(s['complete']);self.assertTrue(any('Fixed references changed' in e for e in s['errors']))
    def test_wrong_backward_count_never_completes(self):
        class Wrong(Analytic):
            def ad(self,sid):r=super().ad(sid);self.calls['backward_calls']-=1;return r
        s=self.workflow(Wrong);self.assertFalse(s['complete']);self.assertTrue(any('All18 scalar reverse calls' in e for e in s['errors']))
    def test_C157_success_cannot_be_silently_substituted(self):
        def changed(*a):c=loader(*a);c['failed157']['decision']='C157_SUCCESS';return c
        s=self.workflow(load=changed);self.assertFalse(s['complete']);self.assertEqual(s['new_model_forwards'],0)
    def test_fixed14_points_and24_requests_before_model(self):
        dump(self.root/'DESIGN_FROZEN_BEFORE_MODEL.json',design());p,d,r,s=freeze(context()['data'],self.root);self.assertEqual(len(p),14);self.assertEqual(len(r),24);self.assertEqual(r[0]['stage'],'preflight');self.assertTrue(all(sha(self.root/n)==h for n,h in s.items()))
    def test_registered_path_rejects_boolean_step_and_zero_direction(self):
        z=np.zeros(DIM,np.float32)
        with self.assertRaises(ValueError):numpy_map(constants(context()['data']),FOCUS[0],True)
        c=context();c['data']['after']=c['data']['before'].copy();dump(self.root/'DESIGN_FROZEN_BEFORE_MODEL.json',design())
        with self.assertRaises(ValueError):freeze(c['data'],self.root)
    def test_FD_sign_scale_and_all_three_reference_losses(self):
        panels={n:dict(absolute_response=(np.arange(6)*.001+a*1e-5+a*a*1e-4).tolist()) for n,a in LABELS.items()};ad=dict(coordinate_derivatives=[1e-5]*6,response_loss_derivatives={n:0. for n in REFERENCES});r=analyse(panels,ad,{n:np.zeros(6) for n in REFERENCES});self.assertEqual(len(r['scales']),3)
        for row in r['scales']:self.assertLess(row['central_minus_AD_Linf'],1e-16);self.assertEqual(set(row['response_loss']),set(REFERENCES))
    def test_missing_scale_or_reference_blocks(self):
        panels={n:dict(absolute_response=[0.]*6) for n in LABELS};ad=dict(coordinate_derivatives=[0.]*6,response_loss_derivatives={n:0. for n in REFERENCES});refs={n:np.zeros(6) for n in REFERENCES};del panels['minus_half']
        with self.assertRaises(ValueError):analyse(panels,ad,refs)
    def test_design_discloses_protocol_change_without_LC2_threshold_change(self):
        d=design();self.assertTrue(d['rounding_boundary']);self.assertFalse(d['LC2_acceptance_thresholds_changed']);self.assertEqual(d['within_run_tolerance'],2e-9);self.assertEqual(d['LC2_acceptance_budgets']['response'],.002);self.assertEqual(REFERENCES,('current_FP64','archived_FP64','archived_FP32'))
    def test_wrong_whole_input_hash_blocks_before_archive_payload(self):
        from inputs159 import load
        p,c=inputs(self.root/'inputs');c['pins']['C157']['sha256']='0'*64
        with self.assertRaises(ValueError):load(p,self.root/'bound',c)
if __name__=='__main__':unittest.main(verbosity=2)
