"""Fast path, arithmetic, protocol and failure accounting tests; all fixtures synthetic."""
import copy,io,tempfile,unittest
from unittest.mock import patch
from core157 import *
from worker157 import run
from fixture157 import context,Analytic,setup_inputs,loader
class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
    def test_fixed_design_counts_and_boundaries(self):
        d=design();self.assertEqual(d['counts']['model_forwards'],26);self.assertEqual(d['counts']['scalar_backward_calls'],16);self.assertFalse(d['LC2_closed']);self.assertEqual(len(FOCUS),2)
    def test_exact_chord_endpoints_and_scalar_direction(self):
        d=context()['data'];sid=FOCUS[0];z=d['points']['zero__'+sid];p=d['points']['plus_one__'+sid]
        np.testing.assert_array_equal(chord(z,p,0.),z.astype(float));np.testing.assert_array_equal(chord(z,p,1.),p.astype(float));self.assertEqual(chord(z,p,.25).dtype,np.float64)
    def test_unregistered_bool_or_float32_wrong_shape_blocks(self):
        z=np.zeros(DIM,np.float32)
        for a in [True,.125,np.nan]:
            with self.assertRaises(ValueError):chord(z,z,a)
        with self.assertRaises(ValueError):chord(z[:6],z,0.)
        with self.assertRaises(ValueError):chord(z.astype(float),z,0.)
    def test_all14_frozen_and_original_path_difference_retained(self):
        d=context()['data'];d['points']['minus_quarter__'+FOCUS[0]][0]+=np.float32(1e-5);dump(self.root/'DESIGN_FROZEN_BEFORE_MODEL.json',design());points,dirs,requests,seals=freeze(d,self.root)
        self.assertEqual(len(points),14);self.assertEqual(len(requests),24);rows=clean_json((self.root/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json').read_bytes());self.assertTrue(any(r['original_point_difference_Linf']>1e-7 for r in rows));self.assertTrue(all(sha(self.root/n)==h for n,h in seals.items()))
    def test_zero_direction_has_no_arbitrary_replacement(self):
        d=context()['data'];sid=FOCUS[0];d['points']['plus_one__'+sid]=d['points']['zero__'+sid].copy();dump(self.root/'DESIGN_FROZEN_BEFORE_MODEL.json',design())
        with self.assertRaises(ValueError):freeze(d,self.root)
    def test_fd_sign_scale_quadratic_and_all_scales(self):
        records={n:dict(absolute_response=(np.arange(6)*.001+a*.00001+a*a*.0001).tolist()) for n,a in LABELS.items()};ad=dict(coordinate_derivatives=[.00001]*6,response_loss_derivatives=dict(local=0.,shared=0.));r=analyse(records,ad,np.zeros(6),np.zeros(6));self.assertEqual(len(r['scales']),3)
        for row in r['scales']:self.assertLess(row['central_minus_AD_Linf'],1e-16)
        del records['minus_one']
        with self.assertRaises(ValueError):analyse(records,ad,np.zeros(6),np.zeros(6))
    def test_reference_cancels_coordinate_not_necessarily_loss(self):
        r=np.array([.002,0,0,0,0,0]);l=losses(r,np.zeros(6),np.array([.0005,0,0,0,0,0]));self.assertNotEqual(l['local'],l['shared'])
    def test_hinge_ties_and_signs_are_not_hidden(self):
        r=describe([.002,-.002,0,0,0,0]);self.assertEqual(r['max_coordinates'],[0,1]);self.assertTrue(r['max_coordinate_tie']);self.assertTrue(r['hinge_active'])
    def run_fixture(self,factory=Analytic,loadfn=loader):
        paths,c=setup_inputs(self.root/'inputs')
        with contextlib.redirect_stdout(io.StringIO()):return run(paths,self.root/'out',c,factory,loadfn)
    def test_full_synthetic_protocol_preserves_all_denominators(self):
        s=self.run_fixture();self.assertTrue(s['complete'],s['errors']);self.assertEqual(s['new_model_forwards'],26);self.assertEqual(s['new_backward_calls'],16);self.assertFalse(s['real_model_execution_claim']);self.assertFalse(s['LC2_closed']);self.assertEqual(len(s['diagnostics']),2)
    def test_replay_failure_blocks_before_any_AD(self):
        class Bad(Analytic):
            def measure(self,sid,n):r=super().measure(sid,n);r['absolute_response'][0]+=.01;return r
        s=self.run_fixture(Bad);self.assertFalse(s['complete']);self.assertEqual(s['new_backward_calls'],0);self.assertTrue(any('endpoint replay differs' in e for e in s['errors']))
    def test_wrong_total_never_completes(self):
        class Wrong(Analytic):
            def ad(self,sid):r=super().ad(sid);self.calls['backward_calls']-=1;return r
        s=self.run_fixture(Wrong);self.assertFalse(s['complete']);self.assertFalse(s['LC2_closed']);self.assertTrue(any('All16 scalar reverse calls' in e for e in s['errors']))
    def test_runtime_mutation_of_sealed_point_blocks(self):
        class Changed(Analytic):
            def ad(self,sid):r=super().ad(sid);self.points[(sid,'minus_one')][0]+=.01;return r
        s=self.run_fixture(Changed);self.assertFalse(s['complete']);self.assertTrue(any('Actual fixed points changed' in e for e in s['errors']))
    def test_bound_input_change_never_completes(self):
        paths,c=setup_inputs(self.root/'inputs')
        def changed(pp,out,cc,budget):pp['C155'].write_bytes(b'changed');return loader(pp,out,cc,budget)
        with contextlib.redirect_stdout(io.StringIO()):s=run(paths,self.root/'out',c,Analytic,changed)
        self.assertFalse(s['complete']);self.assertFalse(s['original_inputs_unchanged']['C155'])
    def test_missing_input_hash_blocks_before_payload(self):
        from inputs157 import load
        paths,c=setup_inputs(self.root/'inputs');c['pins']['C155']['sha256']='0'*64
        with self.assertRaises(ValueError):load(paths,self.root/'bound',c)
if __name__=='__main__':unittest.main(verbosity=2)
